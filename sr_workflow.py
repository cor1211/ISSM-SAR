from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml
from dotenv import load_dotenv

from runtime_logging import configure_root_logging, emit_runtime_log, resolve_runtime_log_level
from sr_publish import (
    build_publish_plan,
    build_requests_session,
    build_s3_client_from_env,
    execute_publish,
    run_preflight,
)


def _resolve_pipeline_script_path() -> Path:
    container_path = Path("/app/sar_pipeline.py")
    if container_path.exists():
        return container_path
    return Path(__file__).resolve().with_name("sar_pipeline.py")


DEFAULT_PIPELINE_CONFIG = "/app/config/pipeline_config_stac_runtime.yaml"


class WorkflowError(RuntimeError):
    pass


class _TeeTextIO:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass
class WorkflowSettings:
    publish_enabled: bool = True
    publish_execute: bool = True
    publish_overwrite: bool = True
    publish_timeout_seconds: int = 30
    publish_continue_on_error: bool = False
    fail_on_no_outputs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publish_enabled": self.publish_enabled,
            "publish_execute": self.publish_execute,
            "publish_overwrite": self.publish_overwrite,
            "publish_timeout_seconds": self.publish_timeout_seconds,
            "publish_continue_on_error": self.publish_continue_on_error,
            "fail_on_no_outputs": self.fail_on_no_outputs,
        }


def emit_workflow_log(level: int, message: str, **fields: Any) -> None:
    emit_runtime_log("sr_workflow", level, message, **fields)


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise WorkflowError(f"Invalid boolean value: {value!r}")


def _parse_int(value: Optional[str], *, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def parse_workflow_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run SR pipeline and publish all produced outputs in one workflow.",
        add_help=True,
    )
    parser.add_argument(
        "--publish",
        dest="publish_enabled",
        action="store_true",
        default=None,
        help="Enable publish stage after pipeline completes successfully.",
    )
    parser.add_argument(
        "--no-publish",
        dest="publish_enabled",
        action="store_false",
        default=None,
        help="Disable publish stage and only run the pipeline.",
    )
    parser.add_argument(
        "--publish-preflight-only",
        action="store_true",
        help="Run publish preflight only instead of execute mode.",
    )
    parser.add_argument(
        "--publish-overwrite",
        action="store_true",
        default=None,
        help="Allow overwrite during publish.",
    )
    parser.add_argument(
        "--publish-timeout-seconds",
        type=int,
        default=None,
        help="HTTP timeout for STAC publish requests.",
    )
    parser.add_argument(
        "--publish-continue-on-error",
        action="store_true",
        default=None,
        help="Continue publishing remaining outputs if one publish fails.",
    )
    parser.add_argument(
        "--fail-on-no-outputs",
        action="store_true",
        default=None,
        help="Exit non-zero if pipeline finishes without any publishable outputs.",
    )
    parser.add_argument(
        "--pipeline-help",
        action="store_true",
        help="Show the underlying pipeline help and exit.",
    )
    args, pipeline_args = parser.parse_known_args(list(argv or sys.argv[1:]))
    return args, pipeline_args


def load_workflow_settings(args: argparse.Namespace, *, env: Optional[Mapping[str, str]] = None) -> WorkflowSettings:
    values = dict(os.environ if env is None else env)
    settings = WorkflowSettings(
        publish_enabled=_parse_bool(values.get("WORKFLOW_PUBLISH_ENABLED"), default=True),
        publish_execute=_parse_bool(values.get("WORKFLOW_PUBLISH_EXECUTE"), default=True),
        publish_overwrite=_parse_bool(values.get("WORKFLOW_PUBLISH_OVERWRITE"), default=True),
        publish_timeout_seconds=_parse_int(values.get("WORKFLOW_PUBLISH_TIMEOUT_SECONDS"), default=30),
        publish_continue_on_error=_parse_bool(values.get("WORKFLOW_PUBLISH_CONTINUE_ON_ERROR"), default=False),
        fail_on_no_outputs=_parse_bool(values.get("WORKFLOW_FAIL_ON_NO_OUTPUTS"), default=False),
    )
    if args.publish_enabled is not None:
        settings.publish_enabled = bool(args.publish_enabled)
    if args.publish_preflight_only:
        settings.publish_enabled = True
        settings.publish_execute = False
    if args.publish_overwrite is not None:
        settings.publish_overwrite = bool(args.publish_overwrite)
    if args.publish_timeout_seconds is not None:
        settings.publish_timeout_seconds = int(args.publish_timeout_seconds)
    if args.publish_continue_on_error is not None:
        settings.publish_continue_on_error = bool(args.publish_continue_on_error)
    if args.fail_on_no_outputs is not None:
        settings.fail_on_no_outputs = bool(args.fail_on_no_outputs)
    return settings


def _extract_option(args: Sequence[str], name: str) -> Optional[str]:
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return str(args[index + 1])
        if arg.startswith(f"{name}="):
            return str(arg.split("=", 1)[1])
    return None


def resolve_pipeline_config_path(pipeline_args: Sequence[str]) -> str:
    return _extract_option(pipeline_args, "--config") or DEFAULT_PIPELINE_CONFIG


def resolve_output_root(pipeline_args: Sequence[str]) -> Path:
    explicit = _extract_option(pipeline_args, "--output-dir")
    if explicit:
        return Path(explicit)
    config_path = Path(resolve_pipeline_config_path(pipeline_args))
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        output_root = (((loaded.get("output") or {}) if isinstance(loaded.get("output"), dict) else {}).get("root_dir"))
        if output_root:
            return Path(str(output_root))
    return Path("/app/runs")


def snapshot_job_dirs(jobs_root: Path) -> Set[Path]:
    if not jobs_root.exists():
        return set()
    return {entry.resolve() for entry in jobs_root.iterdir() if entry.is_dir()}


def discover_created_job_dir(jobs_root: Path, before: Set[Path], *, started_at_epoch: float) -> Path:
    after = snapshot_job_dirs(jobs_root)
    created = sorted(after - before)
    if len(created) == 1:
        return created[0]
    if len(created) > 1:
        created.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        emit_workflow_log(
            logging.WARNING,
            "Multiple new job directories were detected; selecting the most recent one",
            jobs_root=str(jobs_root),
            candidate_count=len(created),
            selected_job_dir=str(created[0]),
        )
        return created[0]

    recent = [
        path
        for path in after
        if (path / "summary.json").exists() and path.stat().st_mtime >= (started_at_epoch - 1.0)
    ]
    if recent:
        recent.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return recent[0]
    raise WorkflowError(f"Could not locate the newly created runtime job under {jobs_root}.")


def load_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        raise WorkflowError(f"summary.json not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def collect_publishable_item_json_paths(summary: Mapping[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for aoi in list(summary.get("aois") or []):
        for period in list((aoi.get("periods") or [])):
            artifacts = period.get("artifacts") or {}
            item_json_path = artifacts.get("item_json_path")
            if not item_json_path:
                continue
            candidate = Path(str(item_json_path))
            if candidate.exists():
                paths.append(candidate)
    unique_paths: List[Path] = []
    seen: Set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def resolve_save_debug_artifacts_for_summary(summary: Mapping[str, Any]) -> Optional[bool]:
    candidates: List[Any] = []

    resolved_config = summary.get("resolved_config") if isinstance(summary, Mapping) else None
    if isinstance(resolved_config, Mapping):
        output_cfg = resolved_config.get("output")
        if isinstance(output_cfg, Mapping):
            candidates.append(output_cfg.get("save_debug_artifacts"))

    run_config = summary.get("run_config") if isinstance(summary, Mapping) else None
    if isinstance(run_config, Mapping):
        candidates.append(run_config.get("save_debug_artifacts"))

    for aoi in list(summary.get("aois") or []):
        for period in list((aoi.get("periods") or [])):
            period_run_config = period.get("run_config")
            if isinstance(period_run_config, Mapping):
                candidates.append(period_run_config.get("save_debug_artifacts"))

    for candidate in candidates:
        parsed = _coerce_optional_bool(candidate)
        if parsed is not None:
            return parsed
    return None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _safe_cleanup_fragment(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return normalized or "artifact"


def _build_cleanup_report(
    *,
    requested: bool,
    status: str,
    save_debug_artifacts: Optional[bool],
    policy: str,
    skipped_reason: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "requested": bool(requested),
        "status": status,
        "policy": policy,
        "save_debug_artifacts": save_debug_artifacts,
        "deleted_file_count": 0,
        "deleted_roles": [],
        "deleted_paths": [],
        "pruned_dirs": [],
        "bytes_freed": 0,
        "warnings": [],
    }
    if skipped_reason:
        report["skipped_reason"] = skipped_reason
    return report


def _prune_empty_dirs(start_dir: Optional[Path], *, stop_at: Path) -> List[str]:
    if start_dir is None:
        return []
    removed: List[str] = []
    stop_at_resolved = stop_at.resolve()
    current = start_dir
    while True:
        try:
            current_resolved = current.resolve()
        except Exception:
            break
        if current_resolved == stop_at_resolved:
            break
        try:
            current.rmdir()
        except OSError:
            break
        removed.append(str(current_resolved))
        parent = current.parent
        if parent == current:
            break
        current = parent
    return removed


def _path_tree_stats(path: Path) -> Tuple[int, int]:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    if not resolved.exists():
        return 0, 0
    if resolved.is_file():
        return 1, int(resolved.stat().st_size)
    if not resolved.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for child in resolved.rglob("*"):
        try:
            if child.is_file():
                file_count += 1
                total_bytes += int(child.stat().st_size)
        except Exception:
            continue
    return file_count, total_bytes


def cleanup_published_local_artifacts(
    *,
    plan: Any,
    job_dir: Path,
) -> Dict[str, Any]:
    cleanup = _build_cleanup_report(
        requested=True,
        status="ok",
        save_debug_artifacts=False,
        policy="cleanup_after_publish_success",
    )

    job_dir_resolved = job_dir.resolve()
    item_json_path = Path(str(getattr(plan, "item_json_path"))).resolve()
    output_dir = item_json_path.parent
    period_dir = output_dir.parent
    if not _path_within(output_dir, job_dir_resolved) or not _path_within(period_dir, job_dir_resolved):
        cleanup["status"] = "warning"
        cleanup["skipped_reason"] = "unsafe_output_dir_outside_job_dir"
        cleanup["warnings"].append("Output directory is outside the workflow job directory; skipping cleanup.")
        return cleanup

    trash_root = period_dir / ".cleanup_trash" / _safe_cleanup_fragment(getattr(plan, "item_id", item_json_path.stem))
    cleanup["trash_dir"] = str(trash_root)
    moved: List[Tuple[str, Path, int, Path]] = []

    for artifact in list(getattr(plan, "artifacts", []) or []):
        role = str(getattr(artifact, "role", "") or "").strip()
        if role not in {"vv", "vh", "item_json"}:
            continue
        local_path = Path(str(getattr(artifact, "local_path"))).resolve()
        if not local_path.exists():
            cleanup["warnings"].append(f"Skipping cleanup for {role}: local artifact does not exist.")
            continue
        if local_path.is_symlink():
            cleanup["warnings"].append(f"Skipping cleanup for {role}: symlink paths are not allowed.")
            continue
        if not local_path.is_file():
            cleanup["warnings"].append(f"Skipping cleanup for {role}: expected a regular file.")
            continue
        if local_path.parent != output_dir:
            cleanup["warnings"].append(f"Skipping cleanup for {role}: artifact is not located under the output directory.")
            continue
        if not _path_within(local_path, job_dir_resolved):
            cleanup["warnings"].append(f"Skipping cleanup for {role}: artifact is outside the workflow job directory.")
            continue

        trash_root.mkdir(parents=True, exist_ok=True)
        trash_path = trash_root / f"{role}__{local_path.name}"
        try:
            size_bytes = local_path.stat().st_size
            local_path.replace(trash_path)
            moved.append((role, trash_path, size_bytes, local_path))
        except Exception as exc:
            cleanup["warnings"].append(f"Failed to move {role} into cleanup trash: {exc}")

    for role, trash_path, size_bytes, original_path in moved:
        try:
            trash_path.unlink()
            cleanup["deleted_file_count"] += 1
            cleanup["bytes_freed"] += int(size_bytes)
            cleanup["deleted_roles"].append(role)
            cleanup["deleted_paths"].append(str(original_path))
        except Exception as exc:
            cleanup["warnings"].append(f"Failed to delete trashed artifact for {role}: {exc}")

    debug_dir = period_dir / "debug"
    if debug_dir.exists():
        if debug_dir.is_symlink():
            cleanup["warnings"].append("Skipping cleanup for debug_dir: symlink paths are not allowed.")
        elif not debug_dir.is_dir():
            cleanup["warnings"].append("Skipping cleanup for debug_dir: expected a directory.")
        elif not _path_within(debug_dir, job_dir_resolved):
            cleanup["warnings"].append("Skipping cleanup for debug_dir: directory is outside the workflow job directory.")
        else:
            trash_root.mkdir(parents=True, exist_ok=True)
            trash_debug_dir = trash_root / "debug"
            try:
                file_count, size_bytes = _path_tree_stats(debug_dir)
                debug_dir.replace(trash_debug_dir)
                shutil.rmtree(trash_debug_dir)
                cleanup["deleted_file_count"] += int(file_count)
                cleanup["bytes_freed"] += int(size_bytes)
                cleanup["deleted_roles"].append("debug_dir")
                cleanup["deleted_paths"].append(str(debug_dir.resolve()))
            except Exception as exc:
                cleanup["warnings"].append(f"Failed to cleanup debug_dir: {exc}")

    cleanup["pruned_dirs"].extend(_prune_empty_dirs(output_dir, stop_at=job_dir_resolved))
    cleanup["pruned_dirs"].extend(_prune_empty_dirs(trash_root, stop_at=job_dir_resolved))
    cleanup["pruned_dirs"].extend(_prune_empty_dirs(period_dir, stop_at=job_dir_resolved))

    if cleanup["warnings"]:
        cleanup["status"] = "warning"
    cleanup["trash_retained"] = trash_root.exists()
    return cleanup


def build_post_publish_cleanup_report(
    *,
    summary: Mapping[str, Any],
    settings: WorkflowSettings,
    publish_report: Mapping[str, Any],
    plan: Optional[Any] = None,
    job_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    save_debug_artifacts = resolve_save_debug_artifacts_for_summary(summary)
    if not settings.publish_execute:
        return _build_cleanup_report(
            requested=False,
            status="skipped",
            save_debug_artifacts=save_debug_artifacts,
            policy="keep_local_outputs",
            skipped_reason="preflight_mode",
        )
    if not bool(publish_report.get("published")):
        return _build_cleanup_report(
            requested=False,
            status="skipped",
            save_debug_artifacts=save_debug_artifacts,
            policy="keep_local_outputs",
            skipped_reason="publish_not_successful",
        )
    if save_debug_artifacts is None:
        return _build_cleanup_report(
            requested=False,
            status="skipped",
            save_debug_artifacts=save_debug_artifacts,
            policy="keep_local_outputs",
            skipped_reason="save_debug_artifacts_unknown",
        )
    if save_debug_artifacts:
        return _build_cleanup_report(
            requested=False,
            status="skipped",
            save_debug_artifacts=save_debug_artifacts,
            policy="keep_local_outputs",
            skipped_reason="save_debug_artifacts_enabled",
        )
    if plan is None or job_dir is None:
        return _build_cleanup_report(
            requested=False,
            status="warning",
            save_debug_artifacts=save_debug_artifacts,
            policy="cleanup_after_publish_success",
            skipped_reason="missing_cleanup_context",
        )
    return cleanup_published_local_artifacts(plan=plan, job_dir=job_dir)


@contextmanager
def capture_workflow_job_logs(job_dir: Path):
    log_path = job_dir / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_fp:
        if log_path.exists() and log_path.stat().st_size > 0:
            log_fp.write("\n")
        tee_stdout = _TeeTextIO(sys.stdout, log_fp)
        tee_stderr = _TeeTextIO(sys.stderr, log_fp)
        root_logger = logging.getLogger()
        redirected_handlers: List[Tuple[logging.StreamHandler, Any]] = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                old_stream = handler.setStream(tee_stderr)
                redirected_handlers.append((handler, old_stream))
        try:
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                yield log_path
        finally:
            for handler, old_stream in redirected_handlers:
                handler.setStream(old_stream)


def write_summary_json(path: Path, summary: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resolved_path_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return str(value)


def attach_period_publish_result(
    summary: Dict[str, Any],
    *,
    item_json_path: Path,
    publish_entry: Mapping[str, Any],
) -> bool:
    target = _resolved_path_text(item_json_path)
    for aoi in list(summary.get("aois") or []):
        for period in list((aoi.get("periods") or [])):
            artifacts = period.get("artifacts") or {}
            candidate = _resolved_path_text(artifacts.get("item_json_path"))
            if candidate and candidate == target:
                period["publish"] = dict(publish_entry)
                return True
    return False


def run_pipeline_subprocess(pipeline_args: Sequence[str]) -> int:
    pipeline_script = _resolve_pipeline_script_path()
    command = [sys.executable, str(pipeline_script), *pipeline_args]
    emit_workflow_log(logging.INFO, "Launching pipeline subprocess", command=command)
    completed = subprocess.run(command, check=False)
    emit_workflow_log(logging.INFO, "Pipeline subprocess finished", exit_code=completed.returncode)
    return int(completed.returncode)


def publish_outputs_for_summary(
    *,
    summary: Dict[str, Any],
    job_dir: Path,
    settings: WorkflowSettings,
) -> Tuple[int, Path]:
    summary_path = Path(str(summary.get("summary_json") or job_dir / "summary.json"))
    item_json_paths = collect_publishable_item_json_paths(summary)
    save_debug_artifacts = resolve_save_debug_artifacts_for_summary(summary)
    publish_settings = settings.to_dict()
    publish_settings["save_debug_artifacts"] = save_debug_artifacts
    publish_settings["cleanup_after_publish_success"] = bool(
        settings.publish_enabled and settings.publish_execute and save_debug_artifacts is False
    )
    workflow_report: Dict[str, Any] = {
        "status": "ok",
        "job_dir": str(job_dir),
        "summary_json": str(summary_path),
        "publish_settings": publish_settings,
        "publishable_item_count": len(item_json_paths),
        "results": [],
        "cleanup_summary": {
            "deleted_file_count": 0,
            "bytes_freed": 0,
            "cleaned_item_count": 0,
            "warning_item_count": 0,
            "skipped_item_count": 0,
        },
    }
    summary["workflow_publish"] = workflow_report

    if not settings.publish_enabled:
        emit_workflow_log(logging.INFO, "Publish stage disabled by workflow settings")
        workflow_report["publish_enabled"] = False
        write_summary_json(summary_path, summary)
        return 0, summary_path

    if not item_json_paths:
        emit_workflow_log(logging.WARNING, "Workflow found no publishable SR outputs", job_dir=str(job_dir))
        workflow_report["publish_enabled"] = True
        workflow_report["published_count"] = 0
        workflow_report["failed_count"] = 0
        if settings.fail_on_no_outputs:
            workflow_report["status"] = "error"
            workflow_report["error"] = "No publishable outputs were produced by the pipeline."
            write_summary_json(summary_path, summary)
            return 1, summary_path
        write_summary_json(summary_path, summary)
        return 0, summary_path

    session = build_requests_session()
    s3_client = build_s3_client_from_env()
    failed_count = 0
    published_count = 0
    with capture_workflow_job_logs(job_dir):
        emit_workflow_log(
            logging.INFO,
            "Starting workflow publish stage",
            publishable_item_count=len(item_json_paths),
            publish_execute=settings.publish_execute,
            publish_overwrite=settings.publish_overwrite,
            publish_timeout_seconds=settings.publish_timeout_seconds,
            publish_continue_on_error=settings.publish_continue_on_error,
            save_debug_artifacts=save_debug_artifacts,
            cleanup_after_publish_success=bool(
                settings.publish_enabled and settings.publish_execute and save_debug_artifacts is False
            ),
        )
        for item_json_path in item_json_paths:
            try:
                plan = build_publish_plan(item_json_path=item_json_path)
                emit_workflow_log(
                    logging.INFO,
                    "Publishing SR output from workflow",
                    item_json_path=str(item_json_path),
                    item_id=plan.item_id,
                    collection_id=plan.collection_id,
                    execute=settings.publish_execute,
                )
                if settings.publish_execute:
                    publish_report = execute_publish(
                        plan=plan,
                        session=session,
                        s3_client=s3_client,
                        overwrite=settings.publish_overwrite,
                        timeout_seconds=settings.publish_timeout_seconds,
                    )
                    publish_report["status"] = "ok"
                else:
                    publish_report = run_preflight(
                        plan=plan,
                        session=session,
                        s3_client=s3_client,
                        overwrite=settings.publish_overwrite,
                        timeout_seconds=settings.publish_timeout_seconds,
                    )
                    publish_report["published"] = False
                    publish_report["status"] = "ok"
                cleanup_report = build_post_publish_cleanup_report(
                    summary=summary,
                    settings=settings,
                    publish_report=publish_report,
                    plan=plan,
                    job_dir=job_dir,
                )
                result_entry = {
                    "item_json_path": str(item_json_path),
                    "item_id": plan.item_id,
                    "collection_id": plan.collection_id,
                    "status": "ok",
                    "published": bool(publish_report.get("published")),
                    "mode": "execute" if settings.publish_execute else "preflight",
                    "cleanup": cleanup_report,
                }
                workflow_report["results"].append(result_entry)
                cleanup_summary = workflow_report["cleanup_summary"]
                cleanup_summary["deleted_file_count"] += int(cleanup_report.get("deleted_file_count") or 0)
                cleanup_summary["bytes_freed"] += int(cleanup_report.get("bytes_freed") or 0)
                if cleanup_report.get("requested"):
                    if cleanup_report.get("status") == "ok":
                        cleanup_summary["cleaned_item_count"] += 1
                    else:
                        cleanup_summary["warning_item_count"] += 1
                else:
                    cleanup_summary["skipped_item_count"] += 1
                attach_period_publish_result(
                    summary,
                    item_json_path=item_json_path,
                    publish_entry={
                        **result_entry,
                        "details": publish_report,
                    },
                )
                published_count += 1
                workflow_report["published_count"] = published_count
                workflow_report["failed_count"] = failed_count
                write_summary_json(summary_path, summary)
                emit_workflow_log(
                    logging.INFO,
                    "Workflow publish completed for item",
                    item_id=plan.item_id,
                    published=bool(publish_report.get("published")),
                )
                if cleanup_report.get("requested"):
                    emit_workflow_log(
                        logging.INFO if cleanup_report.get("status") == "ok" else logging.WARNING,
                        "Post-publish cleanup processed for item",
                        item_id=plan.item_id,
                        cleanup_status=cleanup_report.get("status"),
                        deleted_file_count=cleanup_report.get("deleted_file_count"),
                        bytes_freed=cleanup_report.get("bytes_freed"),
                        warnings=len(cleanup_report.get("warnings") or []),
                    )
                else:
                    emit_workflow_log(
                        logging.INFO,
                        "Post-publish cleanup skipped for item",
                        item_id=plan.item_id,
                        skipped_reason=cleanup_report.get("skipped_reason"),
                    )
            except Exception as exc:
                failed_count += 1
                cleanup_report = _build_cleanup_report(
                    requested=False,
                    status="skipped",
                    save_debug_artifacts=save_debug_artifacts,
                    policy="keep_local_outputs",
                    skipped_reason="publish_not_successful",
                )
                result_entry = {
                    "item_json_path": str(item_json_path),
                    "status": "error",
                    "published": False,
                    "mode": "execute" if settings.publish_execute else "preflight",
                    "error": str(exc),
                    "cleanup": cleanup_report,
                }
                workflow_report["results"].append(result_entry)
                workflow_report["cleanup_summary"]["skipped_item_count"] += 1
                attach_period_publish_result(
                    summary,
                    item_json_path=item_json_path,
                    publish_entry=result_entry,
                )
                emit_workflow_log(
                    logging.ERROR,
                    "Workflow publish failed for item",
                    item_json_path=str(item_json_path),
                    error=str(exc),
                )
                workflow_report["published_count"] = published_count
                workflow_report["failed_count"] = failed_count
                write_summary_json(summary_path, summary)
                if not settings.publish_continue_on_error:
                    workflow_report["status"] = "error"
                    write_summary_json(summary_path, summary)
                    return 1, summary_path

    workflow_report["published_count"] = published_count
    workflow_report["failed_count"] = failed_count
    if failed_count:
        workflow_report["status"] = "error"
    write_summary_json(summary_path, summary)
    return (1 if failed_count else 0), summary_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args, pipeline_args = parse_workflow_args(argv)
    if args.pipeline_help:
        pipeline_script = _resolve_pipeline_script_path()
        completed = subprocess.run([sys.executable, str(pipeline_script), "--help"], check=False)
        return int(completed.returncode)
    if not pipeline_args:
        raise SystemExit("No pipeline arguments were provided. Pass normal sar_pipeline.py arguments to the workflow command.")

    effective_log_level, _ = resolve_runtime_log_level(cli_level=None, config=None)
    configure_root_logging(effective_log_level)
    settings = load_workflow_settings(args)
    output_root = resolve_output_root(pipeline_args)
    jobs_root = output_root / "jobs"
    before = snapshot_job_dirs(jobs_root)
    started_at_epoch = time.time()

    emit_workflow_log(
        logging.INFO,
        "Starting one-shot SR workflow",
        output_root=str(output_root),
        publish_enabled=settings.publish_enabled,
        publish_execute=settings.publish_execute,
        publish_overwrite=settings.publish_overwrite,
        publish_timeout_seconds=settings.publish_timeout_seconds,
        publish_continue_on_error=settings.publish_continue_on_error,
        fail_on_no_outputs=settings.fail_on_no_outputs,
    )

    pipeline_exit_code = run_pipeline_subprocess(pipeline_args)
    if pipeline_exit_code != 0:
        emit_workflow_log(logging.ERROR, "Workflow stopped because pipeline failed", exit_code=pipeline_exit_code)
        return pipeline_exit_code

    job_dir = discover_created_job_dir(jobs_root, before, started_at_epoch=started_at_epoch)
    summary_path = job_dir / "summary.json"
    summary = load_summary(summary_path)
    emit_workflow_log(logging.INFO, "Resolved workflow runtime job", job_dir=str(job_dir), summary_json=str(summary_path))

    publish_exit_code, updated_summary_path = publish_outputs_for_summary(
        summary=summary,
        job_dir=job_dir,
        settings=settings,
    )
    if publish_exit_code != 0:
        emit_workflow_log(
            logging.ERROR,
            "Workflow finished with publish errors",
            summary_json=str(updated_summary_path),
            exit_code=publish_exit_code,
        )
        return publish_exit_code

    emit_workflow_log(
        logging.INFO,
        "Workflow completed successfully",
        summary_json=str(updated_summary_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

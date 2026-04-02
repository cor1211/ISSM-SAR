from __future__ import annotations

import argparse
import json
import logging
import os
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
    publish_enabled: bool = False
    publish_execute: bool = False
    publish_overwrite: bool = False
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
        publish_enabled=_parse_bool(values.get("WORKFLOW_PUBLISH_ENABLED"), default=False),
        publish_execute=_parse_bool(values.get("WORKFLOW_PUBLISH_EXECUTE"), default=False),
        publish_overwrite=_parse_bool(values.get("WORKFLOW_PUBLISH_OVERWRITE"), default=False),
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
    command = [sys.executable, "/app/sar_pipeline.py", *pipeline_args]
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
    workflow_report: Dict[str, Any] = {
        "status": "ok",
        "job_dir": str(job_dir),
        "summary_json": str(summary_path),
        "publish_settings": settings.to_dict(),
        "publishable_item_count": len(item_json_paths),
        "results": [],
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
                result_entry = {
                    "item_json_path": str(item_json_path),
                    "item_id": plan.item_id,
                    "collection_id": plan.collection_id,
                    "status": "ok",
                    "published": bool(publish_report.get("published")),
                    "mode": "execute" if settings.publish_execute else "preflight",
                }
                workflow_report["results"].append(result_entry)
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
            except Exception as exc:
                failed_count += 1
                result_entry = {
                    "item_json_path": str(item_json_path),
                    "status": "error",
                    "published": False,
                    "mode": "execute" if settings.publish_execute else "preflight",
                    "error": str(exc),
                }
                workflow_report["results"].append(result_entry)
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
        completed = subprocess.run([sys.executable, "/app/sar_pipeline.py", "--help"], check=False)
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

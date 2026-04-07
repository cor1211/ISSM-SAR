from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from db_aoi_source import inspect_database_settings
from pipeline_support.json_support import compact_jsonable
from query_stac_download import DEFAULT_COLLECTION, DEFAULT_STAC_API
from runtime_env_overrides import apply_pipeline_env_overrides
from runtime_logging import detect_s3_credential_source, normalize_log_level_name

WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE = "stac_trainlike_composite"
WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE = "gee_trainlike_composite"
SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD = "representative_calendar_period"
SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC = "componentized_parent_mosaic"


def normalize_workflow_mode(value: Any, *, default: str = "") -> str:
    return str(value or default).strip().lower()


def normalize_selection_strategy(
    value: Any,
    *,
    default: str = SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD,
) -> str:
    return str(value or default).strip().lower()


def is_representative_composite_workflow_mode(value: Any) -> bool:
    normalized = normalize_workflow_mode(value, default="")
    return normalized in {WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE, WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE}


def is_canonical_selection_strategy(value: Any) -> bool:
    return (
        normalize_selection_strategy(
            value,
            default=SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD,
        )
        == SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD
    )


def resolve_workflow_backend(value: Any) -> Optional[str]:
    normalized = normalize_workflow_mode(value, default="")
    if normalized == WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE:
        return "stac"
    if normalized == WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE:
        return "gee"
    return None


def resolve_spatial_strategy(train_cfg: Optional[Dict[str, Any]]) -> str:
    cfg = train_cfg or {}
    if not bool(cfg.get("componentize_seed_intersections", True)):
        raise ValueError(
            "Canonical representative runtime requires "
            "trainlike.componentize_seed_intersections=true; whole_aoi is no longer supported."
        )
    if not bool(cfg.get("component_parent_mosaic", True)):
        raise ValueError(
            "Canonical representative runtime requires "
            "trainlike.component_parent_mosaic=true."
        )
    return SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC


def describe_pipeline_profile(
    *,
    workflow_mode: Any,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_workflow_mode = normalize_workflow_mode(workflow_mode, default="")
    representative_composite = is_representative_composite_workflow_mode(normalized_workflow_mode)
    selection_strategy = None
    canonical_selection_strategy = False
    if representative_composite:
        normalized_selection = normalize_selection_strategy((train_cfg or {}).get("selection_strategy"), default="")
        selection_strategy = normalized_selection or None
        canonical_selection_strategy = is_canonical_selection_strategy(selection_strategy)
    spatial_strategy = (
        resolve_spatial_strategy(train_cfg)
        if representative_composite and canonical_selection_strategy
        else None
    )
    runtime_family = "representative_composite" if representative_composite else "unknown"
    return compact_jsonable(
        {
            "workflow_backend": resolve_workflow_backend(normalized_workflow_mode),
            "runtime_family": runtime_family,
            "selection_strategy": selection_strategy,
            "spatial_strategy": spatial_strategy,
            "canonical_selection_strategy": canonical_selection_strategy if representative_composite else None,
        }
    )


def build_effective_runtime_settings(
    *,
    workflow_mode: Any = "",
    train_cfg: Dict[str, Any],
    infer_config: Optional[Dict[str, Any]],
    save_debug_artifacts: bool,
) -> Dict[str, Any]:
    infer_cfg = ((infer_config or {}).get("inference") or {}) if isinstance(infer_config, dict) else {}
    return compact_jsonable(
        {
            "min_scenes_per_half": int(train_cfg.get("min_scenes_per_half", 1)),
            "component_item_min_coverage": float(train_cfg.get("component_item_min_coverage", 1.0)),
            "component_min_area_ratio": float(train_cfg.get("component_min_area_ratio", 0.0)),
            "save_debug_artifacts": bool(save_debug_artifacts),
            "infer_device": (infer_config or {}).get("device") if isinstance(infer_config, dict) else None,
            "infer_patch_size": infer_cfg.get("patch_size"),
            "infer_overlap": infer_cfg.get("overlap"),
            "infer_batch_size": infer_cfg.get("batch_size"),
            "infer_amp": infer_cfg.get("use_amp"),
            "infer_blending": infer_cfg.get("gaussian_blend"),
        }
    )


def _safe_resolved_db_summary(env_path: str | Path) -> Dict[str, Any]:
    inspected = inspect_database_settings(env_path)
    resolved = inspected.get("resolved", {})
    return {
        "env_path": inspected.get("env_path"),
        "missing_keys": inspected.get("missing_keys", []),
        "pghost_present": bool((resolved.get("PGHOST") or {}).get("present")),
        "pgport_present": bool((resolved.get("PGPORT") or {}).get("present")),
        "pguser_present": bool((resolved.get("PGUSER") or {}).get("present")),
        "pgdatabase_present": bool((resolved.get("PGDATABASE") or {}).get("present")),
        "pgpassword_present": bool((resolved.get("PGPASSWORD") or {}).get("present")),
        "pghost": (resolved.get("PGHOST") or {}).get("value"),
        "pgport": (resolved.get("PGPORT") or {}).get("value"),
        "pguser": (resolved.get("PGUSER") or {}).get("value"),
        "pgdatabase": (resolved.get("PGDATABASE") or {}).get("value"),
    }


def build_startup_checks(
    *,
    config: Dict[str, Any],
    source_mode: str,
    db_env_path: Optional[str],
) -> Dict[str, Any]:
    stac_cfg = config.get("stac", {}) or {}
    db_enabled = source_mode in {"db_single", "db_batch"}
    db_summary = _safe_resolved_db_summary(db_env_path or ".env") if db_enabled else None
    s3_access_present = bool(os.getenv("S3_ACCESS_KEY"))
    s3_secret_present = bool(os.getenv("S3_SECRET_KEY"))
    checks = {
        "policy": "warn_early",
        "db": {
            "enabled": db_enabled,
            **(db_summary or {}),
        },
        "stac": {
            "url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
        },
        "s3": {
            "endpoint": os.getenv("S3_ENDPOINT"),
            "credential_source": detect_s3_credential_source(),
            "s3_access_key_present": s3_access_present,
            "s3_secret_key_present": s3_secret_present,
            "explicit_env_present": bool(s3_access_present and s3_secret_present),
        },
    }
    return compact_jsonable(checks)


def apply_runtime_env_overrides(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    overrides: List[Dict[str, Any]] = list(apply_pipeline_env_overrides(config))
    stac_cfg = config.setdefault("stac", {})

    stac_api_url = (os.getenv("STAC_API_URL") or "").strip()
    if stac_api_url:
        stac_cfg["url"] = stac_api_url
        overrides.append(
            {
                "target": "stac.url",
                "source": "env:STAC_API_URL",
            }
        )

    stac_collection = (os.getenv("STAC_COLLECTION") or os.getenv("STAC_COLLECTION_ID") or "").strip()
    if stac_collection:
        stac_cfg["collection"] = stac_collection
        overrides.append(
            {
                "target": "stac.collection",
                "source": ("env:STAC_COLLECTION" if os.getenv("STAC_COLLECTION") else "env:STAC_COLLECTION_ID"),
            }
        )

    if overrides:
        config.setdefault("_runtime", {})["env_overrides"] = compact_jsonable(overrides)
    return overrides


def model_trainlike_semantics() -> Dict[str, Any]:
    return {
        "t1_label": "S1T1",
        "t2_label": "S1T2",
        "t1_role": "post/future window relative to anchor",
        "t2_role": "pre/past window relative to anchor",
        "matches_training_semantics": True,
        "note": "This matches the training convention: model(S1T1_post_future, S1T2_pre_past).",
    }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    dst = Path(path)
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def remove_dir_if_empty(path: str | Path) -> None:
    candidate = Path(path)
    try:
        candidate.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


class _TeeTextIO:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self._streams and getattr(self._streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        if not self._streams:
            raise OSError("No underlying stream available.")
        return self._streams[0].fileno()


@contextmanager
def capture_runtime_job_logs(job_dir: str | Path) -> Iterator[Path]:
    log_path = job_log_path(job_dir)
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


def _storage_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def build_storage_job_id(workflow_mode: str, source_mode: str, *, now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}__{_storage_token(workflow_mode)}__{_storage_token(source_mode)}"


def run_meta_dir(run_dir: str | Path) -> Path:
    return ensure_dir(Path(run_dir))


def period_meta_dir(period_dir: str | Path) -> Path:
    return ensure_dir(Path(period_dir))


def job_meta_dir(job_dir: str | Path) -> Path:
    return ensure_dir(Path(job_dir))


def inputs_dir(base_dir: str | Path) -> Path:
    return ensure_dir(Path(base_dir) / "inputs")


def intermediate_dir(base_dir: str | Path) -> Path:
    return ensure_dir(Path(base_dir) / "intermediate")


def save_debug_artifacts_enabled(config: Dict[str, Any]) -> bool:
    out_cfg = config.get("output", {}) or {}
    return bool(out_cfg.get("save_debug_artifacts", False))


def resolve_optional_debug_dir(
    *,
    persist: bool,
    persistent_path: str | Path,
    transient_path: str | Path,
) -> Path:
    target = Path(persistent_path) if persist else Path(transient_path)
    return ensure_dir(target)


def infer_job_dir_from_runtime_path(path: str | Path) -> Path:
    cursor = Path(path)
    for parent in (cursor, *cursor.parents):
        if parent.name == "aois":
            return parent.parent
        if parent.name == "jobs":
            return parent
    return cursor


def aoi_log_path(run_dir: str | Path) -> Path:
    return job_log_path(infer_job_dir_from_runtime_path(run_dir))


def job_log_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / "job.log"


def job_summary_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / "summary.json"


def aoi_summary_paths(run_dir: str | Path) -> Tuple[Path, Optional[Path]]:
    meta_dir = run_meta_dir(run_dir)
    return meta_dir / "aoi.json", None


def aoi_manifest_path(run_dir: str | Path) -> Path:
    return run_meta_dir(run_dir) / "aoi.json"


def period_summary_paths(period_dir: str | Path) -> Tuple[Path, Optional[Path]]:
    meta_dir = period_meta_dir(period_dir)
    return meta_dir / "period.json", None


def period_manifest_path(period_dir: str | Path) -> Path:
    return period_meta_dir(period_dir) / "period.json"


def job_summary_paths(job_dir: str | Path) -> Tuple[Path, Optional[Path]]:
    meta_dir = job_meta_dir(job_dir)
    return meta_dir / "job.json", None


def debug_dir(base_dir: str | Path) -> Path:
    return ensure_dir(Path(base_dir) / "debug")


def runtime_relpath(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(Path.cwd()))
    except ValueError:
        return str(candidate)


def load_json_if_exists(path: str | Path) -> Optional[Dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return None
    with open(candidate, "r", encoding="utf-8") as f:
        return json.load(f)


def rewrite_runtime_paths(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        for old_prefix, new_prefix in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if value == old_prefix:
                return new_prefix
            if value.startswith(f"{old_prefix}/"):
                return f"{new_prefix}{value[len(old_prefix):]}"
        return value
    if isinstance(value, list):
        return [rewrite_runtime_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_runtime_paths(item, replacements) for key, item in value.items()}
    return value


def merge_move_dir(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return
    ensure_dir(dst_path)
    for child in list(src_path.iterdir()):
        target = dst_path / child.name
        if child.is_dir() and target.exists() and target.is_dir():
            merge_move_dir(child, target)
            remove_dir_if_empty(child)
            continue
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    remove_dir_if_empty(src_path)


def canonicalize_period_debug_layout(period_dir: str | Path) -> Dict[str, str]:
    period_root = Path(period_dir)
    replacements: Dict[str, str] = {}
    moves = [
        (period_root / "inputs" / "window_raw", period_root / "debug" / "window_raw"),
        (period_root / "intermediate" / "composite", period_root / "debug" / "composite"),
        (period_root / "inputs" / "components", period_root / "debug" / "components"),
        (period_root / "intermediate" / "components", period_root / "debug" / "components"),
    ]
    for src, dst in moves:
        if not src.exists():
            continue
        merge_move_dir(src, dst)
        replacements[runtime_relpath(src)] = runtime_relpath(dst)
    remove_dir_if_empty(period_root / "inputs")
    remove_dir_if_empty(period_root / "intermediate")
    return replacements


def build_runtime_period_entry(record: Optional[Dict[str, Any]], period_dir: str | Path) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    period_root = Path(period_dir)
    rewritten = rewrite_runtime_paths(copy.deepcopy(record), canonicalize_period_debug_layout(period_root))
    period_info = rewritten.get("period") or {}
    output_paths = rewritten.get("output_paths") or {}
    output_root = period_root / "output"
    debug_root = period_root / "debug"
    debug_info = compact_jsonable(
        {
            "debug_dir": runtime_relpath(debug_root) if debug_root.exists() else None,
            "window_raw_dir": rewritten.get("window_raw_dir"),
            "composite_dir": rewritten.get("composite_dir"),
        }
    )
    entry = {
        "period_id": rewritten.get("period_id") or period_info.get("period_id") or period_root.name,
        "period_dir": runtime_relpath(period_root),
        "status": rewritten.get("status"),
        "lifecycle_status": rewritten.get("lifecycle_status"),
        "final_status": rewritten.get("final_status"),
        "completion_class": rewritten.get("completion_class"),
        "reason_code": rewritten.get("reason_code"),
        "reason_message": rewritten.get("reason_message"),
        "retry": rewritten.get("retry"),
        "time": {
            "period_mode": rewritten.get("period_mode") or period_info.get("period_mode"),
            "period_start": rewritten.get("period_start") or period_info.get("period_start"),
            "period_end": rewritten.get("period_end") or period_info.get("period_end"),
            "period_anchor_datetime": rewritten.get("period_anchor_datetime") or period_info.get("period_anchor_datetime"),
            "period_split_policy": rewritten.get("period_split_policy") or period_info.get("period_split_policy"),
        },
        "items_in_period": rewritten.get("items_in_period"),
        "items_after_hard_filter": rewritten.get("items_after_hard_filter"),
        "selection": rewritten.get("selection"),
        "componentization": rewritten.get("componentization"),
        "rejected_component_candidates": rewritten.get("rejected_component_candidates"),
        "component_results": rewritten.get("component_results"),
        "downloaded_files": rewritten.get("downloaded_files"),
        "composite": rewritten.get("composite"),
        "artifacts": {
            "output_dir": runtime_relpath(output_root) if output_root.exists() else None,
            "item_json_path": output_paths.get("output_sr_json_path") or rewritten.get("output_sr_geojson_path"),
            "vv_tif_path": output_paths.get("output_sr_vv_tif") or rewritten.get("output_sr_vv_tif"),
            "vh_tif_path": output_paths.get("output_sr_vh_tif") or rewritten.get("output_sr_vh_tif"),
            "public_item_id": rewritten.get("public_item_id"),
        },
        "debug": (debug_info or None),
    }
    return compact_jsonable(entry)


def _derive_period_counts(periods: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(periods),
        "completed": sum(1 for period in periods if period.get("status") == "completed"),
        "skipped": sum(1 for period in periods if period.get("status") == "skipped"),
        "failed": sum(1 for period in periods if period.get("status") == "failed"),
    }


def infer_runtime_status_fields(source: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    status = source.get("status")
    final_status = source.get("final_status")
    reason_code = source.get("reason_code")
    reason_message = source.get("reason_message")
    if status:
        return status, final_status, reason_code, reason_message
    if final_status == "skip":
        return "skipped", final_status, reason_code or "PIPELINE_SKIPPED", reason_message
    if final_status == "pass":
        return "completed", final_status, reason_code or "JOB_COMPLETED", reason_message
    if final_status == "fail":
        return "failed", final_status, reason_code or "PIPELINE_EXECUTION_ERROR", reason_message
    return None, final_status, reason_code, reason_message


def build_runtime_aoi_entry(
    *,
    aoi_dir: str | Path,
    aoi_record: Optional[Dict[str, Any]],
    periods: List[Dict[str, Any]],
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source = aoi_record or fallback or {}
    aoi_root = Path(aoi_dir)
    period_counts = source.get("period_counts") or _derive_period_counts(periods)
    status, final_status, reason_code, reason_message = infer_runtime_status_fields(source)
    return compact_jsonable(
        {
            "aoi_id": source.get("aoi_id") or aoi_root.name,
            "aoi_dir": runtime_relpath(aoi_root),
            "source_ref": source.get("aoi_geojson"),
            "status": status,
            "lifecycle_status": source.get("lifecycle_status"),
            "final_status": final_status,
            "completion_class": source.get("completion_class"),
            "reason_code": reason_code,
            "reason_message": reason_message,
            "retry": source.get("retry"),
            "items_after_hard_filter": source.get("items_after_hard_filter"),
            "compatibility": source.get("compatibility"),
            "period_counts": period_counts,
            "periods": periods,
        }
    )


def collect_runtime_aoi_entry(
    aoi_dir: str | Path,
    *,
    fallback_summary: Optional[Dict[str, Any]] = None,
    fallback_error_message: Optional[str] = None,
) -> Dict[str, Any]:
    aoi_root = Path(aoi_dir)
    aoi_record = load_json_if_exists(aoi_root / "aoi.json")
    period_entries: List[Dict[str, Any]] = []
    periods_root = aoi_root / "periods"
    if periods_root.exists():
        for period_root in sorted(periods_root.iterdir()):
            if not period_root.is_dir():
                continue
            period_record = load_json_if_exists(period_root / "period.json")
            period_entry = build_runtime_period_entry(period_record, period_root)
            if period_entry is not None:
                period_entries.append(period_entry)
            (period_root / "period.json").unlink(missing_ok=True)
    (aoi_root / "aoi.json").unlink(missing_ok=True)
    if aoi_record is None and fallback_summary is None:
        return compact_jsonable(
            {
                "aoi_id": aoi_root.name,
                "aoi_dir": runtime_relpath(aoi_root),
                "status": "failed",
                "final_status": "fail",
                "reason_code": "PIPELINE_EXECUTION_ERROR",
                "reason_message": fallback_error_message or "Pipeline execution failed before summary records were written.",
                "period_counts": _derive_period_counts(period_entries),
                "periods": period_entries,
            }
        )
    return build_runtime_aoi_entry(
        aoi_dir=aoi_root,
        aoi_record=aoi_record,
        periods=period_entries,
        fallback=fallback_summary,
    )


def prepare_storage_job_layout(
    *,
    base_root: str | Path,
    workflow_mode: str,
    source_mode: str,
) -> Dict[str, str]:
    jobs_root = ensure_dir(Path(base_root) / "jobs")
    job_id = build_storage_job_id(workflow_mode, source_mode)
    job_dir = ensure_dir(jobs_root / job_id)
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "aois_dir": str(ensure_dir(job_dir / "aois")),
        "source_mode": source_mode,
        "workflow_mode": workflow_mode,
    }


def prepare_storage_aoi_layout(job_layout: Dict[str, str], aoi_id: str) -> Dict[str, str]:
    aoi_dir = ensure_dir(Path(job_layout["aois_dir"]) / str(aoi_id))
    return {
        "aoi_id": str(aoi_id),
        "aoi_dir": str(aoi_dir),
        "periods_dir": str(ensure_dir(aoi_dir / "periods")),
    }


def infer_effective_input_profile(config: Dict[str, Any]) -> Optional[str]:
    workflow_mode = normalize_workflow_mode(config.get("workflow", {}).get("mode", ""), default="")
    if workflow_mode == WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE:
        return "stac_trainlike_composite_db"
    if workflow_mode == WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE:
        return "gee_s1_db"
    compat_cfg = config.get("compatibility", {}) or {}
    current_profile = str(compat_cfg.get("current_download_profile", "") or "").strip()
    return current_profile or None


def build_resolved_config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = copy.deepcopy(config)
    compat_cfg = snapshot.get("compatibility")
    if isinstance(compat_cfg, dict):
        trained_profile = str(compat_cfg.get("trained_input_profile", "") or "").strip() or None
        configured_current_profile = str(compat_cfg.get("current_download_profile", "") or "").strip() or None
        snapshot["compatibility"] = {
            "trained_input_profile": trained_profile,
            "configured_current_download_profile": configured_current_profile,
            "effective_input_profile": infer_effective_input_profile(snapshot),
            "allow_domain_mismatch": bool(compat_cfg.get("allow_domain_mismatch", False)),
        }
    logging_cfg = snapshot.get("logging")
    if isinstance(logging_cfg, dict):
        snapshot["logging"] = {
            "level": normalize_log_level_name(logging_cfg.get("level")),
            "redact_secrets": bool(logging_cfg.get("redact_secrets", True)),
            "startup_env_checks": bool(logging_cfg.get("startup_env_checks", True)),
        }
    return snapshot


def strip_debug_artifact_paths(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    sanitized = copy.deepcopy(record)
    for key in ("window_raw_dir", "composite_dir", "t1_composite_path", "t2_composite_path"):
        if key in sanitized:
            sanitized[key] = None
    if "downloaded_files" in sanitized:
        sanitized["downloaded_files"] = None
    return compact_jsonable(sanitized)


def copy_input_geojson_to_runtime(src_path: str | Path, dst_path: str | Path) -> Path:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {src}")
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def resolve_pipeline_run_dir(config: Dict[str, Any], run_root: str | Path, aoi_id: str) -> Path:
    runtime_info = config.setdefault("_runtime", {})
    existing_run_dir = runtime_info.get("current_run_dir")
    if existing_run_dir:
        run_dir = ensure_dir(existing_run_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = ensure_dir(Path(run_root) / str(aoi_id) / run_id)
        runtime_info["current_run_dir"] = str(run_dir)
    return run_dir


def resolve_runtime_aoi_id(config: Dict[str, Any], geojson_path: str | Path) -> str:
    runtime_info = config.get("_runtime", {})
    if runtime_info.get("aoi_id"):
        return str(runtime_info["aoi_id"])
    db_record = runtime_info.get("db_aoi_record") or {}
    if db_record.get("id"):
        return str(db_record["id"])
    return Path(str(geojson_path)).stem


def resolve_aoi_source_ref(config: Dict[str, Any], geojson_path: str | Path) -> str:
    runtime_info = config.get("_runtime", {})
    if runtime_info.get("aoi_source_ref"):
        return str(runtime_info["aoi_source_ref"])
    return str(Path(str(geojson_path)).resolve())


__all__ = [
    "SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD",
    "SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC",
    "WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE",
    "WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE",
    "_TeeTextIO",
    "aoi_log_path",
    "aoi_manifest_path",
    "aoi_summary_paths",
    "apply_runtime_env_overrides",
    "build_effective_runtime_settings",
    "build_resolved_config_snapshot",
    "build_runtime_aoi_entry",
    "build_runtime_period_entry",
    "build_startup_checks",
    "build_storage_job_id",
    "canonicalize_period_debug_layout",
    "capture_runtime_job_logs",
    "collect_runtime_aoi_entry",
    "copy_input_geojson_to_runtime",
    "debug_dir",
    "describe_pipeline_profile",
    "ensure_dir",
    "infer_effective_input_profile",
    "infer_job_dir_from_runtime_path",
    "inputs_dir",
    "intermediate_dir",
    "is_canonical_selection_strategy",
    "is_representative_composite_workflow_mode",
    "job_log_path",
    "job_meta_dir",
    "job_summary_path",
    "job_summary_paths",
    "load_json_if_exists",
    "load_yaml",
    "merge_move_dir",
    "model_trainlike_semantics",
    "normalize_selection_strategy",
    "normalize_workflow_mode",
    "period_manifest_path",
    "period_meta_dir",
    "period_summary_paths",
    "prepare_storage_aoi_layout",
    "prepare_storage_job_layout",
    "remove_dir_if_empty",
    "resolve_aoi_source_ref",
    "resolve_optional_debug_dir",
    "resolve_pipeline_run_dir",
    "resolve_runtime_aoi_id",
    "resolve_spatial_strategy",
    "resolve_workflow_backend",
    "rewrite_runtime_paths",
    "run_meta_dir",
    "runtime_relpath",
    "save_debug_artifacts_enabled",
    "strip_debug_artifact_paths",
]

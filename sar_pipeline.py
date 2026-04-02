#!/usr/bin/env python3
"""
AOI GeoJSON -> STAC/GEE query -> preprocessing -> SR inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import rasterio
from affine import Affine
from pyproj import Geod, Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.shutil import copy as rio_copy
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from scipy.ndimage import median_filter
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
import yaml

from db_aoi_source import (
    fetch_active_aois_from_database,
    inspect_database_settings,
    materialize_database_aoi_geojson,
    normalize_aoi_uuid,
)
from query_stac_download import (
    DEFAULT_COLLECTION,
    DEFAULT_STAC_API,
    STACClient,
    S3Downloader,
    build_manifest_for_pair,
    build_representative_period_manifest,
    build_trainlike_anchor_manifest,
    build_selected_pair_info,
    canonical_bbox_from_geometry,
    collect_anchor_window_items,
    collect_items_with_filters,
    collect_items_covering_region,
    collect_period_half_items,
    diagnose_no_pair,
    download_manifest_pair,
    expand_month_periods,
    extract_item_info,
    format_duration_human,
    geodesic_area_wgs84,
    build_seed_intersection_region_candidates,
    item_scene_key,
    load_geojson_aoi,
    normalize_polygonal_geojson_geometry,
    normalize_polygonal_shapely_geometry,
    normalize_representative_pool_mode,
    normalize_datetime_range,
    parse_required_pols,
    select_representative_scene_pools,
    select_asset_href,
    search_pairs_sorted,
    suggest_trainlike_anchors,
)
from runtime_logging import (
    configure_root_logging,
    detect_s3_credential_source,
    DEFAULT_LOG_LEVEL,
    ensure_root_logging,
    emit_runtime_log,
    normalize_log_level_name,
    resolve_runtime_log_level,
)
from runtime_env_overrides import apply_inference_env_overrides, apply_pipeline_env_overrides

ensure_root_logging(DEFAULT_LOG_LEVEL)
logger = logging.getLogger("sar_pipeline")
DEFAULT_MAX_RETRY_ATTEMPTS = 3
AUTO_DATETIME_SENTINELS = {"", "auto", "latest_full_month", "previous_full_month", "auto_previous_full_month"}


def emit_pipeline_log(level: int, message: str, **fields: Any) -> None:
    emit_runtime_log("sar_pipeline", level, message, **fields)


def emit_pipeline_stage(title: str, **fields: Any) -> None:
    normalized = " ".join(str(title or "").strip().upper().split()) or "PIPELINE"
    emit_pipeline_log(logging.INFO, f"================ {normalized} ================", **fields)


def build_effective_runtime_settings(
    *,
    train_cfg: Dict[str, Any],
    infer_config: Optional[Dict[str, Any]],
    save_debug_artifacts: bool,
) -> Dict[str, Any]:
    infer_cfg = ((infer_config or {}).get("inference") or {}) if isinstance(infer_config, dict) else {}
    return compact_jsonable(
        {
            "representative_pool_mode": normalize_representative_pool_mode(
                train_cfg.get("representative_pool_mode", "auto")
            ),
            "min_scenes_per_half": int(train_cfg.get("min_scenes_per_half", 2)),
            "component_item_min_coverage": float(train_cfg.get("component_item_min_coverage", 0.0)),
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


def log_effective_runtime_settings(
    *,
    train_cfg: Dict[str, Any],
    infer_config: Optional[Dict[str, Any]],
    save_debug_artifacts: bool,
) -> None:
    emit_pipeline_stage(
        "Effective Runtime Settings",
        **build_effective_runtime_settings(
            train_cfg=train_cfg,
            infer_config=infer_config,
            save_debug_artifacts=save_debug_artifacts,
        ),
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


def log_startup_checks(startup_checks: Dict[str, Any]) -> None:
    db_checks = startup_checks.get("db") or {}
    stac_checks = startup_checks.get("stac") or {}
    s3_checks = startup_checks.get("s3") or {}
    emit_pipeline_log(
        logging.INFO,
        "Startup environment checks",
        policy=startup_checks.get("policy"),
        db_enabled=db_checks.get("enabled"),
        db_env_path=db_checks.get("env_path"),
        pghost=db_checks.get("pghost"),
        pgport=db_checks.get("pgport"),
        pguser=db_checks.get("pguser"),
        pgdatabase=db_checks.get("pgdatabase"),
        pgpassword_present=db_checks.get("pgpassword_present"),
        db_missing_keys=db_checks.get("missing_keys"),
        stac_url=stac_checks.get("url"),
        stac_collection=stac_checks.get("collection"),
        stac_datetime=stac_checks.get("datetime"),
        s3_endpoint=s3_checks.get("endpoint"),
        s3_credential_source=s3_checks.get("credential_source"),
        s3_access_key_present=s3_checks.get("s3_access_key_present"),
        s3_secret_key_present=s3_checks.get("s3_secret_key_present"),
    )
    if db_checks.get("enabled") and db_checks.get("missing_keys"):
        emit_pipeline_log(
            logging.WARNING,
            "Database environment is incomplete",
            missing_keys=db_checks.get("missing_keys"),
            env_path=db_checks.get("env_path"),
        )
    if s3_checks.get("credential_source") == "none":
        emit_pipeline_log(
            logging.WARNING,
            "S3 credentials are not explicitly configured",
            note="The pipeline will continue and fail only if a downstream S3 download is required.",
        )


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


def log_runtime_env_overrides(config: Dict[str, Any]) -> None:
    for override in (config.get("_runtime", {}) or {}).get("env_overrides", []) or []:
        emit_pipeline_log(
            logging.DEBUG,
            "Applied runtime environment override",
            target=override.get("target"),
            source=override.get("source"),
        )


def log_inference_env_overrides(infer_config: Dict[str, Any]) -> None:
    for override in (infer_config.get("_runtime", {}) or {}).get("env_overrides", []) or []:
        emit_pipeline_log(
            logging.DEBUG,
            "Applied inference environment override",
            target=override.get("target"),
            source=override.get("source"),
        )


def exact_pair_semantics() -> Dict[str, Any]:
    return {
        "t1_label": "T1",
        "t2_label": "T2",
        "t1_role": "later/posterior exact scene",
        "t2_role": "earlier/prior exact scene",
        "matches_training_semantics": True,
        "note": "Exact-pair mode now uses canonical model order T1=later and T2=earlier. This matches temporal input ordering, but the recipe is still exact single-scene rather than train-like composite.",
    }


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


@contextmanager
def capture_runtime_job_logs(job_dir: str | Path):
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
    workflow_mode = str(config.get("workflow", {}).get("mode", "") or "").strip().lower()
    if workflow_mode == "exact_pair":
        return "stac_measurement_raw"
    if workflow_mode == "stac_trainlike_composite":
        return "stac_trainlike_composite_db"
    if workflow_mode == "gee_trainlike_composite":
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


def _shift_year_month(year: int, month: int, months_delta: int) -> Tuple[int, int]:
    total = year * 12 + (month - 1) + months_delta
    return total // 12, (total % 12) + 1


def _utc_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_target_month_datetime_range(target_month: str) -> Tuple[str, Dict[str, Any]]:
    raw = str(target_month or "").strip()
    try:
        year_str, month_str = raw.split("-", 1)
        year = int(year_str)
        month = int(month_str)
        if month < 1 or month > 12:
            raise ValueError
    except Exception as exc:
        raise ValueError(f"Invalid target month `{target_month}`. Expected format YYYY-MM.") from exc
    next_year, next_month = _shift_year_month(year, month, 1)
    start_dt_utc = datetime(year, month, 1, tzinfo=timezone.utc)
    end_dt_utc = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
    resolved_datetime = f"{_utc_rfc3339(start_dt_utc)}/{_utc_rfc3339(end_dt_utc)}"
    return resolved_datetime, {
        "mode": "target_month",
        "source": "target_month_override",
        "target_period_id": f"{year:04d}-{month:02d}",
        "resolved_datetime": resolved_datetime,
    }


def resolve_representative_datetime_filter(
    config: Dict[str, Any],
    *,
    now_utc: Optional[datetime] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve representative-month datetime range.

    Priority:
    1. explicit manual datetime from config/CLI
    2. auto mode: previous full month based on configured local timezone

    Auto mode uses local calendar month only to decide *which month label* to run,
    but always emits canonical UTC month boundaries so downstream representative
    period generation stays unchanged.
    """

    gee_cfg = config.setdefault("gee", {})
    stac_cfg = config.setdefault("stac", {})
    train_cfg = config.setdefault("trainlike", {})
    runtime_info = config.setdefault("_runtime", {})

    target_month = train_cfg.get("target_month")
    if target_month:
        resolved_datetime, resolution = resolve_target_month_datetime_range(str(target_month))
        gee_cfg["datetime"] = resolved_datetime
        stac_cfg["datetime"] = resolved_datetime
        runtime_info["datetime_resolution"] = resolution
        return resolved_datetime, resolution

    configured_datetime = gee_cfg.get("datetime") or stac_cfg.get("datetime")
    normalized_manual = normalize_datetime_range(str(configured_datetime).strip()) if configured_datetime is not None else None
    if normalized_manual and normalized_manual.strip().lower() not in AUTO_DATETIME_SENTINELS:
        existing_resolution = runtime_info.get("datetime_resolution")
        resolution = existing_resolution if (
            isinstance(existing_resolution, dict)
            and existing_resolution.get("resolved_datetime") == normalized_manual
        ) else {
            "mode": "manual",
            "source": "explicit_datetime_override",
            "resolved_datetime": normalized_manual,
        }
        gee_cfg["datetime"] = normalized_manual
        stac_cfg["datetime"] = normalized_manual
        runtime_info["datetime_resolution"] = resolution
        return normalized_manual, resolution

    strategy = str(train_cfg.get("auto_datetime_strategy", "previous_full_month")).strip().lower()
    if not strategy or strategy in {"off", "disabled", "none"}:
        raise ValueError(
            "Representative calendar mode requires either an explicit finite datetime range "
            "or trainlike.auto_datetime_strategy to be enabled."
        )
    if strategy not in {"previous_full_month", "latest_full_month", "auto_previous_full_month"}:
        raise ValueError(
            f"Unsupported trainlike.auto_datetime_strategy: {train_cfg.get('auto_datetime_strategy')}"
        )

    timezone_name = str(train_cfg.get("auto_datetime_timezone", "Asia/Ho_Chi_Minh")).strip() or "Asia/Ho_Chi_Minh"
    try:
        local_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unsupported trainlike.auto_datetime_timezone: {timezone_name}") from exc

    months_back = int(train_cfg.get("auto_datetime_months_back", 1))
    if months_back < 1:
        raise ValueError("trainlike.auto_datetime_months_back must be >= 1")

    now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_local = now_utc.astimezone(local_tz)
    target_year, target_month = _shift_year_month(now_local.year, now_local.month, -months_back)
    next_year, next_month = _shift_year_month(target_year, target_month, 1)
    start_dt_utc = datetime(target_year, target_month, 1, tzinfo=timezone.utc)
    end_dt_utc = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
    resolved_datetime = f"{_utc_rfc3339(start_dt_utc)}/{_utc_rfc3339(end_dt_utc)}"

    resolution = {
        "mode": "auto",
        "strategy": "previous_full_month",
        "timezone": timezone_name,
        "months_back": months_back,
        "reference_now_utc": _utc_rfc3339(now_utc),
        "reference_now_local": now_local.isoformat(),
        "target_period_id": f"{target_year:04d}-{target_month:02d}",
        "resolved_datetime": resolved_datetime,
    }
    gee_cfg["datetime"] = resolved_datetime
    stac_cfg["datetime"] = resolved_datetime
    runtime_info["datetime_resolution"] = resolution
    return resolved_datetime, resolution


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Affine):
        return list(value)[:6]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def compact_jsonable(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, dict):
        return {k: compact_jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [compact_jsonable(v) for v in value if v is not None]
    return value


def classify_skip_reason_code(skip_reason: Optional[str]) -> str:
    text = str(skip_reason or "").strip().lower()
    if not text:
        return "SKIPPED_BY_PIPELINE_RULE"
    if "no valid child aoi intersections remained" in text:
        return "NO_VALID_CHILDREN"
    if "no valid in-period representative scene pools" in text:
        return "NO_VALID_REPRESENTATIVE_POOLS"
    if "scene-count" in text or "scene count" in text:
        return "SCENE_COUNT_BELOW_MIN"
    if "no items" in text:
        return "NO_ITEMS_IN_PERIOD"
    return "SKIPPED_BY_PIPELINE_RULE"


def classify_failure_reason(error_message: Optional[str]) -> Tuple[str, bool]:
    text = str(error_message or "").strip().lower()
    if not text:
        return "PIPELINE_EXECUTION_ERROR", False
    retryable_markers = [
        "503",
        "service unavailable",
        "429",
        "502",
        "504",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    ]
    if any(marker in text for marker in retryable_markers):
        return "BACKEND_SERVICE_UNAVAILABLE", True
    if "inference dependencies are missing" in text or "module not found" in text:
        return "DEPENDENCY_ERROR", False
    if "config" in text and "missing" in text:
        return "INVALID_CONFIG", False
    if "unsupported" in text or "requires" in text:
        return "INVALID_CONFIG", False
    if "geojson not found" in text or "invalid_aoi" in text:
        return "INVALID_AOI_GEOMETRY", False
    if "download" in text:
        return "DOWNLOAD_ERROR", True
    if "infer" in text or "checkpoint" in text or "cuda" in text:
        return "MODEL_INFERENCE_ERROR", False
    if "write" in text or "cog" in text or "tif" in text:
        return "ARTIFACT_WRITE_ERROR", False
    return "PIPELINE_EXECUTION_ERROR", False


def infer_completion_class(summary: Dict[str, Any]) -> str:
    raw_status = str(summary.get("status", "")).strip().lower()
    if raw_status in {"skipped", "failed"}:
        return "none"

    period_counts = summary.get("period_counts")
    if isinstance(period_counts, dict):
        total = int(period_counts.get("total", 0) or 0)
        completed = int(period_counts.get("completed", 0) or 0)
        skipped = int(period_counts.get("skipped", 0) or 0)
        failed = int(period_counts.get("failed", 0) or 0)
        if completed > 0 and skipped == 0 and failed == 0 and completed == total:
            return "full"
        if completed > 0:
            return "partial"
        return "none"

    componentization = summary.get("componentization")
    if isinstance(componentization, dict):
        completed = int(componentization.get("completed_component_count", 0) or 0)
        rejected = int(componentization.get("rejected_component_count", 0) or 0)
        try:
            supported_area_ratio = float(componentization.get("parent_supported_area_ratio"))
        except Exception:
            supported_area_ratio = None
        if completed > 0 and rejected == 0:
            if supported_area_ratio is not None and supported_area_ratio < 0.999999:
                return "partial"
            return "full"
        if completed > 0:
            return "partial"
        return "none"

    if raw_status == "completed":
        return "full"
    return "none"


def has_only_suppressed_component_rejections(summary: Dict[str, Any]) -> bool:
    componentization = summary.get("componentization")
    rejected_candidates = summary.get("rejected_component_candidates")
    if not isinstance(componentization, dict) or not isinstance(rejected_candidates, list):
        return False

    completed = int(componentization.get("completed_component_count", 0) or 0)
    rejected = int(componentization.get("rejected_component_count", 0) or 0)
    if completed <= 0 or rejected <= 0 or len(rejected_candidates) != rejected:
        return False

    return all(str(item.get("status", "")).strip().lower() == "suppressed" for item in rejected_candidates)


def attach_run_log_path(summary: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = summary.get("run_dir")
    if run_dir and not summary.get("run_log_path"):
        summary["run_log_path"] = str(aoi_log_path(run_dir))
    return summary


def apply_execution_contract(
    summary: Dict[str, Any],
    *,
    default_reason_code: Optional[str] = None,
    default_reason_message: Optional[str] = None,
    retry_attempt: int = 1,
    max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS,
) -> Dict[str, Any]:
    attach_run_log_path(summary)
    raw_status = str(summary.get("status", "")).strip().lower()
    if not raw_status:
        period_counts = summary.get("period_counts") or {}
        completed = int(period_counts.get("completed", 0) or 0)
        skipped = int(period_counts.get("skipped", 0) or 0)
        failed = int(period_counts.get("failed", 0) or 0)
        if completed > 0:
            raw_status = "completed"
        elif skipped > 0 and failed == 0:
            raw_status = "skipped"
        elif failed > 0:
            raw_status = "failed"
        else:
            raw_status = "failed"
        summary["status"] = raw_status

    if raw_status == "completed":
        final_status = "pass"
        completion_class = infer_completion_class(summary)
        if default_reason_code:
            reason_code = default_reason_code
        elif completion_class == "partial" and has_only_suppressed_component_rejections(summary):
            reason_code = "COMPLETED_WITH_COMPONENT_SUPPRESSION"
        elif completion_class == "partial":
            reason_code = "COMPLETED_WITH_PARTIAL_SKIPS"
        else:
            reason_code = "COMPLETED_WITH_OUTPUT"

        if default_reason_message:
            reason_message = default_reason_message
        elif reason_code == "COMPLETED_WITH_OUTPUT":
            reason_message = "Execution finished and produced valid output artifacts."
        elif reason_code == "COMPLETED_WITH_COMPONENT_SUPPRESSION":
            reason_message = (
                "Execution finished with valid outputs after intentionally suppressing nested child "
                "components that were fully covered by larger regions."
            )
        else:
            reason_message = "Execution finished with valid outputs, but some periods or child components were skipped or rejected."
        retryable = False
    elif raw_status == "skipped":
        final_status = "skip"
        reason_message = default_reason_message or str(summary.get("skip_reason") or "Pipeline skipped this run because no valid data path remained.")
        reason_code = default_reason_code or classify_skip_reason_code(reason_message)
        retryable = False
    else:
        final_status = "fail"
        failure_text = default_reason_message or str(summary.get("error_message") or summary.get("error") or "Pipeline execution failed.")
        reason_code, retryable = classify_failure_reason(failure_text)
        if default_reason_code:
            reason_code = default_reason_code
        reason_message = failure_text

    retry_state = {
        "attempt": int(summary.get("retry", {}).get("attempt", retry_attempt) or retry_attempt),
        "max_attempts": int(summary.get("retry", {}).get("max_attempts", max_retry_attempts) or max_retry_attempts),
        "retryable": bool(summary.get("retry", {}).get("retryable", retryable)),
        "next_retry_at": summary.get("retry", {}).get("next_retry_at"),
    }
    retry_state["remaining_attempts"] = max(0, retry_state["max_attempts"] - retry_state["attempt"])

    summary["lifecycle_status"] = "finished"
    summary["final_status"] = final_status
    summary["completion_class"] = infer_completion_class(summary)
    summary["reason_code"] = reason_code
    summary["reason_message"] = reason_message
    summary["retry"] = retry_state
    summary["status_contract_version"] = "2026-03-24.v1"
    return summary


def build_period_manifest_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    period = summary.get("period") or {}
    manifest_type = "representative_calendar_period"
    if summary.get("component_results") is not None and "selection" not in summary:
        manifest_type = "representative_calendar_period_parent"
    manifest = {
        "manifest_type": manifest_type,
        "workflow_mode": summary.get("workflow_mode"),
        "selection_strategy": summary.get("selection_strategy"),
        "aoi_geojson": summary.get("aoi_geojson"),
        "period_id": period.get("period_id"),
        "period_mode": period.get("period_mode"),
        "period_start": period.get("period_start"),
        "period_end": period.get("period_end"),
        "period_anchor_datetime": period.get("period_anchor_datetime"),
        "period_split_policy": period.get("period_split_policy"),
        "status": summary.get("status"),
        "lifecycle_status": summary.get("lifecycle_status"),
        "final_status": summary.get("final_status"),
        "completion_class": summary.get("completion_class"),
        "reason_code": summary.get("reason_code"),
        "reason_message": summary.get("reason_message"),
        "retry": summary.get("retry"),
        "summary_json": summary.get("summary_json"),
        "summary_md": summary.get("summary_md"),
        "run_log_path": summary.get("run_log_path"),
        "output_sr_geojson_path": summary.get("output_sr_geojson_path"),
    }
    if summary.get("skip_reason"):
        manifest["skip_reason"] = summary.get("skip_reason")
    if summary.get("items_in_period") is not None:
        manifest["items_in_period"] = summary.get("items_in_period")
    if summary.get("componentization") is not None:
        manifest["componentization"] = summary.get("componentization")
    if summary.get("rejected_component_candidates") is not None:
        manifest["rejected_component_candidates"] = summary.get("rejected_component_candidates")
    if summary.get("component_results") is not None:
        manifest["component_results"] = summary.get("component_results")
    if summary.get("selection") is not None:
        manifest["selection"] = summary.get("selection")
    return manifest


def normalize_runtime_record(record: Dict[str, Any], *, scope: str) -> Dict[str, Any]:
    normalized = copy.deepcopy(record)
    if "run_log_path" in normalized:
        normalized["log_path"] = normalized.get("run_log_path")
    elif "job_log_path" in normalized:
        normalized["log_path"] = normalized.get("job_log_path")

    for key in (
        "summary_json",
        "summary_md",
        "manifest_path",
        "manifest_type",
        "record_type",
        "run_log_path",
        "job_log_path",
    ):
        normalized.pop(key, None)

    normalized["scope"] = scope
    return compact_jsonable(normalized)


def write_period_manifest_from_summary(summary: Dict[str, Any]) -> Optional[Path]:
    manifest_path_value = summary.get("manifest_path")
    if not manifest_path_value:
        return None
    manifest_path = Path(manifest_path_value)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_period_record(summary), f, indent=2, ensure_ascii=False)
    return manifest_path


def build_run_manifest(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_type": "representative_calendar_run",
        "workflow_mode": summary.get("workflow_mode"),
        "selection_strategy": summary.get("selection_strategy"),
        "aoi_geojson": summary.get("aoi_geojson"),
        "run_dir": summary.get("run_dir"),
        "periods_dir": summary.get("periods_dir"),
        "status": summary.get("status"),
        "lifecycle_status": summary.get("lifecycle_status"),
        "final_status": summary.get("final_status"),
        "completion_class": summary.get("completion_class"),
        "reason_code": summary.get("reason_code"),
        "reason_message": summary.get("reason_message"),
        "retry": summary.get("retry"),
        "period_counts": summary.get("period_counts"),
        "period_results": summary.get("period_results"),
        "run_config": summary.get("run_config"),
        "summary_json": summary.get("summary_json"),
        "summary_md": summary.get("summary_md"),
        "run_log_path": summary.get("run_log_path"),
        "output_sr_geojson_path": summary.get("output_sr_geojson_path"),
    }


def build_period_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    record = build_period_manifest_from_summary(summary)
    record.update(copy.deepcopy(summary))
    return normalize_runtime_record(record, scope="period")


def build_aoi_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    record = build_run_manifest(summary)
    record.update(copy.deepcopy(summary))
    return normalize_runtime_record(record, scope="aoi")


def build_failed_representative_run_summary(
    *,
    config: Dict[str, Any],
    geojson_path: str,
    run_dir: Path,
    error: Exception,
) -> Dict[str, Any]:
    workflow_mode = str(config.get("workflow", {}).get("mode", "unknown"))
    train_cfg = config.get("trainlike", {})
    output_cfg = config.get("output", {})
    runtime_info = config.get("_runtime", {})
    run_config = {
        "datetime": config.get("gee", {}).get("datetime") or config.get("stac", {}).get("datetime"),
        "datetime_resolution": runtime_info.get("datetime_resolution"),
        "period_mode": train_cfg.get("period_mode", "month"),
        "period_boundary_policy": train_cfg.get("period_boundary_policy", "clip_inside_period"),
        "period_split_policy": train_cfg.get("period_split_policy", "first_half_vs_second_half"),
        "componentize_seed_intersections": bool(train_cfg.get("componentize_seed_intersections", False)),
        "component_item_min_coverage": train_cfg.get("component_item_min_coverage"),
        "component_min_area_ratio": train_cfg.get("component_min_area_ratio"),
        **build_final_output_trace_config(output_cfg),
    }
    return apply_execution_contract(
        {
            "status": "failed",
            "workflow_mode": workflow_mode,
            "selection_strategy": str(train_cfg.get("selection_strategy", "representative_calendar_period")),
            "aoi_geojson": str(Path(geojson_path)),
            "run_dir": str(run_dir),
            "periods_dir": str(run_dir / "periods"),
            "period_counts": {
                "total": 0,
                "completed": 0,
                "skipped": 0,
                "failed": 1,
            },
            "period_results": [],
            "run_config": run_config,
            "error": str(error),
            "error_message": str(error),
        }
    )


def build_final_output_trace_config(output_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    output_cfg = output_cfg or {}
    return {
        "final_target_crs": output_cfg.get("final_target_crs"),
        "final_target_resolution": output_cfg.get("final_target_resolution"),
        "final_resampling": str(output_cfg.get("final_resampling", "bilinear")),
    }


def build_target_grid(aoi_bbox: List[float], target_crs: str, xres: float, yres: float) -> Dict[str, Any]:
    """Build a canonical metric grid covering the AOI bbox."""
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    minx, miny = transformer.transform(aoi_bbox[0], aoi_bbox[1])
    maxx, maxy = transformer.transform(aoi_bbox[2], aoi_bbox[3])

    left = math.floor(min(minx, maxx) / xres) * xres
    right = math.ceil(max(minx, maxx) / xres) * xres
    bottom = math.floor(min(miny, maxy) / yres) * yres
    top = math.ceil(max(miny, maxy) / yres) * yres
    width = max(1, int(round((right - left) / xres)))
    height = max(1, int(round((top - bottom) / yres)))
    transform = Affine(xres, 0.0, left, 0.0, -yres, top)
    return {
        "crs": target_crs,
        "xres": float(xres),
        "yres": float(yres),
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "width": width,
        "height": height,
        "transform": transform,
        "crs_transform": [float(xres), 0.0, left, 0.0, -float(yres), top],
    }


def build_sr_grid_from_input_grid(input_grid: Dict[str, Any], scale_factor: int = 2) -> Dict[str, Any]:
    """Derive the SR grid from an input composite grid."""
    scale = int(scale_factor)
    if scale < 1:
        raise ValueError("scale_factor must be >= 1.")

    xres = float(input_grid["xres"]) / float(scale)
    yres = float(input_grid["yres"]) / float(scale)
    left = float(input_grid["left"])
    top = float(input_grid["top"])
    width = max(1, int(input_grid["width"]) * scale)
    height = max(1, int(input_grid["height"]) * scale)
    right = left + width * xres
    bottom = top - height * yres
    transform = Affine(xres, 0.0, left, 0.0, -yres, top)
    return {
        "crs": input_grid["crs"],
        "xres": float(xres),
        "yres": float(yres),
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "width": width,
        "height": height,
        "transform": transform,
        "crs_transform": [float(xres), 0.0, left, 0.0, -float(yres), top],
    }


def resolve_resampling(name: str) -> Resampling:
    key = str(name or "bilinear").strip().lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "lanczos": Resampling.lanczos,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported resampling mode: {name}")
    return mapping[key]


def dedupe_items_by_scene(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate scene variants so one acquisition contributes once per window."""
    seen: set[Tuple[str, str, str, str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda it: extract_item_info(it)["datetime"]):
        key = item_scene_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def build_grid_meta(grid: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 2,
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "crs": rasterio.crs.CRS.from_user_input(grid["crs"]),
        "transform": grid["transform"],
        "descriptions": ("S1_VV", "S1_VH"),
    }


def write_geojson(path: str | Path, geometry: Dict[str, Any], properties: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feature = {
        "type": "Feature",
        "properties": to_jsonable(properties or {}),
        "geometry": geometry,
    }
    path.write_text(json.dumps(feature, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def geometry_mask_for_grid(
    geometry_wgs84: Dict[str, Any],
    grid: Dict[str, Any],
    all_touched: bool = False,
) -> np.ndarray:
    target_crs = rasterio.crs.CRS.from_user_input(grid["crs"])
    normalized_geometry_wgs84 = normalize_polygonal_geojson_geometry(geometry_wgs84)
    if normalized_geometry_wgs84 is None:
        raise ValueError("Geometry mask requires a polygonal geometry with non-zero area.")
    geometry_in_grid = transform_geom(
        "EPSG:4326",
        target_crs,
        normalized_geometry_wgs84,
        antimeridian_cutting=True,
        precision=15,
    )
    geometry_in_grid = normalize_polygonal_geojson_geometry(geometry_in_grid)
    if geometry_in_grid is None:
        raise ValueError("Geometry mask cannot be created because the transformed geometry has no polygonal area.")
    mask = rasterize(
        [(geometry_in_grid, 1)],
        out_shape=(int(grid["height"]), int(grid["width"])),
        transform=grid["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(np.uint8)


def apply_geometry_mask_to_multiband(
    data: np.ndarray,
    grid: Dict[str, Any],
    geometry_wgs84: Optional[Dict[str, Any]],
    fill_value: float = np.nan,
) -> np.ndarray:
    if geometry_wgs84 is None:
        return data
    mask = geometry_mask_for_grid(geometry_wgs84, grid).astype(bool)
    masked = data.astype(np.float32).copy()
    masked[:, ~mask] = fill_value
    return masked


def write_geometry_mask_tif(
    geometry_wgs84: Dict[str, Any],
    grid: Dict[str, Any],
    out_path: str | Path,
    compression: str = "DEFLATE",
) -> Path:
    mask = geometry_mask_for_grid(geometry_wgs84, grid)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "crs": rasterio.crs.CRS.from_user_input(grid["crs"]),
        "transform": grid["transform"],
        "compress": compression,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "VALID_MASK")
    return out_path


def write_geometry_mask_like_raster(
    geometry_wgs84: Dict[str, Any],
    reference_raster_path: str | Path,
    out_path: str | Path,
    compression: str = "DEFLATE",
) -> Path:
    with rasterio.open(reference_raster_path, "r") as src:
        grid = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
    return write_geometry_mask_tif(geometry_wgs84, grid, out_path, compression=compression)


def _sr_band_slug(description: Optional[str], band_index: int) -> str:
    desc = str(description or f"SR_BAND_{band_index}").strip().upper()
    if "VV" in desc:
        return "SR_VV"
    if "VH" in desc:
        return "SR_VH"
    if "HH" in desc:
        return "SR_HH"
    if "HV" in desc:
        return "SR_HV"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in desc).strip("_")
    return cleaned or f"SR_BAND_{band_index}"


def _cog_creation_options(compression: str = "DEFLATE", blocksize: int = 512) -> Dict[str, Any]:
    codec = str(compression or "DEFLATE").upper()
    opts: Dict[str, Any] = {
        "BLOCKSIZE": int(blocksize),
        "COMPRESS": codec,
        "BIGTIFF": "IF_SAFER",
        "NUM_THREADS": "ALL_CPUS",
        "OVERVIEWS": "AUTO",
        "OVERVIEW_RESAMPLING": "AVERAGE",
    }
    if codec in {"DEFLATE", "LZW", "ZSTD"}:
        opts["PREDICTOR"] = "3"
    return opts


def _safe_tiff_blocksize(width: int, height: int, preferred: int = 512) -> int:
    candidate = max(16, int(preferred))
    max_dim = max(1, int(min(width, height)))
    while candidate > max_dim and candidate > 16:
        candidate //= 2
    candidate = max(16, (candidate // 16) * 16)
    return candidate


def _crop_masked_extent(
    band_data: np.ndarray,
    valid_mask: np.ndarray,
    transform: Affine,
) -> Tuple[np.ndarray, np.ndarray, Affine]:
    valid_rows, valid_cols = np.where(valid_mask)
    if valid_rows.size == 0 or valid_cols.size == 0:
        raise ValueError("Valid AOI mask contains no valid pixels to crop.")
    row_start = int(valid_rows.min())
    row_stop = int(valid_rows.max()) + 1
    col_start = int(valid_cols.min())
    col_stop = int(valid_cols.max()) + 1
    cropped_band = band_data[row_start:row_stop, col_start:col_stop]
    cropped_mask = valid_mask[row_start:row_stop, col_start:col_stop]
    cropped_transform = transform * Affine.translation(col_start, row_start)
    return cropped_band, cropped_mask, cropped_transform


def _reproject_mask_and_transform(
    valid_mask: np.ndarray,
    *,
    src_crs: Any,
    src_transform: Affine,
    dst_crs: str,
    dst_resolution: Optional[float],
) -> Tuple[np.ndarray, Affine]:
    src_height, src_width = valid_mask.shape
    left, bottom, right, top = array_bounds(src_height, src_width, src_transform)
    transform_kwargs: Dict[str, Any] = {}
    if dst_resolution is not None:
        transform_kwargs["resolution"] = float(dst_resolution)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        src_width,
        src_height,
        left,
        bottom,
        right,
        top,
        **transform_kwargs,
    )
    dst_mask = np.zeros((int(dst_height), int(dst_width)), dtype=np.uint8)
    reproject(
        source=valid_mask.astype(np.uint8),
        destination=dst_mask,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst_mask.astype(bool), dst_transform


def _reproject_mask_to_grid(
    valid_mask: np.ndarray,
    *,
    src_crs: Any,
    src_transform: Affine,
    dst_crs: Any,
    dst_transform: Affine,
    dst_shape: Tuple[int, int],
) -> np.ndarray:
    dst_height, dst_width = dst_shape
    dst_mask = np.zeros((int(dst_height), int(dst_width)), dtype=np.uint8)
    reproject(
        source=valid_mask.astype(np.uint8),
        destination=dst_mask,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst_mask.astype(bool)


def _reproject_band_to_grid(
    band_data: np.ndarray,
    *,
    src_crs: Any,
    src_transform: Affine,
    dst_crs: str,
    dst_transform: Affine,
    dst_shape: Tuple[int, int],
    resampling_name: str,
) -> np.ndarray:
    dst_height, dst_width = dst_shape
    destination = np.full((int(dst_height), int(dst_width)), np.nan, dtype=np.float32)
    reproject(
        source=band_data.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resolve_resampling(resampling_name),
    )
    return destination


def _component_mosaic_priority_key(component_result: Dict[str, Any]) -> Tuple[float, float, int, int, str]:
    selection = component_result.get("selection") or {}
    try:
        relaxation_level = int(selection.get("selected_relaxation_level"))
    except Exception:
        relaxation_level = 999
    pre_scene_count = int(selection.get("pre_scene_count", 0) or 0)
    post_scene_count = int(selection.get("post_scene_count", 0) or 0)
    area_ratio = float(component_result.get("area_ratio_vs_parent", 0.0) or 0.0)
    area_m2 = float(component_result.get("area_m2", 0.0) or 0.0)
    return (
        -area_ratio,
        -area_m2,
        -(pre_scene_count + post_scene_count),
        relaxation_level,
        str(component_result.get("component_id") or ""),
    )


def mosaic_component_sr_multibands_to_parent(
    *,
    component_sources: List[Dict[str, Any]],
    parent_aoi_geometry: Dict[str, Any],
    parent_aoi_bbox: List[float],
    target_crs: str,
    target_resolution: float,
    output_path: str | Path,
    compression: str = "DEFLATE",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> Dict[str, Any]:
    """Merge child SR outputs back onto one parent AOI canvas."""
    if not component_sources:
        raise ValueError("component_sources must not be empty.")
    emit_pipeline_log(
        logging.INFO,
        "Starting parent mosaic from component SR outputs",
        component_source_count=len(component_sources),
        target_crs=target_crs,
        target_resolution=target_resolution,
        output_path=output_path,
    )

    parent_input_grid = build_target_grid(parent_aoi_bbox, target_crs, target_resolution, target_resolution)
    parent_sr_grid = build_sr_grid_from_input_grid(parent_input_grid, scale_factor=2)
    dst_shape = (int(parent_sr_grid["height"]), int(parent_sr_grid["width"]))
    parent_bands = np.full((2, dst_shape[0], dst_shape[1]), np.nan, dtype=np.float32)
    filled_mask = np.zeros(dst_shape, dtype=bool)

    total_parent_pixels = int(dst_shape[0] * dst_shape[1])
    contributing_component_ids: List[str] = []
    contributing_geometries = []
    component_audit: List[Dict[str, Any]] = []

    sorted_components = sorted(component_sources, key=_component_mosaic_priority_key)
    for priority_rank, component in enumerate(sorted_components, start=1):
        component_geometry = component.get("geometry")
        child_sr_path = component.get("sr_multiband_path")
        if not component_geometry or not child_sr_path:
            continue

        child_path = Path(str(child_sr_path))
        if not child_path.exists():
            raise FileNotFoundError(f"Child SR raster missing for mosaic: {child_path}")

        with rasterio.open(child_path, "r") as src:
            if src.count < 2:
                raise ValueError(f"Child SR raster must have 2 bands: {child_path}")
            child_grid = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
            }
            child_valid_mask = geometry_mask_for_grid(component_geometry, child_grid).astype(bool)
            warped_mask = _reproject_mask_to_grid(
                child_valid_mask,
                src_crs=src.crs,
                src_transform=src.transform,
                dst_crs=parent_sr_grid["crs"],
                dst_transform=parent_sr_grid["transform"],
                dst_shape=dst_shape,
            )

            warped_bands: List[np.ndarray] = []
            for band_index in range(1, 3):
                band = src.read(band_index).astype(np.float32)
                band[~child_valid_mask] = np.nan
                warped = _reproject_band_to_grid(
                    band,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=parent_sr_grid["crs"],
                    dst_transform=parent_sr_grid["transform"],
                    dst_shape=dst_shape,
                    resampling_name="nearest",
                )
                warped_bands.append(warped)

        component_valid = warped_mask & np.isfinite(warped_bands[0]) & np.isfinite(warped_bands[1])
        new_pixels = component_valid & ~filled_mask
        new_pixel_count = int(np.count_nonzero(new_pixels))
        new_pixel_ratio = float(new_pixel_count / total_parent_pixels) if total_parent_pixels > 0 else 0.0
        audit_record = {
            "component_id": str(component.get("component_id") or ""),
            "mosaic_priority_rank": priority_rank,
            "contributed_to_parent_mosaic": bool(new_pixel_count > 0),
            "new_pixel_count": new_pixel_count,
            "new_pixel_ratio": new_pixel_ratio,
        }
        if not np.any(new_pixels):
            component_audit.append(compact_jsonable(audit_record))
            emit_pipeline_log(
                logging.DEBUG,
                "Component contributed no new pixels to parent mosaic",
                component_id=component.get("component_id"),
                mosaic_priority_rank=priority_rank,
            )
            continue

        parent_bands[0, new_pixels] = warped_bands[0][new_pixels]
        parent_bands[1, new_pixels] = warped_bands[1][new_pixels]
        filled_mask[new_pixels] = True
        contributing_component_ids.append(str(component.get("component_id")))
        contributing_geometries.append(shape(component_geometry))
        component_audit.append(compact_jsonable(audit_record))

    if not contributing_component_ids:
        emit_pipeline_log(
            logging.WARNING,
            "Parent mosaic produced no supported pixels",
            component_source_count=len(component_sources),
        )
        raise RuntimeError("Parent mosaic produced no supported pixels from child SR outputs.")

    supported_union_geom = unary_union(contributing_geometries)
    supported_geometry_wgs84 = mapping(supported_union_geom)
    parent_area_m2 = geodesic_area_wgs84(shape(parent_aoi_geometry))
    supported_area_m2 = geodesic_area_wgs84(supported_union_geom)
    supported_area_ratio = float(supported_area_m2 / parent_area_m2) if parent_area_m2 > 0 else 0.0

    output_meta = {
        "crs": rasterio.crs.CRS.from_user_input(parent_sr_grid["crs"]),
        "transform": parent_sr_grid["transform"],
        "descriptions": ("SR_VV", "SR_VH"),
    }
    output_file = write_multiband_geotiff(
        output_path,
        parent_bands,
        output_meta,
        compression=compression,
        tiled=tiled,
        blockxsize=blockxsize,
        blockysize=blockysize,
    )
    emit_pipeline_log(
        logging.INFO,
        "Completed parent mosaic",
        parent_mosaic_ordering="largest_first",
        contributing_component_count=len(contributing_component_ids),
        single_child_mosaic=(len(contributing_component_ids) == 1),
        contributing_component_ids=contributing_component_ids,
        supported_area_ratio=supported_area_ratio,
        output_path=output_file,
    )
    return {
        "sr_multiband_path": str(output_file),
        "supported_geometry": supported_geometry_wgs84,
        "supported_area_m2": supported_area_m2,
        "supported_area_ratio": supported_area_ratio,
        "contributing_component_ids": contributing_component_ids,
        "component_audit": component_audit,
        "parent_mosaic_ordering": "largest_first",
        "grid": {
            "crs": parent_sr_grid["crs"],
            "width": parent_sr_grid["width"],
            "height": parent_sr_grid["height"],
            "transform": list(parent_sr_grid["transform"])[:6],
        },
    }


def _write_export_valid_mask(
    *,
    mask_path: Path,
    valid_mask: np.ndarray,
    crs: Any,
    transform: Affine,
    compression: str,
) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": int(valid_mask.shape[1]),
        "height": int(valid_mask.shape[0]),
        "crs": rasterio.crs.CRS.from_user_input(crs),
        "transform": transform,
        "compress": compression,
    }
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(valid_mask.astype(np.uint8), 1)
        dst.set_band_description(1, "VALID_MASK")


def _pixel_size_meters(
    *,
    crs: Optional[Any],
    transform: Affine,
    width: int,
    height: int,
) -> Optional[float]:
    if crs is None:
        return None
    raster_crs = rasterio.crs.CRS.from_user_input(crs)
    try:
        if not raster_crs.is_geographic:
            gsd_x = abs(float(transform.a))
            gsd_y = abs(float(transform.e))
            return min(v for v in (gsd_x, gsd_y) if v > 0) if (gsd_x > 0 or gsd_y > 0) else None
    except Exception:
        pass

    center_col = width / 2.0
    center_row = height / 2.0
    center_x, center_y = transform * (center_col, center_row)
    east_x, east_y = transform * (center_col + 1.0, center_row)
    south_x, south_y = transform * (center_col, center_row + 1.0)
    geod = Geod(ellps="WGS84")
    _, _, dist_x = geod.inv(center_x, center_y, east_x, east_y)
    _, _, dist_y = geod.inv(center_x, center_y, south_x, south_y)
    positive = [value for value in (abs(dist_x), abs(dist_y)) if value > 0]
    return min(positive) if positive else None


def export_masked_sr_band_cogs(
    *,
    sr_multiband_path: str | Path,
    output_dir: str | Path,
    output_basename: str,
    geometry_wgs84: Optional[Dict[str, Any]] = None,
    output_valid_mask_path: Optional[str | Path] = None,
    compression: str = "DEFLATE",
    blocksize: int = 512,
    band_filename_style: str = "legacy",
    crop_to_valid_data: bool = False,
    include_internal_mask: bool = True,
    persist_valid_mask: bool = False,
    final_target_crs: Optional[str] = None,
    final_target_resolution: Optional[float] = None,
    final_resampling_name: str = "bilinear",
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    sr_multiband_path = Path(sr_multiband_path)
    mask_path = Path(output_valid_mask_path) if output_valid_mask_path else None
    with tempfile.TemporaryDirectory(prefix="sr_mask_") as temp_mask_dir:
        if mask_path is None:
            if geometry_wgs84 is None:
                raise ValueError("geometry_wgs84 or output_valid_mask_path is required to export masked SR COGs.")
            mask_path = write_geometry_mask_like_raster(
                geometry_wgs84,
                sr_multiband_path,
                Path(temp_mask_dir) / f"{output_basename}_valid_mask.tif",
                compression=compression,
            )

        with rasterio.open(mask_path) as mask_src:
            valid_mask = mask_src.read(1).astype(bool)

        outputs: Dict[str, Any] = {
            "output_valid_mask_path": None,
            "output_sr_vv_tif": None,
            "output_sr_vh_tif": None,
            "output_sr_band_tifs": [],
        }

        with rasterio.open(sr_multiband_path) as src:
            if src.width != valid_mask.shape[1] or src.height != valid_mask.shape[0]:
                raise ValueError("SR raster and valid mask dimensions do not match.")

            src_crs = src.crs
            source_band_template = src.read(1).astype(np.float32)
            source_band_template[~valid_mask] = np.nan
            source_transform = src.transform
            source_mask = valid_mask
            if crop_to_valid_data:
                source_band_template, source_mask, source_transform = _crop_masked_extent(
                    source_band_template,
                    source_mask,
                    source_transform,
                )

            export_mask = source_mask
            export_transform = source_transform
            export_crs = src_crs
            if final_target_crs:
                requested_crs = rasterio.crs.CRS.from_user_input(final_target_crs)
                if src_crs is None or requested_crs != src_crs or final_target_resolution is not None:
                    export_mask, export_transform = _reproject_mask_and_transform(
                        source_mask,
                        src_crs=src_crs,
                        src_transform=source_transform,
                        dst_crs=str(requested_crs),
                        dst_resolution=final_target_resolution,
                    )
                    export_crs = requested_crs
                    if crop_to_valid_data:
                        export_band_template = np.where(export_mask, 1.0, np.nan).astype(np.float32)
                        export_band_template, export_mask, export_transform = _crop_masked_extent(
                            export_band_template,
                            export_mask,
                            export_transform,
                        )

            if persist_valid_mask:
                persisted_mask_path = Path(output_valid_mask_path) if output_valid_mask_path else (output_dir / f"{output_basename}_valid_mask.tif")
                _write_export_valid_mask(
                    mask_path=persisted_mask_path,
                    valid_mask=export_mask,
                    crs=export_crs,
                    transform=export_transform,
                    compression=compression,
                )
                outputs["output_valid_mask_path"] = str(persisted_mask_path)

            safe_blocksize = _safe_tiff_blocksize(export_mask.shape[1], export_mask.shape[0], preferred=blocksize)
            cog_options = _cog_creation_options(compression=compression, blocksize=safe_blocksize)

            for band_index in range(1, src.count + 1):
                description = src.descriptions[band_index - 1] if src.descriptions else None
                band_slug = _sr_band_slug(description, band_index)
                if band_filename_style == "whole_monthly_public":
                    band_suffix = {"SR_VV": "vv", "SR_VH": "vh"}.get(band_slug, band_slug.lower())
                    dest_path = output_dir / f"{output_basename}_{band_suffix}.tif"
                else:
                    dest_path = output_dir / f"{output_basename}_{band_slug}_x2.tif"
                band_data = src.read(band_index).astype(np.float32)
                band_data[~valid_mask] = np.nan
                if crop_to_valid_data:
                    band_data, _, _ = _crop_masked_extent(
                        band_data,
                        valid_mask,
                        src.transform,
                    )
                if final_target_crs and export_crs is not None and (export_crs != src_crs or final_target_resolution is not None):
                    band_data = _reproject_band_to_grid(
                        band_data,
                        src_crs=src_crs,
                        src_transform=source_transform,
                        dst_crs=str(export_crs),
                        dst_transform=export_transform,
                        dst_shape=export_mask.shape,
                        resampling_name=final_resampling_name,
                    )
                band_data[~export_mask] = np.nan

                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    width=int(export_mask.shape[1]),
                    height=int(export_mask.shape[0]),
                    transform=export_transform,
                    crs=export_crs,
                    compress=str(compression or "DEFLATE"),
                    tiled=True,
                    blockxsize=int(safe_blocksize),
                    blockysize=int(safe_blocksize),
                    nodata=np.nan,
                )

                fd, tmp_name = tempfile.mkstemp(
                    suffix=".tif",
                    prefix=f"{output_basename}_{band_slug}_",
                    dir=str(output_dir),
                )
                os.close(fd)
                Path(tmp_name).unlink(missing_ok=True)
                tmp_path = Path(tmp_name)
                try:
                    with rasterio.open(tmp_path, "w", **profile) as tmp_dst:
                        tmp_dst.write(band_data, 1)
                        tmp_dst.set_band_description(1, band_slug)
                        if include_internal_mask:
                            tmp_dst.write_mask(export_mask.astype(np.uint8) * 255)
                    rio_copy(tmp_path, dest_path, driver="COG", **cog_options)
                finally:
                    tmp_path.unlink(missing_ok=True)

                band_record = {"band": band_slug, "path": str(dest_path)}
                outputs["output_sr_band_tifs"].append(band_record)
                if band_slug == "SR_VV":
                    outputs["output_sr_vv_tif"] = str(dest_path)
                elif band_slug == "SR_VH":
                    outputs["output_sr_vh_tif"] = str(dest_path)

        return outputs


def _local_href(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _join_url(base: str, *parts: str) -> str:
    cleaned = [str(base).rstrip("/")]
    cleaned.extend(str(part).strip("/") for part in parts if str(part).strip("/"))
    return "/".join(cleaned)


def _infer_sr_collection_name(summary: Dict[str, Any]) -> str:
    workflow_mode = str(summary.get("workflow_mode") or "sar_sr")
    if workflow_mode == "exact_pair":
        return "issm-sar-sr-x2-exact"
    if summary.get("period") is not None and summary.get("component") is not None:
        return "issm-sar-sr-x2-monthly-component"
    if summary.get("period") is not None:
        return "issm-sar-sr-x2-monthly"
    if summary.get("anchor") is not None:
        return "issm-sar-sr-x2-anchor-window"
    return "issm-sar-sr-x2"


def _resolve_sr_collection_name(summary: Dict[str, Any]) -> str:
    inferred = _infer_sr_collection_name(summary)
    workflow_mode = str(summary.get("workflow_mode") or "sar_sr")
    if workflow_mode == "exact_pair":
        return os.getenv("SR_COLLECTION_ID_EXACT", inferred)
    if summary.get("period") is not None and summary.get("component") is not None:
        return os.getenv("SR_COLLECTION_ID_MONTHLY_COMPONENT", inferred)
    if summary.get("period") is not None:
        return os.getenv("SR_COLLECTION_ID_MONTHLY", inferred)
    if summary.get("anchor") is not None:
        return os.getenv("SR_COLLECTION_ID_ANCHOR_WINDOW", inferred)
    return os.getenv("SR_COLLECTION_ID_DEFAULT", inferred)


def _summary_aoi_id(summary: Dict[str, Any]) -> Optional[str]:
    explicit = str(summary.get("aoi_id") or "").strip()
    if explicit:
        return explicit

    for candidate_key in ("period_dir", "run_dir"):
        candidate = summary.get(candidate_key)
        if not candidate:
            continue
        try:
            path = Path(str(candidate)).resolve()
        except Exception:
            path = Path(str(candidate))
        parts = list(path.parts)
        if "aois" in parts:
            idx = parts.index("aois")
            if idx + 1 < len(parts):
                derived = str(parts[idx + 1]).strip()
                if derived:
                    return derived

    aoi_geojson = summary.get("aoi_geojson")
    if not aoi_geojson:
        return None
    try:
        stem = Path(str(aoi_geojson)).stem
        if stem and stem != "input_aoi":
            return stem
        return None
    except Exception:
        return None


def _summary_period_token(summary: Dict[str, Any]) -> Optional[str]:
    if summary.get("period"):
        return summary["period"].get("period_id")
    if summary.get("anchor"):
        return summary["anchor"].get("anchor_date") or summary["anchor"].get("pair_id")
    if summary.get("selected_pair"):
        return summary["selected_pair"].get("pair_id")
    return None


def _normalize_public_token(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    chars: List[str] = []
    last_was_sep = False
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_")


def _whole_monthly_sr_item_id(aoi_id: str, period_id: str) -> str:
    normalized_aoi = _normalize_public_token(aoi_id)
    normalized_period = _normalize_public_token(period_id)
    if not normalized_aoi or not normalized_period:
        raise ValueError("Whole-monthly SR item id requires non-empty AOI id and period id.")
    collection_stem = _normalize_public_token(os.getenv("SR_COLLECTION_ID_MONTHLY", "sentinel-1sr-5m-monthly"))
    if not collection_stem:
        collection_stem = "sentinel-1sr-5m-monthly"
    return f"{collection_stem}_{normalized_period}_{normalized_aoi}"


def _summary_whole_monthly_item_id(summary: Dict[str, Any]) -> Optional[str]:
    period = summary.get("period") or {}
    if not period or summary.get("component") is not None:
        return None
    aoi_id = _summary_aoi_id(summary)
    period_id = period.get("period_id")
    if not aoi_id or not period_id:
        return None
    return _whole_monthly_sr_item_id(aoi_id, period_id)


def _summary_period_year_month(summary: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    period_id = _summary_period_token(summary)
    if not period_id:
        return None, None
    text = str(period_id).strip()
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        return text[:4], text[5:7]
    return None, None


def _resolve_sr_item_id(summary: Dict[str, Any], fallback: Optional[str | Path] = None) -> str:
    explicit = summary.get("public_item_id") or summary.get("item_id")
    if explicit:
        return str(explicit)
    monthly_item_id = _summary_whole_monthly_item_id(summary)
    if monthly_item_id:
        return monthly_item_id
    if fallback is not None:
        return Path(str(fallback)).stem
    output_tif = summary.get("output_tif")
    if output_tif:
        return Path(str(output_tif)).stem
    raise ValueError("Unable to resolve SR item id.")


def _resolve_sr_s3_prefix(summary: Dict[str, Any]) -> str:
    if summary.get("period") is not None and summary.get("component") is not None:
        return os.getenv("SR_S3_PREFIX_MONTHLY_COMPONENT", "issm-sar-sr-x2/monthly-component")
    if summary.get("period") is not None:
        return os.getenv("SR_S3_PREFIX_MONTHLY", "issm-sar-sr-x2/monthly")
    if summary.get("anchor") is not None:
        return os.getenv("SR_S3_PREFIX_ANCHOR_WINDOW", "issm-sar-sr-x2/anchor-window")
    if str(summary.get("workflow_mode") or "") == "exact_pair":
        return os.getenv("SR_S3_PREFIX_EXACT", "issm-sar-sr-x2/exact")
    return os.getenv("SR_S3_PREFIX_DEFAULT", "issm-sar-sr-x2")


def _resolve_sr_object_key(summary: Dict[str, Any], item_id: str, filename: str) -> str:
    prefix = _resolve_sr_s3_prefix(summary).strip("/")
    aoi_id = _summary_aoi_id(summary) or "unknown-aoi"
    period_token = _summary_period_token(summary)
    parts = [prefix, aoi_id]
    year, month = _summary_period_year_month(summary)
    if summary.get("period") is not None and summary.get("component") is None and year and month:
        parts.extend([year, month])
    elif period_token:
        parts.append(str(period_token))
    parts.append(item_id)
    parts.append(filename)
    return "/".join(part.strip("/") for part in parts if str(part).strip("/"))


def _resolve_sr_href(
    summary: Dict[str, Any],
    item_id: str,
    filename: str | Path,
    *,
    mode: str,
    published_filename: Optional[str] = None,
) -> str:
    normalized_mode = str(mode or "local").strip().lower()
    if normalized_mode == "local":
        return _local_href(filename)
    object_key = _resolve_sr_object_key(summary, item_id, published_filename or Path(filename).name)
    if normalized_mode == "s3":
        bucket = os.getenv("SR_S3_BUCKET")
        if not bucket:
            raise ValueError("SR_S3_BUCKET is required when SR href mode is 's3'.")
        return f"s3://{bucket}/{object_key}"
    if normalized_mode == "http":
        base_url = os.getenv("SR_PUBLIC_BASE_URL")
        if not base_url:
            raise ValueError("SR_PUBLIC_BASE_URL is required when SR href mode is 'http'.")
        return _join_url(base_url, object_key)
    if normalized_mode == "stac_api":
        root_url = _resolve_sr_stac_root_url()
        collection_name = _resolve_sr_collection_name(summary)
        if not root_url:
            raise ValueError("SR_STAC_ROOT_URL or STAC_API_URL is required when item href mode is 'stac_api'.")
        return _join_url(root_url, "collections", collection_name, "items", item_id)
    raise ValueError(f"Unsupported SR href mode: {mode}")


def _resolve_sr_root_links(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    root_url = _resolve_sr_stac_root_url()
    if not root_url:
        return []
    collection_name = _resolve_sr_collection_name(summary)
    collection_href = _join_url(root_url, "collections", collection_name)
    return [
        {"rel": "collection", "type": "application/json", "href": collection_href},
        {"rel": "parent", "type": "application/json", "href": collection_href},
        {"rel": "root", "type": "application/json", "href": root_url.rstrip("/") + "/"},
    ]


def _resolve_sr_stac_root_url() -> str:
    return str(os.getenv("SR_STAC_ROOT_URL") or os.getenv("STAC_API_URL") or "").strip()


def _summarize_source_items(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        info = extract_item_info(item)
        item_id = str(info.get("id") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        self_href = None
        for link in item.get("links", []):
            if str(link.get("rel") or "").lower() == "self" and link.get("href"):
                self_href = str(link["href"])
                break
        summaries.append(
            {
                "id": item_id,
                "datetime": info.get("datetime"),
                "platform": info.get("platform"),
                "orbit_state": info.get("orbit_state"),
                "relative_orbit": info.get("relative_orbit"),
                "slice_number": info.get("slice_number"),
                "aoi_geometry_coverage": item.get("_aoi_coverage"),
                "aoi_bbox_coverage": item.get("_aoi_bbox_coverage"),
                "coverage_source": item.get("_coverage_source"),
                "self_href": self_href,
            }
        )
    return summaries


def _raster_asset_type(src: rasterio.io.DatasetReader) -> str:
    layout = str(src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT", "")).upper()
    if layout == "COG":
        return "image/tiff; application=geotiff; profile=cloud-optimized"
    return "image/tiff; application=geotiff"


def _raster_asset_metadata(
    path: str | Path,
    *,
    title: str,
    roles: List[str],
    description: Optional[str] = None,
    eo_bands: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    asset_path = Path(path)
    with rasterio.open(asset_path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        gsd = _pixel_size_meters(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
        )
        raster_bands: List[Dict[str, Any]] = []
        for band_index in range(src.count):
            band_meta: Dict[str, Any] = {
                "nodata": src.nodatavals[band_index],
                "data_type": src.dtypes[band_index],
            }
            if gsd is not None:
                band_meta["spatial_resolution"] = gsd
            raster_bands.append(to_jsonable(band_meta))

        left, bottom, right, top = src.bounds
        asset: Dict[str, Any] = {
            "href": _local_href(asset_path),
            "type": _raster_asset_type(src),
            "roles": roles,
            "title": title,
            "proj:bbox": [float(left), float(bottom), float(right), float(top)],
            "proj:shape": [int(src.height), int(src.width)],
            "proj:transform": list(src.transform)[:6],
            "raster:bands": raster_bands,
        }
        if description:
            asset["description"] = description
        if gsd is not None:
            asset["gsd"] = gsd
        if epsg is not None:
            asset["proj:epsg"] = epsg
        if eo_bands:
            asset["eo:bands"] = eo_bands
    return asset


def _json_asset_metadata(path: str | Path, *, title: str, roles: List[str], asset_type: str) -> Dict[str, Any]:
    asset_path = Path(path)
    return {
        "href": _local_href(asset_path),
        "type": asset_type,
        "roles": roles,
        "title": title,
    }


def _strip_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none_values(v) for v in value]
    return value


def write_sr_output_geojson(
    *,
    out_path: str | Path,
    summary: Dict[str, Any],
    geometry_wgs84: Dict[str, Any],
    infer_config: Dict[str, Any],
    source_t1_items: Optional[List[Dict[str, Any]]] = None,
    source_t2_items: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sr_vv_path = summary.get("output_sr_vv_tif")
    sr_vh_path = summary.get("output_sr_vh_tif")
    if not sr_vv_path or not sr_vh_path:
        raise ValueError("SR GeoJSON requires SR_VV and SR_VH outputs.")

    source_t1 = _summarize_source_items(source_t1_items)
    source_t2 = _summarize_source_items(source_t2_items)

    nominal_datetime = None
    start_datetime = None
    end_datetime = None
    pair_id = None
    if summary.get("period"):
        period = summary["period"]
        nominal_datetime = period.get("period_anchor_datetime")
        start_datetime = period.get("period_start")
        end_datetime = period.get("period_end")
        pair_id = summary.get("component", {}).get("pair_id") or f"period_{period.get('period_id')}"
    elif summary.get("anchor"):
        anchor = summary["anchor"]
        nominal_datetime = anchor.get("anchor_datetime")
        pair_id = anchor.get("pair_id")
        if source_t2:
            start_datetime = source_t2[0].get("datetime")
        if source_t1:
            end_datetime = source_t1[-1].get("datetime")
    elif summary.get("selected_pair"):
        selected_pair = summary["selected_pair"]
        nominal_datetime = selected_pair.get("t1_datetime")
        start_datetime = selected_pair.get("t2_datetime")
        end_datetime = selected_pair.get("t1_datetime")
        pair_id = selected_pair.get("pair_id")

    item_id = _resolve_sr_item_id(summary, fallback=out_path)
    collection_name = _resolve_sr_collection_name(summary)
    asset_href_mode = os.getenv("SR_ASSET_HREF_MODE", "local")
    item_href_mode = os.getenv("SR_ITEM_SELF_MODE", "local")
    include_local_source_paths = _env_flag("SR_INCLUDE_LOCAL_SOURCE_PATHS", default=False)
    whole_monthly_public = summary.get("period") is not None and summary.get("component") is None
    sr_vv_publish_name = f"{item_id}_vv.tif" if whole_monthly_public else Path(str(sr_vv_path)).name
    sr_vh_publish_name = f"{item_id}_vh.tif" if whole_monthly_public else Path(str(sr_vh_path)).name

    sr_vv_asset = _raster_asset_metadata(
        sr_vv_path,
        title="SR VV (x2)",
        roles=["data"],
        description="Super-resolved VV backscatter output masked by the valid geometry mask.",
        eo_bands=[{"name": "VV", "description": "Super-resolved VV backscatter"}],
    )
    sr_vh_asset = _raster_asset_metadata(
        sr_vh_path,
        title="SR VH (x2)",
        roles=["data"],
        description="Super-resolved VH backscatter output masked by the valid geometry mask.",
        eo_bands=[{"name": "VH", "description": "Super-resolved VH backscatter"}],
    )
    sr_vv_asset["href"] = _resolve_sr_href(
        summary,
        item_id,
        sr_vv_path,
        mode=asset_href_mode,
        published_filename=sr_vv_publish_name,
    )
    sr_vh_asset["href"] = _resolve_sr_href(
        summary,
        item_id,
        sr_vh_path,
        mode=asset_href_mode,
        published_filename=sr_vh_publish_name,
    )
    assets: Dict[str, Any] = {
        "sr_vv": sr_vv_asset,
        "sr_vh": sr_vh_asset,
    }

    gsd = sr_vv_asset.get("gsd") or sr_vh_asset.get("gsd")
    proj_epsg = sr_vv_asset.get("proj:epsg") or sr_vh_asset.get("proj:epsg")
    input_semantics = summary.get("inference_input_semantics") or {}

    common_properties: Dict[str, Any] = {
        "created": _utc_rfc3339(datetime.now(timezone.utc)),
        "datetime": nominal_datetime,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "gsd": gsd,
        "proj:epsg": proj_epsg,
        "constellation": "sentinel-1",
        "instruments": ["C-SAR"],
        "sar:instrument_mode": "IW",
        "sar:frequency_band": "C",
        "sar:product_type": "GRD",
        "sar:polarizations": ["VV", "VH"],
        "sr:method": "ISSM-SAR",
        "sr:model": "ISSM-SAR x2 dual-polarization",
        "sr:scale_factor": 2,
        "sr:product_version": os.getenv("SR_PRODUCT_VERSION"),
        "sr:publisher": os.getenv("SR_PUBLISHER"),
        "sr:license": os.getenv("SR_LICENSE"),
    }
    if whole_monthly_public:
        properties: Dict[str, Any] = {
            **common_properties,
            "source:aoi_id": _summary_aoi_id(summary),
            "source:s1t1_items": source_t1,
            "source:s1t2_items": source_t2,
            "source:s1t1_scene_ids": [item["id"] for item in source_t1],
            "source:s1t2_scene_ids": [item["id"] for item in source_t2],
        }
    else:
        properties = {
            **common_properties,
            "sr:model_vv_checkpoint": Path(str(infer_config.get("ckpt_path_vv", ""))).name or None,
            "sr:model_vh_checkpoint": Path(str(infer_config.get("ckpt_path_vh", ""))).name or None,
            "source:bands_used": ["VV", "VH"],
            "source:workflow_mode": summary.get("workflow_mode"),
            "source:selection_strategy": summary.get("selection_strategy"),
            "source:pair_id": pair_id,
            "source:aoi_id": _summary_aoi_id(summary),
            "source:aoi_geojson": str(Path(summary["aoi_geojson"]).resolve()) if include_local_source_paths and summary.get("aoi_geojson") else None,
            "source:backend": "stac" if str(summary.get("workflow_mode", "")).startswith("stac") or summary.get("workflow_mode") == "exact_pair" else "gee",
            "source:input_semantics": input_semantics,
            "source:s1t1_label": input_semantics.get("t1_label"),
            "source:s1t2_label": input_semantics.get("t2_label"),
            "source:s1t1_role": input_semantics.get("t1_role"),
            "source:s1t2_role": input_semantics.get("t2_role"),
            "source:s1t1_scene_count": len(source_t1),
            "source:s1t2_scene_count": len(source_t2),
            "source:s1t1_items": source_t1,
            "source:s1t2_items": source_t2,
            "source:s1t1_scene_ids": [item["id"] for item in source_t1],
            "source:s1t2_scene_ids": [item["id"] for item in source_t2],
        }
    if summary.get("period"):
        period = summary["period"]
        properties.update(
            {
                "source:period_id": period.get("period_id"),
                "source:period_mode": period.get("period_mode"),
                "source:period_start": period.get("period_start"),
                "source:period_end": period.get("period_end"),
                "source:period_anchor_datetime": period.get("period_anchor_datetime"),
            }
        )
    if summary.get("selection") and not whole_monthly_public:
        selection = summary["selection"]
        properties.update(
            {
                "source:selected_relaxation_level": selection.get("selected_relaxation_level"),
                "source:selected_relaxation_name": selection.get("selected_relaxation_name"),
                "source:required_scene_count": selection.get("required_scene_count"),
                "source:scene_signature_mode": selection.get("scene_signature_mode"),
                "source:scene_signature_value": selection.get("scene_signature_value"),
                "source:latest_input_datetime": selection.get("latest_input_datetime"),
            }
        )
    if summary.get("component"):
        component = summary["component"]
        properties.update(
            {
                "source:component_id": component.get("component_id"),
                "source:component_pair_id": component.get("pair_id"),
                "source:component_area_m2": component.get("area_m2"),
                "source:component_area_ratio_vs_parent": component.get("area_ratio_vs_parent"),
                "source:component_seed_item_ids": component.get("seed_item_ids"),
            }
        )
    if summary.get("selected_pair"):
        selected_pair = summary["selected_pair"]
        properties.update(
            {
                "source:exact_pair_id": selected_pair.get("pair_id"),
                "source:exact_t1_id": selected_pair.get("t1_id"),
                "source:exact_t2_id": selected_pair.get("t2_id"),
                "source:exact_t1_datetime": selected_pair.get("t1_datetime"),
                "source:exact_t2_datetime": selected_pair.get("t2_datetime"),
            }
        )

    links: List[Dict[str, Any]] = _resolve_sr_root_links(summary)
    links.append(
        {
            "rel": "self",
            "type": "application/json" if whole_monthly_public else "application/geo+json",
            "href": _resolve_sr_href(
                summary,
                item_id,
                out_path,
                mode=item_href_mode,
                published_filename=f"{item_id}.json" if whole_monthly_public else Path(out_path).name,
            ),
        }
    )
    seen_source_hrefs: set[str] = set()
    for item in source_t1 + source_t2:
        href = item.get("self_href")
        if not href or href in seen_source_hrefs:
            continue
        seen_source_hrefs.add(href)
        links.append(
            {
                "rel": "derived_from",
                "type": "application/geo+json",
                "href": href,
                "title": item.get("id"),
            }
        )

    feature = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
        ],
        "id": item_id,
        "collection": collection_name,
        "geometry": geometry_wgs84,
        "bbox": canonical_bbox_from_geometry(geometry_wgs84),
        "properties": _strip_none_values(properties),
        "assets": _strip_none_values(assets),
        "links": links,
    }
    out_path.write_text(json.dumps(to_jsonable(feature), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out_path


def attach_sr_output_geojson(
    *,
    summary: Dict[str, Any],
    geometry_wgs84: Dict[str, Any],
    infer_config: Dict[str, Any],
    source_t1_items: Optional[List[Dict[str, Any]]] = None,
    source_t2_items: Optional[List[Dict[str, Any]]] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    output_tif = summary.get("output_tif")
    if out_path is None:
        preferred_source = summary.get("output_sr_vv_tif") or summary.get("output_sr_vh_tif") or output_tif
        if not preferred_source:
            raise ValueError("SR metadata path requires at least one SR output band or explicit out_path.")
        item_id = _resolve_sr_item_id(summary, fallback=preferred_source)
        preferred_dir = Path(str(preferred_source)).parent
        out_path = preferred_dir / f"{item_id}.json"
    geojson_path = write_sr_output_geojson(
        out_path=out_path,
        summary=summary,
        geometry_wgs84=geometry_wgs84,
        infer_config=infer_config,
        source_t1_items=source_t1_items,
        source_t2_items=source_t2_items,
    )
    summary["output_sr_geojson_path"] = str(geojson_path)
    return geojson_path


def _component_candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, str]:
    return (
        -float(candidate["area_m2"]),
        candidate["bbox"][0],
        candidate["bbox"][1],
        str(candidate["candidate_region_key"]),
    )


def _component_tolerant_containment_metrics(container: Dict[str, Any], inner: Dict[str, Any]) -> Dict[str, float | bool]:
    container_geom = shape(container["geometry"])
    inner_geom = shape(inner["geometry"])
    inner_area_m2 = float(inner.get("area_m2") or geodesic_area_wgs84(inner_geom) or 0.0)
    if container_geom.is_empty or inner_geom.is_empty:
        return {
            "contained": False,
            "difference_area_m2": inner_area_m2,
            "difference_area_ratio": 1.0 if inner_area_m2 > 0 else 0.0,
            "tolerance_m2": 0.0,
        }

    exact_contains = bool(container_geom.equals(inner_geom) or container_geom.covers(inner_geom))
    difference_geom = normalize_polygonal_shapely_geometry(inner_geom.difference(container_geom))
    difference_area_m2 = float(geodesic_area_wgs84(difference_geom))
    tolerance_m2 = max(1.0, 1e-8 * max(inner_area_m2, 0.0))
    difference_area_ratio = float(difference_area_m2 / inner_area_m2) if inner_area_m2 > 0 else 0.0
    return {
        "contained": bool(exact_contains or difference_area_m2 <= tolerance_m2),
        "difference_area_m2": difference_area_m2,
        "difference_area_ratio": difference_area_ratio,
        "tolerance_m2": tolerance_m2,
    }


def _component_geometry_contains(container: Dict[str, Any], inner: Dict[str, Any]) -> bool:
    return bool(_component_tolerant_containment_metrics(container, inner)["contained"])


def _component_rejection_message(reasons: List[str]) -> str:
    if "CONTAINED_BY_VALID_LARGER_REGION" in reasons:
        return "Suppressed because a larger child already covers this region within geometric tolerance."
    if "NO_VALID_REPRESENTATIVE_SELECTION" in reasons:
        return "Rejected because no valid representative pre/post scene pools remained for this child region."
    if "PRE_SCENE_COUNT_BELOW_MIN" in reasons and "POST_SCENE_COUNT_BELOW_MIN" in reasons:
        return "Rejected because both pre and post pools fell below the minimum scene count."
    if "PRE_SCENE_COUNT_BELOW_MIN" in reasons:
        return "Rejected because the pre-scene pool fell below the minimum scene count."
    if "POST_SCENE_COUNT_BELOW_MIN" in reasons:
        return "Rejected because the post-scene pool fell below the minimum scene count."
    if "NO_PRE_ITEMS_MEET_REGION_COVERAGE_THRESHOLD" in reasons:
        return "Rejected because no pre-scene met the child coverage threshold."
    if "NO_POST_ITEMS_MEET_REGION_COVERAGE_THRESHOLD" in reasons:
        return "Rejected because no post-scene met the child coverage threshold."
    if "REGION_AREA_RATIO_BELOW_MIN" in reasons:
        return "Rejected because the child region was smaller than the configured minimum parent-area ratio."
    return "Rejected because the child region did not pass the component-selection rules."


def _component_selection_decision_summary(component: Dict[str, Any], *, contributed: Optional[bool] = None) -> Dict[str, Any]:
    selection = component.get("selection") or {}
    summary = {
        "selected_relaxation_name": selection.get("selected_relaxation_name"),
        "scene_signature_mode": selection.get("scene_signature_mode"),
        "pre_scene_count": selection.get("pre_scene_count"),
        "post_scene_count": selection.get("post_scene_count"),
        "area_ratio_vs_parent": component.get("area_ratio_vs_parent"),
    }
    if contributed is not None:
        summary["contributed_to_parent_mosaic"] = bool(contributed)
    return compact_jsonable(summary)


def select_seed_intersection_component_candidates(
    *,
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    parent_aoi_geometry: Dict[str, Any],
    parent_aoi_bbox: List[float],
    period: Dict[str, Any],
    min_scenes_per_half: int,
    auto_relax_inside_period: bool,
    require_same_orbit_direction: bool,
    representative_pool_mode: str = "auto",
    component_item_min_coverage: float,
    component_min_area_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = build_seed_intersection_region_candidates(
        pre_items=pre_items,
        post_items=post_items,
        parent_aoi_geometry=parent_aoi_geometry,
        parent_aoi_bbox=parent_aoi_bbox,
        min_region_coverage=component_item_min_coverage,
        min_region_area_ratio=component_min_area_ratio,
    )
    emit_pipeline_log(
        logging.INFO,
        "Built component child candidates",
        period_id=period.get("period_id"),
        candidate_count=len(candidates),
        pre_items=len(pre_items),
        post_items=len(post_items),
        component_item_min_coverage=component_item_min_coverage,
        component_min_area_ratio=component_min_area_ratio,
    )

    successful: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    anchor_dt = datetime.fromisoformat(period["period_anchor_datetime"].replace("Z", "+00:00"))

    for candidate in candidates:
        candidate_record = {
            "candidate_region_key": candidate["candidate_region_key"],
            "geometry": candidate["geometry"],
            "bbox": candidate["bbox"],
            "area_m2": candidate["area_m2"],
            "area_ratio_vs_parent": candidate["area_ratio_vs_parent"],
            "seed_item_ids": sorted(candidate["seed_item_ids"]),
            "seed_item_datetimes": sorted(candidate["seed_item_datetimes"]),
            "membership_coverage_threshold": candidate.get("membership_coverage_threshold", component_item_min_coverage),
            "pre_considered_items": candidate.get("pre_considered_items", []),
            "post_considered_items": candidate.get("post_considered_items", []),
            "pre_accepted_items": candidate.get("pre_accepted_items", []),
            "post_accepted_items": candidate.get("post_accepted_items", []),
            "pre_considered_item_ids": candidate.get("pre_considered_item_ids", []),
            "post_considered_item_ids": candidate.get("post_considered_item_ids", []),
            "pre_considered_item_count": candidate.get("pre_considered_item_count", 0),
            "post_considered_item_count": candidate.get("post_considered_item_count", 0),
            "pre_considered_scene_count": candidate.get("pre_considered_scene_count", 0),
            "post_considered_scene_count": candidate.get("post_considered_scene_count", 0),
            "pre_considered_coverage_stats": candidate.get("pre_considered_coverage_stats", {}),
            "post_considered_coverage_stats": candidate.get("post_considered_coverage_stats", {}),
            "pre_covering_item_ids": candidate["pre_covering_item_ids"],
            "post_covering_item_ids": candidate["post_covering_item_ids"],
            "pre_covering_item_count": candidate["pre_covering_item_count"],
            "post_covering_item_count": candidate["post_covering_item_count"],
            "pre_covering_scene_count": candidate["pre_covering_scene_count"],
            "post_covering_scene_count": candidate["post_covering_scene_count"],
            "pre_covering_coverage_stats": candidate.get("pre_covering_coverage_stats", {}),
            "post_covering_coverage_stats": candidate.get("post_covering_coverage_stats", {}),
            "status": "candidate",
        }
        candidate_reasons = list(candidate.get("reject_reasons", []))

        pre_covering_items = dedupe_items_by_scene(copy.deepcopy(candidate["pre_covering_items"]))
        post_covering_items = dedupe_items_by_scene(copy.deepcopy(candidate["post_covering_items"]))
        candidate_min_scene_count = 1 if auto_relax_inside_period else int(min_scenes_per_half)
        if candidate.get("pre_considered_scene_count", 0) > 0 and len(pre_covering_items) == 0:
            candidate_reasons.append("NO_PRE_ITEMS_MEET_REGION_COVERAGE_THRESHOLD")
        if candidate.get("post_considered_scene_count", 0) > 0 and len(post_covering_items) == 0:
            candidate_reasons.append("NO_POST_ITEMS_MEET_REGION_COVERAGE_THRESHOLD")
        if len(pre_covering_items) < candidate_min_scene_count:
            if candidate.get("pre_considered_scene_count", 0) >= candidate_min_scene_count:
                candidate_reasons.append("PRE_REGION_COVERAGE_FILTER_REDUCED_SCENE_COUNT_BELOW_MIN")
            candidate_reasons.append("PRE_SCENE_COUNT_BELOW_MIN")
        if len(post_covering_items) < candidate_min_scene_count:
            if candidate.get("post_considered_scene_count", 0) >= candidate_min_scene_count:
                candidate_reasons.append("POST_REGION_COVERAGE_FILTER_REDUCED_SCENE_COUNT_BELOW_MIN")
            candidate_reasons.append("POST_SCENE_COUNT_BELOW_MIN")

        if candidate_reasons:
            candidate_record["status"] = "rejected"
            candidate_record["reject_reasons"] = candidate_reasons
            rejected.append(candidate_record)
            emit_pipeline_log(
                logging.DEBUG,
                "Rejected component child candidate before representative selection",
                candidate_region_key=candidate["candidate_region_key"],
                bbox=candidate["bbox"],
                area_ratio_vs_parent=candidate["area_ratio_vs_parent"],
                reject_reasons=candidate_reasons,
            )
            continue

        selection = select_representative_scene_pools(
            pre_items=pre_covering_items,
            post_items=post_covering_items,
            aoi_geometry=candidate["geometry"],
            aoi_bbox=candidate["bbox"],
            anchor_dt=anchor_dt,
            min_scenes_per_half=min_scenes_per_half,
            auto_relax_inside_period=auto_relax_inside_period,
            require_same_orbit_direction=require_same_orbit_direction,
            representative_pool_mode=representative_pool_mode,
        )
        if selection is None:
            candidate_record["status"] = "rejected"
            candidate_record["reason_code"] = "NO_VALID_REPRESENTATIVE_SELECTION"
            candidate_record["reason_message"] = (
                "No valid representative pre/post pools remained after signature filtering for this child region."
            )
            candidate_record["reject_reasons"] = [
                "NO_VALID_REPRESENTATIVE_SELECTION",
                "No valid representative pre/post pools remained after signature filtering for this child region.",
            ]
            rejected.append(candidate_record)
            emit_pipeline_log(
                logging.DEBUG,
                "Rejected component child candidate after representative selection",
                candidate_region_key=candidate["candidate_region_key"],
                bbox=candidate["bbox"],
                area_ratio_vs_parent=candidate["area_ratio_vs_parent"],
                reject_reasons=candidate_record["reject_reasons"],
            )
            continue

        candidate_record["status"] = "selected"
        candidate_record["selection"] = selection
        candidate_record["why_kept"] = (
            "Kept because the child passed coverage checks and retained a valid representative pre/post scene pool."
        )
        candidate_record["human_summary"] = (
            f"Selected child region with {selection.get('pre_scene_count')} pre scenes and "
            f"{selection.get('post_scene_count')} post scenes."
        )
        candidate_record["decision_summary"] = _component_selection_decision_summary(candidate_record)
        successful.append(candidate_record)

    successful.sort(key=_component_candidate_sort_key)
    kept: List[Dict[str, Any]] = []
    for candidate in successful:
        containment = next(
            (
                (existing, _component_tolerant_containment_metrics(existing, candidate))
                for existing in kept
                if _component_geometry_contains(existing, candidate)
            ),
            None,
        )
        container = containment[0] if containment is not None else None
        if container is not None:
            metrics = containment[1]
            rejected.append(
                {
                    "candidate_region_key": candidate["candidate_region_key"],
                    "geometry": candidate["geometry"],
                    "bbox": candidate["bbox"],
                    "area_m2": candidate["area_m2"],
                    "area_ratio_vs_parent": candidate["area_ratio_vs_parent"],
                    "seed_item_ids": candidate["seed_item_ids"],
                    "seed_item_datetimes": candidate["seed_item_datetimes"],
                    "status": "suppressed",
                    "reason_code": "SUPPRESSED_AS_NESTED_CHILD",
                    "reject_reasons": ["CONTAINED_BY_VALID_LARGER_REGION"],
                    "suppressed_by_region_key": container["candidate_region_key"],
                    "containment_difference_area_m2": metrics["difference_area_m2"],
                    "containment_difference_area_ratio": metrics["difference_area_ratio"],
                    "containment_tolerance_m2": metrics["tolerance_m2"],
                    "why_rejected": "Suppressed because a larger child already covered this region within tolerance.",
                    "reason_message": "Suppressed because a larger child already covered this region within tolerance.",
                    "human_summary": (
                        "Suppressed as a near-nested child because an earlier larger child already "
                        "covered the same region within geometric tolerance."
                    ),
                }
            )
            emit_pipeline_log(
                logging.DEBUG,
                "Rejected component child candidate because a larger valid region contains it within tolerance",
                candidate_region_key=candidate["candidate_region_key"],
                suppressed_by_region_key=container["candidate_region_key"],
                area_ratio_vs_parent=candidate["area_ratio_vs_parent"],
                difference_area_m2=metrics["difference_area_m2"],
                difference_area_ratio=metrics["difference_area_ratio"],
                tolerance_m2=metrics["tolerance_m2"],
            )
            continue
        kept.append(candidate)

    for idx, candidate in enumerate(kept, start=1):
        candidate["component_id"] = f"child_{idx:03d}"
        candidate["pair_id"] = f"period_{period['period_id']}__{candidate['component_id']}"
        candidate["decision_summary"] = _component_selection_decision_summary(candidate)
        emit_pipeline_log(
            logging.INFO,
            "Selected component child candidate",
            component_id=candidate["component_id"],
            bbox=candidate["bbox"],
            area_ratio_vs_parent=candidate["area_ratio_vs_parent"],
            pre_scene_count=((candidate.get("selection") or {}).get("pre_scene_count")),
            post_scene_count=((candidate.get("selection") or {}).get("post_scene_count")),
        )
    component_id_by_region_key = {
        candidate["candidate_region_key"]: candidate["component_id"]
        for candidate in kept
        if candidate.get("candidate_region_key") and candidate.get("component_id")
    }
    for rejected_candidate in rejected:
        if rejected_candidate.get("status") != "suppressed":
            rejected_candidate["why_rejected"] = _component_rejection_message(
                list(rejected_candidate.get("reject_reasons", []) or [])
            )
            rejected_candidate.setdefault(
                "reason_code",
                str((rejected_candidate.get("reject_reasons") or ["REJECTED_COMPONENT_CANDIDATE"])[0]),
            )
            rejected_candidate.setdefault("reason_message", rejected_candidate["why_rejected"])
            rejected_candidate.setdefault("human_summary", rejected_candidate["why_rejected"])
            continue
        suppressed_by_key = rejected_candidate.get("suppressed_by_region_key")
        if suppressed_by_key in component_id_by_region_key:
            rejected_candidate["suppressed_by_component_id"] = component_id_by_region_key[suppressed_by_key]
        rejected_candidate.setdefault("reason_message", rejected_candidate.get("why_rejected"))
    for idx, rejected_candidate in enumerate(rejected, start=1):
        if not rejected_candidate.get("component_id"):
            prefix = "suppressed_child" if rejected_candidate.get("status") == "suppressed" else "rejected_child"
            rejected_candidate["component_id"] = f"{prefix}_{idx:03d}"
    emit_pipeline_log(
        logging.INFO,
        "Component child selection summary",
        period_id=period.get("period_id"),
        kept_count=len(kept),
        rejected_count=len(rejected),
        suppressed_count=sum(1 for item in rejected if item.get("status") == "suppressed"),
    )
    return kept, rejected


def align_single_band_to_grid(path: str | Path, grid: Dict[str, Any], resampling: Resampling) -> np.ndarray:
    """Reproject one 1-band raster to the canonical grid."""
    path = Path(path)
    ref_crs = rasterio.crs.CRS.from_user_input(grid["crs"])
    ref_transform = grid["transform"]
    ref_width = int(grid["width"])
    ref_height = int(grid["height"])

    with rasterio.open(path, "r") as src:
        if src.count < 1:
            raise ValueError(f"Raster has no readable bands: {path}")
        if src.crs is None:
            raise ValueError(f"Raster missing CRS and cannot be aligned: {path}")

        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst


def write_multiband_geotiff(
    path: str | Path,
    data: np.ndarray,
    meta: Dict[str, Any],
    compression: str = "DEFLATE",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> Path:
    """Write a multi-band float32 GeoTIFF with band descriptions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": bands,
        "width": width,
        "height": height,
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "compress": compression,
        "tiled": tiled,
    }
    if tiled:
        profile["blockxsize"] = blockxsize
        profile["blockysize"] = blockysize

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        descs = meta.get("descriptions") or ()
        for idx, desc in enumerate(descs, start=1):
            if idx <= bands and desc:
                dst.set_band_description(idx, str(desc))
    return path


def build_circular_footprint(radius_pixels: float) -> np.ndarray:
    """Approximate GEE focal_median(circle, meters) with a circular footprint."""
    radius = max(0.0, float(radius_pixels))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    r = max(1, int(math.ceil(radius)))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    footprint = (xx * xx + yy * yy) <= (radius * radius)
    if not np.any(footprint):
        footprint[r, r] = True
    return footprint


def nanmedian_stack(arrays: List[np.ndarray]) -> np.ndarray:
    """Median across aligned scenes while ignoring nodata."""
    if not arrays:
        raise ValueError("Cannot composite an empty array stack.")
    stack = np.stack(arrays, axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comp = np.nanmedian(stack, axis=0)
    if np.all(np.isnan(comp)):
        raise RuntimeError("Composite produced only NaN values.")
    if np.any(np.isnan(comp)):
        finite = comp[np.isfinite(comp)]
        fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
        comp = np.nan_to_num(comp, nan=fill_value)
    return comp.astype(np.float32)


def apply_focal_median_db(arr: np.ndarray, radius_m: float, resolution_m: float) -> np.ndarray:
    """Apply a circular median filter in the same order as the GEE train/test recipe."""
    if radius_m <= 0:
        return arr.astype(np.float32)
    radius_px = float(radius_m) / float(resolution_m)
    footprint = build_circular_footprint(radius_px)
    return median_filter(arr.astype(np.float32), footprint=footprint, mode="nearest").astype(np.float32)


def choose_anchor_candidate(
    items: List[Dict[str, Any]],
    aoi_bbox: List[float],
    aoi_geometry: Dict[str, Any],
    train_cfg: Dict[str, Any],
    pair_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """Choose the best anchor candidate, optionally relaxing scene-count requirements."""
    min_scenes = int(train_cfg.get("min_scenes_per_window", 1))
    auto_relax = bool(train_cfg.get("auto_relax_min_scenes", True))
    before_days = float(train_cfg.get("window_before_days", 30.0))
    after_days = float(train_cfg.get("window_after_days", 30.0))
    same_orbit_direction = bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False)))
    min_delta_hours = float(train_cfg.get("anchor_min_delta_hours", pair_cfg.get("min_delta_hours", 24.0)))

    for required_count in range(min_scenes, 0, -1):
        candidates = suggest_trainlike_anchors(
            items=items,
            aoi_bbox=aoi_bbox,
            aoi_geometry=aoi_geometry,
            window_before_days=before_days,
            window_after_days=after_days,
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
            min_delta_hours=min_delta_hours,
            same_orbit_direction=same_orbit_direction,
            min_scenes_per_window=required_count,
        )
        if candidates:
            pick_index = max(1, int(train_cfg.get("anchor_pick_index", 1)))
            pick_index = min(pick_index, len(candidates))
            return candidates[pick_index - 1], required_count
        if not auto_relax:
            break
    raise RuntimeError(
        "No valid STAC anchor candidate found for the requested windows. "
        "Try widening the STAC datetime search range or lowering min_scenes_per_window."
    )


def sanitize_scene_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def download_window_assets(
    items: List[Dict[str, Any]],
    out_dir: Path,
    aoi_geometry: Dict[str, Any],
    required_pols: List[str],
    downloader: S3Downloader,
) -> Dict[str, List[Path]]:
    """Download AOI subsets for all scenes in a window, grouped by polarization."""
    grouped: Dict[str, List[Path]] = {pol.lower(): [] for pol in required_pols}
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        info = extract_item_info(item)
        dt_token = sanitize_scene_token((info["datetime"] or "").replace(":", "").replace("Z", ""))
        item_token = sanitize_scene_token(info["id"] or "scene")
        for pol in required_pols:
            asset_info = select_asset_href(item, pol)
            if asset_info is None:
                raise RuntimeError(f"Item {info['id']} is missing asset for polarization {pol}.")
            asset_key, href = asset_info
            local_path = out_dir / f"{dt_token}__{item_token}__{pol.lower()}.tif"
            emit_pipeline_log(
                logging.DEBUG,
                "Downloading AOI subset for selected scene",
                item_id=info["id"],
                item_datetime=info["datetime"],
                polarization=pol,
                asset_key=asset_key,
                local_path=local_path,
            )
            ok = downloader.download_aoi_subset_from_href(href, str(local_path), aoi_geometry)
            if not ok:
                raise RuntimeError(f"Failed to download AOI subset for item={info['id']} pol={pol}.")
            grouped[pol.lower()].append(local_path)
    return grouped


def compose_window_to_multiband(
    grouped_paths: Dict[str, List[Path]],
    grid: Dict[str, Any],
    resampling_name: str,
    focal_radius_m: float,
    out_path: Path,
    output_cfg: Dict[str, Any],
    mask_geometry_wgs84: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Align all scenes in one window, composite them, smooth them, and write one 2-band TIFF."""
    resampling = resolve_resampling(resampling_name)
    meta = build_grid_meta(grid)
    resolution_m = max(float(grid["xres"]), float(grid["yres"]))
    band_order = [("vv", "S1_VV"), ("vh", "S1_VH")]
    bands: List[np.ndarray] = []
    band_counts: Dict[str, int] = {}

    for pol_key, desc in band_order:
        paths = grouped_paths.get(pol_key, [])
        if not paths:
            raise RuntimeError(f"Window composite is missing polarization {pol_key.upper()}.")
        aligned = [align_single_band_to_grid(p, grid, resampling) for p in paths]
        composite = nanmedian_stack(aligned)
        composite = apply_focal_median_db(composite, focal_radius_m, resolution_m)
        bands.append(composite)
        band_counts[pol_key] = len(paths)

    meta["descriptions"] = tuple(desc for _, desc in band_order)
    stacked = np.stack(bands, axis=0)
    stacked = apply_geometry_mask_to_multiband(stacked, grid, mask_geometry_wgs84)
    valid_mask = geometry_mask_for_grid(mask_geometry_wgs84, grid) if mask_geometry_wgs84 is not None else np.ones((int(grid["height"]), int(grid["width"])), dtype=np.uint8)
    out_file = write_multiband_geotiff(
        out_path,
        stacked,
        meta,
        compression=output_cfg.get("compression", "DEFLATE"),
        tiled=bool(output_cfg.get("tiled", True)),
        blockxsize=int(output_cfg.get("blockxsize", 256)),
        blockysize=int(output_cfg.get("blockysize", 256)),
    )
    return out_file, {
        "band_descriptions": list(meta["descriptions"]),
        "scene_counts": band_counts,
        "grid": {
            "crs": grid["crs"],
            "width": grid["width"],
            "height": grid["height"],
            "transform": list(grid["transform"])[:6],
        },
        "valid_pixel_count": int(valid_mask.sum()),
        "valid_pixel_ratio": float(valid_mask.mean()),
    }


def check_domain_compatibility(config: Dict[str, Any], current_profile_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Warn or fail when inference input domain mismatches the trained model domain."""
    compat_cfg = config.get("compatibility", {})
    trained_profile = str(compat_cfg.get("trained_input_profile", "") or "").strip()
    current_profile = str(current_profile_override or compat_cfg.get("current_download_profile", "") or "").strip()
    allow_mismatch = bool(compat_cfg.get("allow_domain_mismatch", False))
    if not trained_profile or not current_profile:
        return None

    mismatch = trained_profile == "gee_s1_db" and current_profile == "stac_measurement_raw"
    if not mismatch:
        return {
            "trained_input_profile": trained_profile,
            "current_download_profile": current_profile,
            "allow_domain_mismatch": allow_mismatch,
            "domain_mismatch": False,
        }

    message = (
        "Domain mismatch detected: model expects 'gee_s1_db' style inputs, but the pipeline is "
        "currently downloading 'stac_measurement_raw'. Raw STAC measurement GeoTIFFs are not "
        "radiometrically equivalent to GEE Sentinel-1 dB, so poor SR quality is expected."
    )
    if not allow_mismatch:
        raise RuntimeError(
            message
            + " Refusing to run by default. Either switch the imagery source to GEE-compatible dB "
            + "products or set compatibility.allow_domain_mismatch=true if you only want a diagnostic run."
        )
    logger.warning(message)
    return {
        "trained_input_profile": trained_profile,
        "current_download_profile": current_profile,
        "allow_domain_mismatch": allow_mismatch,
        "domain_mismatch": True,
        "message": message,
    }


def build_query_namespace(config: Dict[str, Any], geojson_path: str) -> argparse.Namespace:
    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    return argparse.Namespace(
        stac_url=stac_cfg.get("url", DEFAULT_STAC_API),
        collection=stac_cfg.get("collection", DEFAULT_COLLECTION),
        bbox=None,
        geojson=geojson_path,
        datetime=stac_cfg.get("datetime"),
        limit=int(stac_cfg.get("limit", 300)),
        orbit=pair_cfg.get("orbit"),
        rel_orbit=pair_cfg.get("rel_orbit"),
        pols=pair_cfg.get("pols", "VV,VH"),
    )


def select_best_pair(
    client: STACClient,
    config: Dict[str, Any],
    geojson_path: str,
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any], Dict[str, Any], str, Optional[Dict[str, Any]]]:
    pair_cfg = config.get("pairing", {})
    query_args = build_query_namespace(config, geojson_path)
    required_pols = parse_required_pols(query_args.pols)
    if required_pols != ["VV", "VH"]:
        raise ValueError("Pipeline end-to-end currently requires pols=VV,VH.")

    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, query_args, required_pols)
    strict_pairs = search_pairs_sorted(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
        max_delta_days=int(pair_cfg.get("max_delta_days", 10)),
        min_delta_hours=float(pair_cfg.get("min_delta_hours", 24.0)),
        strict_slice=bool(pair_cfg.get("strict_slice", False)),
        same_orbit_direction=bool(pair_cfg.get("same_orbit_direction", False)),
    )
    if strict_pairs:
        return items, aoi_bbox, aoi_geometry, strict_pairs[0], "strict", None

    diag = diagnose_no_pair(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        strict_slice=bool(pair_cfg.get("strict_slice", False)),
        min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
        max_delta_days=int(pair_cfg.get("max_delta_days", 10)),
        min_delta_hours=float(pair_cfg.get("min_delta_hours", 24.0)),
        same_orbit_direction=bool(pair_cfg.get("same_orbit_direction", False)),
    )

    if not bool(pair_cfg.get("auto_relax", False)):
        raise RuntimeError(
            f"No valid strict pair found for {geojson_path}. reason={diag['reason']} item_count={diag['item_count']}"
        )

    relax_profiles = [
        ("balanced", 30),
        ("loose", 90),
    ]
    for profile_name, max_days in relax_profiles:
        candidates = search_pairs_sorted(
            items=items,
            aoi_bbox=aoi_bbox,
            aoi_geometry=aoi_geometry,
            min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
            max_delta_days=max_days,
            min_delta_hours=float(pair_cfg.get("min_delta_hours", 24.0)),
            strict_slice=bool(pair_cfg.get("strict_slice", False)),
            same_orbit_direction=bool(pair_cfg.get("same_orbit_direction", False)),
        )
        if candidates:
            return items, aoi_bbox, aoi_geometry, candidates[0], profile_name, diag

    raise RuntimeError(
        f"No valid pair found for {geojson_path} after relax. reason={diag['reason']} item_count={diag['item_count']}"
    )


def expected_single_band_paths(
    raw_dir: Path,
    manifest: Dict[str, Any],
    t1_prefix: str,
    t2_prefix: str,
) -> Dict[str, Path]:
    pair_id = manifest["pair_id"]
    return {
        "t1_vv": raw_dir / f"{t1_prefix}{pair_id}_vv.tif",
        "t1_vh": raw_dir / f"{t1_prefix}{pair_id}_vh.tif",
        "t2_vv": raw_dir / f"{t2_prefix}{pair_id}_vv.tif",
        "t2_vh": raw_dir / f"{t2_prefix}{pair_id}_vh.tif",
    }


def write_run_summary(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    attach_run_log_path(summary)
    summary["aoi_id"] = _summary_aoi_id(summary)
    run_config = summary.setdefault("run_config", {})
    if isinstance(run_config, dict):
        run_config.setdefault("mode", summary.get("workflow_mode"))
        run_config.setdefault("selection_strategy", summary.get("selection_strategy"))
    json_path, md_path = aoi_summary_paths(run_dir)
    summary["summary_json"] = str(json_path)
    summary["summary_md"] = None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_aoi_record(summary), f, indent=2, ensure_ascii=False)
    return json_path, None


def write_trainlike_run_summary(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    """Write JSON summary for the train-like composite workflow."""
    attach_run_log_path(summary)
    summary["aoi_id"] = _summary_aoi_id(summary)
    run_config = summary.setdefault("run_config", {})
    if isinstance(run_config, dict):
        run_config.setdefault("mode", summary.get("workflow_mode"))
        run_config.setdefault("selection_strategy", summary.get("selection_strategy"))
    json_path, md_path = aoi_summary_paths(run_dir)
    summary["summary_json"] = str(json_path)
    summary["summary_md"] = None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_aoi_record(summary), f, indent=2, ensure_ascii=False)
    return json_path, None


def write_representative_period_summary(period_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    """Write JSON summary for one representative calendar period."""
    json_path, _ = period_summary_paths(period_dir)
    summary["summary_json"] = str(json_path)
    summary["summary_md"] = None
    attach_run_log_path(summary)
    apply_execution_contract(summary)
    summary["aoi_id"] = _summary_aoi_id(summary)
    summary["period_id"] = _summary_period_token(summary)
    run_config = summary.setdefault("run_config", {})
    if isinstance(run_config, dict):
        run_config.setdefault("mode", summary.get("workflow_mode"))
        run_config.setdefault("selection_strategy", summary.get("selection_strategy"))
    if summary.get("output_paths") is None:
        summary["output_paths"] = {
            "output_tif": summary.get("output_tif"),
            "output_sr_vv_tif": summary.get("output_sr_vv_tif"),
            "output_sr_vh_tif": summary.get("output_sr_vh_tif"),
            "output_sr_json_path": summary.get("output_sr_geojson_path"),
            "output_valid_mask_path": summary.get("output_valid_mask_path"),
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_period_record(summary), f, indent=2, ensure_ascii=False)
    return json_path, None


def write_representative_job_summary(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    """Write top-level JSON summary for representative calendar periods."""
    json_path, _ = aoi_summary_paths(run_dir)
    summary["summary_json"] = str(json_path)
    summary["summary_md"] = None
    attach_run_log_path(summary)
    apply_execution_contract(summary)
    summary["aoi_id"] = _summary_aoi_id(summary)
    run_config = summary.setdefault("run_config", {})
    if isinstance(run_config, dict):
        run_config.setdefault("mode", summary.get("workflow_mode"))
        run_config.setdefault("selection_strategy", summary.get("selection_strategy"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(build_aoi_record(summary), f, indent=2, ensure_ascii=False)
    return json_path, None


def run_exact_pair_pipeline(config: Dict[str, Any], geojson_path: str, output_root: Optional[str], cache_staging: bool, device: Optional[str]) -> Dict[str, Any]:
    try:
        from infer_production import SARInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Please install the production inference environment before running sar_pipeline.py."
        ) from exc

    aoi_path = Path(geojson_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {aoi_path}")
    aoi_id = resolve_runtime_aoi_id(config, aoi_path)

    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    dl_cfg = config.get("download", {})
    staging_cfg = config.get("staging", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="stac_measurement_raw")
    save_debug_artifacts = save_debug_artifacts_enabled(config)

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("Pipeline end-to-end currently requires pols=VV,VH.")

    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_dir = resolve_pipeline_run_dir(config, run_root, aoi_id)
    staging_dir = intermediate_dir(run_dir) / staging_cfg.get("dir_name", "staging")
    output_dir = ensure_dir(Path(run_dir) / out_cfg.get("output_dir_name", "output"))

    client = STACClient(stac_cfg.get("url", DEFAULT_STAC_API))
    items, aoi_bbox, aoi_geometry, chosen_pair, selected_profile, diag = select_best_pair(client, config, str(aoi_path))
    manifest = build_manifest_for_pair(chosen_pair, required_pols)
    if manifest is None:
        raise RuntimeError("Selected pair is missing required VV/VH asset hrefs.")
    manifest["selection_profile"] = selected_profile
    manifest["aoi_geojson"] = str(aoi_path)
    manifest["aoi_bbox"] = aoi_bbox

    manifest_path = aoi_manifest_path(run_dir)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    infer_overrides = apply_inference_env_overrides(infer_config)
    if infer_overrides:
        infer_config.setdefault("_runtime", {})["env_overrides"] = compact_jsonable(infer_overrides)
        log_inference_env_overrides(infer_config)
    if device:
        infer_config["device"] = device

    cache_enabled = cache_staging or bool(staging_cfg.get("cache_aligned_inputs", False))
    inferencer = SARInferencer(infer_config)
    with tempfile.TemporaryDirectory(prefix="sr_exact_pair_") as transient_dir:
        transient_root = Path(transient_dir)
        raw_dir = resolve_optional_debug_dir(
            persist=save_debug_artifacts,
            persistent_path=inputs_dir(run_dir) / dl_cfg.get("raw_dir_name", "raw"),
            transient_path=transient_root / dl_cfg.get("raw_dir_name", "raw"),
        )
        downloaded_paths = download_manifest_pair(
            manifest=manifest,
            required_pols=required_pols,
            out_dir=str(raw_dir),
            t1_prefix=dl_cfg.get("t1_prefix", "s1t1_"),
            t2_prefix=dl_cfg.get("t2_prefix", "s1t2_"),
            subset_aoi=not bool(dl_cfg.get("full_item", False)),
            aoi_geometry=aoi_geometry,
        )
        if len(downloaded_paths) != 4:
            raise RuntimeError(f"Expected 4 downloaded files, got {len(downloaded_paths)}")

        expected = expected_single_band_paths(
            raw_dir,
            manifest,
            dl_cfg.get("t1_prefix", "s1t1_"),
            dl_cfg.get("t2_prefix", "s1t2_"),
        )
        missing = [str(path) for path in expected.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing expected raw single-band files: {missing}")
        transient_output_tif = Path(transient_dir) / f"{aoi_id}__{manifest['pair_id']}_SR_x2.tif"
        inferencer.run_pair_from_single_band_files(
            t1_vv=expected["t1_vv"],
            t1_vh=expected["t1_vh"],
            t2_vv=expected["t2_vv"],
            t2_vh=expected["t2_vh"],
            out_path=transient_output_tif,
            config={
                "resampling": staging_cfg.get("resampling", "bilinear"),
                "target_crs": staging_cfg.get("target_crs"),
                "target_resolution": staging_cfg.get("target_resolution"),
                **exact_pair_semantics(),
            },
            cache_dir=staging_dir if cache_enabled else None,
            identifier=manifest["pair_id"],
        )
        packaged_outputs = export_masked_sr_band_cogs(
            sr_multiband_path=transient_output_tif,
            output_dir=output_dir,
            output_basename=f"{aoi_id}__{manifest['pair_id']}",
            geometry_wgs84=aoi_geometry,
            compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
            blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
            persist_valid_mask=False,
            final_target_crs=out_cfg.get("final_target_crs"),
            final_target_resolution=out_cfg.get("final_target_resolution"),
            final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
        )

    selected_pair_info = build_selected_pair_info(chosen_pair, manifest)
    summary = {
        "workflow_mode": "exact_pair",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "raw_dir": (str(raw_dir) if save_debug_artifacts else None),
        "staging_dir": str(staging_dir if cache_enabled else ""),
        "output_tif": None,
        "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
        "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
        "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
        "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
        "selection_profile": selected_profile,
        "selected_pair": selected_pair_info,
        "inference_input_semantics": exact_pair_semantics(),
        "manifest_path": str(manifest_path),
        "downloaded_files": ([str(p) for p in downloaded_paths] if save_debug_artifacts else None),
        "items_after_hard_filter": len(items),
        "run_config": {
            "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
            "limit": int(stac_cfg.get("limit", 300)),
            "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
            "min_delta_hours": float(pair_cfg.get("min_delta_hours", 24.0)),
            "max_delta_days": int(pair_cfg.get("max_delta_days", 10)),
            "selection_priority": "latest_input_datetime",
            "same_orbit_direction": bool(pair_cfg.get("same_orbit_direction", False)),
            "auto_relax": bool(pair_cfg.get("auto_relax", False)),
            "resampling": staging_cfg.get("resampling", "bilinear"),
            "target_crs": staging_cfg.get("target_crs"),
            "target_resolution": staging_cfg.get("target_resolution"),
            "cache_staging": cache_enabled,
            "save_debug_artifacts": save_debug_artifacts,
            "device": infer_config.get("device"),
        },
    }
    if diag is not None:
        summary["initial_diagnostics"] = diag
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    attach_sr_output_geojson(
        summary=summary,
        geometry_wgs84=aoi_geometry,
        infer_config=infer_config,
        source_t1_items=[chosen_pair.get("t2_item")] if chosen_pair.get("t2_item") is not None else None,
        source_t2_items=[chosen_pair.get("t1_item")] if chosen_pair.get("t1_item") is not None else None,
    )
    summary_json, summary_md = aoi_summary_paths(run_dir)
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = (str(summary_md) if summary_md else None)
    write_run_summary(run_dir, summary)
    return summary


def run_stac_representative_calendar_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """AOI -> STAC monthly buckets -> first-half/second-half composite -> inference."""
    aoi_path = Path(geojson_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {aoi_path}")
    aoi_id = resolve_runtime_aoi_id(config, aoi_path)

    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    train_cfg = config.get("trainlike", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="stac_trainlike_composite_db")

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("STAC representative calendar pipeline currently requires pols=VV,VH.")

    datetime_filter, datetime_resolution = resolve_representative_datetime_filter(config)

    period_mode = str(train_cfg.get("period_mode", "month")).strip().lower()
    if period_mode != "month":
        raise ValueError(f"Unsupported representative period_mode: {train_cfg.get('period_mode')}")
    period_boundary_policy = str(train_cfg.get("period_boundary_policy", "clip_inside_period")).strip().lower()
    if period_boundary_policy != "clip_inside_period":
        raise ValueError(
            f"Unsupported representative period_boundary_policy: {train_cfg.get('period_boundary_policy')}"
        )
    period_split_policy = str(train_cfg.get("period_split_policy", "first_half_vs_second_half")).strip().lower()
    if period_split_policy != "first_half_vs_second_half":
        raise ValueError(
            f"Unsupported representative period_split_policy: {train_cfg.get('period_split_policy')}"
        )

    allow_partial_periods = bool(train_cfg.get("allow_partial_periods", False))
    min_scenes_per_half = int(train_cfg.get("min_scenes_per_half", 2))
    auto_relax_inside_period = bool(train_cfg.get("auto_relax_inside_period", True))
    same_orbit_direction = bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False)))
    representative_pool_mode = normalize_representative_pool_mode(train_cfg.get("representative_pool_mode", "auto"))
    componentize_seed_intersections = bool(train_cfg.get("componentize_seed_intersections", False))
    component_parent_mosaic = bool(train_cfg.get("component_parent_mosaic", True))
    component_item_min_coverage = float(train_cfg.get("component_item_min_coverage", 0.0))
    component_min_area_ratio = float(train_cfg.get("component_min_area_ratio", 0.0))
    save_debug_artifacts = save_debug_artifacts_enabled(config)

    periods = expand_month_periods(datetime_filter, allow_partial_periods=allow_partial_periods)
    if not periods:
        raise RuntimeError(
            "Representative calendar mode found no eligible full calendar months in the requested datetime range. "
            "Widen the range or set trainlike.allow_partial_periods=true if partial months are acceptable."
        )
    emit_pipeline_log(
        logging.INFO,
        "Resolved representative monthly run configuration",
        period_count=len(periods),
        datetime=datetime_filter,
        datetime_resolution=datetime_resolution,
        representative_pool_mode=representative_pool_mode,
        componentize_seed_intersections=componentize_seed_intersections,
        component_parent_mosaic=component_parent_mosaic,
        save_debug_artifacts=save_debug_artifacts,
    )

    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_dir = resolve_pipeline_run_dir(config, run_root, aoi_id)
    periods_root = ensure_dir(run_dir / "periods")

    emit_pipeline_stage(
        "STAC Query",
        stac_url=stac_cfg.get("url", DEFAULT_STAC_API),
        collection=stac_cfg.get("collection", DEFAULT_COLLECTION),
        datetime=datetime_filter,
        limit=int(stac_cfg.get("limit", 300)),
    )
    client = STACClient(stac_cfg.get("url", DEFAULT_STAC_API))
    query_args = build_query_namespace(config, str(aoi_path))
    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, query_args, required_pols)
    if not items:
        emit_pipeline_log(
            logging.WARNING,
            "Representative monthly run found no items after hard filters",
            aoi_id=aoi_id,
            datetime=datetime_filter,
            stac_url=stac_cfg.get("url", DEFAULT_STAC_API),
            collection=stac_cfg.get("collection", DEFAULT_COLLECTION),
        )
        skip_reason = (
            "No items passed hard filters for the resolved STAC AOI + datetime query window. "
            "Every representative period in this run is skipped."
        )
        period_results: List[Dict[str, Any]] = []
        for period in periods:
            period_dir = ensure_dir(periods_root / period["period_id"])
            period_summary = {
                "status": "skipped",
                "workflow_mode": "stac_trainlike_composite",
                "selection_strategy": "representative_calendar_period",
                "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                "run_dir": str(run_dir),
                "period_dir": str(period_dir),
                "manifest_path": str(period_manifest_path(period_dir)),
                "period": {
                    "period_id": period["period_id"],
                    "period_mode": period["period_mode"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    "period_anchor_datetime": period["period_anchor_datetime"],
                    "period_split_policy": period_split_policy,
                },
                "items_in_period": {"pre": 0, "post": 0},
                "skip_reason": skip_reason,
                "run_config": {
                    "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
                    "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
                    "datetime": stac_cfg.get("datetime"),
                    "datetime_resolution": datetime_resolution,
                    "limit": int(stac_cfg.get("limit", 300)),
                    "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
                    "pols": ",".join(required_pols),
                    "period_mode": period_mode,
                    "period_boundary_policy": period_boundary_policy,
                    "period_split_policy": period_split_policy,
                    "allow_partial_periods": allow_partial_periods,
                    "min_scenes_per_half": min_scenes_per_half,
                    "auto_relax_inside_period": auto_relax_inside_period,
                    "same_orbit_direction": same_orbit_direction,
                    "representative_pool_mode": representative_pool_mode,
                    "componentize_seed_intersections": componentize_seed_intersections,
                    "component_item_min_coverage": component_item_min_coverage,
                    "component_min_area_ratio": component_min_area_ratio,
                    "save_debug_artifacts": save_debug_artifacts,
                    **build_final_output_trace_config(out_cfg),
                },
            }
            summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
            period_results.append(
                {
                    "period_id": period["period_id"],
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "pre_scene_count": 0,
                    "post_scene_count": 0,
                    "summary_json": str(summary_json),
                    "summary_md": (str(summary_md) if summary_md else None),
                }
            )

        summary = {
            "status": "skipped",
            "workflow_mode": "stac_trainlike_composite",
            "selection_strategy": "representative_calendar_period",
            "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
            "run_dir": str(run_dir),
            "periods_dir": str(periods_root),
            "items_after_hard_filter": 0,
            "period_counts": {
                "total": len(periods),
                "completed": 0,
                "skipped": len(periods),
                "failed": 0,
            },
            "period_results": period_results,
            "skip_reason": skip_reason,
            "run_config": {
                "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
                "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
                "datetime": stac_cfg.get("datetime"),
                "datetime_resolution": datetime_resolution,
                "limit": int(stac_cfg.get("limit", 300)),
                "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
                "pols": ",".join(required_pols),
                "period_mode": period_mode,
                "period_boundary_policy": period_boundary_policy,
                "period_split_policy": period_split_policy,
                "allow_partial_periods": allow_partial_periods,
                "min_scenes_per_half": min_scenes_per_half,
                "auto_relax_inside_period": auto_relax_inside_period,
                "same_orbit_direction": same_orbit_direction,
                "representative_pool_mode": representative_pool_mode,
                "componentize_seed_intersections": componentize_seed_intersections,
                "component_item_min_coverage": component_item_min_coverage,
                "component_min_area_ratio": component_min_area_ratio,
                "save_debug_artifacts": save_debug_artifacts,
                **build_final_output_trace_config(out_cfg),
            },
        }
        if compatibility_info is not None:
            summary["compatibility"] = compatibility_info
        summary_json, summary_md = write_representative_job_summary(run_dir, summary)
        summary["summary_json"] = str(summary_json)
        summary["summary_md"] = (str(summary_md) if summary_md else None)
        return summary

    target_crs = str(train_cfg.get("target_crs", "EPSG:3857"))
    target_resolution = float(train_cfg.get("target_resolution", 10.0))
    resampling_name = str(train_cfg.get("resampling", config.get("staging", {}).get("resampling", "bilinear")))
    focal_radius_m = float(train_cfg.get("focal_median_radius_m", 15.0))
    try:
        from infer_production import SARInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Please install the production inference environment before running sar_pipeline.py."
        ) from exc
    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    infer_overrides = apply_inference_env_overrides(infer_config)
    if infer_overrides:
        infer_config.setdefault("_runtime", {})["env_overrides"] = compact_jsonable(infer_overrides)
        log_inference_env_overrides(infer_config)
    if device:
        infer_config["device"] = device
    log_effective_runtime_settings(
        train_cfg=train_cfg,
        infer_config=infer_config,
        save_debug_artifacts=save_debug_artifacts,
    )
    inferencer = SARInferencer(infer_config)

    period_results: List[Dict[str, Any]] = []
    for period in periods:
        period_pair_id = f"period_{period['period_id']}"
        period_dir = ensure_dir(periods_root / period["period_id"])
        manifest_path = period_manifest_path(period_dir)
        output_dir = ensure_dir(period_dir / out_cfg.get("output_dir_name", "output"))
        emit_pipeline_log(
            logging.INFO,
            "Processing representative period",
            period_id=period["period_id"],
            period_start=period["period_start"],
            period_end=period["period_end"],
            period_anchor_datetime=period["period_anchor_datetime"],
        )

        anchor_dt = datetime.fromisoformat(period["period_anchor_datetime"].replace("Z", "+00:00"))
        pre_items_full, post_items_full = collect_period_half_items(
            items=items,
            aoi_geometry=aoi_geometry,
            aoi_bbox=aoi_bbox,
            period_start=datetime.fromisoformat(period["period_start"].replace("Z", "+00:00")),
            period_anchor=anchor_dt,
            period_end=datetime.fromisoformat(period["period_end"].replace("Z", "+00:00")),
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
        )
        pre_items = dedupe_items_by_scene(pre_items_full)
        post_items = dedupe_items_by_scene(post_items_full)
        emit_pipeline_log(
            logging.INFO,
            "Split month into pre/post scene pools",
            period_id=period["period_id"],
            pre_items=len(pre_items),
            post_items=len(post_items),
            pre_scene_ids=[extract_item_info(item)["id"] for item in pre_items],
            post_scene_ids=[extract_item_info(item)["id"] for item in post_items],
        )

        if componentize_seed_intersections:
            emit_pipeline_stage(
                "Component Selection",
                period_id=period["period_id"],
                pre_items=len(pre_items),
                post_items=len(post_items),
            )
            selected_components, rejected_components = select_seed_intersection_component_candidates(
                pre_items=pre_items,
                post_items=post_items,
                parent_aoi_geometry=aoi_geometry,
                parent_aoi_bbox=aoi_bbox,
                period=period,
                min_scenes_per_half=min_scenes_per_half,
                auto_relax_inside_period=auto_relax_inside_period,
                require_same_orbit_direction=same_orbit_direction,
                representative_pool_mode=representative_pool_mode,
                component_item_min_coverage=component_item_min_coverage,
                component_min_area_ratio=component_min_area_ratio,
            )
            emit_pipeline_log(
                logging.INFO,
                "Component child selection finished for period",
                period_id=period["period_id"],
                selected_components=len(selected_components),
                rejected_components=len(rejected_components),
            )
            if not selected_components:
                skip_reason = (
                    "No valid child AOI intersections remained for this month after component seed generation, "
                    "strict region-coverage filtering, scene-count checks, and valid-larger-region suppression."
                )
                emit_pipeline_log(
                    logging.WARNING,
                    "Skipping period because no valid child components remained",
                    period_id=period["period_id"],
                    reason_code="NO_VALID_CHILDREN",
                    pre_items=len(pre_items),
                    post_items=len(post_items),
                    rejected_components=len(rejected_components),
                )
                period_summary = {
                    "status": "skipped",
                    "workflow_mode": "stac_trainlike_composite",
                    "selection_strategy": "representative_calendar_period",
                    "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                    "run_dir": str(run_dir),
                    "period_dir": str(period_dir),
                    "period": {
                        "period_id": period["period_id"],
                        "period_mode": period["period_mode"],
                        "period_start": period["period_start"],
                        "period_end": period["period_end"],
                        "period_anchor_datetime": period["period_anchor_datetime"],
                        "period_split_policy": period_split_policy,
                    },
                    "items_in_period": {
                        "pre": len(pre_items),
                        "post": len(post_items),
                    },
                    "skip_reason": skip_reason,
                    "manifest_path": str(manifest_path),
                    "componentization": {
                        "enabled": True,
                        "mode": "seed_item_intersections",
                        "delivery_mode": "parent_mosaic",
                        "item_min_region_coverage": component_item_min_coverage,
                        "min_area_ratio": component_min_area_ratio,
                        "completed_component_count": 0,
                        "rejected_component_count": len(rejected_components),
                        "suppressed_component_count": sum(
                            1 for item in rejected_components if item.get("status") == "suppressed"
                        ),
                        "parent_supported_area_ratio": 0.0,
                        "parent_mosaic_ordering": "largest_first",
                        "decision_summary": {
                            "suppression_policy": "largest_first_tolerant_nested_pruning",
                            "suppressed_component_count": sum(
                                1 for item in rejected_components if item.get("status") == "suppressed"
                            ),
                            "completed_component_count": 0,
                            "parent_mosaic_ordering": "largest_first",
                        },
                    },
                    "rejected_component_candidates": rejected_components,
                    "human_summary": (
                        "No child component survived selection, so the representative period was skipped "
                        "before parent mosaic execution."
                    ),
                }
                summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
                period_results.append(
                    {
                        "period_id": period["period_id"],
                        "status": "skipped",
                        "skip_reason": skip_reason,
                        "pre_scene_count": len(pre_items),
                        "post_scene_count": len(post_items),
                        "component_count_completed": 0,
                        "component_count_rejected": len(rejected_components),
                        "summary_json": str(summary_json),
                        "summary_md": (str(summary_md) if summary_md else None),
                    }
                )
                continue

            downloader = S3Downloader()
            component_results: List[Dict[str, Any]] = []
            parent_source_t1_items: List[Dict[str, Any]] = []
            parent_source_t2_items: List[Dict[str, Any]] = []
            with tempfile.TemporaryDirectory(prefix=f"sr_components_{period['period_id']}_") as transient_dir:
                transient_root = Path(transient_dir)
                if save_debug_artifacts:
                    component_inputs_root = ensure_dir(inputs_dir(period_dir) / "components")
                    component_intermediate_root = ensure_dir(intermediate_dir(period_dir) / "components")
                    emit_pipeline_log(
                        logging.INFO,
                        "Persisting component debug artifacts",
                        period_id=period["period_id"],
                        debug_inputs_root=component_inputs_root,
                        debug_intermediate_root=component_intermediate_root,
                    )
                else:
                    component_inputs_root = ensure_dir(transient_root / "inputs" / "components")
                    component_intermediate_root = ensure_dir(transient_root / "intermediate" / "components")
                    emit_pipeline_log(
                        logging.INFO,
                        "Using temporary directories for component debug artifacts",
                        period_id=period["period_id"],
                        temporary_root=transient_root,
                        auto_cleanup=True,
                    )
                component_mosaic_sources: List[Dict[str, Any]] = []
                emit_pipeline_stage(
                    "Component Execution",
                    period_id=period["period_id"],
                    selected_component_count=len(selected_components),
                )
                for component in selected_components:
                    selection = component["selection"]
                    component_id = component["component_id"]
                    component_pair_id = component["pair_id"]
                    component_geometry = component["geometry"]
                    component_bbox = component["bbox"]
                    emit_pipeline_log(
                        logging.INFO,
                        "Processing selected component child",
                        period_id=period["period_id"],
                        component_id=component_id,
                        bbox=component_bbox,
                        area_ratio_vs_parent=component["area_ratio_vs_parent"],
                        pre_scene_count=selection.get("pre_scene_count"),
                        post_scene_count=selection.get("post_scene_count"),
                    )

                    child_window_raw_dir = ensure_dir(component_inputs_root / component_id / train_cfg.get("window_raw_dir_name", "window_raw"))
                    child_composite_dir = ensure_dir(component_intermediate_root / component_id / train_cfg.get("composite_dir_name", "composite"))

                    child_manifest = build_representative_period_manifest(
                        period=period,
                        selection=selection,
                        aoi_bbox=component_bbox,
                        geojson_path=str(aoi_path),
                        required_pols=required_pols,
                    )
                    child_manifest["pair_id"] = component_pair_id
                    child_manifest["parent_pair_id"] = f"period_{period['period_id']}"
                    child_manifest["component_id"] = component_id
                    child_manifest["component_mode"] = "seed_item_intersections"
                    child_manifest["component_geometry"] = component_geometry
                    child_manifest["component_bbox"] = component_bbox
                    child_manifest["component_area_m2"] = component["area_m2"]
                    child_manifest["component_area_ratio_vs_parent"] = component["area_ratio_vs_parent"]
                    child_manifest["seed_item_ids"] = component["seed_item_ids"]
                    child_manifest["component_membership"] = {
                        "item_min_region_coverage": component.get("membership_coverage_threshold", component_item_min_coverage),
                        "pre_considered_items": component.get("pre_considered_items", []),
                        "post_considered_items": component.get("post_considered_items", []),
                        "pre_accepted_items": component.get("pre_accepted_items", []),
                        "post_accepted_items": component.get("post_accepted_items", []),
                        "pre_considered_scene_count": component.get("pre_considered_scene_count", 0),
                        "post_considered_scene_count": component.get("post_considered_scene_count", 0),
                        "pre_accepted_scene_count": component.get("pre_covering_scene_count", 0),
                        "post_accepted_scene_count": component.get("post_covering_scene_count", 0),
                        "pre_considered_coverage_stats": component.get("pre_considered_coverage_stats", {}),
                        "post_considered_coverage_stats": component.get("post_considered_coverage_stats", {}),
                        "pre_accepted_coverage_stats": component.get("pre_covering_coverage_stats", {}),
                        "post_accepted_coverage_stats": component.get("post_covering_coverage_stats", {}),
                    }
                    child_manifest["period_split_policy"] = period_split_policy
                    child_manifest["period_boundary_policy"] = period_boundary_policy

                    pre_paths = download_window_assets(
                        selection["pre_items"],
                        child_window_raw_dir / "pre",
                        component_geometry,
                        required_pols,
                        downloader,
                    )
                    post_paths = download_window_assets(
                        selection["post_items"],
                        child_window_raw_dir / "post",
                        component_geometry,
                        required_pols,
                        downloader,
                    )

                    child_grid = build_target_grid(component_bbox, target_crs, target_resolution, target_resolution)
                    t1_composite_path, post_meta = compose_window_to_multiband(
                        grouped_paths=post_paths,
                        grid=child_grid,
                        resampling_name=resampling_name,
                        focal_radius_m=focal_radius_m,
                        out_path=child_composite_dir / f"s1t1_{component_pair_id}.tif",
                        output_cfg=out_cfg,
                        mask_geometry_wgs84=component_geometry,
                    )
                    t2_composite_path, pre_meta = compose_window_to_multiband(
                        grouped_paths=pre_paths,
                        grid=child_grid,
                        resampling_name=resampling_name,
                        focal_radius_m=focal_radius_m,
                        out_path=child_composite_dir / f"s1t2_{component_pair_id}.tif",
                        output_cfg=out_cfg,
                        mask_geometry_wgs84=component_geometry,
                    )

                    transient_output_tif = transient_root / f"{component_pair_id}_SR_x2.tif"
                    inferencer.run_pair_from_multiband_files(
                        identifier=component_pair_id,
                        t1_path=t1_composite_path,
                        t2_path=t2_composite_path,
                        out_path=transient_output_tif,
                        config=model_trainlike_semantics(),
                    )

                    component_record = {
                        "component_id": component_id,
                        "pair_id": component_pair_id,
                        "status": "completed",
                        "area_m2": component["area_m2"],
                        "area_ratio_vs_parent": component["area_ratio_vs_parent"],
                        "pre_scene_count": child_manifest["pre_scene_count"],
                        "post_scene_count": child_manifest["post_scene_count"],
                        "selected_relaxation_name": child_manifest["selected_relaxation_name"],
                        "scene_signature_mode": child_manifest["scene_signature_mode"],
                        "seed_item_ids": component["seed_item_ids"],
                        "seed_item_datetimes": component["seed_item_datetimes"],
                        "geometry": component_geometry,
                        "bbox": component_bbox,
                        "why_kept": component.get("why_kept"),
                        "human_summary": component.get("human_summary"),
                        "decision_summary": component.get("decision_summary"),
                        "selection": {
                            "selection_priority": child_manifest.get("selection_priority", "balanced_period_representation"),
                            "selected_relaxation_level": child_manifest["selected_relaxation_level"],
                            "selected_relaxation_name": child_manifest["selected_relaxation_name"],
                            "required_scene_count": child_manifest["required_scene_count"],
                            "scene_signature_mode": child_manifest["scene_signature_mode"],
                            "scene_signature_value": child_manifest["scene_signature_value"],
                            "pre_scene_count": child_manifest["pre_scene_count"],
                            "post_scene_count": child_manifest["post_scene_count"],
                            "pre_unique_datetime_count": child_manifest["pre_unique_datetime_count"],
                            "post_unique_datetime_count": child_manifest["post_unique_datetime_count"],
                            "pre_union_coverage": child_manifest.get("pre_union_coverage"),
                            "post_union_coverage": child_manifest.get("post_union_coverage"),
                            "combined_union_coverage": child_manifest.get("combined_union_coverage"),
                            "pre_anchor_gap_hours": child_manifest["pre_anchor_gap_hours"],
                            "post_anchor_gap_hours": child_manifest["post_anchor_gap_hours"],
                            "latest_input_datetime": child_manifest.get("latest_input_datetime"),
                            "witness_support_pair": {
                                "support_t1_id": child_manifest.get("support_t1_id"),
                                "support_t2_id": child_manifest.get("support_t2_id"),
                                "support_t1_datetime": child_manifest.get("support_t1_datetime"),
                                "support_t2_datetime": child_manifest.get("support_t2_datetime"),
                                "support_pair_delta_hours": child_manifest.get("support_pair_delta_hours"),
                                "support_pair_delta_days": child_manifest.get("support_pair_delta_days"),
                                "support_pair_orbit_state": child_manifest.get("support_pair_orbit_state"),
                                "support_pair_relative_orbit": child_manifest.get("support_pair_relative_orbit"),
                                "support_t1_aoi_coverage": child_manifest.get("support_t1_aoi_coverage"),
                                "support_t2_aoi_coverage": child_manifest.get("support_t2_aoi_coverage"),
                                "support_t1_aoi_bbox_coverage": child_manifest.get("support_t1_aoi_bbox_coverage"),
                                "support_t2_aoi_bbox_coverage": child_manifest.get("support_t2_aoi_bbox_coverage"),
                            },
                            "pre_scenes": child_manifest["pre_scenes"],
                            "post_scenes": child_manifest["post_scenes"],
                        },
                        "component_membership": child_manifest.get("component_membership", {}),
                        "window_raw_dir": str(child_window_raw_dir),
                        "downloaded_files": {
                            "pre": {pol: [str(p) for p in paths] for pol, paths in pre_paths.items()},
                            "post": {pol: [str(p) for p in paths] for pol, paths in post_paths.items()},
                        },
                        "composite_dir": str(child_composite_dir),
                        "t1_composite_path": str(t1_composite_path),
                        "t2_composite_path": str(t2_composite_path),
                        "composite": {
                            "grid": post_meta["grid"],
                            "pre": pre_meta,
                            "post": post_meta,
                        },
                    }
                    component_results.append(
                        component_record if save_debug_artifacts else strip_debug_artifact_paths(component_record)
                    )
                    component_mosaic_sources.append(
                        {
                            "component_id": component_id,
                            "area_ratio_vs_parent": component["area_ratio_vs_parent"],
                            "area_m2": component["area_m2"],
                            "selection": component_record["selection"],
                            "geometry": component_geometry,
                            "sr_multiband_path": str(transient_output_tif),
                        }
                    )
                    parent_source_t1_items.extend(selection["post_items"])
                    parent_source_t2_items.extend(selection["pre_items"])

                public_item_id = _whole_monthly_sr_item_id(aoi_id, period["period_id"])
                emit_pipeline_stage(
                    "Parent Mosaic",
                    period_id=period["period_id"],
                    selected_component_count=len(component_mosaic_sources),
                    parent_mosaic_ordering="largest_first",
                )
                emit_pipeline_log(
                    logging.INFO,
                    "Running parent mosaic from completed component outputs",
                    period_id=period["period_id"],
                    contributing_candidates=len(component_mosaic_sources),
                    public_item_id=public_item_id,
                )
                parent_transient_output_tif = transient_root / f"{public_item_id}_parent_SR_x2.tif"
                parent_mosaic = mosaic_component_sr_multibands_to_parent(
                    component_sources=component_mosaic_sources,
                    parent_aoi_geometry=aoi_geometry,
                    parent_aoi_bbox=aoi_bbox,
                    target_crs=target_crs,
                    target_resolution=target_resolution,
                    output_path=parent_transient_output_tif,
                    compression=out_cfg.get("compression", "DEFLATE"),
                    tiled=bool(out_cfg.get("tiled", True)),
                    blockxsize=int(out_cfg.get("blockxsize", 256)),
                    blockysize=int(out_cfg.get("blockysize", 256)),
                )
                packaged_outputs = export_masked_sr_band_cogs(
                    sr_multiband_path=parent_mosaic["sr_multiband_path"],
                    output_dir=output_dir,
                    output_basename=public_item_id,
                    geometry_wgs84=parent_mosaic["supported_geometry"],
                    compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
                    blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
                    band_filename_style="whole_monthly_public",
                    crop_to_valid_data=False,
                    include_internal_mask=False,
                    persist_valid_mask=False,
                    final_target_crs=out_cfg.get("final_target_crs"),
                    final_target_resolution=out_cfg.get("final_target_resolution"),
                    final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
                )

            component_audit_by_id = {
                str(entry.get("component_id")): entry for entry in parent_mosaic.get("component_audit", []) or []
            }
            for component_record in component_results:
                audit = component_audit_by_id.get(str(component_record.get("component_id"))) or {}
                contributed = bool(audit.get("contributed_to_parent_mosaic", False))
                component_record["mosaic_priority_rank"] = audit.get("mosaic_priority_rank")
                component_record["contributed_to_parent_mosaic"] = contributed
                component_record["new_pixel_count"] = int(audit.get("new_pixel_count", 0) or 0)
                component_record["new_pixel_ratio"] = float(audit.get("new_pixel_ratio", 0.0) or 0.0)
                component_record["decision_summary"] = compact_jsonable(
                    {
                        **(component_record.get("decision_summary") or {}),
                        "mosaic_priority_rank": audit.get("mosaic_priority_rank"),
                        "contributed_to_parent_mosaic": contributed,
                        "new_pixel_count": component_record["new_pixel_count"],
                        "new_pixel_ratio": component_record["new_pixel_ratio"],
                    }
                )
                component_record["human_summary"] = (
                    f"Child kept with {component_record['pre_scene_count']} pre scenes and "
                    f"{component_record['post_scene_count']} post scenes; "
                    f"{'contributed new parent pixels' if contributed else 'did not add new parent pixels after largest-first mosaic'}."
                )

            suppressed_component_count = sum(1 for item in rejected_components if item.get("status") == "suppressed")
            period_summary = {
                "status": "completed",
                "workflow_mode": "stac_trainlike_composite",
                "selection_strategy": "representative_calendar_period",
                "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                "run_dir": str(run_dir),
                "period_dir": str(period_dir),
                "manifest_path": str(manifest_path),
                "output_tif": None,
                "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
                "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
                "public_item_id": public_item_id,
                "items_after_hard_filter": len(items),
                "period": {
                    "period_id": period["period_id"],
                    "period_mode": period["period_mode"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    "period_anchor_datetime": period["period_anchor_datetime"],
                    "period_split_policy": period_split_policy,
                },
                "componentization": {
                    "enabled": True,
                    "mode": "seed_item_intersections",
                    "delivery_mode": "parent_mosaic",
                    "item_min_region_coverage": component_item_min_coverage,
                    "min_area_ratio": component_min_area_ratio,
                    "completed_component_count": len(component_results),
                    "rejected_component_count": len(rejected_components),
                    "suppressed_component_count": suppressed_component_count,
                    "parent_supported_area_m2": parent_mosaic["supported_area_m2"],
                    "parent_supported_area_ratio": parent_mosaic["supported_area_ratio"],
                    "parent_supported_bbox": canonical_bbox_from_geometry(parent_mosaic["supported_geometry"]),
                    "parent_contributing_component_ids": parent_mosaic["contributing_component_ids"],
                    "parent_mosaic_ordering": parent_mosaic.get("parent_mosaic_ordering", "largest_first"),
                    "component_parent_mosaic": component_parent_mosaic,
                    "decision_summary": {
                        "suppression_policy": "largest_first_tolerant_nested_pruning",
                        "suppressed_component_count": suppressed_component_count,
                        "completed_component_count": len(component_results),
                        "parent_contributing_component_ids": parent_mosaic["contributing_component_ids"],
                        "parent_mosaic_ordering": parent_mosaic.get("parent_mosaic_ordering", "largest_first"),
                    },
                },
                "component_results": component_results,
                "rejected_component_candidates": rejected_components,
                "run_config": {
                    "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
                    "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
                    "datetime": stac_cfg.get("datetime"),
                    "datetime_resolution": datetime_resolution,
                    "limit": int(stac_cfg.get("limit", 300)),
                    "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
                    "pols": ",".join(required_pols),
                    "period_mode": period_mode,
                    "period_boundary_policy": period_boundary_policy,
                    "period_split_policy": period_split_policy,
                    "allow_partial_periods": allow_partial_periods,
                    "min_scenes_per_half": min_scenes_per_half,
                    "auto_relax_inside_period": auto_relax_inside_period,
                    "same_orbit_direction": same_orbit_direction,
                    "representative_pool_mode": representative_pool_mode,
                    "target_crs": target_crs,
                    "target_resolution": target_resolution,
                    "resampling": resampling_name,
                    "focal_median_radius_m": focal_radius_m,
                    "device": infer_config.get("device"),
                    "cache_staging": cache_staging,
                    "componentize_seed_intersections": componentize_seed_intersections,
                    "component_parent_mosaic": component_parent_mosaic,
                    "component_item_min_coverage": component_item_min_coverage,
                    "component_min_area_ratio": component_min_area_ratio,
                    "save_debug_artifacts": save_debug_artifacts,
                    **build_final_output_trace_config(out_cfg),
                },
                "human_summary": (
                    f"Processed {len(component_results)} child components after suppressing "
                    f"{suppressed_component_count} near-nested candidates. Parent mosaic used largest-first ordering "
                    f"and received new pixels from {len(parent_mosaic['contributing_component_ids'])} child components."
                ),
            }
            if compatibility_info is not None:
                period_summary["compatibility"] = compatibility_info
            attach_sr_output_geojson(
                summary=period_summary,
                geometry_wgs84=parent_mosaic["supported_geometry"],
                infer_config=infer_config,
                source_t1_items=parent_source_t1_items,
                source_t2_items=parent_source_t2_items,
            )
            summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
            emit_pipeline_stage(
                "Output Summary",
                period_id=period["period_id"],
                status=period_summary.get("status"),
                public_item_id=public_item_id,
                completed_component_count=len(component_results),
                suppressed_component_count=suppressed_component_count,
                contributing_component_ids=parent_mosaic["contributing_component_ids"],
            )
            period_results.append(
                {
                    "period_id": period["period_id"],
                    "status": "completed",
                    "component_count_completed": len(component_results),
                    "component_count_rejected": len(rejected_components),
                    "output_tif": None,
                    "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                    "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                    "output_sr_geojson_path": period_summary.get("output_sr_geojson_path"),
                    "manifest_path": str(manifest_path),
                    "summary_json": str(summary_json),
                    "summary_md": (str(summary_md) if summary_md else None),
                    "component_results": component_results,
                    "component_delivery_mode": "parent_mosaic",
                }
            )
            emit_pipeline_log(
                logging.INFO,
                "Completed representative period with component parent mosaic",
                period_id=period["period_id"],
                public_item_id=public_item_id,
                completed_components=len(component_results),
                rejected_components=len(rejected_components),
                parent_supported_area_ratio=parent_mosaic["supported_area_ratio"],
                output_sr_vv_tif=packaged_outputs["output_sr_vv_tif"],
                output_sr_vh_tif=packaged_outputs["output_sr_vh_tif"],
            )
            continue

        selection = select_representative_scene_pools(
            pre_items=pre_items,
            post_items=post_items,
            aoi_geometry=aoi_geometry,
            aoi_bbox=aoi_bbox,
            anchor_dt=anchor_dt,
            min_scenes_per_half=min_scenes_per_half,
            auto_relax_inside_period=auto_relax_inside_period,
            require_same_orbit_direction=same_orbit_direction,
            representative_pool_mode=representative_pool_mode,
        )
        if selection is None:
            skip_reason = (
                "No valid in-period representative scene pools found after the relaxation ladder. "
                "The month is missing balanced first-half/second-half coverage under the configured constraints."
            )
            emit_pipeline_log(
                logging.WARNING,
                "Skipping period because no representative pools remained",
                period_id=period["period_id"],
                reason_code="NO_VALID_REPRESENTATIVE_POOLS",
                pre_items=len(pre_items),
                post_items=len(post_items),
                representative_pool_mode=representative_pool_mode,
            )
            period_summary = {
                "status": "skipped",
                "workflow_mode": "stac_trainlike_composite",
                "selection_strategy": "representative_calendar_period",
                "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                "run_dir": str(run_dir),
                "period_dir": str(period_dir),
                "period": {
                    "period_id": period["period_id"],
                    "period_mode": period["period_mode"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    "period_anchor_datetime": period["period_anchor_datetime"],
                    "period_split_policy": period_split_policy,
                },
                "items_in_period": {
                    "pre": len(pre_items),
                    "post": len(post_items),
                },
                "skip_reason": skip_reason,
                "manifest_path": str(manifest_path),
                "run_config": {
                    "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
                    "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
                    "datetime": stac_cfg.get("datetime"),
                    "datetime_resolution": datetime_resolution,
                    "limit": int(stac_cfg.get("limit", 300)),
                    "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
                    "pols": ",".join(required_pols),
                    "period_mode": period_mode,
                    "period_boundary_policy": period_boundary_policy,
                    "period_split_policy": period_split_policy,
                    "allow_partial_periods": allow_partial_periods,
                    "min_scenes_per_half": min_scenes_per_half,
                    "auto_relax_inside_period": auto_relax_inside_period,
                    "same_orbit_direction": same_orbit_direction,
                    "representative_pool_mode": representative_pool_mode,
                    "target_crs": target_crs,
                    "target_resolution": target_resolution,
                    "resampling": resampling_name,
                    "focal_median_radius_m": focal_radius_m,
                    "device": infer_config.get("device"),
                    "cache_staging": cache_staging,
                    "componentize_seed_intersections": componentize_seed_intersections,
                    "component_item_min_coverage": component_item_min_coverage,
                    "component_min_area_ratio": component_min_area_ratio,
                    "save_debug_artifacts": save_debug_artifacts,
                    **build_final_output_trace_config(out_cfg),
                },
            }
            summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
            period_summary["summary_json"] = str(summary_json)
            period_summary["summary_md"] = (str(summary_md) if summary_md else None)
            period_results.append(
                {
                    "period_id": period["period_id"],
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "pre_scene_count": len(pre_items),
                    "post_scene_count": len(post_items),
                    "summary_json": str(summary_json),
                    "summary_md": (str(summary_md) if summary_md else None),
                }
            )
            emit_pipeline_log(
                logging.WARNING,
                "Representative period skipped",
                period_id=period["period_id"],
                reason_code="NO_VALID_REPRESENTATIVE_POOLS",
                pre_scene_count=len(pre_items),
                post_scene_count=len(post_items),
            )
            continue

        manifest = build_representative_period_manifest(
            period=period,
            selection=selection,
            aoi_bbox=aoi_bbox,
            geojson_path=str(aoi_path),
            required_pols=required_pols,
        )
        manifest["period_split_policy"] = period_split_policy
        manifest["period_boundary_policy"] = period_boundary_policy
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        public_item_id = _whole_monthly_sr_item_id(aoi_id, period["period_id"])
        downloader = S3Downloader()
        with tempfile.TemporaryDirectory(prefix="sr_period_") as transient_dir:
            transient_root = Path(transient_dir)
            if save_debug_artifacts:
                window_raw_dir = ensure_dir(inputs_dir(period_dir) / train_cfg.get("window_raw_dir_name", "window_raw"))
                composite_dir = ensure_dir(intermediate_dir(period_dir) / train_cfg.get("composite_dir_name", "composite"))
                emit_pipeline_log(
                    logging.INFO,
                    "Persisting period debug artifacts",
                    period_id=period["period_id"],
                    window_raw_dir=window_raw_dir,
                    composite_dir=composite_dir,
                )
            else:
                window_raw_dir = ensure_dir(transient_root / train_cfg.get("window_raw_dir_name", "window_raw"))
                composite_dir = ensure_dir(transient_root / train_cfg.get("composite_dir_name", "composite"))
                emit_pipeline_log(
                    logging.INFO,
                    "Using temporary directories for period debug artifacts",
                    period_id=period["period_id"],
                    temporary_root=transient_root,
                    auto_cleanup=True,
                )

            pre_paths = download_window_assets(selection["pre_items"], window_raw_dir / "pre", aoi_geometry, required_pols, downloader)
            post_paths = download_window_assets(selection["post_items"], window_raw_dir / "post", aoi_geometry, required_pols, downloader)

            grid = build_target_grid(aoi_bbox, target_crs, target_resolution, target_resolution)
            t1_composite_path, post_meta = compose_window_to_multiband(
                grouped_paths=post_paths,
                grid=grid,
                resampling_name=resampling_name,
                focal_radius_m=focal_radius_m,
                out_path=composite_dir / f"s1t1_{manifest['pair_id']}.tif",
                output_cfg=out_cfg,
            )
            t2_composite_path, pre_meta = compose_window_to_multiband(
                grouped_paths=pre_paths,
                grid=grid,
                resampling_name=resampling_name,
                focal_radius_m=focal_radius_m,
                out_path=composite_dir / f"s1t2_{manifest['pair_id']}.tif",
                output_cfg=out_cfg,
            )

            transient_output_tif = transient_root / f"{aoi_id}__{manifest['pair_id']}_SR_x2.tif"
            inferencer.run_pair_from_multiband_files(
                identifier=manifest["pair_id"],
                t1_path=t1_composite_path,
                t2_path=t2_composite_path,
                out_path=transient_output_tif,
                config=model_trainlike_semantics(),
            )
            packaged_outputs = export_masked_sr_band_cogs(
                sr_multiband_path=transient_output_tif,
                output_dir=output_dir,
                output_basename=public_item_id,
                geometry_wgs84=aoi_geometry,
                compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
                blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
                band_filename_style="whole_monthly_public",
                crop_to_valid_data=True,
                include_internal_mask=False,
                persist_valid_mask=False,
                final_target_crs=out_cfg.get("final_target_crs"),
                final_target_resolution=out_cfg.get("final_target_resolution"),
                final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
            )

        period_summary = {
            "status": "completed",
            "workflow_mode": "stac_trainlike_composite",
            "selection_strategy": "representative_calendar_period",
            "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
            "run_dir": str(run_dir),
            "period_dir": str(period_dir),
            "manifest_path": str(manifest_path),
            "window_raw_dir": (str(window_raw_dir) if save_debug_artifacts else None),
            "composite_dir": (str(composite_dir) if save_debug_artifacts else None),
            "t1_composite_path": (str(t1_composite_path) if save_debug_artifacts else None),
            "t2_composite_path": (str(t2_composite_path) if save_debug_artifacts else None),
            "output_tif": None,
            "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
            "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
            "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
            "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
            "public_item_id": public_item_id,
            "inference_input_semantics": model_trainlike_semantics(),
            "items_after_hard_filter": len(items),
            "period": {
                "period_id": period["period_id"],
                "period_mode": period["period_mode"],
                "period_start": period["period_start"],
                "period_end": period["period_end"],
                "period_anchor_datetime": period["period_anchor_datetime"],
                "period_split_policy": period_split_policy,
            },
            "selection": {
                "selection_priority": manifest.get("selection_priority", "balanced_period_representation"),
                "selected_relaxation_level": manifest["selected_relaxation_level"],
                "selected_relaxation_name": manifest["selected_relaxation_name"],
                "required_scene_count": manifest["required_scene_count"],
                "scene_signature_mode": manifest["scene_signature_mode"],
                "scene_signature_value": manifest["scene_signature_value"],
                "pre_scene_count": manifest["pre_scene_count"],
                "post_scene_count": manifest["post_scene_count"],
                "pre_unique_datetime_count": manifest["pre_unique_datetime_count"],
                "post_unique_datetime_count": manifest["post_unique_datetime_count"],
                "pre_union_coverage": manifest.get("pre_union_coverage"),
                "post_union_coverage": manifest.get("post_union_coverage"),
                "combined_union_coverage": manifest.get("combined_union_coverage"),
                "pre_anchor_gap_hours": manifest["pre_anchor_gap_hours"],
                "post_anchor_gap_hours": manifest["post_anchor_gap_hours"],
                "latest_input_datetime": manifest.get("latest_input_datetime"),
                "witness_support_pair": {
                    "support_t1_id": manifest.get("support_t1_id"),
                    "support_t2_id": manifest.get("support_t2_id"),
                    "support_t1_datetime": manifest.get("support_t1_datetime"),
                    "support_t2_datetime": manifest.get("support_t2_datetime"),
                    "support_pair_delta_hours": manifest.get("support_pair_delta_hours"),
                    "support_pair_delta_days": manifest.get("support_pair_delta_days"),
                    "support_pair_orbit_state": manifest.get("support_pair_orbit_state"),
                    "support_pair_relative_orbit": manifest.get("support_pair_relative_orbit"),
                    "support_t1_aoi_coverage": manifest.get("support_t1_aoi_coverage"),
                    "support_t2_aoi_coverage": manifest.get("support_t2_aoi_coverage"),
                    "support_t1_aoi_bbox_coverage": manifest.get("support_t1_aoi_bbox_coverage"),
                    "support_t2_aoi_bbox_coverage": manifest.get("support_t2_aoi_bbox_coverage"),
                },
                "pre_scenes": manifest["pre_scenes"],
                "post_scenes": manifest["post_scenes"],
            },
            "composite": {
                "grid": post_meta["grid"],
                "pre": pre_meta,
                "post": post_meta,
            },
            "downloaded_files": (
                {
                    "pre": {pol: [str(p) for p in paths] for pol, paths in pre_paths.items()},
                    "post": {pol: [str(p) for p in paths] for pol, paths in post_paths.items()},
                }
                if save_debug_artifacts
                else None
            ),
            "run_config": {
                "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
                "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
                "datetime": stac_cfg.get("datetime"),
                "datetime_resolution": datetime_resolution,
                "limit": int(stac_cfg.get("limit", 300)),
                "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
                "pols": ",".join(required_pols),
                "period_mode": period_mode,
                "period_boundary_policy": period_boundary_policy,
                "period_split_policy": period_split_policy,
                "allow_partial_periods": allow_partial_periods,
                "min_scenes_per_half": min_scenes_per_half,
                "auto_relax_inside_period": auto_relax_inside_period,
                "same_orbit_direction": same_orbit_direction,
                "representative_pool_mode": representative_pool_mode,
                "target_crs": target_crs,
                "target_resolution": target_resolution,
                "resampling": resampling_name,
                "focal_median_radius_m": focal_radius_m,
                "device": infer_config.get("device"),
                "cache_staging": cache_staging,
                "componentize_seed_intersections": componentize_seed_intersections,
                "component_item_min_coverage": component_item_min_coverage,
                "component_min_area_ratio": component_min_area_ratio,
                "save_debug_artifacts": save_debug_artifacts,
                **build_final_output_trace_config(out_cfg),
            },
        }
        if compatibility_info is not None:
            period_summary["compatibility"] = compatibility_info
        attach_sr_output_geojson(
            summary=period_summary,
            geometry_wgs84=aoi_geometry,
            infer_config=infer_config,
            source_t1_items=selection["post_items"],
            source_t2_items=selection["pre_items"],
        )
        summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
        period_summary["summary_json"] = str(summary_json)
        period_summary["summary_md"] = (str(summary_md) if summary_md else None)
        period_results.append(
            {
                "period_id": period["period_id"],
                "status": "completed",
                "selected_relaxation_level": manifest["selected_relaxation_level"],
                "selected_relaxation_name": manifest["selected_relaxation_name"],
                "pre_scene_count": manifest["pre_scene_count"],
                "post_scene_count": manifest["post_scene_count"],
                "pre_unique_datetime_count": manifest["pre_unique_datetime_count"],
                "post_unique_datetime_count": manifest["post_unique_datetime_count"],
                "output_tif": None,
                "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                "output_sr_geojson_path": period_summary.get("output_sr_geojson_path"),
                "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
                "manifest_path": str(manifest_path),
                "summary_json": str(summary_json),
                "summary_md": (str(summary_md) if summary_md else None),
            }
        )
        emit_pipeline_log(
            logging.INFO,
            "Completed representative period",
            period_id=period["period_id"],
            selected_relaxation_name=manifest.get("selected_relaxation_name"),
            pre_scene_count=manifest.get("pre_scene_count"),
            post_scene_count=manifest.get("post_scene_count"),
            output_sr_vv_tif=packaged_outputs["output_sr_vv_tif"],
            output_sr_vh_tif=packaged_outputs["output_sr_vh_tif"],
        )

    summary = {
        "workflow_mode": "stac_trainlike_composite",
        "selection_strategy": "representative_calendar_period",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "periods_dir": str(periods_root),
        "items_after_hard_filter": len(items),
        "period_counts": {
            "total": len(periods),
            "completed": sum(1 for p in period_results if p["status"] == "completed"),
            "skipped": sum(1 for p in period_results if p["status"] == "skipped"),
            "failed": sum(1 for p in period_results if p["status"] == "failed"),
        },
        "period_results": period_results,
        "run_config": {
            "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
            "datetime_resolution": datetime_resolution,
            "limit": int(stac_cfg.get("limit", 300)),
            "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
            "pols": ",".join(required_pols),
            "period_mode": period_mode,
            "period_boundary_policy": period_boundary_policy,
            "period_split_policy": period_split_policy,
            "allow_partial_periods": allow_partial_periods,
            "min_scenes_per_half": min_scenes_per_half,
            "auto_relax_inside_period": auto_relax_inside_period,
            "same_orbit_direction": same_orbit_direction,
            "representative_pool_mode": representative_pool_mode,
            "target_crs": target_crs,
            "target_resolution": target_resolution,
            "resampling": resampling_name,
            "focal_median_radius_m": focal_radius_m,
            "device": infer_config.get("device"),
            "cache_staging": cache_staging,
            "componentize_seed_intersections": componentize_seed_intersections,
            "component_item_min_coverage": component_item_min_coverage,
            "component_min_area_ratio": component_min_area_ratio,
            "save_debug_artifacts": save_debug_artifacts,
            **build_final_output_trace_config(out_cfg),
        },
    }
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    summary_json, summary_md = write_representative_job_summary(run_dir, summary)
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = (str(summary_md) if summary_md else None)
    return summary


def _build_gee_scene_collection(collection_id: str, scene_items: List[Dict[str, Any]]) -> Any:
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError("earthengine-api is required for gee_trainlike_composite mode.") from exc

    images = [ee.Image(f"{collection_id}/{extract_item_info(item)['id']}") for item in scene_items]
    return ee.ImageCollection.fromImages(images)


def run_gee_representative_calendar_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    try:
        from infer_production import SARInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Please install the production inference environment before running sar_pipeline.py."
        ) from exc

    from gee_compare_download import build_export_params, init_gee, rewrite_with_descriptions, validate_pair
    from gee_trainlike_download import (
        build_collection,
        build_trainlike_image,
        clip_geometry,
        collection_scene_items,
        download_gee_image,
    )

    aoi_path = Path(geojson_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {aoi_path}")
    aoi_id = resolve_runtime_aoi_id(config, aoi_path)

    gee_cfg = config.get("gee", {})
    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    train_cfg = config.get("trainlike", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="gee_s1_db")

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("GEE train-like pipeline currently requires pols=VV,VH.")

    gee_project = str(gee_cfg.get("project") or "").strip()
    if not gee_project:
        raise RuntimeError("Missing GEE project. Set gee.project in config/pipeline_config.yaml or pass --gee-project.")
    init_gee(gee_project, authenticate=bool(gee_cfg.get("authenticate", False)))

    datetime_filter, datetime_resolution = resolve_representative_datetime_filter(config)
    period_mode = str(train_cfg.get("period_mode", "month")).strip().lower()
    if period_mode != "month":
        raise ValueError(f"Unsupported representative period_mode: {train_cfg.get('period_mode')}")
    period_boundary_policy = str(train_cfg.get("period_boundary_policy", "clip_inside_period")).strip().lower()
    if period_boundary_policy != "clip_inside_period":
        raise ValueError(
            f"Unsupported representative period_boundary_policy: {train_cfg.get('period_boundary_policy')}"
        )
    period_split_policy = str(train_cfg.get("period_split_policy", "first_half_vs_second_half")).strip().lower()
    if period_split_policy != "first_half_vs_second_half":
        raise ValueError(
            f"Unsupported representative period_split_policy: {train_cfg.get('period_split_policy')}"
        )

    allow_partial_periods = bool(train_cfg.get("allow_partial_periods", False))
    min_scenes_per_half = int(train_cfg.get("min_scenes_per_half", 2))
    auto_relax_inside_period = bool(train_cfg.get("auto_relax_inside_period", True))
    same_orbit_direction = bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False)))
    representative_pool_mode = normalize_representative_pool_mode(train_cfg.get("representative_pool_mode", "auto"))
    componentize_seed_intersections = bool(train_cfg.get("componentize_seed_intersections", False))
    component_parent_mosaic = bool(train_cfg.get("component_parent_mosaic", True))
    component_item_min_coverage = float(train_cfg.get("component_item_min_coverage", 0.0))
    component_min_area_ratio = float(train_cfg.get("component_min_area_ratio", 0.0))
    save_debug_artifacts = save_debug_artifacts_enabled(config)
    target_crs = str(train_cfg.get("target_crs", "EPSG:3857"))
    target_resolution = float(train_cfg.get("target_resolution", 10.0))
    focal_radius_m = float(train_cfg.get("focal_median_radius_m", 15.0))
    clip_mode_name = str(train_cfg.get("clip_mode", "geometry"))
    orbit_pass = str(train_cfg.get("orbit_pass", "BOTH"))
    band_names = list(gee_cfg.get("band_names", ["VV", "VH"]))
    output_descs = list(gee_cfg.get("output_band_descriptions", ["S1_VV", "S1_VH"]))
    collection_id = gee_cfg.get("collection", "COPERNICUS/S1_GRD")

    periods = expand_month_periods(datetime_filter, allow_partial_periods=allow_partial_periods)
    if not periods:
        raise RuntimeError(
            "No calendar periods were generated from the datetime range. "
            "Use a finite range like `2025-01-01/2025-12-31`."
        )

    aoi_bbox, aoi_geometry = load_geojson_aoi(aoi_path)
    filter_start = periods[0]["period_start"]
    filter_end = periods[-1]["period_end"]
    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_dir = resolve_pipeline_run_dir(config, run_root, aoi_id)
    periods_root = ensure_dir(run_dir / "periods")

    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError("earthengine-api is required for gee_trainlike_composite mode.") from exc
    filter_geom = ee.Geometry(aoi_geometry)
    clip_geom = clip_geometry(clip_mode_name, aoi_bbox, aoi_geometry)
    emit_pipeline_stage(
        "GEE Query",
        collection=collection_id,
        datetime=datetime_filter,
        orbit_pass=orbit_pass,
    )
    full_collection = build_collection(
        collection_id=collection_id,
        filter_geom=filter_geom,
        start_dt=datetime.fromisoformat(filter_start.replace("Z", "+00:00")),
        end_dt=datetime.fromisoformat(filter_end.replace("Z", "+00:00")),
        orbit_pass=orbit_pass,
    )
    gee_items = collection_scene_items(full_collection, aoi_bbox)

    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    infer_overrides = apply_inference_env_overrides(infer_config)
    if infer_overrides:
        infer_config.setdefault("_runtime", {})["env_overrides"] = compact_jsonable(infer_overrides)
        log_inference_env_overrides(infer_config)
    if device:
        infer_config["device"] = device
    log_effective_runtime_settings(
        train_cfg=train_cfg,
        infer_config=infer_config,
        save_debug_artifacts=save_debug_artifacts,
    )
    inferencer = SARInferencer(infer_config)

    period_results: List[Dict[str, Any]] = []
    for period in periods:
        period_dir = ensure_dir(periods_root / period["period_id"])
        output_dir = ensure_dir(period_dir / out_cfg.get("output_dir_name", "output"))
        manifest_path = period_manifest_path(period_dir)
        anchor_dt = datetime.fromisoformat(period["period_anchor_datetime"].replace("Z", "+00:00"))
        period_start = datetime.fromisoformat(period["period_start"].replace("Z", "+00:00"))
        period_end = datetime.fromisoformat(period["period_end"].replace("Z", "+00:00"))

        pre_items, post_items = collect_period_half_items(
            items=gee_items,
            aoi_geometry=aoi_geometry,
            aoi_bbox=aoi_bbox,
            period_start=period_start,
            period_anchor=anchor_dt,
            period_end=period_end,
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
        )

        if componentize_seed_intersections:
            emit_pipeline_stage(
                "Component Selection",
                period_id=period["period_id"],
                pre_items=len(pre_items),
                post_items=len(post_items),
            )
            selected_components, rejected_components = select_seed_intersection_component_candidates(
                pre_items=pre_items,
                post_items=post_items,
                parent_aoi_geometry=aoi_geometry,
                parent_aoi_bbox=aoi_bbox,
                period=period,
                min_scenes_per_half=min_scenes_per_half,
                auto_relax_inside_period=auto_relax_inside_period,
                require_same_orbit_direction=same_orbit_direction,
                representative_pool_mode=representative_pool_mode,
                component_item_min_coverage=component_item_min_coverage,
                component_min_area_ratio=component_min_area_ratio,
            )
            if not selected_components:
                skip_reason = (
                    "No valid child AOI intersections remained for this month after component seed generation, "
                    "strict region-coverage filtering, scene-count checks, and valid-larger-region suppression."
                )
                period_summary = {
                    "status": "skipped",
                    "workflow_mode": "gee_trainlike_composite",
                    "selection_strategy": "representative_calendar_period",
                    "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                    "run_dir": str(run_dir),
                    "period_dir": str(period_dir),
                    "period": {
                        "period_id": period["period_id"],
                        "period_mode": period["period_mode"],
                        "period_start": period["period_start"],
                        "period_end": period["period_end"],
                        "period_anchor_datetime": period["period_anchor_datetime"],
                        "period_split_policy": period_split_policy,
                    },
                    "items_in_period": {
                        "pre": len(pre_items),
                        "post": len(post_items),
                    },
                    "skip_reason": skip_reason,
                    "manifest_path": str(manifest_path),
                    "componentization": {
                        "enabled": True,
                        "mode": "seed_item_intersections",
                        "delivery_mode": "parent_mosaic",
                        "item_min_region_coverage": component_item_min_coverage,
                        "min_area_ratio": component_min_area_ratio,
                        "completed_component_count": 0,
                        "rejected_component_count": len(rejected_components),
                        "suppressed_component_count": sum(
                            1 for item in rejected_components if item.get("status") == "suppressed"
                        ),
                        "parent_supported_area_ratio": 0.0,
                        "parent_mosaic_ordering": "largest_first",
                        "decision_summary": {
                            "suppression_policy": "largest_first_tolerant_nested_pruning",
                            "suppressed_component_count": sum(
                                1 for item in rejected_components if item.get("status") == "suppressed"
                            ),
                            "completed_component_count": 0,
                            "parent_mosaic_ordering": "largest_first",
                        },
                    },
                    "rejected_component_candidates": rejected_components,
                    "human_summary": (
                        "No child component survived selection, so the representative period was skipped "
                        "before parent mosaic execution."
                    ),
                }
                summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
                period_results.append(
                    {
                        "period_id": period["period_id"],
                        "status": "skipped",
                        "skip_reason": skip_reason,
                        "pre_scene_count": len(pre_items),
                        "post_scene_count": len(post_items),
                        "component_count_completed": 0,
                        "component_count_rejected": len(rejected_components),
                        "summary_json": str(summary_json),
                        "summary_md": (str(summary_md) if summary_md else None),
                    }
                )
                continue

            component_results: List[Dict[str, Any]] = []
            parent_source_t1_items: List[Dict[str, Any]] = []
            parent_source_t2_items: List[Dict[str, Any]] = []
            with tempfile.TemporaryDirectory(prefix=f"sr_components_{period['period_id']}_") as transient_dir:
                transient_root = Path(transient_dir)
                component_intermediate_root = resolve_optional_debug_dir(
                    persist=save_debug_artifacts,
                    persistent_path=intermediate_dir(period_dir) / "components",
                    transient_path=transient_root / "components",
                )
                component_mosaic_sources: List[Dict[str, Any]] = []
                emit_pipeline_stage(
                    "Component Execution",
                    period_id=period["period_id"],
                    selected_component_count=len(selected_components),
                )
                for component in selected_components:
                    selection = component["selection"]
                    component_id = component["component_id"]
                    component_pair_id = component["pair_id"]
                    component_geometry = component["geometry"]
                    component_bbox = component["bbox"]
                    child_composite_dir = ensure_dir(component_intermediate_root / component_id / "composite")
                    child_manifest = build_representative_period_manifest(
                        period=period,
                        selection=selection,
                        aoi_bbox=component_bbox,
                        geojson_path=str(aoi_path),
                        required_pols=required_pols,
                    )
                    child_manifest["pair_id"] = component_pair_id
                    child_manifest["parent_pair_id"] = f"period_{period['period_id']}"
                    child_manifest["component_id"] = component_id
                    child_manifest["component_mode"] = "seed_item_intersections"
                    child_manifest["component_geometry"] = component_geometry
                    child_manifest["component_bbox"] = component_bbox
                    child_manifest["component_area_m2"] = component["area_m2"]
                    child_manifest["component_area_ratio_vs_parent"] = component["area_ratio_vs_parent"]
                    child_manifest["seed_item_ids"] = component["seed_item_ids"]
                    child_manifest["component_membership"] = {
                        "item_min_region_coverage": component.get("membership_coverage_threshold", component_item_min_coverage),
                        "pre_considered_items": component.get("pre_considered_items", []),
                        "post_considered_items": component.get("post_considered_items", []),
                        "pre_accepted_items": component.get("pre_accepted_items", []),
                        "post_accepted_items": component.get("post_accepted_items", []),
                        "pre_considered_scene_count": component.get("pre_considered_scene_count", 0),
                        "post_considered_scene_count": component.get("post_considered_scene_count", 0),
                        "pre_accepted_scene_count": component.get("pre_covering_scene_count", 0),
                        "post_accepted_scene_count": component.get("post_covering_scene_count", 0),
                        "pre_considered_coverage_stats": component.get("pre_considered_coverage_stats", {}),
                        "post_considered_coverage_stats": component.get("post_considered_coverage_stats", {}),
                        "pre_accepted_coverage_stats": component.get("pre_covering_coverage_stats", {}),
                        "post_accepted_coverage_stats": component.get("post_covering_coverage_stats", {}),
                    }
                    child_manifest["period_split_policy"] = period_split_policy
                    child_manifest["period_boundary_policy"] = period_boundary_policy
                    child_manifest["gee_project"] = gee_project
                    child_manifest["gee_collection"] = collection_id

                    component_clip_geom = clip_geometry(clip_mode_name, component_bbox, component_geometry)
                    pre_collection = _build_gee_scene_collection(collection_id, selection["pre_items"])
                    post_collection = _build_gee_scene_collection(collection_id, selection["post_items"])
                    t1_image = build_trainlike_image(post_collection, component_clip_geom, focal_radius_m)
                    t2_image = build_trainlike_image(pre_collection, component_clip_geom, focal_radius_m)
                    child_grid = build_target_grid(component_bbox, target_crs, target_resolution, target_resolution)
                    t1_composite_path = child_composite_dir / f"s1t1_{component_pair_id}.tif"
                    t2_composite_path = child_composite_dir / f"s1t2_{component_pair_id}.tif"
                    download_gee_image(
                        t1_image,
                        build_export_params(f"s1t1_{component_pair_id}", child_grid, band_names),
                        t1_composite_path,
                    )
                    download_gee_image(
                        t2_image,
                        build_export_params(f"s1t2_{component_pair_id}", child_grid, band_names),
                        t2_composite_path,
                    )
                    rewrite_with_descriptions(t1_composite_path, output_descs, child_grid)
                    rewrite_with_descriptions(t2_composite_path, output_descs, child_grid)
                    validation = validate_pair(child_composite_dir, component_pair_id, child_grid, output_descs)

                    transient_output_tif = transient_root / f"{component_pair_id}_SR_x2.tif"
                    inferencer.run_pair_from_multiband_files(
                        identifier=component_pair_id,
                        t1_path=t1_composite_path,
                        t2_path=t2_composite_path,
                        out_path=transient_output_tif,
                        config=model_trainlike_semantics(),
                    )

                    pre_count = len(selection["pre_items"])
                    post_count = len(selection["post_items"])
                    composite_meta = {
                        "grid": {
                            "crs": child_grid["crs"],
                            "width": child_grid["width"],
                            "height": child_grid["height"],
                            "transform": list(child_grid["transform"])[:6],
                        },
                        "pre": {
                            "scene_counts": {"vv": pre_count, "vh": pre_count},
                            "band_descriptions": output_descs,
                            "grid": {
                                "crs": child_grid["crs"],
                                "width": child_grid["width"],
                                "height": child_grid["height"],
                                "transform": list(child_grid["transform"])[:6],
                            },
                        },
                        "post": {
                            "scene_counts": {"vv": post_count, "vh": post_count},
                            "band_descriptions": output_descs,
                            "grid": {
                                "crs": child_grid["crs"],
                                "width": child_grid["width"],
                                "height": child_grid["height"],
                                "transform": list(child_grid["transform"])[:6],
                            },
                        },
                    }
                    component_record = {
                        "component_id": component_id,
                        "pair_id": component_pair_id,
                        "status": "completed",
                        "area_m2": component["area_m2"],
                        "area_ratio_vs_parent": component["area_ratio_vs_parent"],
                        "pre_scene_count": child_manifest["pre_scene_count"],
                        "post_scene_count": child_manifest["post_scene_count"],
                        "selected_relaxation_name": child_manifest["selected_relaxation_name"],
                        "scene_signature_mode": child_manifest["scene_signature_mode"],
                        "seed_item_ids": component["seed_item_ids"],
                        "seed_item_datetimes": component["seed_item_datetimes"],
                        "geometry": component_geometry,
                        "bbox": component_bbox,
                        "why_kept": component.get("why_kept"),
                        "human_summary": component.get("human_summary"),
                        "decision_summary": component.get("decision_summary"),
                        "validation": validation,
                        "selection": {
                            "selection_priority": child_manifest.get("selection_priority", "balanced_period_representation"),
                            "selected_relaxation_level": child_manifest["selected_relaxation_level"],
                            "selected_relaxation_name": child_manifest["selected_relaxation_name"],
                            "required_scene_count": child_manifest["required_scene_count"],
                            "scene_signature_mode": child_manifest["scene_signature_mode"],
                            "scene_signature_value": child_manifest["scene_signature_value"],
                            "pre_scene_count": child_manifest["pre_scene_count"],
                            "post_scene_count": child_manifest["post_scene_count"],
                            "pre_unique_datetime_count": child_manifest["pre_unique_datetime_count"],
                            "post_unique_datetime_count": child_manifest["post_unique_datetime_count"],
                            "pre_union_coverage": child_manifest.get("pre_union_coverage"),
                            "post_union_coverage": child_manifest.get("post_union_coverage"),
                            "combined_union_coverage": child_manifest.get("combined_union_coverage"),
                            "pre_anchor_gap_hours": child_manifest["pre_anchor_gap_hours"],
                            "post_anchor_gap_hours": child_manifest["post_anchor_gap_hours"],
                            "latest_input_datetime": child_manifest.get("latest_input_datetime"),
                            "witness_support_pair": {
                                "support_t1_id": child_manifest.get("support_t1_id"),
                                "support_t2_id": child_manifest.get("support_t2_id"),
                                "support_t1_datetime": child_manifest.get("support_t1_datetime"),
                                "support_t2_datetime": child_manifest.get("support_t2_datetime"),
                                "support_pair_delta_hours": child_manifest.get("support_pair_delta_hours"),
                                "support_pair_delta_days": child_manifest.get("support_pair_delta_days"),
                                "support_pair_orbit_state": child_manifest.get("support_pair_orbit_state"),
                                "support_pair_relative_orbit": child_manifest.get("support_pair_relative_orbit"),
                                "support_t1_aoi_coverage": child_manifest.get("support_t1_aoi_coverage"),
                                "support_t2_aoi_coverage": child_manifest.get("support_t2_aoi_coverage"),
                                "support_t1_aoi_bbox_coverage": child_manifest.get("support_t1_aoi_bbox_coverage"),
                                "support_t2_aoi_bbox_coverage": child_manifest.get("support_t2_aoi_bbox_coverage"),
                            },
                            "pre_scenes": child_manifest["pre_scenes"],
                            "post_scenes": child_manifest["post_scenes"],
                        },
                        "component_membership": child_manifest.get("component_membership", {}),
                        "composite_dir": str(child_composite_dir),
                        "t1_composite_path": str(t1_composite_path),
                        "t2_composite_path": str(t2_composite_path),
                        "composite": composite_meta,
                    }
                    component_results.append(
                        component_record if save_debug_artifacts else strip_debug_artifact_paths(component_record)
                    )
                    component_mosaic_sources.append(
                        {
                            "component_id": component_id,
                            "area_ratio_vs_parent": component["area_ratio_vs_parent"],
                            "area_m2": component["area_m2"],
                            "selection": component_record["selection"],
                            "geometry": component_geometry,
                            "sr_multiband_path": str(transient_output_tif),
                        }
                    )
                    parent_source_t1_items.extend(selection["post_items"])
                    parent_source_t2_items.extend(selection["pre_items"])

                public_item_id = _whole_monthly_sr_item_id(aoi_id, period["period_id"])
                emit_pipeline_stage(
                    "Parent Mosaic",
                    period_id=period["period_id"],
                    selected_component_count=len(component_mosaic_sources),
                    parent_mosaic_ordering="largest_first",
                )
                parent_transient_output_tif = transient_root / f"{public_item_id}_parent_SR_x2.tif"
                parent_mosaic = mosaic_component_sr_multibands_to_parent(
                    component_sources=component_mosaic_sources,
                    parent_aoi_geometry=aoi_geometry,
                    parent_aoi_bbox=aoi_bbox,
                    target_crs=target_crs,
                    target_resolution=target_resolution,
                    output_path=parent_transient_output_tif,
                    compression=out_cfg.get("compression", "DEFLATE"),
                    tiled=bool(out_cfg.get("tiled", True)),
                    blockxsize=int(out_cfg.get("blockxsize", 256)),
                    blockysize=int(out_cfg.get("blockysize", 256)),
                )
                packaged_outputs = export_masked_sr_band_cogs(
                    sr_multiband_path=parent_mosaic["sr_multiband_path"],
                    output_dir=output_dir,
                    output_basename=public_item_id,
                    geometry_wgs84=parent_mosaic["supported_geometry"],
                    compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
                    blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
                    band_filename_style="whole_monthly_public",
                    crop_to_valid_data=False,
                    include_internal_mask=False,
                    persist_valid_mask=False,
                    final_target_crs=out_cfg.get("final_target_crs"),
                    final_target_resolution=out_cfg.get("final_target_resolution"),
                    final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
                )

            component_audit_by_id = {
                str(entry.get("component_id")): entry for entry in parent_mosaic.get("component_audit", []) or []
            }
            for component_record in component_results:
                audit = component_audit_by_id.get(str(component_record.get("component_id"))) or {}
                contributed = bool(audit.get("contributed_to_parent_mosaic", False))
                component_record["mosaic_priority_rank"] = audit.get("mosaic_priority_rank")
                component_record["contributed_to_parent_mosaic"] = contributed
                component_record["new_pixel_count"] = int(audit.get("new_pixel_count", 0) or 0)
                component_record["new_pixel_ratio"] = float(audit.get("new_pixel_ratio", 0.0) or 0.0)
                component_record["decision_summary"] = compact_jsonable(
                    {
                        **(component_record.get("decision_summary") or {}),
                        "mosaic_priority_rank": audit.get("mosaic_priority_rank"),
                        "contributed_to_parent_mosaic": contributed,
                        "new_pixel_count": component_record["new_pixel_count"],
                        "new_pixel_ratio": component_record["new_pixel_ratio"],
                    }
                )
                component_record["human_summary"] = (
                    f"Child kept with {component_record['pre_scene_count']} pre scenes and "
                    f"{component_record['post_scene_count']} post scenes; "
                    f"{'contributed new parent pixels' if contributed else 'did not add new parent pixels after largest-first mosaic'}."
                )

            suppressed_component_count = sum(1 for item in rejected_components if item.get("status") == "suppressed")
            period_summary = {
                "status": "completed",
                "workflow_mode": "gee_trainlike_composite",
                "selection_strategy": "representative_calendar_period",
                "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                "run_dir": str(run_dir),
                "period_dir": str(period_dir),
                "manifest_path": str(manifest_path),
                "output_tif": None,
                "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
                "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
                "public_item_id": public_item_id,
                "items_after_hard_filter": len(gee_items),
                "period": {
                    "period_id": period["period_id"],
                    "period_mode": period["period_mode"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    "period_anchor_datetime": period["period_anchor_datetime"],
                    "period_split_policy": period_split_policy,
                },
                "componentization": {
                    "enabled": True,
                    "mode": "seed_item_intersections",
                    "delivery_mode": "parent_mosaic",
                    "item_min_region_coverage": component_item_min_coverage,
                    "min_area_ratio": component_min_area_ratio,
                    "completed_component_count": len(component_results),
                    "rejected_component_count": len(rejected_components),
                    "suppressed_component_count": suppressed_component_count,
                    "parent_supported_area_m2": parent_mosaic["supported_area_m2"],
                    "parent_supported_area_ratio": parent_mosaic["supported_area_ratio"],
                    "parent_supported_bbox": canonical_bbox_from_geometry(parent_mosaic["supported_geometry"]),
                    "parent_contributing_component_ids": parent_mosaic["contributing_component_ids"],
                    "parent_mosaic_ordering": parent_mosaic.get("parent_mosaic_ordering", "largest_first"),
                    "component_parent_mosaic": component_parent_mosaic,
                    "decision_summary": {
                        "suppression_policy": "largest_first_tolerant_nested_pruning",
                        "suppressed_component_count": suppressed_component_count,
                        "completed_component_count": len(component_results),
                        "parent_contributing_component_ids": parent_mosaic["contributing_component_ids"],
                        "parent_mosaic_ordering": parent_mosaic.get("parent_mosaic_ordering", "largest_first"),
                    },
                },
                "component_results": component_results,
                "rejected_component_candidates": rejected_components,
                "run_config": {
                    "gee_project": gee_project,
                    "collection": collection_id,
                    "datetime": datetime_filter,
                    "datetime_resolution": datetime_resolution,
                    "pols": ",".join(required_pols),
                    "period_mode": period_mode,
                    "period_boundary_policy": period_boundary_policy,
                    "period_split_policy": period_split_policy,
                    "allow_partial_periods": allow_partial_periods,
                    "min_scenes_per_half": min_scenes_per_half,
                    "auto_relax_inside_period": auto_relax_inside_period,
                    "same_orbit_direction": same_orbit_direction,
                    "representative_pool_mode": representative_pool_mode,
                    "orbit_pass": orbit_pass,
                    "clip_mode": clip_mode_name,
                    "target_crs": target_crs,
                    "target_resolution": target_resolution,
                    "focal_median_radius_m": focal_radius_m,
                    "device": infer_config.get("device"),
                    "cache_staging": cache_staging,
                    "componentize_seed_intersections": componentize_seed_intersections,
                    "component_parent_mosaic": component_parent_mosaic,
                    "component_item_min_coverage": component_item_min_coverage,
                    "component_min_area_ratio": component_min_area_ratio,
                    "save_debug_artifacts": save_debug_artifacts,
                    **build_final_output_trace_config(out_cfg),
                },
                "human_summary": (
                    f"Processed {len(component_results)} child components after suppressing "
                    f"{suppressed_component_count} near-nested candidates. Parent mosaic used largest-first ordering "
                    f"and received new pixels from {len(parent_mosaic['contributing_component_ids'])} child components."
                ),
            }
            if compatibility_info is not None:
                period_summary["compatibility"] = compatibility_info
            attach_sr_output_geojson(
                summary=period_summary,
                geometry_wgs84=parent_mosaic["supported_geometry"],
                infer_config=infer_config,
                source_t1_items=parent_source_t1_items,
                source_t2_items=parent_source_t2_items,
            )
            summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
            emit_pipeline_stage(
                "Output Summary",
                period_id=period["period_id"],
                status=period_summary.get("status"),
                public_item_id=public_item_id,
                completed_component_count=len(component_results),
                suppressed_component_count=suppressed_component_count,
                contributing_component_ids=parent_mosaic["contributing_component_ids"],
            )
            period_results.append(
                {
                    "period_id": period["period_id"],
                    "status": "completed",
                    "component_count_completed": len(component_results),
                    "component_count_rejected": len(rejected_components),
                    "output_tif": None,
                    "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                    "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                    "output_sr_geojson_path": period_summary.get("output_sr_geojson_path"),
                    "manifest_path": str(manifest_path),
                    "summary_json": str(summary_json),
                    "summary_md": (str(summary_md) if summary_md else None),
                    "component_results": component_results,
                    "component_delivery_mode": "parent_mosaic",
                }
            )
            continue

        selection = select_representative_scene_pools(
            pre_items=pre_items,
            post_items=post_items,
            aoi_geometry=aoi_geometry,
            aoi_bbox=aoi_bbox,
            anchor_dt=anchor_dt,
            min_scenes_per_half=min_scenes_per_half,
            auto_relax_inside_period=auto_relax_inside_period,
            require_same_orbit_direction=same_orbit_direction,
            representative_pool_mode=representative_pool_mode,
        )

        if selection is None:
            skip_reason = (
                "No valid in-period representative scene pools found after the relaxation ladder. "
                "The month is missing balanced first-half/second-half coverage under the configured constraints."
            )
            period_summary = {
                "status": "skipped",
                "workflow_mode": "gee_trainlike_composite",
                "selection_strategy": "representative_calendar_period",
                "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
                "run_dir": str(run_dir),
                "period_dir": str(period_dir),
                "period": {
                    "period_id": period["period_id"],
                    "period_mode": period["period_mode"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    "period_anchor_datetime": period["period_anchor_datetime"],
                    "period_split_policy": period_split_policy,
                },
                "items_in_period": {
                    "pre": len(pre_items),
                    "post": len(post_items),
                },
                "skip_reason": skip_reason,
                "manifest_path": str(manifest_path),
                "run_config": {
                    "gee_project": gee_project,
                    "collection": collection_id,
                    "datetime": datetime_filter,
                    "datetime_resolution": datetime_resolution,
                    "pols": ",".join(required_pols),
                    "period_mode": period_mode,
                    "period_boundary_policy": period_boundary_policy,
                    "period_split_policy": period_split_policy,
                    "allow_partial_periods": allow_partial_periods,
                    "min_scenes_per_half": min_scenes_per_half,
                    "auto_relax_inside_period": auto_relax_inside_period,
                    "same_orbit_direction": same_orbit_direction,
                    "representative_pool_mode": representative_pool_mode,
                    "orbit_pass": orbit_pass,
                    "clip_mode": clip_mode_name,
                    "target_crs": target_crs,
                    "target_resolution": target_resolution,
                    "focal_median_radius_m": focal_radius_m,
                    "device": infer_config.get("device"),
                    "cache_staging": cache_staging,
                    "componentize_seed_intersections": componentize_seed_intersections,
                    "component_item_min_coverage": component_item_min_coverage,
                    "component_min_area_ratio": component_min_area_ratio,
                    "save_debug_artifacts": save_debug_artifacts,
                    **build_final_output_trace_config(out_cfg),
                },
            }
            summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
            period_summary["summary_json"] = str(summary_json)
            period_summary["summary_md"] = (str(summary_md) if summary_md else None)
            period_results.append(
                {
                    "period_id": period["period_id"],
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "pre_scene_count": len(pre_items),
                    "post_scene_count": len(post_items),
                    "summary_json": str(summary_json),
                    "summary_md": (str(summary_md) if summary_md else None),
                }
            )
            continue

        manifest = build_representative_period_manifest(
            period=period,
            selection=selection,
            aoi_bbox=aoi_bbox,
            geojson_path=str(aoi_path),
            required_pols=required_pols,
        )
        manifest["period_split_policy"] = period_split_policy
        manifest["period_boundary_policy"] = period_boundary_policy
        manifest["gee_project"] = gee_project
        manifest["gee_collection"] = collection_id
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        public_item_id = _whole_monthly_sr_item_id(aoi_id, period["period_id"])
        with tempfile.TemporaryDirectory(prefix="sr_period_") as transient_dir:
            transient_root = Path(transient_dir)
            composite_dir = resolve_optional_debug_dir(
                persist=save_debug_artifacts,
                persistent_path=intermediate_dir(period_dir) / "composite",
                transient_path=transient_root / "composite",
            )
            pre_collection = _build_gee_scene_collection(collection_id, selection["pre_items"])
            post_collection = _build_gee_scene_collection(collection_id, selection["post_items"])
            t1_image = build_trainlike_image(post_collection, clip_geom, focal_radius_m)
            t2_image = build_trainlike_image(pre_collection, clip_geom, focal_radius_m)
            grid = build_target_grid(aoi_bbox, target_crs, target_resolution, target_resolution)

            t1_composite_path = composite_dir / f"s1t1_{manifest['pair_id']}.tif"
            t2_composite_path = composite_dir / f"s1t2_{manifest['pair_id']}.tif"
            download_gee_image(
                t1_image,
                build_export_params(f"s1t1_{manifest['pair_id']}", grid, band_names),
                t1_composite_path,
            )
            download_gee_image(
                t2_image,
                build_export_params(f"s1t2_{manifest['pair_id']}", grid, band_names),
                t2_composite_path,
            )
            rewrite_with_descriptions(t1_composite_path, output_descs, grid)
            rewrite_with_descriptions(t2_composite_path, output_descs, grid)
            validation = validate_pair(composite_dir, manifest["pair_id"], grid, output_descs)
            transient_output_tif = Path(transient_dir) / f"{aoi_id}__{manifest['pair_id']}_SR_x2.tif"
            inferencer.run_pair_from_multiband_files(
                identifier=manifest["pair_id"],
                t1_path=t1_composite_path,
                t2_path=t2_composite_path,
                out_path=transient_output_tif,
                config=model_trainlike_semantics(),
            )
            packaged_outputs = export_masked_sr_band_cogs(
                sr_multiband_path=transient_output_tif,
                output_dir=output_dir,
                output_basename=public_item_id,
                geometry_wgs84=aoi_geometry,
                compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
                blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
                band_filename_style="whole_monthly_public",
                crop_to_valid_data=True,
                include_internal_mask=False,
                persist_valid_mask=False,
                final_target_crs=out_cfg.get("final_target_crs"),
                final_target_resolution=out_cfg.get("final_target_resolution"),
                final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
            )

        pre_count = len(selection["pre_items"])
        post_count = len(selection["post_items"])
        composite_meta = {
            "grid": {
                "crs": grid["crs"],
                "width": grid["width"],
                "height": grid["height"],
                "transform": list(grid["transform"])[:6],
            },
            "pre": {
                "scene_counts": {"vv": pre_count, "vh": pre_count},
                "band_descriptions": output_descs,
                "grid": {
                    "crs": grid["crs"],
                    "width": grid["width"],
                    "height": grid["height"],
                    "transform": list(grid["transform"])[:6],
                },
            },
            "post": {
                "scene_counts": {"vv": post_count, "vh": post_count},
                "band_descriptions": output_descs,
                "grid": {
                    "crs": grid["crs"],
                    "width": grid["width"],
                    "height": grid["height"],
                    "transform": list(grid["transform"])[:6],
                },
            },
        }
        period_summary = {
            "status": "completed",
            "workflow_mode": "gee_trainlike_composite",
            "selection_strategy": "representative_calendar_period",
            "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
            "run_dir": str(run_dir),
            "period_dir": str(period_dir),
            "manifest_path": str(manifest_path),
            "composite_dir": (str(composite_dir) if save_debug_artifacts else None),
            "t1_composite_path": (str(t1_composite_path) if save_debug_artifacts else None),
            "t2_composite_path": (str(t2_composite_path) if save_debug_artifacts else None),
            "output_tif": None,
            "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
            "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
            "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
            "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
            "public_item_id": public_item_id,
            "validation": validation,
            "inference_input_semantics": model_trainlike_semantics(),
            "items_after_hard_filter": len(gee_items),
            "period": {
                "period_id": period["period_id"],
                "period_mode": period["period_mode"],
                "period_start": period["period_start"],
                "period_end": period["period_end"],
                "period_anchor_datetime": period["period_anchor_datetime"],
                "period_split_policy": period_split_policy,
            },
            "selection": {
                "selection_priority": manifest.get("selection_priority", "balanced_period_representation"),
                "selected_relaxation_level": manifest["selected_relaxation_level"],
                "selected_relaxation_name": manifest["selected_relaxation_name"],
                "required_scene_count": manifest["required_scene_count"],
                "scene_signature_mode": manifest["scene_signature_mode"],
                "scene_signature_value": manifest["scene_signature_value"],
                "pre_scene_count": manifest["pre_scene_count"],
                "post_scene_count": manifest["post_scene_count"],
                "pre_unique_datetime_count": manifest["pre_unique_datetime_count"],
                "post_unique_datetime_count": manifest["post_unique_datetime_count"],
                "pre_union_coverage": manifest.get("pre_union_coverage"),
                "post_union_coverage": manifest.get("post_union_coverage"),
                "combined_union_coverage": manifest.get("combined_union_coverage"),
                "pre_anchor_gap_hours": manifest["pre_anchor_gap_hours"],
                "post_anchor_gap_hours": manifest["post_anchor_gap_hours"],
                "latest_input_datetime": manifest.get("latest_input_datetime"),
                "witness_support_pair": {
                    "support_t1_id": manifest.get("support_t1_id"),
                    "support_t2_id": manifest.get("support_t2_id"),
                    "support_t1_datetime": manifest.get("support_t1_datetime"),
                    "support_t2_datetime": manifest.get("support_t2_datetime"),
                    "support_pair_delta_hours": manifest.get("support_pair_delta_hours"),
                    "support_pair_delta_days": manifest.get("support_pair_delta_days"),
                    "support_pair_orbit_state": manifest.get("support_pair_orbit_state"),
                    "support_pair_relative_orbit": manifest.get("support_pair_relative_orbit"),
                    "support_t1_aoi_coverage": manifest.get("support_t1_aoi_coverage"),
                    "support_t2_aoi_coverage": manifest.get("support_t2_aoi_coverage"),
                    "support_t1_aoi_bbox_coverage": manifest.get("support_t1_aoi_bbox_coverage"),
                    "support_t2_aoi_bbox_coverage": manifest.get("support_t2_aoi_bbox_coverage"),
                },
                "pre_scenes": manifest["pre_scenes"],
                "post_scenes": manifest["post_scenes"],
            },
            "composite": composite_meta,
            "run_config": {
                "gee_project": gee_project,
                "collection": collection_id,
                "datetime": datetime_filter,
                "datetime_resolution": datetime_resolution,
                "pols": ",".join(required_pols),
                "period_mode": period_mode,
                "period_boundary_policy": period_boundary_policy,
                "period_split_policy": period_split_policy,
                "allow_partial_periods": allow_partial_periods,
                "min_scenes_per_half": min_scenes_per_half,
                "auto_relax_inside_period": auto_relax_inside_period,
                "same_orbit_direction": same_orbit_direction,
                "representative_pool_mode": representative_pool_mode,
                "orbit_pass": orbit_pass,
                "clip_mode": clip_mode_name,
                "target_crs": target_crs,
                "target_resolution": target_resolution,
                "focal_median_radius_m": focal_radius_m,
                "device": infer_config.get("device"),
                "cache_staging": cache_staging,
                "componentize_seed_intersections": componentize_seed_intersections,
                "component_item_min_coverage": component_item_min_coverage,
                "component_min_area_ratio": component_min_area_ratio,
                "save_debug_artifacts": save_debug_artifacts,
                **build_final_output_trace_config(out_cfg),
            },
        }
        if compatibility_info is not None:
            period_summary["compatibility"] = compatibility_info
        attach_sr_output_geojson(
            summary=period_summary,
            geometry_wgs84=aoi_geometry,
            infer_config=infer_config,
            source_t1_items=selection["post_items"],
            source_t2_items=selection["pre_items"],
        )
        summary_json, summary_md = write_representative_period_summary(period_dir, period_summary)
        period_summary["summary_json"] = str(summary_json)
        period_summary["summary_md"] = (str(summary_md) if summary_md else None)
        period_results.append(
            {
                "period_id": period["period_id"],
                "status": "completed",
                "selected_relaxation_level": manifest["selected_relaxation_level"],
                "selected_relaxation_name": manifest["selected_relaxation_name"],
                "pre_scene_count": manifest["pre_scene_count"],
                "post_scene_count": manifest["post_scene_count"],
                "pre_unique_datetime_count": manifest["pre_unique_datetime_count"],
                "post_unique_datetime_count": manifest["post_unique_datetime_count"],
                "output_tif": None,
                "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
                "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
                "output_sr_geojson_path": period_summary.get("output_sr_geojson_path"),
                "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
                "manifest_path": str(manifest_path),
                "summary_json": str(summary_json),
                "summary_md": (str(summary_md) if summary_md else None),
            }
        )

    summary = {
        "workflow_mode": "gee_trainlike_composite",
        "selection_strategy": "representative_calendar_period",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "periods_dir": str(periods_root),
        "items_after_hard_filter": len(gee_items),
        "period_counts": {
            "total": len(periods),
            "completed": sum(1 for p in period_results if p["status"] == "completed"),
            "skipped": sum(1 for p in period_results if p["status"] == "skipped"),
            "failed": sum(1 for p in period_results if p["status"] == "failed"),
        },
        "period_results": period_results,
        "run_config": {
            "gee_project": gee_project,
            "collection": collection_id,
            "datetime": datetime_filter,
            "datetime_resolution": datetime_resolution,
            "pols": ",".join(required_pols),
            "period_mode": period_mode,
            "period_boundary_policy": period_boundary_policy,
            "period_split_policy": period_split_policy,
            "allow_partial_periods": allow_partial_periods,
            "min_scenes_per_half": min_scenes_per_half,
            "auto_relax_inside_period": auto_relax_inside_period,
            "same_orbit_direction": same_orbit_direction,
            "representative_pool_mode": representative_pool_mode,
            "orbit_pass": orbit_pass,
            "clip_mode": clip_mode_name,
            "target_crs": target_crs,
            "target_resolution": target_resolution,
            "focal_median_radius_m": focal_radius_m,
            "device": infer_config.get("device"),
            "cache_staging": cache_staging,
            "componentize_seed_intersections": componentize_seed_intersections,
            "component_item_min_coverage": component_item_min_coverage,
            "component_min_area_ratio": component_min_area_ratio,
            "save_debug_artifacts": save_debug_artifacts,
            **build_final_output_trace_config(out_cfg),
        },
    }
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    summary_json, summary_md = write_representative_job_summary(run_dir, summary)
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = (str(summary_md) if summary_md else None)
    return summary


def run_stac_trainlike_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """AOI -> STAC timeline -> anchor -> multi-scene window download -> local composite -> inference."""
    train_cfg = config.get("trainlike", {})
    selection_strategy = str(train_cfg.get("selection_strategy", "latest_anchor_by_support_pair")).strip().lower()
    if selection_strategy == "representative_calendar_period":
        return run_stac_representative_calendar_pipeline(config, geojson_path, output_root, cache_staging, device)

    try:
        from infer_production import SARInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Please install the production inference environment before running sar_pipeline.py."
        ) from exc

    aoi_path = Path(geojson_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {aoi_path}")
    aoi_id = resolve_runtime_aoi_id(config, aoi_path)

    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="stac_trainlike_composite_db")
    save_debug_artifacts = save_debug_artifacts_enabled(config)

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("STAC train-like pipeline currently requires pols=VV,VH.")

    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_dir = resolve_pipeline_run_dir(config, run_root, aoi_id)
    manifest_path = aoi_manifest_path(run_dir)
    output_dir = ensure_dir(run_dir / out_cfg.get("output_dir_name", "output"))

    client = STACClient(stac_cfg.get("url", DEFAULT_STAC_API))
    query_args = build_query_namespace(config, str(aoi_path))
    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, query_args, required_pols)
    if not items:
        raise RuntimeError(f"No STAC items passed hard filters for {aoi_path}.")

    anchor_candidate, required_scene_count = choose_anchor_candidate(items, aoi_bbox, aoi_geometry, train_cfg, pair_cfg)
    manifest = build_trainlike_anchor_manifest(anchor_candidate, aoi_bbox, str(aoi_path), required_pols)
    manifest["required_scene_count"] = required_scene_count
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    anchor_dt = datetime.fromisoformat(manifest["anchor_datetime"].replace("Z", "+00:00"))
    pre_items_full, post_items_full = collect_anchor_window_items(
        items=items,
        aoi_geometry=aoi_geometry,
        aoi_bbox=aoi_bbox,
        anchor_dt=anchor_dt,
        window_before_days=float(manifest["window_before_days"]),
        window_after_days=float(manifest["window_after_days"]),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 0.0)),
    )
    pre_items = dedupe_items_by_scene(pre_items_full)
    post_items = dedupe_items_by_scene(post_items_full)
    if len(pre_items) < required_scene_count or len(post_items) < required_scene_count:
        raise RuntimeError(
            "Selected anchor no longer satisfies the required scene count after item reconstruction."
        )

    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    infer_overrides = apply_inference_env_overrides(infer_config)
    if infer_overrides:
        infer_config.setdefault("_runtime", {})["env_overrides"] = compact_jsonable(infer_overrides)
        log_inference_env_overrides(infer_config)
    if device:
        infer_config["device"] = device
    inferencer = SARInferencer(infer_config)
    target_crs = str(train_cfg.get("target_crs", "EPSG:3857"))
    target_resolution = float(train_cfg.get("target_resolution", 10.0))
    resampling_name = str(train_cfg.get("resampling", config.get("staging", {}).get("resampling", "bilinear")))
    focal_radius_m = float(train_cfg.get("focal_median_radius_m", 15.0))
    pair_id = manifest["pair_id"]
    with tempfile.TemporaryDirectory(prefix="sr_anchor_") as transient_dir:
        transient_root = Path(transient_dir)
        window_raw_dir = resolve_optional_debug_dir(
            persist=save_debug_artifacts,
            persistent_path=inputs_dir(run_dir) / train_cfg.get("window_raw_dir_name", "window_raw"),
            transient_path=transient_root / train_cfg.get("window_raw_dir_name", "window_raw"),
        )
        composite_dir = resolve_optional_debug_dir(
            persist=save_debug_artifacts,
            persistent_path=intermediate_dir(run_dir) / train_cfg.get("composite_dir_name", "composite"),
            transient_path=transient_root / train_cfg.get("composite_dir_name", "composite"),
        )
        downloader = S3Downloader()
        pre_paths = download_window_assets(pre_items, window_raw_dir / "pre", aoi_geometry, required_pols, downloader)
        post_paths = download_window_assets(post_items, window_raw_dir / "post", aoi_geometry, required_pols, downloader)

        grid = build_target_grid(aoi_bbox, target_crs, target_resolution, target_resolution)
        t1_composite_path, post_meta = compose_window_to_multiband(
            grouped_paths=post_paths,
            grid=grid,
            resampling_name=resampling_name,
            focal_radius_m=focal_radius_m,
            out_path=composite_dir / f"s1t1_{pair_id}.tif",
            output_cfg=out_cfg,
        )
        t2_composite_path, pre_meta = compose_window_to_multiband(
            grouped_paths=pre_paths,
            grid=grid,
            resampling_name=resampling_name,
            focal_radius_m=focal_radius_m,
            out_path=composite_dir / f"s1t2_{pair_id}.tif",
            output_cfg=out_cfg,
        )
        transient_output_tif = transient_root / f"{aoi_id}__{pair_id}_SR_x2.tif"
        inferencer.run_pair_from_multiband_files(
            identifier=pair_id,
            t1_path=t1_composite_path,
            t2_path=t2_composite_path,
            out_path=transient_output_tif,
            config=model_trainlike_semantics(),
        )
        packaged_outputs = export_masked_sr_band_cogs(
            sr_multiband_path=transient_output_tif,
            output_dir=output_dir,
            output_basename=f"{aoi_id}__{pair_id}",
            geometry_wgs84=aoi_geometry,
            compression=infer_config.get("output", {}).get("compression", "DEFLATE"),
            blocksize=int(infer_config.get("output", {}).get("blockxsize", 512)),
            persist_valid_mask=False,
            final_target_crs=out_cfg.get("final_target_crs"),
            final_target_resolution=out_cfg.get("final_target_resolution"),
            final_resampling_name=str(out_cfg.get("final_resampling", "bilinear")),
        )

    summary = {
        "workflow_mode": "stac_trainlike_composite",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "anchor_manifest_path": str(manifest_path),
        "window_raw_dir": (str(window_raw_dir) if save_debug_artifacts else None),
        "composite_dir": (str(composite_dir) if save_debug_artifacts else None),
        "t1_composite_path": (str(t1_composite_path) if save_debug_artifacts else None),
        "t2_composite_path": (str(t2_composite_path) if save_debug_artifacts else None),
        "output_tif": None,
        "output_sr_vv_tif": packaged_outputs["output_sr_vv_tif"],
        "output_sr_vh_tif": packaged_outputs["output_sr_vh_tif"],
        "output_sr_band_tifs": packaged_outputs["output_sr_band_tifs"],
        "output_valid_mask_path": packaged_outputs["output_valid_mask_path"],
        "inference_input_semantics": model_trainlike_semantics(),
        "items_after_hard_filter": len(items),
        "anchor": {
            "selection_priority": manifest.get("selection_priority", "latest_input_datetime"),
            "anchor_strategy": manifest["anchor_strategy"],
            "anchor_datetime": manifest["anchor_datetime"],
            "latest_input_datetime": manifest.get("latest_input_datetime"),
            "window_before_days": manifest["window_before_days"],
            "window_after_days": manifest["window_after_days"],
            "required_scene_count": required_scene_count,
            "support_t1_id": manifest.get("t1_id"),
            "support_t2_id": manifest.get("t2_id"),
            "support_t1_datetime": manifest.get("t1_datetime"),
            "support_t2_datetime": manifest.get("t2_datetime"),
            "support_later_id": manifest.get("later_id", manifest.get("t1_id")),
            "support_earlier_id": manifest.get("earlier_id", manifest.get("t2_id")),
            "support_later_datetime": manifest.get("later_datetime", manifest.get("t1_datetime")),
            "support_earlier_datetime": manifest.get("earlier_datetime", manifest.get("t2_datetime")),
            "support_pair_delta_hours": manifest.get("support_pair_delta_hours"),
            "support_pair_delta_days": manifest.get("support_pair_delta_days"),
            "pre_scene_count": len(pre_items),
            "post_scene_count": len(post_items),
            "pre_scenes": manifest.get("pre_scenes", []),
            "post_scenes": manifest.get("post_scenes", []),
            "model_input_semantics": model_trainlike_semantics(),
        },
        "composite": {
            "grid": post_meta["grid"],
            "pre": pre_meta,
            "post": post_meta,
        },
        "downloaded_files": (
            {
                "pre": {pol: [str(p) for p in paths] for pol, paths in pre_paths.items()},
                "post": {pol: [str(p) for p in paths] for pol, paths in post_paths.items()},
            }
            if save_debug_artifacts
            else None
        ),
        "run_config": {
            "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
            "datetime_resolution": datetime_resolution,
            "limit": int(stac_cfg.get("limit", 300)),
            "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 0.0)),
            "pols": ",".join(required_pols),
            "window_before_days": float(manifest["window_before_days"]),
            "window_after_days": float(manifest["window_after_days"]),
            "min_scenes_per_window": required_scene_count,
            "selection_priority": "latest_input_datetime",
            "same_orbit_direction": bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False))),
            "target_crs": target_crs,
            "target_resolution": target_resolution,
            "resampling": resampling_name,
            "focal_median_radius_m": focal_radius_m,
            "device": infer_config.get("device"),
            "cache_staging": cache_staging,
            "save_debug_artifacts": save_debug_artifacts,
        },
    }
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    attach_sr_output_geojson(
        summary=summary,
        geometry_wgs84=aoi_geometry,
        infer_config=infer_config,
        source_t1_items=post_items,
        source_t2_items=pre_items,
    )
    summary_json, summary_md = aoi_summary_paths(run_dir)
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = (str(summary_md) if summary_md else None)
    write_trainlike_run_summary(run_dir, summary)
    return summary


def run_pipeline(config: Dict[str, Any], geojson_path: str, output_root: Optional[str], cache_staging: bool, device: Optional[str]) -> Dict[str, Any]:
    workflow_cfg = config.get("workflow", {})
    mode = str(workflow_cfg.get("mode", "exact_pair")).strip().lower()
    try:
        if mode == "exact_pair":
            return run_exact_pair_pipeline(config, geojson_path, output_root, cache_staging, device)
        if mode == "stac_trainlike_composite":
            return run_stac_trainlike_pipeline(config, geojson_path, output_root, cache_staging, device)
        if mode == "gee_trainlike_composite":
            selection_strategy = str(config.get("trainlike", {}).get("selection_strategy", "representative_calendar_period")).strip().lower()
            if selection_strategy != "representative_calendar_period":
                raise ValueError(
                    "gee_trainlike_composite currently supports only trainlike.selection_strategy=representative_calendar_period."
                )
            return run_gee_representative_calendar_pipeline(config, geojson_path, output_root, cache_staging, device)
        raise ValueError(f"Unsupported workflow.mode: {workflow_cfg.get('mode')}")
    except Exception as exc:
        if mode not in {"stac_trainlike_composite", "gee_trainlike_composite"}:
            raise
        aoi_path = Path(geojson_path)
        aoi_id = resolve_runtime_aoi_id(config, aoi_path)
        run_root = ensure_dir(output_root or config.get("output", {}).get("root_dir", "runs/pipeline"))
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        runtime_info = config.setdefault("_runtime", {})
        existing_run_dir = runtime_info.get("current_run_dir")
        if existing_run_dir:
            run_dir = ensure_dir(existing_run_dir)
        else:
            run_dir = ensure_dir(run_root / aoi_id / f"{run_id}__failed")
        failure_summary = build_failed_representative_run_summary(
            config=config,
            geojson_path=geojson_path,
            run_dir=run_dir,
            error=exc,
        )
        write_representative_job_summary(run_dir, failure_summary)
        raise


def _print_pipeline_completion(summary: Dict[str, Any]) -> None:
    print("[PIPELINE] completed")
    print(f"  Mode: {summary['workflow_mode']}")
    print(f"  AOI: {summary['aoi_geojson']}")
    if summary["workflow_mode"] == "exact_pair":
        pair = summary["selected_pair"]
        print(f"  Pair: {pair['pair_id']}")
        print(f"  Latest input: {pair['latest_input_datetime']}")
        print(f"  Delta: {format_duration_human(pair['delta_seconds'])}")
        print(f"  AOI geometry coverage min: {pair['aoi_coverage_min']:.3f}")
        print(f"  AOI bbox coverage min (diagnostic): {pair['aoi_bbox_coverage_min']:.3f}")
        print(f"  Output: {summary['output_tif']}")
    elif summary.get("selection_strategy") == "representative_calendar_period":
        counts = summary["period_counts"]
        print(f"  Selection strategy: {summary['selection_strategy']}")
        print(f"  Periods: total={counts['total']} completed={counts['completed']} skipped={counts['skipped']}")
        first_output = next((p.get("output_tif") for p in summary["period_results"] if p.get("status") == "completed"), None)
        if first_output:
            print(f"  First output: {first_output}")
    else:
        anchor = summary["anchor"]
        print(f"  Anchor: {anchor['anchor_datetime']}")
        print(f"  Latest input: {anchor.get('latest_input_datetime')}")
        print(f"  Window: -{anchor['window_before_days']}d / +{anchor['window_after_days']}d")
        print(f"  Scenes: pre={anchor['pre_scene_count']} post={anchor['post_scene_count']}")
        print(f"  Output: {summary['output_tif']}")
    runtime_summary = summary.get("summary_json")
    runtime_run_dir = summary.get("run_dir") or summary.get("period_dir")
    if runtime_run_dir:
        runtime_summary = str(job_summary_path(infer_job_dir_from_runtime_path(runtime_run_dir)))
    print(f"  Summary: {runtime_summary}")


def execute_pipeline_request(
    *,
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
    config_path: str,
    mode_label: Optional[str],
    capture_output: bool = True,
) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    def _run_request_body() -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
        summary_local: Optional[Dict[str, Any]] = None
        exit_code_local = 0
        error_message_local: Optional[str] = None
        runtime_info = config.get("_runtime", {}) or {}
        logger.info("=" * 72)
        logger.info("CLI pipeline started")
        logger.info(
            "Runtime context | job_id=%s source_mode=%s aoi_id=%s effective_log_level=%s log_level_source=%s",
            runtime_info.get("job_id"),
            runtime_info.get("source_mode"),
            runtime_info.get("aoi_id"),
            runtime_info.get("effective_log_level"),
            runtime_info.get("log_level_source"),
        )
        logger.info("AOI: %s", geojson_path)
        logger.info("Mode override: %s", mode_label or "(from config)")
        logger.info("Config: %s", config_path)
        logger.info("=" * 72)
        try:
            summary_local = run_pipeline(
                config=config,
                geojson_path=geojson_path,
                output_root=output_root,
                cache_staging=cache_staging,
                device=device,
            )
        except KeyboardInterrupt:
            print("[PIPELINE] interrupted by user")
            exit_code_local = 130
            error_message_local = "Interrupted by user."
        except Exception as exc:
            print(f"[PIPELINE] failed: {exc}")
            exit_code_local = 1
            error_message_local = str(exc)
        else:
            _print_pipeline_completion(summary_local)
        logger.info("=" * 72)
        logger.info("CLI pipeline finished with exit code %s", exit_code_local)
        logger.info("=" * 72)
        return summary_local, exit_code_local, error_message_local

    if not capture_output:
        summary, exit_code, error_message = _run_request_body()
        config.setdefault("_runtime", {})["last_pipeline_summary"] = summary
        config.setdefault("_runtime", {})["last_pipeline_error"] = error_message
        return summary, exit_code, error_message

    fd, transcript_name = tempfile.mkstemp(prefix="sar_pipeline_", suffix=".log")
    os.close(fd)
    transcript_path = Path(transcript_name)
    summary: Optional[Dict[str, Any]] = None
    exit_code = 0
    error_message: Optional[str] = None
    try:
        with open(transcript_path, "w", encoding="utf-8", buffering=1) as transcript_fp:
            tee_stdout = _TeeTextIO(sys.stdout, transcript_fp)
            tee_stderr = _TeeTextIO(sys.stderr, transcript_fp)
            root_logger = logging.getLogger()
            redirected_handlers: List[Tuple[logging.StreamHandler, Any]] = []
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    old_stream = handler.setStream(tee_stderr)
                    redirected_handlers.append((handler, old_stream))
            try:
                with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                    summary, exit_code, error_message = _run_request_body()
            finally:
                for handler, old_stream in redirected_handlers:
                    handler.setStream(old_stream)
        config.setdefault("_runtime", {})["last_pipeline_summary"] = summary
        config.setdefault("_runtime", {})["last_pipeline_error"] = error_message
    finally:
        runtime_run_dir = config.get("_runtime", {}).get("current_run_dir")
        final_log_path: Optional[Path] = None
        if summary is not None and summary.get("run_log_path"):
            final_log_path = Path(str(summary["run_log_path"]))
        elif runtime_run_dir:
            final_log_path = aoi_log_path(runtime_run_dir)
        if final_log_path is not None:
            final_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_path, "r", encoding="utf-8") as src:
                transcript_text = src.read()
            if transcript_text:
                with open(final_log_path, "a", encoding="utf-8") as dst:
                    if final_log_path.exists() and final_log_path.stat().st_size > 0:
                        dst.write("\n")
                    dst.write(transcript_text)
        transcript_path.unlink(missing_ok=True)
    return summary, exit_code, error_message


def _summarize_job_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    completed = sum(1 for result in results if result.get("status") == "completed")
    failed = sum(1 for result in results if result.get("status") == "failed")
    interrupted = sum(1 for result in results if result.get("status") == "interrupted")
    skipped = sum(1 for result in results if result.get("status") == "skipped")
    if interrupted > 0:
        status = "interrupted"
        final_status = "fail"
        reason_code = "JOB_INTERRUPTED"
        reason_message = "Execution was interrupted before all AOIs finished."
    elif failed > 0:
        status = "failed"
        final_status = "fail"
        reason_code = "AOI_RUNS_FAILED"
        reason_message = "One or more AOI runs failed."
    elif completed > 0 and failed == 0 and interrupted == 0:
        status = "completed"
        if completed == total:
            final_status = "pass"
            reason_code = "JOB_COMPLETED"
            reason_message = "All AOI runs completed successfully."
        else:
            final_status = "pass"
            reason_code = "JOB_COMPLETED_WITH_SKIPS"
            reason_message = "Job completed, but some AOIs were skipped."
    else:
        status = "skipped"
        final_status = "skip"
        reason_code = "NO_VALID_AOI_TO_RUN"
        reason_message = "No valid AOI remained to run."
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "interrupted": interrupted,
        "skipped": skipped,
        "status": status,
        "final_status": final_status,
        "reason_code": reason_code,
        "reason_message": reason_message,
    }


def write_runtime_job_summary(job_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
    json_path = job_summary_path(job_dir)
    log_path = job_log_path(job_dir)
    summary["summary_json"] = str(json_path)
    summary["log_path"] = str(log_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_jsonable(summary), f, indent=2, ensure_ascii=False)
    return json_path, None


def build_runtime_job_record(
    *,
    job_layout: Dict[str, str],
    config_path: str,
    resolved_config: Optional[Dict[str, Any]],
    log_level: Optional[str],
    startup_checks: Optional[Dict[str, Any]],
    db_env_path: Optional[str],
    requested_aoi_id: Optional[str],
    aois: List[Dict[str, Any]],
    selection_mode: Optional[str] = None,
    db_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result_rows = [
        {
            "status": aoi.get("status"),
            "final_status": aoi.get("final_status"),
            "reason_code": aoi.get("reason_code"),
        }
        for aoi in aois
    ]
    counts = _summarize_job_results(result_rows)
    record = {
        "summary_version": "2026-03-30.v2",
        "job_id": job_layout["job_id"],
        "source_mode": job_layout["source_mode"],
        "workflow_mode": job_layout["workflow_mode"],
        "job_dir": job_layout["job_dir"],
        "aois_dir": job_layout["aois_dir"],
        "config_path": config_path,
        "log_level": log_level,
        "startup_checks": startup_checks,
        "db_env_path": db_env_path,
        "requested_aoi_id": requested_aoi_id,
        "resolved_config": resolved_config,
        "counts": counts,
        "status": counts["status"],
        "final_status": counts["final_status"],
        "reason_code": counts["reason_code"],
        "reason_message": counts["reason_message"],
        "log_path": str(job_log_path(job_layout["job_dir"])),
        "aois": aois,
    }
    if selection_mode is not None:
        record["selection_mode"] = selection_mode
    if db_counts is not None:
        record["db_counts"] = db_counts
    return compact_jsonable(record)


def run_database_aoi_batch(
    *,
    config: Dict[str, Any],
    records: List[Dict[str, Any]],
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
    config_path: str,
    mode_label: Optional[str],
    db_env_path: str,
    selection_mode: str,
    effective_log_level: Optional[str],
    startup_checks: Optional[Dict[str, Any]],
    requested_aoi_id: Optional[str] = None,
    job_layout_override: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
) -> Tuple[Dict[str, Any], int]:
    workflow_mode = str(config.get("workflow", {}).get("mode", "unknown"))
    default_root = ensure_dir(output_root or config.get("output", {}).get("root_dir", "runs/pipeline"))
    job_layout = job_layout_override or prepare_storage_job_layout(
        base_root=default_root,
        workflow_mode=workflow_mode,
        source_mode="db_batch",
    )
    job_dir = Path(job_layout["job_dir"])
    resolved_config = build_resolved_config_snapshot(config)
    emit_pipeline_log(
        logging.INFO,
        "Starting database AOI batch run",
        job_id=job_layout["job_id"],
        workflow_mode=workflow_mode,
        record_count=len(records),
        selection_mode=selection_mode,
    )

    aoi_entries: List[Dict[str, Any]] = []
    batch_exit_code = 0

    for index, record in enumerate(records, start=1):
        emit_pipeline_log(
            logging.INFO,
            "Processing AOI from database batch",
            index=index,
            total=len(records),
            aoi_id=record.get("id"),
            geometry_is_valid=record.get("geometry_is_valid"),
        )
        if not record.get("geometry_is_valid", False):
            emit_pipeline_log(
                logging.WARNING,
                "Skipping ACTIVE AOI with invalid geometry",
                aoi_id=record.get("id"),
                invalid_reason=record.get("geometry_invalid_reason"),
                geometry_type=record.get("geometry_type"),
                geometry_srid=record.get("geometry_srid"),
            )
            aoi_entries.append(
                compact_jsonable(
                    {
                        "aoi_id": record["id"],
                        "aoi_dir": runtime_relpath(Path(job_layout["aois_dir"]) / record["id"]),
                        "source_ref": f"db://public.aois/{record['id']}",
                        "status": "skipped",
                        "final_status": "skip",
                        "reason_code": "INVALID_AOI_GEOMETRY",
                        "reason_message": (
                            f"Skipped because geom is invalid: {record.get('geometry_invalid_reason') or 'unknown reason'}"
                        ),
                        "db_status": record.get("status"),
                        "geometry_is_valid": record.get("geometry_is_valid"),
                        "geometry_invalid_reason": record.get("geometry_invalid_reason"),
                        "geometry_type": record.get("geometry_type"),
                        "geometry_srid": record.get("geometry_srid"),
                        "period_counts": {"total": 0, "completed": 0, "skipped": 0, "failed": 0},
                        "periods": [],
                    }
                )
            )
            continue

        aoi_layout = prepare_storage_aoi_layout(job_layout, record["id"])
        with tempfile.TemporaryDirectory(prefix=f"aoi_{record['id']}_") as temp_geojson_dir:
            aoi_geojson_path = materialize_database_aoi_geojson(
                record,
                temp_geojson_dir,
                filename="input_aoi.geojson",
            )
            aoi_config = copy.deepcopy(config)
            runtime = aoi_config.setdefault("_runtime", {})
            runtime["current_run_dir"] = aoi_layout["aoi_dir"]
            runtime["job_dir"] = job_layout["job_dir"]
            runtime["job_id"] = job_layout["job_id"]
            runtime["source_mode"] = job_layout["source_mode"]
            runtime["aoi_id"] = aoi_layout["aoi_id"]
            runtime["aoi_source_ref"] = f"db://public.aois/{record['id']}"
            runtime["effective_log_level"] = effective_log_level
            runtime["log_level_source"] = config.get("_runtime", {}).get("log_level_source")
            runtime["startup_checks"] = startup_checks
            runtime["db_aoi_record"] = {
                "id": record["id"],
                "status": record.get("status"),
                "name": record.get("name"),
                "geometry_type": record.get("geometry_type"),
                "geometry_srid": record.get("geometry_srid"),
            }
            summary, exit_code, error_message = execute_pipeline_request(
                config=aoi_config,
                geojson_path=str(aoi_geojson_path),
                output_root=str(aoi_layout["aoi_dir"]),
                cache_staging=cache_staging,
                device=device,
                config_path=config_path,
                mode_label=mode_label,
                capture_output=capture_output,
            )
        aoi_entry = collect_runtime_aoi_entry(
            aoi_layout["aoi_dir"],
            fallback_summary=summary,
            fallback_error_message=error_message,
        )
        if not aoi_entry.get("source_ref"):
            aoi_entry["source_ref"] = f"db://public.aois/{record['id']}"
        aoi_entry["db_status"] = record.get("status")
        aoi_entry["geometry_type"] = record.get("geometry_type")
        aoi_entry["geometry_srid"] = record.get("geometry_srid")
        aoi_entries.append(compact_jsonable(aoi_entry))
        if exit_code == 130:
            batch_exit_code = 130
            break
        if exit_code != 0:
            batch_exit_code = 1

    emit_pipeline_log(
        logging.INFO,
        "Finished database AOI batch orchestration",
        job_id=job_layout["job_id"],
        completed_runs=sum(1 for entry in aoi_entries if entry.get("status") == "completed"),
        skipped_runs=sum(1 for entry in aoi_entries if entry.get("status") == "skipped"),
        failed_runs=sum(1 for entry in aoi_entries if entry.get("status") == "failed"),
        interrupted=(batch_exit_code == 130),
    )

    batch_summary = build_runtime_job_record(
        job_layout=job_layout,
        config_path=config_path,
        resolved_config=resolved_config,
        log_level=effective_log_level,
        startup_checks=startup_checks,
        db_env_path=db_env_path,
        requested_aoi_id=requested_aoi_id,
        aois=aoi_entries,
        selection_mode=selection_mode,
        db_counts={
            "fetched_active": len(records),
            "valid_geometry": sum(1 for record in records if record.get("geometry_is_valid", False)),
            "skipped_invalid_geometry": sum(1 for record in records if not record.get("geometry_is_valid", False)),
        },
    )
    json_path, md_path = write_runtime_job_summary(job_dir, batch_summary)
    batch_summary["summary_json"] = str(json_path)
    batch_summary["summary_md"] = (str(md_path) if md_path else None)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact_jsonable(batch_summary), f, indent=2, ensure_ascii=False)
    return batch_summary, batch_exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="AOI -> STAC/GEE -> preprocess -> ISSM-SAR inference pipeline")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--geojson", help="Path to input AOI GeoJSON")
    source_group.add_argument("--db-aoi-id", help="Load one ACTIVE AOI from public.aois by UUID and run the pipeline on it.")
    source_group.add_argument(
        "--db-all-active-aois",
        action="store_true",
        help="Query all ACTIVE AOIs from public.aois and process them sequentially.",
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Path to pipeline config")
    parser.add_argument("--mode", default=None, help="Override workflow mode: exact_pair, stac_trainlike_composite, or gee_trainlike_composite")
    parser.add_argument("--db-env-path", default=".env", help="Path to .env file used to resolve PG* database settings.")
    parser.add_argument("--db-limit", type=int, default=None, help="Optional safety limit for --db-all-active-aois.")
    parser.add_argument(
        "--datetime",
        default=None,
        help="Override query datetime range. For representative-month mode, omit this flag or pass `auto` to resolve the latest completed month automatically.",
    )
    parser.add_argument(
        "--target-month",
        default=None,
        help="Representative-month shortcut for backfill, formatted as YYYY-MM. This expands to the exact UTC month range automatically.",
    )
    parser.add_argument("--min-delta-hours", type=float, default=None, help="Override minimum time delta")
    parser.add_argument("--max-delta-days", type=int, default=None, help="Override maximum time delta")
    parser.add_argument("--min-aoi-coverage", type=float, default=None, help="Override minimum AOI geometry coverage; hard gate is coverage > threshold")
    parser.add_argument("--same-orbit-direction", action="store_true", help="Require same orbit direction within the train-like selection logic")
    parser.add_argument(
        "--representative-pool-mode",
        choices=["auto", "orbit_only", "mixed"],
        default=None,
        help=(
            "Override representative monthly pre/post pool selection. "
            "`auto` keeps the existing relaxation ladder, "
            "`orbit_only` never relaxes into mixed-orbit pools, "
            "`mixed` forces one shared pre/post pool regardless of orbit grouping."
        ),
    )
    parser.add_argument("--auto-relax", action="store_true", help="Enable balanced/loose time window fallback")
    parser.add_argument("--window-before-days", type=float, default=None, help="Override STAC train-like pre-window length")
    parser.add_argument("--window-after-days", type=float, default=None, help="Override STAC train-like post-window length")
    parser.add_argument("--min-scenes-per-window", type=int, default=None, help="Override minimum unique scenes per STAC window")
    parser.add_argument("--target-crs", default=None, help="Override train-like target CRS, e.g. EPSG:3857")
    parser.add_argument("--target-resolution", type=float, default=None, help="Override train-like target pixel size in meters")
    parser.add_argument("--focal-median-radius-m", type=float, default=None, help="Override train-like focal median radius in meters")
    parser.add_argument("--device", default=None, help="Override inference device")
    parser.add_argument("--output-dir", default=None, help="Override pipeline run root directory")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override runtime log level. Precedence: --log-level > PIPELINE_LOG_LEVEL > config.logging.level > INFO.",
    )
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--save-debug-data",
        dest="save_debug_data",
        action="store_true",
        help="Persist raw/composite debug artifacts into the runtime tree.",
    )
    debug_group.add_argument(
        "--no-save-debug-data",
        dest="save_debug_data",
        action="store_false",
        help="Do not persist raw/composite debug artifacts. This is the safe default.",
    )
    parser.set_defaults(save_debug_data=None)
    parser.add_argument("--cache-staging", action="store_true", help="Persist aligned 2-band inputs in staging dir")
    parser.add_argument("--gee-project", default=None, help="Override GEE project id")
    parser.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() if required for GEE")
    parser.add_argument(
        "--auto-datetime-strategy",
        default=None,
        help="Override representative-month auto datetime strategy, e.g. previous_full_month.",
    )
    parser.add_argument(
        "--auto-datetime-months-back",
        type=int,
        default=None,
        help="Override how many completed months to step back when datetime is auto-resolved.",
    )
    parser.add_argument(
        "--auto-datetime-timezone",
        default=None,
        help="Override timezone used to decide which calendar month is considered the latest completed month.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    config.setdefault("workflow", {})
    config.setdefault("stac", {})
    config.setdefault("gee", {})
    config.setdefault("pairing", {})
    config.setdefault("trainlike", {})
    config.setdefault("output", {})
    config.setdefault("logging", {})
    config["logging"].setdefault("level", DEFAULT_LOG_LEVEL)
    config["logging"].setdefault("redact_secrets", True)
    config["logging"].setdefault("startup_env_checks", True)
    config["output"].setdefault("save_debug_artifacts", False)
    effective_log_level, log_level_source = resolve_runtime_log_level(cli_level=args.log_level, config=config)
    configure_root_logging(effective_log_level)
    config["logging"]["level"] = effective_log_level
    config.setdefault("_runtime", {})["effective_log_level"] = effective_log_level
    config.setdefault("_runtime", {})["log_level_source"] = log_level_source
    apply_runtime_env_overrides(config)
    if args.mode is not None:
        config["workflow"]["mode"] = args.mode
    if args.db_limit is not None and args.db_limit < 1:
        parser.error("--db-limit must be >= 1.")
    if args.target_month is not None and args.datetime is not None:
        parser.error("Use only one of --datetime or --target-month.")
    if args.datetime is not None:
        config["stac"]["datetime"] = args.datetime
        config["gee"]["datetime"] = args.datetime
    if args.target_month is not None:
        resolved_datetime, resolution = resolve_target_month_datetime_range(args.target_month)
        config["trainlike"]["target_month"] = args.target_month
        config["stac"]["datetime"] = resolved_datetime
        config["gee"]["datetime"] = resolved_datetime
        config.setdefault("_runtime", {})["datetime_resolution"] = resolution
    if args.auto_datetime_strategy is not None:
        config["trainlike"]["auto_datetime_strategy"] = args.auto_datetime_strategy
    if args.auto_datetime_months_back is not None:
        config["trainlike"]["auto_datetime_months_back"] = args.auto_datetime_months_back
    if args.auto_datetime_timezone is not None:
        config["trainlike"]["auto_datetime_timezone"] = args.auto_datetime_timezone
    if args.min_delta_hours is not None:
        config["pairing"]["min_delta_hours"] = args.min_delta_hours
    if args.max_delta_days is not None:
        config["pairing"]["max_delta_days"] = args.max_delta_days
    if args.min_aoi_coverage is not None:
        config["pairing"]["min_aoi_coverage"] = args.min_aoi_coverage
    if args.same_orbit_direction:
        config["pairing"]["same_orbit_direction"] = True
        config["trainlike"]["same_orbit_direction"] = True
    if args.representative_pool_mode is not None:
        config["trainlike"]["representative_pool_mode"] = args.representative_pool_mode
    if args.auto_relax:
        config["pairing"]["auto_relax"] = True
    if args.window_before_days is not None:
        config["trainlike"]["window_before_days"] = args.window_before_days
    if args.window_after_days is not None:
        config["trainlike"]["window_after_days"] = args.window_after_days
    if args.min_scenes_per_window is not None:
        config["trainlike"]["min_scenes_per_window"] = args.min_scenes_per_window
    if args.target_crs is not None:
        config["trainlike"]["target_crs"] = args.target_crs
    if args.target_resolution is not None:
        config["trainlike"]["target_resolution"] = args.target_resolution
    if args.focal_median_radius_m is not None:
        config["trainlike"]["focal_median_radius_m"] = args.focal_median_radius_m
    if args.log_level is not None:
        config["logging"]["level"] = normalize_log_level_name(args.log_level)
    if args.save_debug_data is not None:
        config["output"]["save_debug_artifacts"] = bool(args.save_debug_data)
    if args.gee_project is not None:
        config["gee"]["project"] = args.gee_project
    if args.authenticate:
        config["gee"]["authenticate"] = True
    if args.geojson:
        workflow_mode = str(config.get("workflow", {}).get("mode", "unknown"))
        default_root = ensure_dir(args.output_dir or config.get("output", {}).get("root_dir", "runs/pipeline"))
        job_layout = prepare_storage_job_layout(
            base_root=default_root,
            workflow_mode=workflow_mode,
            source_mode="geojson_single",
        )
        aoi_id = Path(args.geojson).stem
        aoi_layout = prepare_storage_aoi_layout(job_layout, aoi_id)
        runtime = config.setdefault("_runtime", {})
        runtime["current_run_dir"] = aoi_layout["aoi_dir"]
        runtime["job_dir"] = job_layout["job_dir"]
        runtime["job_id"] = job_layout["job_id"]
        runtime["source_mode"] = job_layout["source_mode"]
        runtime["aoi_id"] = aoi_layout["aoi_id"]
        runtime["aoi_source_ref"] = str(Path(args.geojson).resolve())
        runtime["effective_log_level"] = effective_log_level
        runtime["log_level_source"] = log_level_source
        runtime["startup_checks"] = build_startup_checks(
            config=config,
            source_mode=job_layout["source_mode"],
            db_env_path=None,
        )
        resolved_config = build_resolved_config_snapshot(config)
        with capture_runtime_job_logs(job_layout["job_dir"]):
            log_runtime_env_overrides(config)
            emit_pipeline_log(
                logging.INFO,
                "Prepared runtime job",
                job_id=job_layout["job_id"],
                source_mode=job_layout["source_mode"],
                workflow_mode=workflow_mode,
                config_path=args.config,
                effective_log_level=effective_log_level,
                log_level_source=log_level_source,
                save_debug_artifacts=config["output"].get("save_debug_artifacts"),
                output_root=aoi_layout["aoi_dir"],
                target_month=config.get("trainlike", {}).get("target_month"),
                datetime=config.get("stac", {}).get("datetime") or config.get("gee", {}).get("datetime"),
            )
            if config.get("logging", {}).get("startup_env_checks", True):
                log_startup_checks(runtime["startup_checks"])
            summary, exit_code, error_message = execute_pipeline_request(
                config=config,
                geojson_path=str(Path(args.geojson).resolve()),
                output_root=aoi_layout["aoi_dir"],
                cache_staging=args.cache_staging,
                device=args.device,
                config_path=args.config,
                mode_label=args.mode,
                capture_output=False,
            )
        aoi_entry = collect_runtime_aoi_entry(
            aoi_layout["aoi_dir"],
            fallback_summary=summary,
            fallback_error_message=error_message,
        )
        single_job_summary = build_runtime_job_record(
            job_layout=job_layout,
            config_path=args.config,
            resolved_config=resolved_config,
            log_level=effective_log_level,
            startup_checks=runtime.get("startup_checks"),
            db_env_path=None,
            requested_aoi_id=None,
            aois=[aoi_entry],
        )
        write_runtime_job_summary(Path(job_layout["job_dir"]), single_job_summary)
        if exit_code:
            sys.exit(exit_code)
        return

    if args.db_aoi_id:
        workflow_mode = str(config.get("workflow", {}).get("mode", "unknown"))
        single_root = ensure_dir(args.output_dir or config.get("output", {}).get("root_dir", "runs/pipeline"))
        job_layout = prepare_storage_job_layout(
            base_root=single_root,
            workflow_mode=workflow_mode,
            source_mode="db_single",
        )
        startup_checks = build_startup_checks(
            config=config,
            source_mode=job_layout["source_mode"],
            db_env_path=args.db_env_path,
        )
        with capture_runtime_job_logs(job_layout["job_dir"]):
            log_runtime_env_overrides(config)
            emit_pipeline_log(
                logging.INFO,
                "Prepared runtime job",
                job_id=job_layout["job_id"],
                source_mode=job_layout["source_mode"],
                workflow_mode=workflow_mode,
                config_path=args.config,
                effective_log_level=effective_log_level,
                log_level_source=log_level_source,
                save_debug_artifacts=config["output"].get("save_debug_artifacts"),
                output_root=job_layout["job_dir"],
                target_month=config.get("trainlike", {}).get("target_month"),
                datetime=config.get("stac", {}).get("datetime") or config.get("gee", {}).get("datetime"),
                requested_aoi_id=args.db_aoi_id,
            )
            if config.get("logging", {}).get("startup_env_checks", True):
                log_startup_checks(startup_checks)
            emit_pipeline_stage(
                "DB AOI",
                requested_aoi_id=args.db_aoi_id,
                db_env_path=args.db_env_path,
            )
            record_list = fetch_active_aois_from_database(
                aoi_id=normalize_aoi_uuid(args.db_aoi_id),
                env_path=args.db_env_path,
            )
            if not record_list:
                emit_pipeline_log(
                    logging.ERROR,
                    "Requested ACTIVE AOI was not found in database",
                    requested_aoi_id=args.db_aoi_id,
                    db_env_path=args.db_env_path,
                )
                raise RuntimeError(f"No ACTIVE AOI found in public.aois for id={args.db_aoi_id}.")
            record = record_list[0]
            if not record.get("geometry_is_valid", False):
                emit_pipeline_log(
                    logging.ERROR,
                    "Requested ACTIVE AOI has invalid geometry",
                    requested_aoi_id=args.db_aoi_id,
                    invalid_reason=record.get("geometry_invalid_reason"),
                )
                raise RuntimeError(
                    f"AOI {record['id']} is ACTIVE but geom is invalid: {record.get('geometry_invalid_reason') or 'unknown reason'}"
                )
            aoi_layout = prepare_storage_aoi_layout(job_layout, record["id"])
            aoi_config = copy.deepcopy(config)
            runtime = aoi_config.setdefault("_runtime", {})
            runtime["current_run_dir"] = aoi_layout["aoi_dir"]
            runtime["job_dir"] = job_layout["job_dir"]
            runtime["job_id"] = job_layout["job_id"]
            runtime["source_mode"] = job_layout["source_mode"]
            runtime["aoi_id"] = aoi_layout["aoi_id"]
            runtime["aoi_source_ref"] = f"db://public.aois/{record['id']}"
            runtime["effective_log_level"] = effective_log_level
            runtime["log_level_source"] = log_level_source
            runtime["startup_checks"] = startup_checks
            runtime["db_aoi_record"] = {
                "id": record["id"],
                "status": record.get("status"),
                "name": record.get("name"),
                "geometry_type": record.get("geometry_type"),
                "geometry_srid": record.get("geometry_srid"),
            }
            resolved_config = build_resolved_config_snapshot(aoi_config)
            with tempfile.TemporaryDirectory(prefix=f"aoi_{record['id']}_") as temp_geojson_dir:
                geojson_path = materialize_database_aoi_geojson(record, temp_geojson_dir, filename="input_aoi.geojson")
                summary, exit_code, error_message = execute_pipeline_request(
                    config=aoi_config,
                    geojson_path=str(geojson_path),
                    output_root=str(aoi_layout["aoi_dir"]),
                    cache_staging=args.cache_staging,
                    device=args.device,
                    config_path=args.config,
                    mode_label=args.mode,
                    capture_output=False,
                )
        aoi_entry = collect_runtime_aoi_entry(
            aoi_layout["aoi_dir"],
            fallback_summary=summary,
            fallback_error_message=error_message,
        )
        if not aoi_entry.get("source_ref"):
            aoi_entry["source_ref"] = f"db://public.aois/{record['id']}"
        single_job_summary = build_runtime_job_record(
            job_layout=job_layout,
            config_path=args.config,
            resolved_config=resolved_config,
            log_level=effective_log_level,
            startup_checks=startup_checks,
            db_env_path=args.db_env_path,
            requested_aoi_id=record["id"],
            aois=[compact_jsonable(aoi_entry)],
        )
        write_runtime_job_summary(Path(job_layout["job_dir"]), single_job_summary)
        if exit_code:
            sys.exit(exit_code)
        return

    workflow_mode = str(config.get("workflow", {}).get("mode", "unknown"))
    default_root = ensure_dir(args.output_dir or config.get("output", {}).get("root_dir", "runs/pipeline"))
    preview_job_layout = prepare_storage_job_layout(
        base_root=default_root,
        workflow_mode=workflow_mode,
        source_mode="db_batch",
    )
    startup_checks = build_startup_checks(
        config=config,
        source_mode=preview_job_layout["source_mode"],
        db_env_path=args.db_env_path,
    )
    with capture_runtime_job_logs(preview_job_layout["job_dir"]):
        log_runtime_env_overrides(config)
        emit_pipeline_log(
            logging.INFO,
            "Prepared runtime job",
            job_id=preview_job_layout["job_id"],
            source_mode=preview_job_layout["source_mode"],
            workflow_mode=workflow_mode,
            config_path=args.config,
            effective_log_level=effective_log_level,
            log_level_source=log_level_source,
            save_debug_artifacts=config["output"].get("save_debug_artifacts"),
            output_root=preview_job_layout["job_dir"],
            target_month=config.get("trainlike", {}).get("target_month"),
            datetime=config.get("stac", {}).get("datetime") or config.get("gee", {}).get("datetime"),
            requested_db_limit=args.db_limit,
        )
        if config.get("logging", {}).get("startup_env_checks", True):
            log_startup_checks(startup_checks)
        records = fetch_active_aois_from_database(
            limit=args.db_limit,
            env_path=args.db_env_path,
        )
        if not records:
            emit_pipeline_log(
                logging.ERROR,
                "Database batch query returned no ACTIVE AOIs",
                db_env_path=args.db_env_path,
                requested_limit=args.db_limit,
            )
            raise RuntimeError("No ACTIVE AOIs were returned from public.aois.")
        batch_summary, batch_exit_code = run_database_aoi_batch(
            config=config,
            records=records,
            output_root=args.output_dir,
            cache_staging=args.cache_staging,
            device=args.device,
            config_path=args.config,
            mode_label=args.mode,
            db_env_path=args.db_env_path,
            selection_mode="all_active",
            effective_log_level=effective_log_level,
            startup_checks=startup_checks,
            job_layout_override=preview_job_layout,
            capture_output=False,
        )
    print("[DB BATCH] completed")
    print(f"  ACTIVE rows fetched: {batch_summary['db_counts']['fetched_active']}")
    print(f"  Valid AOIs queued: {batch_summary['db_counts']['valid_geometry']}")
    print(f"  Invalid AOIs skipped: {batch_summary['db_counts']['skipped_invalid_geometry']}")
    print(f"  Completed runs: {batch_summary['counts']['completed']}")
    print(f"  Failed runs: {batch_summary['counts']['failed']}")
    print(f"  Batch summary: {batch_summary['summary_json']}")
    if batch_exit_code:
        sys.exit(batch_exit_code)


if __name__ == "__main__":
    main()

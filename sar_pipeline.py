#!/usr/bin/env python3
"""
AOI GeoJSON -> STAC/GEE query -> preprocessing -> SR inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from shapely.geometry import shape

from db_aoi_source import (
    fetch_active_aois_from_database,
    materialize_database_aoi_geojson,
    normalize_aoi_uuid,
)
from pipeline_support.contract_support import (
    apply_execution_contract,
    attach_run_log_path,
    build_aoi_record,
    build_period_record,
    classify_failure_reason,
)
from pipeline_support.json_support import compact_jsonable
from query_stac_download import (
    DEFAULT_COLLECTION,
    DEFAULT_STAC_API,
    STACClient,
    S3Downloader,
    build_representative_period_manifest,
    canonical_bbox_from_geometry,
    collect_items_with_filters,
    collect_period_half_items,
    expand_month_periods,
    extract_item_info,
    geodesic_area_wgs84,
    build_seed_intersection_region_candidates,
    load_geojson_aoi,
    normalize_polygonal_shapely_geometry,
    normalize_representative_pool_mode,
    normalize_datetime_range,
    parse_required_pols,
    select_representative_scene_pools,
    select_asset_href,
)
from pipeline_support.runtime_support import (
    SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD,
    SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC,
    WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE,
    WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE,
    _TeeTextIO,
    aoi_manifest_path,
    aoi_log_path,
    aoi_summary_paths,
    apply_runtime_env_overrides,
    build_effective_runtime_settings,
    build_resolved_config_snapshot,
    build_startup_checks,
    capture_runtime_job_logs,
    collect_runtime_aoi_entry,
    describe_pipeline_profile,
    ensure_dir,
    infer_job_dir_from_runtime_path,
    inputs_dir,
    intermediate_dir,
    is_canonical_selection_strategy,
    is_representative_composite_workflow_mode,
    job_log_path,
    job_summary_path,
    load_yaml,
    model_trainlike_semantics,
    normalize_selection_strategy,
    normalize_workflow_mode,
    period_manifest_path,
    period_summary_paths,
    prepare_storage_aoi_layout,
    prepare_storage_job_layout,
    resolve_aoi_source_ref,
    resolve_pipeline_run_dir,
    resolve_runtime_aoi_id,
    resolve_spatial_strategy,
    runtime_relpath,
    save_debug_artifacts_enabled,
)
from pipeline_support.raster_support import (
    align_single_band_to_grid,
    apply_focal_median_db,
    apply_geometry_mask_to_multiband,
    build_grid_meta,
    build_target_grid,
    dedupe_items_by_scene,
    export_masked_sr_band_cogs,
    geometry_mask_for_grid,
    mosaic_component_sr_multibands_to_parent,
    nanmedian_stack,
    resolve_resampling,
    write_multiband_geotiff,
)
from pipeline_support.sr_packaging_support import (
    _summary_aoi_id,
    _summary_period_token,
    _whole_monthly_sr_item_id,
    attach_sr_output_geojson,
)
from runtime_logging import (
    configure_root_logging,
    DEFAULT_LOG_LEVEL,
    ensure_root_logging,
    emit_runtime_log,
    normalize_log_level_name,
    resolve_runtime_log_level,
)
from runtime_env_overrides import apply_inference_env_overrides

ensure_root_logging(DEFAULT_LOG_LEVEL)
logger = logging.getLogger("sar_pipeline")
AUTO_DATETIME_SENTINELS = {"", "auto", "latest_full_month", "previous_full_month", "auto_previous_full_month"}


def emit_pipeline_log(level: int, message: str, **fields: Any) -> None:
    emit_runtime_log("sar_pipeline", level, message, **fields)


def emit_pipeline_stage(title: str, **fields: Any) -> None:
    normalized = " ".join(str(title or "").strip().upper().split()) or "PIPELINE"
    emit_pipeline_log(logging.INFO, f"================ {normalized} ================", **fields)


def log_effective_runtime_settings(
    *,
    workflow_mode: Any = "",
    train_cfg: Dict[str, Any],
    infer_config: Optional[Dict[str, Any]],
    save_debug_artifacts: bool,
) -> None:
    emit_pipeline_stage(
        "Effective Runtime Settings",
        **build_effective_runtime_settings(
            workflow_mode=workflow_mode,
            train_cfg=train_cfg,
            infer_config=infer_config,
            save_debug_artifacts=save_debug_artifacts,
        ),
    )


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


def _suppressed_component_count(rejected_components: List[Dict[str, Any]]) -> int:
    return sum(1 for item in rejected_components if item.get("status") == "suppressed")


def _build_component_skip_componentization(
    *,
    component_item_min_coverage: float,
    component_min_area_ratio: float,
    rejected_components: List[Dict[str, Any]],
) -> Dict[str, Any]:
    suppressed_count = _suppressed_component_count(rejected_components)
    return {
        "enabled": True,
        "mode": "seed_item_intersections",
        "delivery_mode": "parent_mosaic",
        "item_min_region_coverage": component_item_min_coverage,
        "min_area_ratio": component_min_area_ratio,
        "completed_component_count": 0,
        "rejected_component_count": len(rejected_components),
        "suppressed_component_count": suppressed_count,
        "parent_supported_area_ratio": 0.0,
        "parent_mosaic_ordering": "largest_first",
        "decision_summary": {
            "suppression_policy": "largest_first_tolerant_nested_pruning",
            "suppressed_component_count": suppressed_count,
            "completed_component_count": 0,
            "parent_mosaic_ordering": "largest_first",
        },
    }


def _build_period_counts(period_results: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(period_results),
        "completed": sum(1 for p in period_results if p["status"] == "completed"),
        "skipped": sum(1 for p in period_results if p["status"] == "skipped"),
        "failed": sum(1 for p in period_results if p["status"] == "failed"),
    }


def _finalize_representative_job_summary(
    run_dir: Path,
    summary: Dict[str, Any],
    compatibility_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    summary_json, summary_md = write_representative_job_summary(run_dir, summary)
    summary["summary_json"] = str(summary_json)
    summary["summary_md"] = (str(summary_md) if summary_md else None)
    return summary


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
    representative_pool_mode: str = "mixed",
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


def run_stac_representative_componentized_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """Canonical STAC representative pipeline for componentized parent-mosaic delivery."""
    return run_stac_representative_calendar_pipeline(config, geojson_path, output_root, cache_staging, device)


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
    min_scenes_per_half = int(train_cfg.get("min_scenes_per_half", 1))
    auto_relax_inside_period = bool(train_cfg.get("auto_relax_inside_period", True))
    same_orbit_direction = bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False)))
    representative_pool_mode = normalize_representative_pool_mode(train_cfg.get("representative_pool_mode", "mixed"))
    componentize_seed_intersections = bool(train_cfg.get("componentize_seed_intersections", False))
    component_parent_mosaic = bool(train_cfg.get("component_parent_mosaic", True))
    component_item_min_coverage = float(train_cfg.get("component_item_min_coverage", 1.0))
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
            "period_counts": _build_period_counts(period_results),
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
        return _finalize_representative_job_summary(run_dir, summary, compatibility_info)

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
        workflow_mode=WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE,
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
                        "componentization": _build_component_skip_componentization(
                            component_item_min_coverage=component_item_min_coverage,
                            component_min_area_ratio=component_min_area_ratio,
                            rejected_components=rejected_components,
                        ),
                    "rejected_component_candidates": rejected_components,
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
                component_inputs_root = ensure_dir(inputs_dir(period_dir) / "components")
                component_intermediate_root = ensure_dir(intermediate_dir(period_dir) / "components")
                emit_pipeline_log(
                    logging.INFO,
                    "Persisting component debug artifacts",
                    period_id=period["period_id"],
                    debug_inputs_root=component_inputs_root,
                    debug_intermediate_root=component_intermediate_root,
                    cleanup_after_publish_success=(not save_debug_artifacts),
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
                    component_results.append(component_record)
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
            suppressed_component_count = _suppressed_component_count(rejected_components)
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


    summary = {
        "workflow_mode": "stac_trainlike_composite",
        "selection_strategy": "representative_calendar_period",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "periods_dir": str(periods_root),
        "items_after_hard_filter": len(items),
        "period_counts": _build_period_counts(period_results),
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
    return _finalize_representative_job_summary(run_dir, summary, compatibility_info)


def _build_gee_scene_collection(collection_id: str, scene_items: List[Dict[str, Any]]) -> Any:
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError("earthengine-api is required for gee_trainlike_composite mode.") from exc

    images = [ee.Image(f"{collection_id}/{extract_item_info(item)['id']}") for item in scene_items]
    return ee.ImageCollection.fromImages(images)


def run_gee_representative_componentized_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """Canonical GEE representative pipeline for componentized parent-mosaic delivery."""
    return run_gee_representative_calendar_pipeline(config, geojson_path, output_root, cache_staging, device)


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
    min_scenes_per_half = int(train_cfg.get("min_scenes_per_half", 1))
    auto_relax_inside_period = bool(train_cfg.get("auto_relax_inside_period", True))
    same_orbit_direction = bool(train_cfg.get("same_orbit_direction", pair_cfg.get("same_orbit_direction", False)))
    representative_pool_mode = normalize_representative_pool_mode(train_cfg.get("representative_pool_mode", "mixed"))
    componentize_seed_intersections = bool(train_cfg.get("componentize_seed_intersections", False))
    component_parent_mosaic = bool(train_cfg.get("component_parent_mosaic", True))
    component_item_min_coverage = float(train_cfg.get("component_item_min_coverage", 1.0))
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
        workflow_mode=WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE,
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
                    "componentization": _build_component_skip_componentization(
                        component_item_min_coverage=component_item_min_coverage,
                        component_min_area_ratio=component_min_area_ratio,
                        rejected_components=rejected_components,
                    ),
                    "rejected_component_candidates": rejected_components,
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
                component_intermediate_root = ensure_dir(intermediate_dir(period_dir) / "components")
                emit_pipeline_log(
                    logging.INFO,
                    "Persisting component debug artifacts",
                    period_id=period["period_id"],
                    debug_intermediate_root=component_intermediate_root,
                    cleanup_after_publish_success=(not save_debug_artifacts),
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
                    component_results.append(component_record)
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
            suppressed_component_count = _suppressed_component_count(rejected_components)
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


    summary = {
        "workflow_mode": "gee_trainlike_composite",
        "selection_strategy": "representative_calendar_period",
        "aoi_geojson": resolve_aoi_source_ref(config, aoi_path),
        "run_dir": str(run_dir),
        "periods_dir": str(periods_root),
        "items_after_hard_filter": len(gee_items),
        "period_counts": _build_period_counts(period_results),
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
    return _finalize_representative_job_summary(run_dir, summary, compatibility_info)


def run_stac_trainlike_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """Dispatch canonical STAC trainlike requests."""
    train_cfg = config.get("trainlike", {})
    selection_strategy = normalize_selection_strategy(
        train_cfg.get("selection_strategy", SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD),
        default=SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD,
    )
    if not is_canonical_selection_strategy(selection_strategy):
        raise ValueError(
            "stac_trainlike_composite currently supports only "
            "trainlike.selection_strategy=representative_calendar_period."
        )
    spatial_strategy = resolve_spatial_strategy(train_cfg)
    if spatial_strategy == SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC:
        return run_stac_representative_componentized_pipeline(
            config,
            geojson_path,
            output_root,
            cache_staging,
            device,
        )
    raise ValueError(f"Unsupported canonical STAC spatial strategy: {spatial_strategy}")


def run_gee_trainlike_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """Dispatch canonical GEE trainlike requests."""
    train_cfg = config.get("trainlike", {})
    selection_strategy = normalize_selection_strategy(
        train_cfg.get("selection_strategy", SELECTION_STRATEGY_REPRESENTATIVE_CALENDAR_PERIOD)
    )
    if not is_canonical_selection_strategy(selection_strategy):
        raise ValueError(
            "gee_trainlike_composite currently supports only "
            "trainlike.selection_strategy=representative_calendar_period."
        )
    spatial_strategy = resolve_spatial_strategy(train_cfg)
    if spatial_strategy == SPATIAL_STRATEGY_COMPONENTIZED_PARENT_MOSAIC:
        return run_gee_representative_componentized_pipeline(
            config,
            geojson_path,
            output_root,
            cache_staging,
            device,
        )
    raise ValueError(f"Unsupported canonical GEE spatial strategy: {spatial_strategy}")


def run_representative_composite_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """Dispatch canonical representative composite pipelines by backend."""
    workflow_mode = normalize_workflow_mode(
        (config.get("workflow", {}) or {}).get("mode", WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE),
        default=WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE,
    )
    if workflow_mode == WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE:
        return run_stac_trainlike_pipeline(config, geojson_path, output_root, cache_staging, device)
    if workflow_mode == WORKFLOW_MODE_GEE_TRAINLIKE_COMPOSITE:
        return run_gee_trainlike_pipeline(config, geojson_path, output_root, cache_staging, device)
    raise ValueError(f"Unsupported representative composite workflow.mode: {config.get('workflow', {}).get('mode')}")


def run_pipeline(config: Dict[str, Any], geojson_path: str, output_root: Optional[str], cache_staging: bool, device: Optional[str]) -> Dict[str, Any]:
    workflow_cfg = config.get("workflow", {})
    mode = normalize_workflow_mode(
        workflow_cfg.get("mode", WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE),
        default=WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE,
    )

    try:
        if is_representative_composite_workflow_mode(mode):
            return run_representative_composite_pipeline(
                config,
                geojson_path,
                output_root,
                cache_staging,
                device,
            )
        raise ValueError(f"Unsupported workflow.mode: {workflow_cfg.get('mode')}")
    except Exception as exc:
        if not is_representative_composite_workflow_mode(mode):
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
    workflow_mode = normalize_workflow_mode(
        summary.get("workflow_mode"),
        default=WORKFLOW_MODE_STAC_TRAINLIKE_COMPOSITE,
    )
    print(f"  Mode: {summary['workflow_mode']}")
    print(f"  AOI: {summary['aoi_geojson']}")
    if is_representative_composite_workflow_mode(workflow_mode):
        counts = summary["period_counts"]
        print(f"  Selection strategy: {summary.get('selection_strategy')}")
        print(f"  Periods: total={counts['total']} completed={counts['completed']} skipped={counts['skipped']}")
        first_output = next((p.get("output_tif") for p in summary["period_results"] if p.get("status") == "completed"), None)
        if first_output:
            print(f"  First output: {first_output}")
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
        "pipeline_profile": describe_pipeline_profile(
            workflow_mode=job_layout["workflow_mode"],
            train_cfg=(resolved_config or {}).get("trainlike", {}) if isinstance(resolved_config, dict) else {},
        ),
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
    workflow_profile = describe_pipeline_profile(
        workflow_mode=workflow_mode,
        train_cfg=config.get("trainlike", {}),
    )
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
        **workflow_profile,
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
    parser.add_argument(
        "--config",
        default="config/pipeline_config_stac_runtime.yaml",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Override workflow mode: "
            "stac_trainlike_composite or gee_trainlike_composite."
        ),
    )
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
        workflow_profile = describe_pipeline_profile(
            workflow_mode=workflow_mode,
            train_cfg=config.get("trainlike", {}),
        )
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
                **workflow_profile,
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
        workflow_profile = describe_pipeline_profile(
            workflow_mode=workflow_mode,
            train_cfg=config.get("trainlike", {}),
        )
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
                **workflow_profile,
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
    workflow_profile = describe_pipeline_profile(
        workflow_mode=workflow_mode,
        train_cfg=config.get("trainlike", {}),
    )
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
            **workflow_profile,
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

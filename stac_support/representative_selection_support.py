from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from runtime_logging import emit_runtime_log
from stac_support.stac_geometry_support import (
    annotate_items_for_aoi,
    item_aoi_bbox_coverage_value,
    item_aoi_coverage_value,
    item_coverage_source,
    items_union_coverage,
)
from stac_support.stac_item_support import (
    extract_item_info,
    summarize_unique_scenes,
    unique_datetime_count_from_items,
)
from stac_support.stac_time_support import parse_datetime_utc

REPRESENTATIVE_POOL_MODE_MIXED = "mixed"
_ALLOWED_REPRESENTATIVE_POOL_MODES = {REPRESENTATIVE_POOL_MODE_MIXED}

def normalize_representative_pool_mode(value: Optional[str]) -> str:
    """Normalize representative monthly pool-selection mode."""
    mode = str(value or REPRESENTATIVE_POOL_MODE_MIXED).strip().lower()
    if mode not in _ALLOWED_REPRESENTATIVE_POOL_MODES:
        allowed = ", ".join(sorted(_ALLOWED_REPRESENTATIVE_POOL_MODES))
        raise ValueError(f"Unsupported representative_pool_mode: {value!r}. Expected one of: {allowed}.")
    return mode

def collect_period_half_items(
    items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]],
    period_start: datetime,
    period_anchor: datetime,
    period_end: datetime,
    min_aoi_coverage: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect pre/post items inside one representative period."""
    pre_items: List[Dict[str, Any]] = []
    post_items: List[Dict[str, Any]] = []
    annotate_items_for_aoi(items, aoi_geometry, aoi_bbox=aoi_bbox)
    for item in items:
        cov = item_aoi_coverage_value(item)
        if cov <= min_aoi_coverage:
            continue
        info = extract_item_info(item)
        dt = parse_datetime_utc(info["datetime"])
        if period_start <= dt < period_anchor:
            pre_items.append(item)
        elif period_anchor <= dt < period_end:
            post_items.append(item)
    return pre_items, post_items

def _group_signature(info: Dict[str, Any], mode: str) -> Tuple[str, ...]:
    if mode == REPRESENTATIVE_POOL_MODE_MIXED:
        return ("mixed",)
    raise ValueError(f"Unsupported representative signature mode: {mode}")

def _stable_signature_token(signature_mode: str, signature_key: Tuple[str, ...]) -> str:
    return f"{signature_mode}:{'|'.join(signature_key)}"

def _latest_pre_gap_hours(pre_items: List[Dict[str, Any]], anchor_dt: datetime) -> float:
    if not pre_items:
        return float("inf")
    latest_pre = max(parse_datetime_utc(extract_item_info(item)["datetime"]) for item in pre_items)
    return float((anchor_dt - latest_pre).total_seconds() / 3600.0)

def _earliest_post_gap_hours(post_items: List[Dict[str, Any]], anchor_dt: datetime) -> float:
    if not post_items:
        return float("inf")
    earliest_post = min(parse_datetime_utc(extract_item_info(item)["datetime"]) for item in post_items)
    return float((earliest_post - anchor_dt).total_seconds() / 3600.0)

def select_witness_support_pair(
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]],
    anchor_dt: datetime,
) -> Optional[Dict[str, Any]]:
    """Pick one support pair for QA/provenance after pools are chosen."""
    annotate_items_for_aoi(pre_items, aoi_geometry, aoi_bbox=aoi_bbox)
    annotate_items_for_aoi(post_items, aoi_geometry, aoi_bbox=aoi_bbox)
    best_pair: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int, int, float, str, str]] = None
    for pre_item in pre_items:
        pre_info = extract_item_info(pre_item)
        pre_dt = parse_datetime_utc(pre_info["datetime"])
        pre_cov = item_aoi_coverage_value(pre_item)
        for post_item in post_items:
            post_info = extract_item_info(post_item)
            post_dt = parse_datetime_utc(post_info["datetime"])
            post_cov = item_aoi_coverage_value(post_item)
            if pre_dt >= anchor_dt or post_dt < anchor_dt:
                continue
            delta_hours = float((post_dt - pre_dt).total_seconds() / 3600.0)
            same_rel_orbit = int(pre_info.get("relative_orbit") != post_info.get("relative_orbit"))
            same_orbit_state = int(str(pre_info.get("orbit_state") or "").lower() != str(post_info.get("orbit_state") or "").lower())
            rank_key = (
                delta_hours,
                same_rel_orbit,
                same_orbit_state,
                -min(pre_cov, post_cov),
                pre_info["id"],
                post_info["id"],
            )
            if best_key is None or rank_key < best_key:
                best_key = rank_key
                best_pair = {
                    "support_t1_id": post_info["id"],
                    "support_t2_id": pre_info["id"],
                    "support_t1_datetime": post_info["datetime"],
                    "support_t2_datetime": pre_info["datetime"],
                    "support_pair_delta_hours": delta_hours,
                    "support_pair_delta_days": delta_hours / 24.0,
                    "support_t1_orbit_state": post_info.get("orbit_state"),
                    "support_t2_orbit_state": pre_info.get("orbit_state"),
                    "support_pair_orbit_state": post_info.get("orbit_state")
                    if str(pre_info.get("orbit_state") or "").lower() == str(post_info.get("orbit_state") or "").lower()
                    else f"{pre_info.get('orbit_state')}->{post_info.get('orbit_state')}",
                    "support_pair_relative_orbit": post_info.get("relative_orbit")
                    if pre_info.get("relative_orbit") == post_info.get("relative_orbit")
                    else [pre_info.get("relative_orbit"), post_info.get("relative_orbit")],
                    "support_t1_aoi_coverage": post_cov,
                    "support_t2_aoi_coverage": pre_cov,
                    "support_t1_aoi_bbox_coverage": item_aoi_bbox_coverage_value(post_item),
                    "support_t2_aoi_bbox_coverage": item_aoi_bbox_coverage_value(pre_item),
                    "support_t1_coverage_source": item_coverage_source(post_item),
                    "support_t2_coverage_source": item_coverage_source(pre_item),
                }
    return best_pair

def _representative_signature_candidates(
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]],
    anchor_dt: datetime,
    min_scenes_per_half: int,
    signature_mode: str,
) -> List[Dict[str, Any]]:
    annotate_items_for_aoi(pre_items, aoi_geometry, aoi_bbox=aoi_bbox)
    annotate_items_for_aoi(post_items, aoi_geometry, aoi_bbox=aoi_bbox)
    pre_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    post_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for item in pre_items:
        pre_groups[_group_signature(extract_item_info(item), signature_mode)].append(item)
    for item in post_items:
        post_groups[_group_signature(extract_item_info(item), signature_mode)].append(item)

    candidates: List[Dict[str, Any]] = []
    for signature_key in sorted(set(pre_groups.keys()) & set(post_groups.keys())):
        pre_group = sorted(pre_groups[signature_key], key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]))
        post_group = sorted(post_groups[signature_key], key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]))
        pre_scenes = summarize_unique_scenes(pre_group)
        post_scenes = summarize_unique_scenes(post_group)
        pre_scene_count = len(pre_scenes)
        post_scene_count = len(post_scenes)
        if pre_scene_count < min_scenes_per_half or post_scene_count < min_scenes_per_half:
            continue
        pre_unique_dt = unique_datetime_count_from_items(pre_group)
        post_unique_dt = unique_datetime_count_from_items(post_group)
        pre_anchor_gap = _latest_pre_gap_hours(pre_group, anchor_dt)
        post_anchor_gap = _earliest_post_gap_hours(post_group, anchor_dt)
        post_latest_dt = max(scene["datetime"] for scene in post_scenes)
        pre_union_coverage = items_union_coverage(pre_group, aoi_geometry)
        post_union_coverage = items_union_coverage(post_group, aoi_geometry)
        combined_union_coverage = items_union_coverage(pre_group + post_group, aoi_geometry)
        candidates.append(
            {
                "signature_mode": signature_mode,
                "signature_key": signature_key,
                "signature_token": _stable_signature_token(signature_mode, signature_key),
                "pre_items": pre_group,
                "post_items": post_group,
                "pre_scenes": pre_scenes,
                "post_scenes": post_scenes,
                "pre_scene_count": pre_scene_count,
                "post_scene_count": post_scene_count,
                "pre_unique_datetime_count": pre_unique_dt,
                "post_unique_datetime_count": post_unique_dt,
                "pre_anchor_gap_hours": pre_anchor_gap,
                "post_anchor_gap_hours": post_anchor_gap,
                "pre_union_coverage": pre_union_coverage,
                "post_union_coverage": post_union_coverage,
                "combined_union_coverage": combined_union_coverage,
                "post_latest_scene_datetime": post_latest_dt,
            }
        )
    candidates.sort(
        key=lambda cand: (
            -min(cand["pre_scene_count"], cand["post_scene_count"]),
            abs(cand["pre_scene_count"] - cand["post_scene_count"]),
            -min(cand["pre_unique_datetime_count"], cand["post_unique_datetime_count"]),
            -min(cand["pre_union_coverage"], cand["post_union_coverage"]),
            -cand["combined_union_coverage"],
            min(cand["pre_anchor_gap_hours"], cand["post_anchor_gap_hours"]),
            _neg_timestamp(cand["post_latest_scene_datetime"]),
            cand["signature_token"],
        )
    )
    return candidates

def select_representative_scene_pools(
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]],
    anchor_dt: datetime,
    min_scenes_per_half: int,
    auto_relax_inside_period: bool,
    require_same_orbit_direction: bool = False,
    representative_pool_mode: str = REPRESENTATIVE_POOL_MODE_MIXED,
) -> Optional[Dict[str, Any]]:
    """Select monthly representative scene pools using the canonical mixed pool."""
    representative_pool_mode = normalize_representative_pool_mode(representative_pool_mode)
    emit_runtime_log(
        "query_stac_download",
        logging.DEBUG,
        "Selecting representative scene pools",
        representative_pool_mode=representative_pool_mode,
        pre_items=len(pre_items),
        post_items=len(post_items),
        min_scenes_per_half=min_scenes_per_half,
        auto_relax_inside_period=auto_relax_inside_period,
        require_same_orbit_direction=require_same_orbit_direction,
        anchor_datetime=anchor_dt.isoformat().replace("+00:00", "Z"),
    )

    if require_same_orbit_direction:
        raise ValueError(
            "representative_pool_mode='mixed' cannot be combined with require_same_orbit_direction=True."
        )
    levels = [
        {
            "level": 100,
            "level_name": "forced_mixed_orbit_all_pre_post",
            "signature_mode": REPRESENTATIVE_POOL_MODE_MIXED,
            "required_scene_count": 1 if auto_relax_inside_period else max(1, int(min_scenes_per_half)),
        }
    ]

    for level_cfg in levels:
        candidates = _representative_signature_candidates(
            pre_items=pre_items,
            post_items=post_items,
            aoi_geometry=aoi_geometry,
            aoi_bbox=aoi_bbox,
            anchor_dt=anchor_dt,
            min_scenes_per_half=level_cfg["required_scene_count"],
            signature_mode=level_cfg["signature_mode"],
        )
        if not candidates:
            emit_runtime_log(
                "query_stac_download",
                logging.DEBUG,
                "Representative selection level produced no candidates",
                level_name=level_cfg["level_name"],
                signature_mode=level_cfg["signature_mode"],
                required_scene_count=level_cfg["required_scene_count"],
            )
            continue
        chosen = candidates[0]
        witness_pair = select_witness_support_pair(
            chosen["pre_items"],
            chosen["post_items"],
            aoi_geometry,
            aoi_bbox,
            anchor_dt,
        )
        chosen.update(
            {
                "selected_relaxation_level": level_cfg["level"],
                "selected_relaxation_name": level_cfg["level_name"],
                "required_scene_count": level_cfg["required_scene_count"],
                "scene_signature_mode": level_cfg["signature_mode"],
                "scene_signature_value": list(chosen["signature_key"]),
                "witness_support_pair": witness_pair,
            }
        )
        emit_runtime_log(
            "query_stac_download",
            logging.DEBUG,
            "Selected representative scene pools",
            selected_relaxation_name=level_cfg["level_name"],
            selected_relaxation_level=level_cfg["level"],
            scene_signature_mode=level_cfg["signature_mode"],
            required_scene_count=level_cfg["required_scene_count"],
            pre_scene_count=chosen["pre_scene_count"],
            post_scene_count=chosen["post_scene_count"],
        )
        return chosen
    emit_runtime_log(
        "query_stac_download",
        logging.WARNING,
        "No valid representative scene pools remained",
        representative_pool_mode=representative_pool_mode,
        pre_items=len(pre_items),
        post_items=len(post_items),
        min_scenes_per_half=min_scenes_per_half,
    )
    return None

def build_representative_period_manifest(
    *,
    period: Dict[str, Any],
    selection: Dict[str, Any],
    aoi_bbox: List[float],
    geojson_path: Optional[str],
    required_pols: List[str],
) -> Dict[str, Any]:
    """Build manifest for one representative calendar period."""
    witness = selection.get("witness_support_pair") or {}
    return {
        "manifest_type": "representative_calendar_period",
        "anchor_source": "fixed_period_midpoint",
        "anchor_strategy": "calendar_period_midpoint",
        "selection_strategy": "representative_calendar_period",
        "selection_priority": "balanced_period_representation",
        "period_id": period["period_id"],
        "period_start": period["period_start"],
        "period_end": period["period_end"],
        "period_anchor_datetime": period["period_anchor_datetime"],
        "anchor_datetime": period["period_anchor_datetime"],
        "pair_id": f"period_{period['period_id']}",
        "aoi_bbox": aoi_bbox,
        "aoi_geojson": str(Path(geojson_path).resolve()) if geojson_path else None,
        "required_polarizations": required_pols,
        "required_scene_count": selection["required_scene_count"],
        "selected_relaxation_level": selection["selected_relaxation_level"],
        "selected_relaxation_name": selection["selected_relaxation_name"],
        "scene_signature_mode": selection["scene_signature_mode"],
        "scene_signature_value": selection["scene_signature_value"],
        "pre_scene_count": selection["pre_scene_count"],
        "post_scene_count": selection["post_scene_count"],
        "pre_unique_datetime_count": selection["pre_unique_datetime_count"],
        "post_unique_datetime_count": selection["post_unique_datetime_count"],
        "pre_anchor_gap_hours": selection["pre_anchor_gap_hours"],
        "post_anchor_gap_hours": selection["post_anchor_gap_hours"],
        "pre_union_coverage": selection.get("pre_union_coverage", 0.0),
        "post_union_coverage": selection.get("post_union_coverage", 0.0),
        "combined_union_coverage": selection.get("combined_union_coverage", 0.0),
        "latest_input_datetime": selection["post_latest_scene_datetime"],
        "pre_scenes": selection["pre_scenes"],
        "post_scenes": selection["post_scenes"],
        "support_t1_id": witness.get("support_t1_id"),
        "support_t2_id": witness.get("support_t2_id"),
        "support_t1_datetime": witness.get("support_t1_datetime"),
        "support_t2_datetime": witness.get("support_t2_datetime"),
        "support_pair_delta_hours": witness.get("support_pair_delta_hours"),
        "support_pair_delta_days": witness.get("support_pair_delta_days"),
        "support_pair_orbit_state": witness.get("support_pair_orbit_state"),
        "support_pair_relative_orbit": witness.get("support_pair_relative_orbit"),
        "support_t1_aoi_coverage": witness.get("support_t1_aoi_coverage"),
        "support_t2_aoi_coverage": witness.get("support_t2_aoi_coverage"),
        "support_t1_aoi_bbox_coverage": witness.get("support_t1_aoi_bbox_coverage"),
        "support_t2_aoi_bbox_coverage": witness.get("support_t2_aoi_bbox_coverage"),
        "support_t1_coverage_source": witness.get("support_t1_coverage_source"),
        "support_t2_coverage_source": witness.get("support_t2_coverage_source"),
        "support_pair_semantics": {
            "t1_role": "later/posterior witness scene from second half of period",
            "t2_role": "earlier/prior witness scene from first half of period",
        },
        "model_input_semantics": {
            "s1t1_role": "post/later second half of the same calendar period",
            "s1t2_role": "pre/earlier first half of the same calendar period",
        },
    }

def _neg_timestamp(value: str) -> float:
    return -parse_datetime_utc(value).timestamp()

#!/usr/bin/env python3
"""
=============================================================
 STAC Query -> Best Pair Selection -> S3/Rasterio Pipeline
=============================================================

Muc dich:
  - Doc AOI tu .geojson hoac --bbox
  - Query STAC Item Sentinel-1 GRD
  - Loc theo cac tieu chi model dau vao (IW, GRD, polarization)
  - Chon cap T1/T2 tot nhat (Delta t nho nhat, cung AOI, du polarization, overlap tot)
  - Lay href asset tren S3, kiem tra bang rasterio, tuy chon download
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import rasterio
import requests
from dotenv import load_dotenv
from pyproj import Geod
from rasterio.features import bounds as geometry_bounds
from rasterio.session import AWSSession
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import GeometryCollection, box as shapely_box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from runtime_logging import detect_s3_credential_source, emit_runtime_log, safe_text_snippet

try:
    import boto3
except ImportError:
    boto3 = None

load_dotenv()


DEFAULT_STAC_API = (os.getenv("STAC_API_URL") or "").strip() or None
DEFAULT_COLLECTION = "sentinel-1-grd"
WGS84_GEOD = Geod(ellps="WGS84")
_BBOX_WARN_TOL = 1e-7
_ALLOWED_REPRESENTATIVE_POOL_MODES = {"auto", "orbit_only", "mixed"}
logger = logging.getLogger("query_stac_download")


def parse_datetime_utc(value: str) -> datetime:
    """Parse RFC3339 datetime ve timezone UTC."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def normalize_datetime_range(datetime_range: Optional[str]) -> Optional[str]:
    """Chuan hoa datetime range ve dang RFC3339."""
    if not datetime_range:
        return None
    if "/" not in datetime_range:
        return datetime_range

    parts = datetime_range.split("/")
    formatted_parts: List[str] = []
    for idx, part in enumerate(parts):
        p = part.strip()
        if p == ".." or "T" in p:
            formatted_parts.append(p)
        else:
            suffix = "T00:00:00Z" if idx == 0 else "T23:59:59Z"
            formatted_parts.append(f"{p}{suffix}")
    return "/".join(formatted_parts)


def normalize_representative_pool_mode(value: Optional[str]) -> str:
    """Normalize representative monthly pool-selection mode."""
    mode = str(value or "auto").strip().lower()
    if mode not in _ALLOWED_REPRESENTATIVE_POOL_MODES:
        allowed = ", ".join(sorted(_ALLOWED_REPRESENTATIVE_POOL_MODES))
        raise ValueError(f"Unsupported representative_pool_mode: {value!r}. Expected one of: {allowed}.")
    return mode


def _iter_coords(node: Any) -> Iterable[Tuple[float, float]]:
    """Duyet de quy cac toa do [lon, lat] trong GeoJSON geometry."""
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
            yield float(node[0]), float(node[1])
            return
        for child in node:
            yield from _iter_coords(child)


def _repair_shapely_geometry(geom):
    if geom is None or geom.is_empty:
        return GeometryCollection()
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        repaired = make_valid(geom)
        if repaired is not None and not repaired.is_empty:
            geom = repaired
    except Exception:
        pass
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        repaired = geom.buffer(0)
        if repaired is not None and not repaired.is_empty:
            geom = repaired
    except Exception:
        pass
    if geom is None or geom.is_empty:
        return GeometryCollection()
    return geom


def _shape_from_geojson(geometry: Optional[Dict[str, Any]]):
    if not geometry:
        return GeometryCollection()
    try:
        return _repair_shapely_geometry(shape(geometry))
    except Exception:
        return GeometryCollection()


def _iter_polygonal_parts(geom) -> Iterable[Any]:
    """Yield polygonal members from arbitrary Shapely geometries."""
    if geom is None or getattr(geom, "is_empty", True):
        return
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        yield geom
        return
    if geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not getattr(part, "is_empty", True):
                yield part
        return
    if geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygonal_parts(part)


def normalize_polygonal_shapely_geometry(geom):
    """Keep only polygonal area from a geometry and merge it into a transform-safe shape."""
    geom = _repair_shapely_geometry(geom)
    if geom is None or getattr(geom, "is_empty", True):
        return GeometryCollection()

    polygonal_parts = [part for part in _iter_polygonal_parts(geom) if not getattr(part, "is_empty", True)]
    if not polygonal_parts:
        return GeometryCollection()

    merged = unary_union(polygonal_parts) if len(polygonal_parts) > 1 else polygonal_parts[0]
    merged = _repair_shapely_geometry(merged)
    if merged is None or getattr(merged, "is_empty", True):
        return GeometryCollection()
    if getattr(merged, "geom_type", None) in {"Polygon", "MultiPolygon"}:
        return merged

    polygonal_parts = [part for part in _iter_polygonal_parts(merged) if not getattr(part, "is_empty", True)]
    if not polygonal_parts:
        return GeometryCollection()
    merged = unary_union(polygonal_parts) if len(polygonal_parts) > 1 else polygonal_parts[0]
    return _repair_shapely_geometry(merged)


def normalize_polygonal_geojson_geometry(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a Polygon/MultiPolygon GeoJSON geometry, dropping line/point-only members."""
    geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(geometry))
    if geom.is_empty:
        return None
    return mapping(geom)


def geodesic_area_wgs84(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    try:
        area, _ = WGS84_GEOD.geometry_area_perimeter(geom)
        return abs(float(area))
    except Exception:
        if getattr(geom, "geom_type", None) == "GeometryCollection":
            return sum(geodesic_area_wgs84(part) for part in geom.geoms)
        raise


def canonical_bbox_from_geometry(geometry: Dict[str, Any]) -> List[float]:
    geom = _shape_from_geojson(geometry)
    if geom.is_empty:
        raise ValueError("Khong the tinh bbox tu geometry rong/khong hop le.")
    minx, miny, maxx, maxy = geom.bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _bbox_is_meaningfully_different(left: List[float], right: List[float], tol: float = _BBOX_WARN_TOL) -> bool:
    return any(abs(float(a) - float(b)) > tol for a, b in zip(left, right))


def load_geojson_aoi(geojson_path: str | Path) -> Tuple[List[float], Dict[str, Any]]:
    """Doc GeoJSON va tra ve (bbox, geometry) de query STAC."""
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file geojson: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geoms: List[Dict[str, Any]] = []
    if data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            geom = feat.get("geometry")
            if geom:
                geoms.append(geom)
    elif data.get("type") == "Feature":
        geom = data.get("geometry")
        if geom:
            geoms.append(geom)
    elif data.get("type"):
        geoms.append(data)

    if not geoms:
        raise ValueError(f"Khong tim thay geometry hop le trong {path}")

    raw_geometry = geoms[0] if len(geoms) == 1 else {"type": "GeometryCollection", "geometries": geoms}
    aoi_geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(raw_geometry))
    if aoi_geom.is_empty:
        raise ValueError(f"Khong chuyen duoc AOI geometry thanh hinh hop le trong {path}")

    aoi_geometry = mapping(aoi_geom)
    bbox = canonical_bbox_from_geometry(aoi_geometry)

    provided_bbox = data.get("bbox")
    if isinstance(provided_bbox, list) and len(provided_bbox) == 4:
        provided = [float(v) for v in provided_bbox]
        if _bbox_is_meaningfully_different(provided, bbox):
            print(
                f"[AOI] Warning: top-level bbox in {path} differs from geometry bounds and will be ignored. "
                f"provided={provided} geometry_bbox={bbox}"
            )

    return bbox, aoi_geometry


def bbox_intersection(b1: List[float], b2: List[float]) -> float:
    """Dien tich giao nhau giua 2 bbox [minx, miny, maxx, maxy]."""
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    return (x_right - x_left) * (y_top - y_bottom)


def bbox_area(b: List[float]) -> float:
    """Dien tich bbox."""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_overlap_ratio(bbox1: List[float], bbox2: List[float]) -> float:
    """Overlap ratio = intersection / min(area1, area2)."""
    inter = bbox_intersection(bbox1, bbox2)
    if inter <= 0:
        return 0.0
    a1 = bbox_area(bbox1)
    a2 = bbox_area(bbox2)
    min_area = min(a1, a2)
    return inter / min_area if min_area > 0 else 0.0


def coverage_ratio(reference_bbox: List[float], item_bbox: List[float]) -> float:
    """Ti le AOI duoc item bao phu = intersection / area(reference_bbox)."""
    inter = bbox_intersection(reference_bbox, item_bbox)
    ref_area = bbox_area(reference_bbox)
    return inter / ref_area if ref_area > 0 else 0.0


def bbox_intersection_bounds(b1: List[float], b2: List[float]) -> Optional[List[float]]:
    """Tra ve bbox giao nhau giua 2 bbox, hoac None neu khong giao."""
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return None
    return [x_left, y_bottom, x_right, y_top]


def bbox_to_geometry(bbox: List[float]) -> Dict[str, Any]:
    """Chuyen bbox [minx, miny, maxx, maxy] thanh Polygon GeoJSON."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def _resolve_item_geometry(item: Dict[str, Any]):
    item_geometry = item.get("geometry")
    geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(item_geometry))
    if not geom.is_empty:
        return geom, "geometry", mapping(geom)

    info = extract_item_info(item)
    bbox = info.get("bbox", [])
    if len(bbox) == 4:
        bbox_geom = bbox_to_geometry(bbox)
        geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(bbox_geom))
        if not geom.is_empty:
            return geom, "bbox_fallback", mapping(geom)

    return GeometryCollection(), "none", None


def _compute_item_aoi_geometry_metrics_prepared(
    item: Dict[str, Any],
    aoi_geom,
    aoi_area: float,
    aoi_bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    item_geom, coverage_source, resolved_geojson = _resolve_item_geometry(item)
    coverage = 0.0
    if aoi_area > 0 and not aoi_geom.is_empty and not item_geom.is_empty:
        coverage = geodesic_area_wgs84(aoi_geom.intersection(item_geom)) / aoi_area

    bbox = extract_item_info(item).get("bbox", [])
    bbox_coverage = coverage_ratio(aoi_bbox, bbox) if aoi_bbox is not None and len(bbox) == 4 else 0.0
    resolved_bbox: Optional[List[float]] = None
    if not item_geom.is_empty:
        minx, miny, maxx, maxy = item_geom.bounds
        resolved_bbox = [float(minx), float(miny), float(maxx), float(maxy)]

    return {
        "aoi_coverage": float(coverage),
        "aoi_bbox_coverage": float(bbox_coverage),
        "coverage_source": coverage_source,
        "resolved_geometry_geojson": resolved_geojson,
        "resolved_shapely_geometry": item_geom,
        "resolved_bbox": resolved_bbox,
    }


def compute_item_aoi_geometry_metrics(
    item: Dict[str, Any],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    aoi_geom = _shape_from_geojson(aoi_geometry)
    aoi_area = geodesic_area_wgs84(aoi_geom)
    return _compute_item_aoi_geometry_metrics_prepared(item, aoi_geom, aoi_area, aoi_bbox=aoi_bbox)


def annotate_items_for_aoi(
    items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]] = None,
) -> None:
    aoi_geom = _shape_from_geojson(aoi_geometry)
    aoi_area = geodesic_area_wgs84(aoi_geom)
    for item in items:
        metrics = _compute_item_aoi_geometry_metrics_prepared(item, aoi_geom, aoi_area, aoi_bbox=aoi_bbox)
        item["_aoi_coverage"] = metrics["aoi_coverage"]
        item["_aoi_bbox_coverage"] = metrics["aoi_bbox_coverage"]
        item["_coverage_source"] = metrics["coverage_source"]
        item["_resolved_geometry_geojson"] = metrics["resolved_geometry_geojson"]
        item["_resolved_shapely_geometry"] = metrics["resolved_shapely_geometry"]
        item["_resolved_bbox"] = metrics["resolved_bbox"]


def item_aoi_coverage_value(item: Dict[str, Any]) -> float:
    return float(item.get("_aoi_coverage", 0.0))


def item_aoi_bbox_coverage_value(item: Dict[str, Any]) -> float:
    return float(item.get("_aoi_bbox_coverage", 0.0))


def item_coverage_source(item: Dict[str, Any]) -> str:
    return str(item.get("_coverage_source", "none"))


def items_union_coverage(items: List[Dict[str, Any]], aoi_geometry: Dict[str, Any]) -> float:
    if not items:
        return 0.0
    aoi_geom = _shape_from_geojson(aoi_geometry)
    aoi_area = geodesic_area_wgs84(aoi_geom)
    if aoi_geom.is_empty or aoi_area <= 0:
        return 0.0
    geoms = []
    for item in items:
        geom = item.get("_resolved_shapely_geometry")
        if geom is not None and not getattr(geom, "is_empty", True):
            geoms.append(geom)
    if not geoms:
        return 0.0
    union_geom = unary_union(geoms)
    return geodesic_area_wgs84(aoi_geom.intersection(union_geom)) / aoi_area


def compute_item_region_coverage_metrics(
    item: Dict[str, Any],
    region_geometry: Dict[str, Any],
    region_bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute how much of a child region is covered by one item geometry."""
    return compute_item_aoi_geometry_metrics(item, region_geometry, aoi_bbox=region_bbox)


def _coverage_stats_from_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize coverage ratios recorded for one candidate region."""
    coverages = [float(rec.get("coverage", 0.0)) for rec in records]
    if not coverages:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
        }
    return {
        "count": len(coverages),
        "min": float(min(coverages)),
        "max": float(max(coverages)),
        "avg": float(sum(coverages) / len(coverages)),
    }


def collect_items_covering_region(
    items: List[Dict[str, Any]],
    region_geometry: Dict[str, Any],
    region_bbox: Optional[List[float]] = None,
    min_region_coverage: float = 0.0,
) -> Dict[str, Any]:
    """Split items into considered vs accepted membership for one child region."""
    threshold = float(min_region_coverage)
    considered_items: List[Dict[str, Any]] = []
    accepted_items: List[Dict[str, Any]] = []
    considered_records: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    for item in items:
        metrics = compute_item_region_coverage_metrics(item, region_geometry, region_bbox=region_bbox)
        coverage = float(metrics["aoi_coverage"])
        info = extract_item_info(item)
        record = {
            "id": info.get("id"),
            "datetime": info.get("datetime"),
            "coverage": coverage,
            "aoi_bbox_coverage": float(metrics.get("aoi_bbox_coverage", 0.0)),
            "coverage_source": metrics.get("coverage_source"),
        }
        if coverage > 0.0:
            considered_items.append(item)
            considered_records.append(record)
        if coverage >= threshold and coverage > 0.0:
            accepted_items.append(item)
            accepted_records.append(record)
    return {
        "coverage_threshold": threshold,
        "considered_items": considered_items,
        "accepted_items": accepted_items,
        "considered_records": considered_records,
        "accepted_records": accepted_records,
        "considered_item_ids": [str(rec["id"]) for rec in considered_records],
        "accepted_item_ids": [str(rec["id"]) for rec in accepted_records],
        "considered_item_count": len(considered_records),
        "accepted_item_count": len(accepted_records),
        "considered_scene_count": len(summarize_unique_scenes(considered_items)),
        "accepted_scene_count": len(summarize_unique_scenes(accepted_items)),
        "considered_coverage_stats": _coverage_stats_from_records(considered_records),
        "accepted_coverage_stats": _coverage_stats_from_records(accepted_records),
    }


def build_seed_intersection_region_candidates(
    *,
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    parent_aoi_geometry: Dict[str, Any],
    parent_aoi_bbox: Optional[List[float]],
    min_region_coverage: float = 0.0,
    min_region_area_ratio: float = 0.0,
    min_region_area_m2: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build child-region candidates R_X = AOI ∩ footprint(X) for one period.

    Candidate generation stays intentionally simple:
      - each period item can seed one child region
      - all items that cover that region above `min_region_coverage` join the region
      - exact-equal regions are merged by geometry key while preserving all seed ids
    """
    parent_geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(parent_aoi_geometry))
    parent_area = geodesic_area_wgs84(parent_geom)
    if parent_geom.is_empty or parent_area <= 0:
        return []

    seed_items = sorted(
        pre_items + post_items,
        key=lambda item: (extract_item_info(item).get("datetime") or "", extract_item_info(item).get("id") or ""),
    )
    by_geometry_key: Dict[str, Dict[str, Any]] = {}

    for seed_item in seed_items:
        seed_info = extract_item_info(seed_item)
        seed_geom = seed_item.get("_resolved_shapely_geometry")
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            seed_geom, _, _ = _resolve_item_geometry(seed_item)
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            continue

        region_geom = normalize_polygonal_shapely_geometry(parent_geom.intersection(seed_geom))
        if region_geom.is_empty:
            continue

        region_geojson = mapping(region_geom)
        region_bbox = canonical_bbox_from_geometry(region_geojson)
        region_area = geodesic_area_wgs84(region_geom)
        area_ratio = (region_area / parent_area) if parent_area > 0 else 0.0
        geometry_key = region_geom.wkb_hex

        candidate = by_geometry_key.get(geometry_key)
        if candidate is None:
            pre_region_membership = collect_items_covering_region(
                pre_items,
                region_geojson,
                region_bbox=region_bbox,
                min_region_coverage=min_region_coverage,
            )
            post_region_membership = collect_items_covering_region(
                post_items,
                region_geojson,
                region_bbox=region_bbox,
                min_region_coverage=min_region_coverage,
            )
            pre_covering_items = pre_region_membership["accepted_items"]
            post_covering_items = post_region_membership["accepted_items"]
            reject_reasons: List[str] = []
            if region_area <= 0:
                reject_reasons.append("EMPTY_REGION")
            if float(min_region_area_m2) > 0 and region_area < float(min_region_area_m2):
                reject_reasons.append("REGION_AREA_BELOW_MIN_M2")
            if float(min_region_area_ratio) > 0 and area_ratio < float(min_region_area_ratio):
                reject_reasons.append("REGION_AREA_RATIO_BELOW_MIN")
            candidate = {
                "candidate_region_key": geometry_key,
                "geometry": region_geojson,
                "bbox": region_bbox,
                "area_m2": region_area,
                "area_ratio_vs_parent": area_ratio,
                "seed_item_ids": [],
                "seed_item_datetimes": [],
                "membership_coverage_threshold": float(min_region_coverage),
                "pre_considered_items": pre_region_membership["considered_records"],
                "post_considered_items": post_region_membership["considered_records"],
                "pre_accepted_items": pre_region_membership["accepted_records"],
                "post_accepted_items": post_region_membership["accepted_records"],
                "pre_considered_item_ids": pre_region_membership["considered_item_ids"],
                "post_considered_item_ids": post_region_membership["considered_item_ids"],
                "pre_considered_item_count": pre_region_membership["considered_item_count"],
                "post_considered_item_count": post_region_membership["considered_item_count"],
                "pre_considered_scene_count": pre_region_membership["considered_scene_count"],
                "post_considered_scene_count": post_region_membership["considered_scene_count"],
                "pre_considered_coverage_stats": pre_region_membership["considered_coverage_stats"],
                "post_considered_coverage_stats": post_region_membership["considered_coverage_stats"],
                "pre_covering_items": pre_covering_items,
                "post_covering_items": post_covering_items,
                "pre_covering_item_ids": pre_region_membership["accepted_item_ids"],
                "post_covering_item_ids": post_region_membership["accepted_item_ids"],
                "pre_covering_item_count": pre_region_membership["accepted_item_count"],
                "post_covering_item_count": post_region_membership["accepted_item_count"],
                "pre_covering_scene_count": pre_region_membership["accepted_scene_count"],
                "post_covering_scene_count": post_region_membership["accepted_scene_count"],
                "pre_covering_coverage_stats": pre_region_membership["accepted_coverage_stats"],
                "post_covering_coverage_stats": post_region_membership["accepted_coverage_stats"],
                "parent_aoi_bbox": parent_aoi_bbox,
                "parent_aoi_area_m2": parent_area,
                "region_item_min_coverage": float(min_region_coverage),
                "reject_reasons": reject_reasons,
                "_region_shapely_geometry": region_geom,
            }
            by_geometry_key[geometry_key] = candidate

        candidate["seed_item_ids"].append(seed_info["id"])
        candidate["seed_item_datetimes"].append(seed_info["datetime"])

    candidates = list(by_geometry_key.values())
    candidates.sort(
        key=lambda cand: (
            -float(cand["area_m2"]),
            cand["bbox"][0],
            cand["bbox"][1],
            ",".join(sorted(cand["seed_item_ids"])),
        )
    )
    return candidates


def parse_required_pols(raw: Optional[str]) -> List[str]:
    """Parse --pols thanh danh sach polarization upper-case."""
    if not raw:
        return ["VV"]
    pols = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return pols or ["VV"]


def extract_item_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """Trich xuat cac field chinh tu STAC Item."""
    props = item.get("properties", {})
    return {
        "id": item.get("id", ""),
        "datetime": props.get("datetime", ""),
        "platform": props.get("platform", ""),
        "datatake_id": props.get("s1:datatake_id"),
        "orbit_state": props.get("sat:orbit_state", ""),
        "relative_orbit": props.get("sat:relative_orbit"),
        "polarizations": props.get("sar:polarizations", []),
        "instrument_mode": props.get("sar:instrument_mode", ""),
        "product_type": props.get("sar:product_type", ""),
        "orbit_source": props.get("s1:orbit_source", ""),
        "slice_number": props.get("s1:slice_number"),
        "total_slices": props.get("s1:total_slices"),
        "bbox": item.get("bbox", []),
        "assets": item.get("assets", {}),
    }


def is_raster_asset(asset_key: str, asset: Dict[str, Any]) -> bool:
    """Nhan dien asset raster co the doc bang rasterio."""
    href = str(asset.get("href", "")).lower()
    media_type = str(asset.get("type", "")).lower()
    key = asset_key.lower()

    if href.endswith(".tif") or href.endswith(".tiff"):
        return True
    if "geotiff" in media_type or "cog" in media_type:
        return True
    if key in {"vv", "vh", "hh", "hv"}:
        return True
    return False


def infer_asset_pol(asset_key: str, asset: Dict[str, Any]) -> Optional[str]:
    """Suy luan polarization tu key/href."""
    key = asset_key.lower()
    href_name = Path(urlparse(str(asset.get("href", ""))).path).name.lower()
    merged = f"{key} {href_name}"

    if re.search(r"(^|[^a-z0-9])vv([^a-z0-9]|$)", merged):
        return "VV"
    if re.search(r"(^|[^a-z0-9])vh([^a-z0-9]|$)", merged):
        return "VH"
    if re.search(r"(^|[^a-z0-9])hh([^a-z0-9]|$)", merged):
        return "HH"
    if re.search(r"(^|[^a-z0-9])hv([^a-z0-9]|$)", merged):
        return "HV"
    return None


def select_asset_href(item: Dict[str, Any], pol: str) -> Optional[Tuple[str, str]]:
    """
    Chon href theo polarization.
    Uu tien:
      1) key = vv/vh...
      2) suy tu key/href
      3) fallback: asset raster dau tien neu item chi co 1 asset raster
    """
    pol = pol.upper()
    assets = item.get("assets", {})
    if not assets:
        return None

    for key, asset in assets.items():
        if key.upper() == pol and is_raster_asset(key, asset):
            href = str(asset.get("href", ""))
            if href:
                return key, href

    for key, asset in assets.items():
        if not is_raster_asset(key, asset):
            continue
        if infer_asset_pol(key, asset) == pol:
            href = str(asset.get("href", ""))
            if href:
                return key, href

    raster_assets = [(k, a) for k, a in assets.items() if is_raster_asset(k, a)]
    if len(raster_assets) == 1:
        k, a = raster_assets[0]
        href = str(a.get("href", ""))
        if href:
            return k, href
    return None


class STACClient:
    """Lightweight STAC API client su dung requests."""

    def __init__(self, api_url: str):
        self.api_url = str(api_url or "").strip().rstrip("/")
        if not self.api_url:
            raise ValueError("STAC API URL is required. Set STAC_API_URL or pass --stac-url.")
        self.session = requests.Session()
        emit_runtime_log("query_stac_download", logging.INFO, "Initialized STAC client", stac_url=self.api_url)

    def search_items(
        self,
        collection: str,
        bbox: Optional[List[float]] = None,
        intersects: Optional[Dict[str, Any]] = None,
        datetime_range: Optional[str] = None,
        limit: int = 200,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query STAC /search, fallback qua GET /collections/{id}/items."""
        datetime_range = normalize_datetime_range(datetime_range)
        search_url = f"{self.api_url}/search"
        base_payload: Dict[str, Any] = {
            "collections": [collection],
            "limit": max(1, int(limit)),
        }
        if datetime_range:
            base_payload["datetime"] = datetime_range
        if query:
            base_payload["query"] = query

        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Starting STAC search",
            collection=collection,
            datetime=datetime_range,
            limit=max(1, int(limit)),
            has_bbox=bool(bbox),
            has_intersects=bool(intersects),
            has_query=bool(query),
        )

        def _try_post(payload: Dict[str, Any], tag: str) -> Optional[List[Dict[str, Any]]]:
            request_started = datetime.now()
            try:
                resp = self.session.post(search_url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
                emit_runtime_log(
                    "query_stac_download",
                    logging.INFO,
                    "STAC POST /search succeeded",
                    search_mode=tag,
                    item_count=len(features),
                    duration_ms=duration_ms,
                    http_status=resp.status_code,
                )
                return features
            except Exception as e:
                status_code = None
                err_text = ""
                try:
                    if "resp" in locals():
                        status_code = resp.status_code
                        err_text = safe_text_snippet(resp.text, limit=300)
                except Exception:
                    err_text = ""
                emit_runtime_log(
                    "query_stac_download",
                    logging.WARNING,
                    "STAC POST /search failed",
                    search_mode=tag,
                    http_status=status_code,
                    error=e,
                    response_body=err_text or None,
                )
                return None

        # Luu y: nhieu STAC backend khong chap nhan gui dong thoi bbox + intersects.
        # Vi vay chi gui 1 trong 2 moi lan thu.
        if intersects:
            payload_intersects = dict(base_payload)
            payload_intersects["intersects"] = intersects
            out = _try_post(payload_intersects, "intersects")
            if out is not None:
                return out

            if bbox:
                payload_bbox = dict(base_payload)
                payload_bbox["bbox"] = bbox
                out = _try_post(payload_bbox, "bbox-fallback")
                if out is not None:
                    return out
        else:
            payload_bbox = dict(base_payload)
            if bbox:
                payload_bbox["bbox"] = bbox
            out = _try_post(payload_bbox, "bbox")
            if out is not None:
                return out

        emit_runtime_log(
            "query_stac_download",
            logging.WARNING,
            "Falling back to STAC GET items after POST /search failed",
            collection=collection,
            datetime=datetime_range,
            has_bbox=bool(bbox),
            has_query=bool(query),
        )

        items_url = f"{self.api_url}/collections/{collection}/items"
        params: Dict[str, Any] = {"limit": max(1, int(limit))}
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        if datetime_range:
            params["datetime"] = datetime_range
        if query:
            # Fallback GET khong ho tro query day du tren moi STAC implementation.
            # Van gui de tan dung neu server co support.
            params["query"] = json.dumps(query)

        try:
            request_started = datetime.now()
            resp = self.session.get(items_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "STAC GET items succeeded",
                item_count=len(features),
                duration_ms=duration_ms,
                http_status=resp.status_code,
            )
            return features
        except Exception as e:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "Unable to query STAC API",
                error=e,
                collection=collection,
                datetime=datetime_range,
                has_bbox=bool(bbox),
            )
            return []

    def get_item(self, collection: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Lay item theo ID, uu tien endpoint chuan /collections/{id}/items/{item_id}."""
        direct_url = f"{self.api_url}/collections/{collection}/items/{item_id}"
        try:
            resp = self.session.get(direct_url, timeout=30)
            if resp.status_code == 200:
                emit_runtime_log(
                    "query_stac_download",
                    logging.DEBUG,
                    "Fetched STAC item by direct endpoint",
                    collection=collection,
                    item_id=item_id,
                    http_status=resp.status_code,
                )
                return resp.json()
        except Exception:
            pass

        # Fallback 1: STAC ids extension
        try:
            payload = {"collections": [collection], "ids": [item_id], "limit": 1}
            resp = self.session.post(f"{self.api_url}/search", json=payload, timeout=30)
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    emit_runtime_log(
                        "query_stac_download",
                        logging.DEBUG,
                        "Fetched STAC item by ids extension fallback",
                        collection=collection,
                        item_id=item_id,
                    )
                    return features[0]
        except Exception:
            pass

        # Fallback 2: query id eq
        emit_runtime_log(
            "query_stac_download",
            logging.DEBUG,
            "Falling back to STAC id query",
            collection=collection,
            item_id=item_id,
        )
        features = self.search_items(
            collection=collection,
            limit=1,
            query={"id": {"eq": item_id}},
        )
        return features[0] if features else None


def apply_hard_filters(
    items: List[Dict[str, Any]],
    orbit_state: Optional[str],
    relative_orbit: Optional[int],
    required_pols: List[str],
    instrument_mode: str = "IW",
    product_type: str = "GRD",
) -> List[Dict[str, Any]]:
    """Loc cac dieu kien bat buoc cho input model."""
    result: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = defaultdict(int)
    rejection_item_ids: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        info = extract_item_info(item)
        item_id = str(info.get("id") or "unknown")

        if instrument_mode and info["instrument_mode"].upper() != instrument_mode.upper():
            rejection_counts["instrument_mode_mismatch"] += 1
            rejection_item_ids["instrument_mode_mismatch"].append(item_id)
            continue
        if product_type and info["product_type"].upper() != product_type.upper():
            rejection_counts["product_type_mismatch"] += 1
            rejection_item_ids["product_type_mismatch"].append(item_id)
            continue
        if orbit_state and str(info["orbit_state"]).lower() != orbit_state.lower():
            rejection_counts["orbit_state_mismatch"] += 1
            rejection_item_ids["orbit_state_mismatch"].append(item_id)
            continue
        if relative_orbit is not None and info["relative_orbit"] != relative_orbit:
            rejection_counts["relative_orbit_mismatch"] += 1
            rejection_item_ids["relative_orbit_mismatch"].append(item_id)
            continue

        item_pols = [str(p).upper() for p in info["polarizations"]]
        if not all(pol in item_pols for pol in required_pols):
            rejection_counts["missing_polarization"] += 1
            rejection_item_ids["missing_polarization"].append(item_id)
            continue

        # Item phai co href cho tat ca polarization can dung
        if any(select_asset_href(item, pol) is None for pol in required_pols):
            rejection_counts["missing_asset_href"] += 1
            rejection_item_ids["missing_asset_href"].append(item_id)
            continue

        result.append(item)

    emit_runtime_log(
        "query_stac_download",
        logging.INFO,
        "Applied hard filters to STAC items",
        input_items=len(items),
        kept_items=len(result),
        rejected_items=max(0, len(items) - len(result)),
        rejection_counts=dict(sorted(rejection_counts.items())),
        orbit_state=orbit_state,
        relative_orbit=relative_orbit,
        required_pols=[str(pol).upper() for pol in required_pols],
        instrument_mode=instrument_mode,
        product_type=product_type,
    )
    if logger.isEnabledFor(logging.DEBUG):
        for reason_key, item_ids in sorted(rejection_item_ids.items()):
            emit_runtime_log(
                "query_stac_download",
                logging.DEBUG,
                "Hard-filter rejection bucket",
                reason=reason_key,
                item_ids=sorted(item_ids),
            )
    return result


def find_pairs(
    items: List[Dict[str, Any]],
    aoi_bbox: List[float],
    aoi_geometry: Dict[str, Any],
    min_overlap: float,
    min_aoi_coverage: float,
    max_delta_days: int,
    min_delta_hours: float,
    strict_slice: bool,
    same_orbit_direction: bool,
) -> List[Dict[str, Any]]:
    """
    Tim cap T1/T2 hop le.
    Dieu kien:
      - AOI geometry coverage cua T1 va T2 > min_aoi_coverage
      - min_delta_hours <= Delta t <= max_delta_days
      - Datatake duoc ghi lai de chan doan, khong dung lam hard filter
      - bbox_overlap chi duoc luu lai de report/ranking, khong la hard filter
    """
    pairs: List[Dict[str, Any]] = []
    min_delta_sec = max(0.0, float(min_delta_hours) * 3600.0)
    max_delta_sec = max(0.0, float(max_delta_days) * 86400.0)
    sorted_items = sorted(items, key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]))
    annotate_items_for_aoi(sorted_items, aoi_geometry, aoi_bbox=aoi_bbox)
    print(f"[PAIR] Candidate items sau filter: {len(sorted_items)}")

    for i in range(len(sorted_items) - 1):
        item_t1 = sorted_items[i]
        info_t1 = extract_item_info(item_t1)
        dt1 = parse_datetime_utc(info_t1["datetime"])

        for j in range(i + 1, len(sorted_items)):
            item_t2 = sorted_items[j]
            info_t2 = extract_item_info(item_t2)
            dt2 = parse_datetime_utc(info_t2["datetime"])

            if same_orbit_direction:
                orbit_t1 = str(info_t1["orbit_state"] or "").lower()
                orbit_t2 = str(info_t2["orbit_state"] or "").lower()
                if not orbit_t1 or not orbit_t2 or orbit_t1 != orbit_t2:
                    continue

            delta_sec = (dt2 - dt1).total_seconds()
            if delta_sec < min_delta_sec:
                continue
            if delta_sec > max_delta_sec:
                break

            bbox1 = info_t1["bbox"]
            bbox2 = info_t2["bbox"]
            if len(bbox1) != 4 or len(bbox2) != 4:
                continue

            overlap = bbox_overlap_ratio(bbox1, bbox2)

            cover_t1 = item_aoi_coverage_value(item_t1)
            cover_t2 = item_aoi_coverage_value(item_t2)
            if cover_t1 <= min_aoi_coverage or cover_t2 <= min_aoi_coverage:
                continue
            bbox_cover_t1 = item_aoi_bbox_coverage_value(item_t1)
            bbox_cover_t2 = item_aoi_bbox_coverage_value(item_t2)
            pair_union_coverage = items_union_coverage([item_t1, item_t2], aoi_geometry)

            pairs.append(
                {
                    "group_key": None,
                    "t1_item": item_t1,
                    "t2_item": item_t2,
                    "t1_id": info_t1["id"],
                    "t2_id": info_t2["id"],
                    "t1_datetime": info_t1["datetime"],
                    "t2_datetime": info_t2["datetime"],
                    "t1_datatake_id": info_t1["datatake_id"],
                    "t2_datatake_id": info_t2["datatake_id"],
                    "delta_seconds": delta_sec,
                    "delta_hours": delta_sec / 3600.0,
                    "delta_days": delta_sec / 86400.0,
                    "bbox_overlap": overlap,
                    "aoi_union_coverage_pair": pair_union_coverage,
                    "aoi_bbox_coverage_t1": bbox_cover_t1,
                    "aoi_bbox_coverage_t2": bbox_cover_t2,
                    "aoi_coverage_t1": cover_t1,
                    "aoi_coverage_t2": cover_t2,
                    "coverage_source_t1": item_coverage_source(item_t1),
                    "coverage_source_t2": item_coverage_source(item_t2),
                    "t1_orbit_state": info_t1["orbit_state"],
                    "t2_orbit_state": info_t2["orbit_state"],
                    "orbit_state": info_t1["orbit_state"],
                    "relative_orbit": info_t1["relative_orbit"],
                    "slice_number": info_t1["slice_number"],
                    "t1_orbit_source": info_t1["orbit_source"],
                    "t2_orbit_source": info_t2["orbit_source"],
                }
            )
    print(f"[PAIR] Tong cap hop le: {len(pairs)}")
    return pairs


def pair_rank_key(pair: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, str, str]:
    """Ranking cap exact-pair theo uu tien recency.

    Uu tien:
      1) latest input moi nhat, tuong ung T2 trong exact pair
      2) T1 moi nhat
      3) delta nho hon
      4) min AOI geometry coverage lon hon
      5) pair union coverage lon hon
      6) bbox overlap lon hon
      7) ID on dinh
    """
    return (
        _neg_timestamp(pair["t2_datetime"]),
        _neg_timestamp(pair["t1_datetime"]),
        pair["delta_seconds"],
        -min(pair.get("aoi_coverage_t1", 0.0), pair.get("aoi_coverage_t2", 0.0)),
        -pair.get("aoi_union_coverage_pair", 0.0),
        -pair["bbox_overlap"],
        pair["t1_id"],
        pair["t2_id"],
    )


def build_pair_identifier(pair: Dict[str, Any]) -> str:
    """Sinh identifier de naming file T1/T2 khop infer_production.py."""
    rel_orbit = pair.get("relative_orbit")
    slice_num = pair.get("slice_number")
    t1_dt = parse_datetime_utc(pair["t1_datetime"]).strftime("%Y%m%dT%H%M%S")
    t2_dt = parse_datetime_utc(pair["t2_datetime"]).strftime("%Y%m%dT%H%M%S")
    raw = f"orb{rel_orbit if rel_orbit is not None else 'x'}_sl{slice_num if slice_num is not None else 'x'}_{t1_dt}_{t2_dt}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw)


def exact_pair_later_first(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Project an internal chronological pair into canonical model order: T1=later, T2=earlier."""
    return {
        "t1_id": pair["t2_id"],
        "t2_id": pair["t1_id"],
        "t1_datetime": pair["t2_datetime"],
        "t2_datetime": pair["t1_datetime"],
        "later_id": pair["t2_id"],
        "earlier_id": pair["t1_id"],
        "later_datetime": pair["t2_datetime"],
        "earlier_datetime": pair["t1_datetime"],
        "t1_datatake_id": pair.get("t2_datatake_id"),
        "t2_datatake_id": pair.get("t1_datatake_id"),
        "t1_orbit_state": pair.get("t2_orbit_state"),
        "t2_orbit_state": pair.get("t1_orbit_state"),
        "relative_orbit": pair.get("relative_orbit"),
        "slice_number": pair.get("slice_number"),
        "delta_seconds": pair["delta_seconds"],
        "delta_hours": pair["delta_hours"],
        "delta_days": pair["delta_days"],
        "bbox_overlap": pair["bbox_overlap"],
        "aoi_union_coverage_pair": pair.get("aoi_union_coverage_pair", 0.0),
        "aoi_bbox_coverage_t1": pair["aoi_bbox_coverage_t2"],
        "aoi_bbox_coverage_t2": pair["aoi_bbox_coverage_t1"],
        "aoi_coverage_t1": pair["aoi_coverage_t2"],
        "aoi_coverage_t2": pair["aoi_coverage_t1"],
        "coverage_source_t1": pair.get("coverage_source_t2", "none"),
        "coverage_source_t2": pair.get("coverage_source_t1", "none"),
        "latest_input_datetime": pair["t2_datetime"],
    }


def support_pair_later_first(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Project a support pair candidate into canonical model order: T1=later, T2=earlier."""
    return {
        "t1_id": candidate["support_t2_id"],
        "t2_id": candidate["support_t1_id"],
        "t1_datetime": candidate["support_t2_datetime"],
        "t2_datetime": candidate["support_t1_datetime"],
        "later_id": candidate["support_t2_id"],
        "earlier_id": candidate["support_t1_id"],
        "later_datetime": candidate["support_t2_datetime"],
        "earlier_datetime": candidate["support_t1_datetime"],
        "t1_orbit_state": candidate.get("support_t2_orbit_state"),
        "t2_orbit_state": candidate.get("support_t1_orbit_state"),
    }


def build_manifest_for_pair(pair: Dict[str, Any], required_pols: List[str]) -> Optional[Dict[str, Any]]:
    """Build manifest bao gom href theo tung polarization cho T1/T2."""
    # Internal pair ordering is chronological earlier -> later.
    # Public manifest ordering is canonical model order: T1=later, T2=earlier.
    t1_item = pair["t2_item"]
    t2_item = pair["t1_item"]
    canonical = exact_pair_later_first(pair)

    assets: Dict[str, Dict[str, Any]] = {}
    for pol in required_pols:
        t1_pick = select_asset_href(t1_item, pol)
        t2_pick = select_asset_href(t2_item, pol)
        if not t1_pick or not t2_pick:
            return None
        t1_key, t1_href = t1_pick
        t2_key, t2_href = t2_pick
        assets[pol] = {
            "t1_asset_key": t1_key,
            "t1_href": t1_href,
            "t2_asset_key": t2_key,
            "t2_href": t2_href,
        }

    pair_id = build_pair_identifier(pair)
    return {
        "pair_id": pair_id,
        "t1_id": canonical["t1_id"],
        "t2_id": canonical["t2_id"],
        "t1_datetime": canonical["t1_datetime"],
        "t2_datetime": canonical["t2_datetime"],
        "later_id": canonical["later_id"],
        "earlier_id": canonical["earlier_id"],
        "later_datetime": canonical["later_datetime"],
        "earlier_datetime": canonical["earlier_datetime"],
        "pair_semantics": {
            "t1_role": "later/posterior exact scene",
            "t2_role": "earlier/prior exact scene",
            "matches_training_semantics": True,
            "recipe_type": "exact_single_scene",
        },
        "latest_input_datetime": canonical["latest_input_datetime"],
        "t1_datatake_id": canonical.get("t1_datatake_id"),
        "t2_datatake_id": canonical.get("t2_datatake_id"),
        "delta_hours": pair["delta_hours"],
        "delta_days": pair["delta_days"],
        "bbox_overlap": pair["bbox_overlap"],
        "aoi_union_coverage_pair": canonical.get("aoi_union_coverage_pair", 0.0),
        "aoi_bbox_coverage_t1": canonical["aoi_bbox_coverage_t1"],
        "aoi_bbox_coverage_t2": canonical["aoi_bbox_coverage_t2"],
        "aoi_coverage_t1": canonical["aoi_coverage_t1"],
        "aoi_coverage_t2": canonical["aoi_coverage_t2"],
        "coverage_source_t1": canonical.get("coverage_source_t1", "none"),
        "coverage_source_t2": canonical.get("coverage_source_t2", "none"),
        "t1_orbit_state": canonical.get("t1_orbit_state"),
        "t2_orbit_state": canonical.get("t2_orbit_state"),
        "orbit_state": pair["orbit_state"],
        "relative_orbit": pair["relative_orbit"],
        "slice_number": pair["slice_number"],
        "assets": assets,
    }


class S3Downloader:
    """Download asset tu S3 bang boto3."""

    def __init__(self):
        if boto3 is None:
            raise RuntimeError("Chua cai boto3. Vui long cai 'pip install boto3' de dung tinh nang download.")
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")
        client_kwargs: Dict[str, Any] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        s3_endpoint = os.getenv("S3_ENDPOINT")
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint
        credential_source = detect_s3_credential_source()
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Initialized S3 downloader",
            s3_endpoint=s3_endpoint,
            credential_source=credential_source,
            s3_access_key_present=bool(access_key),
            s3_secret_key_present=bool(secret_key),
            explicit_env_present=bool(access_key and secret_key),
        )
        if credential_source == "none":
            emit_runtime_log(
                "query_stac_download",
                logging.WARNING,
                "S3 credentials are not explicitly configured",
                credential_source=credential_source,
                note="The pipeline will continue and only fail if S3 access is required later.",
            )
        self.client = boto3.client("s3", **client_kwargs)

    @staticmethod
    def parse_href_to_bucket_key(href: str) -> Optional[Tuple[str, str]]:
        """Parse href (s3:// hoac http(s)) -> (bucket, key)."""
        parsed = urlparse(href)
        if parsed.scheme == "s3":
            return parsed.netloc, parsed.path.lstrip("/")

        if parsed.scheme in ("http", "https"):
            path = parsed.path.lstrip("/")
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

            # Virtual-host style: https://bucket.endpoint/key
            host = parsed.netloc.split(":")[0]
            host_parts = host.split(".")
            if len(host_parts) >= 2 and host_parts[0] not in {"s3", "minio", "storage"} and path:
                return host_parts[0], path
            return None
        return None

    @staticmethod
    def _key_tail(key: Optional[str], parts: int = 3) -> Optional[str]:
        if not key:
            return None
        pieces = [piece for piece in str(key).split("/") if piece]
        if not pieces:
            return None
        return "/".join(pieces[-max(1, int(parts)):])

    def download_from_href(self, href: str, local_path: str) -> bool:
        """Tai file tu href ve local_path."""
        parsed = self.parse_href_to_bucket_key(href)
        if not parsed:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "Failed to parse asset href for download",
                href=href,
                local_path=local_path,
            )
            return False
        bucket, key = parsed

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Starting S3 asset download",
            href=href,
            bucket=bucket,
            key=key,
            local_path=local_path,
        )

        try:
            head = self.client.head_object(Bucket=bucket, Key=key)
            size_mb = head["ContentLength"] / (1024 * 1024)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "Resolved S3 object metadata",
                bucket=bucket,
                key=key,
                size_mb=f"{size_mb:.1f}",
            )

            local_file = Path(local_path)
            if local_file.exists() and local_file.stat().st_size == int(head["ContentLength"]):
                emit_runtime_log(
                    "query_stac_download",
                    logging.INFO,
                    "Skipping download because local file already matches remote size",
                    bucket=bucket,
                    key=key,
                    local_path=local_path,
                )
                return True

            downloaded = [0]
            total_size = max(1, int(head["ContentLength"]))
            progress_state = {"last_print": -5}

            def progress(bytes_amount: int) -> None:
                downloaded[0] += bytes_amount
                pct = downloaded[0] / total_size * 100
                bucket_pct = int(pct // 5) * 5
                if bucket_pct > progress_state["last_print"]:
                    progress_state["last_print"] = bucket_pct
                    emit_runtime_log(
                        "query_stac_download",
                        logging.DEBUG,
                        "S3 download progress",
                        bucket=bucket,
                        key=key,
                        progress_pct=f"{pct:.1f}",
                    )

            self.client.download_file(bucket, key, local_path, Callback=progress)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "Completed S3 asset download",
                bucket=bucket,
                key=key,
                local_path=local_path,
            )
            return True
        except Exception as e:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "S3 asset download failed",
                href=href,
                bucket=bucket,
                key=key,
                local_path=local_path,
                error=e,
            )
            return False

    def download_aoi_subset_from_href(
        self,
        href: str,
        local_path: str,
        aoi_geometry_wgs84: Dict[str, Any],
    ) -> bool:
        """
        Doc remote raster va chi ghi ra bbox giao nhau giua AOI va item bounds.
        Khong tai full item ve local.
        """
        raster_path = href_to_rasterio_path(href)
        env_kwargs = build_rasterio_env_kwargs()

        try:
            access_key = os.getenv("S3_ACCESS_KEY")
            secret_key = os.getenv("S3_SECRET_KEY")
            aws_session = None
            if boto3 is not None and access_key and secret_key:
                b3_session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                aws_session = AWSSession(b3_session)
            elif boto3 is None and access_key and secret_key:
                env_kwargs["AWS_ACCESS_KEY_ID"] = access_key
                env_kwargs["AWS_SECRET_ACCESS_KEY"] = secret_key

            env = rasterio.Env(session=aws_session, **env_kwargs) if aws_session is not None else rasterio.Env(**env_kwargs)
            with env:
                with rasterio.open(raster_path, "r") as src:
                    if src.crs is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because raster has no CRS",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    clip_geometry_wgs84 = normalize_polygonal_geojson_geometry(aoi_geometry_wgs84)
                    if clip_geometry_wgs84 is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because clip geometry has no polygonal area",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    aoi_in_src = transform_geom("EPSG:4326", src.crs, clip_geometry_wgs84, antimeridian_cutting=True, precision=15)
                    aoi_in_src = normalize_polygonal_geojson_geometry(aoi_in_src)
                    if aoi_in_src is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because transformed clip geometry has no polygonal area",
                            href=href,
                            local_path=local_path,
                            raster_crs=src.crs,
                        )
                        return False
                    aoi_bounds_src = list(geometry_bounds(aoi_in_src))
                    src_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    clip_bbox = bbox_intersection_bounds(aoi_bounds_src, src_bounds)
                    if clip_bbox is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.WARNING,
                            "Subset skipped because AOI does not intersect raster bounds",
                            href=href,
                            local_path=local_path,
                            raster_crs=src.crs,
                        )
                        return False

                    # Snap outward theo pixel grid:
                    # - floor offsets (col/row start)
                    # - ceil max extents (col/row end)
                    # de dam bao subset bao kin AOI bounds, tranh bi "hut" 1 phan pixel o goc/canh.
                    raw_win = from_bounds(*clip_bbox, transform=src.transform)
                    col_off = int(math.floor(raw_win.col_off))
                    row_off = int(math.floor(raw_win.row_off))
                    col_max = int(math.ceil(raw_win.col_off + raw_win.width))
                    row_max = int(math.ceil(raw_win.row_off + raw_win.height))

                    col_off = max(0, col_off)
                    row_off = max(0, row_off)
                    col_max = min(src.width, col_max)
                    row_max = min(src.height, row_max)

                    win = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=max(0, col_max - col_off),
                        height=max(0, row_max - row_off),
                    )
                    if win.width < 1 or win.height < 1:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because clipped raster window is empty",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    data = src.read(window=win)
                    out_transform = src.window_transform(win)

                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=int(win.width),
                        height=int(win.height),
                        transform=out_transform,
                    )
                    # Tranh loi tile size khi cua so subset nho hon block size goc.
                    profile.pop("tiled", None)
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)

                    out_file = Path(local_path)
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_file, "w", **profile) as dst:
                        dst.write(data)
            parsed = self.parse_href_to_bucket_key(href)
            emit_runtime_log(
                "query_stac_download",
                logging.DEBUG,
                "Completed AOI subset download",
                bucket=(parsed[0] if parsed else None),
                key_tail=(self._key_tail(parsed[1]) if parsed else None),
                local_name=Path(local_path).name,
                raster_name=Path(raster_path).name,
            )
            return True
        except Exception as e:
            parsed = self.parse_href_to_bucket_key(href)
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "AOI subset download failed",
                href=href,
                bucket=(parsed[0] if parsed else None),
                key=(parsed[1] if parsed else None),
                local_path=local_path,
                error=e,
            )
            return False

    def download_item_assets(
        self,
        assets: Dict[str, Any],
        out_dir: str,
        item_id: str,
        asset_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Tai cac asset key can thiet cua 1 item."""
        downloaded_paths: List[str] = []
        keys = asset_keys or list(assets.keys())
        for key in keys:
            if key not in assets:
                emit_runtime_log(
                    "query_stac_download",
                    logging.WARNING,
                    "Requested asset key is missing from STAC item",
                    asset_key=key,
                    item_id=item_id,
                )
                continue
            asset = assets[key]
            href = str(asset.get("href", ""))
            if not href:
                continue

            ext = Path(urlparse(href).path).suffix or ".tif"
            local_path = str(Path(out_dir) / f"{item_id}_{key}{ext}")
            if self.download_from_href(href, local_path):
                downloaded_paths.append(local_path)
        return downloaded_paths


def href_to_rasterio_path(href: str) -> str:
    """Chuyen href thanh duong dan GDAL VSI neu can."""
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}/{parsed.path.lstrip('/')}"
    if parsed.scheme in ("http", "https"):
        return f"/vsicurl/{href}"
    return href


def build_rasterio_env_kwargs() -> Dict[str, Any]:
    """Build cac env cho rasterio de doc duoc MinIO/S3."""
    env: Dict[str, Any] = {}
    endpoint = os.getenv("S3_ENDPOINT")

    if endpoint:
        p = urlparse(endpoint)
        if p.scheme:
            env["AWS_HTTPS"] = "YES" if p.scheme == "https" else "NO"
            env["AWS_S3_ENDPOINT"] = p.netloc
        else:
            env["AWS_S3_ENDPOINT"] = endpoint
        env["AWS_VIRTUAL_HOSTING"] = "FALSE"

    return env


def probe_rasterio_href(label: str, href: str) -> bool:
    """Thu mo href bang rasterio va in metadata chinh."""
    parsed = urlparse(href)
    if parsed.scheme == "s3" and boto3 is None:
        print(f"[RASTERIO] {label} -> Loi: can cai boto3 de doc authenticated s3:// href")
        return False

    raster_path = href_to_rasterio_path(href)
    env_kwargs = build_rasterio_env_kwargs()
    try:
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")
        aws_session = None
        if boto3 is not None and access_key and secret_key:
            b3_session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            aws_session = AWSSession(b3_session)
        elif boto3 is None and access_key and secret_key:
            # Fallback cho moi truong chua co boto3 package.
            env_kwargs["AWS_ACCESS_KEY_ID"] = access_key
            env_kwargs["AWS_SECRET_ACCESS_KEY"] = secret_key

        env = rasterio.Env(session=aws_session, **env_kwargs) if aws_session is not None else rasterio.Env(**env_kwargs)
        with env:
            with rasterio.open(raster_path, "r") as src:
                print(f"[RASTERIO] {label}")
                print(f"  path: {raster_path}")
                print(f"  size: {src.width}x{src.height}, bands={src.count}, dtype={src.dtypes[0]}")
                print(f"  crs: {src.crs}")
                print(f"  transform: {src.transform}")
        return True
    except Exception as e:
        print(f"[RASTERIO] {label} -> Loi: {e}")
        return False


def resolve_spatial_filter(args: argparse.Namespace) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]]]:
    """
    Uu tien --geojson.
    Neu co geojson, su dung bbox + intersects cho query.
    """
    if getattr(args, "geojson", None):
        bbox, geometry = load_geojson_aoi(args.geojson)
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Resolved AOI spatial filter from GeoJSON",
            geojson_path=args.geojson,
            bbox=bbox,
        )
        return bbox, geometry
    return args.bbox, None


def collect_items_with_filters(
    client: STACClient,
    args: argparse.Namespace,
    required_pols: List[str],
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """Chay query + hard filter, tra ve items, AOI bbox canonical, va AOI geometry canonical."""
    bbox, intersects = resolve_spatial_filter(args)
    if bbox is None:
        raise ValueError("Can --bbox hoac --geojson de xac dinh AOI.")
    aoi_geometry = intersects or bbox_to_geometry(bbox)

    temporal_items_count: Optional[int] = None
    if intersects is not None:
        temporal_only_items = client.search_items(
            collection=args.collection,
            datetime_range=args.datetime,
            limit=args.limit,
        )
        temporal_items_count = len(temporal_only_items)
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Temporal-only STAC query completed",
            collection=args.collection,
            datetime=args.datetime,
            temporal_items=temporal_items_count,
        )

    items = client.search_items(
        collection=args.collection,
        bbox=bbox,
        intersects=intersects,
        datetime_range=args.datetime,
        limit=args.limit,
    )
    emit_runtime_log(
        "query_stac_download",
        logging.INFO,
        "Spatial STAC query completed",
        collection=args.collection,
        datetime=args.datetime,
        spatial_items=len(items),
        temporal_items=temporal_items_count,
        bbox=bbox,
        used_intersects=bool(intersects),
    )
    if not items:
        return [], bbox, aoi_geometry

    filtered = apply_hard_filters(
        items=items,
        orbit_state=args.orbit,
        relative_orbit=args.rel_orbit,
        required_pols=required_pols,
    )
    annotate_items_for_aoi(filtered, aoi_geometry, aoi_bbox=bbox)
    emit_runtime_log(
        "query_stac_download",
        logging.INFO,
        "STAC query pipeline counts",
        temporal_items=temporal_items_count,
        spatial_items=len(items),
        hard_filtered_items=len(filtered),
    )
    return filtered, bbox, aoi_geometry


def print_pairs_table(pairs: List[Dict[str, Any]], top_k: int = 10) -> None:
    """In bang tom tat pair candidates."""
    if not pairs:
        print("[PAIR] Khong co cap hop le.")
        return

    print("=" * 184)
    print(
        f"{'#':>3} {'latest_in':<22} {'dt(h)':>8} {'overlap':>8} {'AOImin':>8} "
        f"{'dtake':>8} {'orbit':>11} {'T1 ID':<48} {'T2 ID':<48}"
    )
    print("=" * 184)
    for idx, p in enumerate(pairs[:top_k], 1):
        canonical = exact_pair_later_first(p)
        min_cov = min(canonical["aoi_coverage_t1"], canonical["aoi_coverage_t2"])
        datatake_label = "same" if canonical.get("t1_datatake_id") == canonical.get("t2_datatake_id") else "diff"
        orbit_label = f"{str(canonical.get('t1_orbit_state') or '?')[:4]}/{str(canonical.get('t2_orbit_state') or '?')[:4]}"
        print(
            f"{idx:>3} {canonical['latest_input_datetime'][:22]:<22} {p['delta_hours']:>8.2f} {p['bbox_overlap']:>8.1%} {min_cov:>8.1%} "
            f"{datatake_label:>8} {orbit_label:>11} {canonical['t1_id'][:48]:<48} {canonical['t2_id'][:48]:<48}"
        )


def search_pairs_sorted(
    items: List[Dict[str, Any]],
    aoi_bbox: List[float],
    aoi_geometry: Dict[str, Any],
    min_overlap: float,
    min_aoi_coverage: float,
    max_delta_days: int,
    min_delta_hours: float,
    strict_slice: bool,
    same_orbit_direction: bool,
) -> List[Dict[str, Any]]:
    """Helper: tim pairs va sort theo ranking key."""
    pairs = find_pairs(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        min_overlap=min_overlap,
        min_aoi_coverage=min_aoi_coverage,
        max_delta_days=max_delta_days,
        min_delta_hours=min_delta_hours,
        strict_slice=strict_slice,
        same_orbit_direction=same_orbit_direction,
    )
    return sorted(pairs, key=pair_rank_key)


def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    """Diem giua giua hai moc thoi gian UTC."""
    return dt1 + (dt2 - dt1) / 2


def parse_finite_datetime_range(datetime_range: Optional[str]) -> Tuple[datetime, datetime]:
    """Parse finite RFC3339 range and return UTC bounds."""
    normalized = normalize_datetime_range(datetime_range)
    if not normalized or "/" not in normalized:
        raise ValueError("Representative calendar selection requires a finite datetime range `start/end`.")
    start_raw, end_raw = [part.strip() for part in normalized.split("/", 1)]
    if start_raw == ".." or end_raw == "..":
        raise ValueError("Representative calendar selection does not support open datetime ranges (`..`).")
    start_dt = parse_datetime_utc(start_raw)
    end_dt = parse_datetime_utc(end_raw)
    if start_dt == datetime.min.replace(tzinfo=timezone.utc) or end_dt == datetime.min.replace(tzinfo=timezone.utc):
        raise ValueError(f"Invalid datetime range: {datetime_range}")
    if start_dt >= end_dt:
        raise ValueError(f"Datetime range must satisfy start < end, got: {datetime_range}")
    return start_dt, end_dt


def floor_month_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def add_month_utc(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)


def expand_month_periods(datetime_range: Optional[str], allow_partial_periods: bool = False) -> List[Dict[str, Any]]:
    """Expand a finite datetime range into calendar-month periods."""
    start_dt, end_dt = parse_finite_datetime_range(datetime_range)
    inclusive_full_end = end_dt
    if end_dt.hour == 23 and end_dt.minute == 59 and end_dt.second == 59 and end_dt.microsecond == 0:
        inclusive_full_end = end_dt + timedelta(seconds=1)
    periods: List[Dict[str, Any]] = []
    cursor = floor_month_utc(start_dt)
    while cursor < inclusive_full_end:
        next_month = add_month_utc(cursor)
        if allow_partial_periods:
            period_start = max(cursor, start_dt)
            period_end = min(next_month, inclusive_full_end)
            is_full = period_start == cursor and period_end == next_month
            if period_start < period_end:
                period_anchor = midpoint_datetime(period_start, period_end)
                periods.append(
                    {
                        "period_id": period_start.strftime("%Y-%m"),
                        "period_mode": "month",
                        "period_start": period_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": period_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "is_full_period": is_full,
                    }
                )
        else:
            if start_dt <= cursor and next_month <= inclusive_full_end:
                period_anchor = midpoint_datetime(cursor, next_month)
                periods.append(
                    {
                        "period_id": cursor.strftime("%Y-%m"),
                        "period_mode": "month",
                        "period_start": cursor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": next_month.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "is_full_period": True,
                    }
                )
        cursor = next_month
    return periods


def item_scene_key(item: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Key de khong dem trung scene khi danh gia window anchor."""
    info = extract_item_info(item)
    return (
        str(info["datetime"] or ""),
        str(info["platform"] or ""),
        str(info["orbit_state"] or ""),
        str(info["relative_orbit"] if info["relative_orbit"] is not None else ""),
        str(info["slice_number"] if info["slice_number"] is not None else ""),
    )


def summarize_unique_scenes(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rut gon danh sach scene unique de mo ta noi dung cua moi window."""
    seen: set[Tuple[str, str, str, str, str]] = set()
    scenes: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"])):
        key = item_scene_key(item)
        if key in seen:
            continue
        seen.add(key)
        info = extract_item_info(item)
        scenes.append(
            {
                "id": info["id"],
                "datetime": info["datetime"],
                "platform": info["platform"],
                "orbit_state": info["orbit_state"],
                "relative_orbit": info["relative_orbit"],
                "slice_number": info["slice_number"],
            }
        )
    return scenes


def scene_datetime_values(items: List[Dict[str, Any]]) -> List[str]:
    return [extract_item_info(item)["datetime"] for item in items if extract_item_info(item).get("datetime")]


def unique_datetime_count_from_items(items: List[Dict[str, Any]]) -> int:
    return len({dt for dt in scene_datetime_values(items) if dt})


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
    orbit_state = str(info.get("orbit_state") or "").lower() or "unknown"
    rel_orbit = "" if info.get("relative_orbit") is None else str(info["relative_orbit"])
    if mode == "orbit_state_relative_orbit":
        return (orbit_state, rel_orbit)
    if mode == "orbit_state_only":
        return (orbit_state,)
    if mode == "mixed":
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
    representative_pool_mode: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Select monthly representative scene pools with an in-period relaxation ladder."""
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

    default_levels = [
        {
            "level": 0,
            "level_name": "same_orbit_state_and_relative_orbit",
            "signature_mode": "orbit_state_relative_orbit",
            "required_scene_count": max(2, int(min_scenes_per_half)),
        },
        {
            "level": 1,
            "level_name": "same_orbit_state_only",
            "signature_mode": "orbit_state_only",
            "required_scene_count": max(2, int(min_scenes_per_half)),
        },
        {
            "level": 2,
            "level_name": "same_orbit_state_one_scene_min",
            "signature_mode": "orbit_state_only",
            "required_scene_count": 1,
        },
        {
            "level": 3,
            "level_name": "mixed_orbit_allowed",
            "signature_mode": "mixed",
            "required_scene_count": 1,
        },
    ]
    if representative_pool_mode == "mixed":
        if require_same_orbit_direction:
            raise ValueError(
                "representative_pool_mode='mixed' cannot be combined with require_same_orbit_direction=True."
            )
        levels = [
            {
                "level": 100,
                "level_name": "forced_mixed_orbit_all_pre_post",
                "signature_mode": "mixed",
                "required_scene_count": max(1, int(min_scenes_per_half)),
            }
        ]
    elif representative_pool_mode == "orbit_only":
        levels = list(default_levels[:3])
        if not auto_relax_inside_period:
            levels = levels[:1]
    else:
        levels = list(default_levels)
        if require_same_orbit_direction:
            levels = [level for level in levels if level["signature_mode"] != "mixed"]
        if not auto_relax_inside_period:
            levels = levels[:1]

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
            pre_ids=[scene.get("id") for scene in chosen["pre_scenes"]],
            post_ids=[scene.get("id") for scene in chosen["post_scenes"]],
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
        "period_mode": period["period_mode"],
        "period_start": period["period_start"],
        "period_end": period["period_end"],
        "period_anchor_datetime": period["period_anchor_datetime"],
        "anchor_datetime": period["period_anchor_datetime"],
        "pair_id": f"period_{period['period_id']}",
        "period_boundary_policy": "clip_inside_period",
        "period_split_policy": "first_half_vs_second_half",
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


def collect_anchor_window_items(
    items: List[Dict[str, Any]],
    aoi_geometry: Dict[str, Any],
    aoi_bbox: Optional[List[float]],
    anchor_dt: datetime,
    window_before_days: float,
    window_after_days: float,
    min_aoi_coverage: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Lay item giao AOI geometry trong 2 cua so quanh anchor."""
    pre_start = anchor_dt - timedelta(days=float(window_before_days))
    post_end = anchor_dt + timedelta(days=float(window_after_days))
    pre_items: List[Dict[str, Any]] = []
    post_items: List[Dict[str, Any]] = []
    annotate_items_for_aoi(items, aoi_geometry, aoi_bbox=aoi_bbox)

    for item in items:
        cov = item_aoi_coverage_value(item)
        if cov <= min_aoi_coverage:
            continue
        info = extract_item_info(item)
        dt = parse_datetime_utc(info["datetime"])
        if pre_start <= dt < anchor_dt:
            pre_items.append(item)
        elif anchor_dt <= dt <= post_end:
            post_items.append(item)
    return pre_items, post_items


def _neg_timestamp(value: str) -> float:
    return -parse_datetime_utc(value).timestamp()


def anchor_rank_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float, str, str]:
    """Xep hang anchor cho train-like model theo uu tien recency.

    Uu tien:
      1) post/latest input moi nhat
      2) anchor moi nhat
      3) support exact scene moi nhat
      4) pre/latest scene moi nhat
      5) support pair gap nho hon
      6) ID on dinh

    Scene count khong con la tieu chi rank chinh; no chi la nguong toi thieu.
    """
    return (
        _neg_timestamp(candidate["post_latest_scene_datetime"]),
        _neg_timestamp(candidate["anchor_datetime"]),
        _neg_timestamp(candidate["support_t2_datetime"]),
        _neg_timestamp(candidate["pre_latest_scene_datetime"]),
        candidate["support_pair_delta_seconds"],
        candidate["support_t1_id"],
        candidate["support_t2_id"],
    )


def suggest_trainlike_anchors(
    items: List[Dict[str, Any]],
    aoi_bbox: List[float],
    aoi_geometry: Dict[str, Any],
    window_before_days: float,
    window_after_days: float,
    min_aoi_coverage: float,
    min_delta_hours: float,
    same_orbit_direction: bool,
    min_scenes_per_window: int,
) -> List[Dict[str, Any]]:
    """De xuat anchor toi uu cho train-like windows khi chi co STAC timeline.

    Anchor duoc suy ra tu midpoint cua cac support pair trong STAC. Sau do danh gia
    2 window quanh anchor de uu tien latest input moi nhat. Scene count chi la nguong hop le.
    """
    candidates: List[Dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]))
    annotate_items_for_aoi(sorted_items, aoi_geometry, aoi_bbox=aoi_bbox)
    min_delta_sec = max(0.0, float(min_delta_hours) * 3600.0)
    max_support_gap_sec = max(0.0, float(window_before_days + window_after_days) * 86400.0)

    for i in range(len(sorted_items) - 1):
        item_t1 = sorted_items[i]
        info_t1 = extract_item_info(item_t1)
        bbox1 = info_t1["bbox"]
        if len(bbox1) != 4 or item_aoi_coverage_value(item_t1) <= min_aoi_coverage:
            continue
        dt1 = parse_datetime_utc(info_t1["datetime"])

        for j in range(i + 1, len(sorted_items)):
            item_t2 = sorted_items[j]
            info_t2 = extract_item_info(item_t2)
            bbox2 = info_t2["bbox"]
            if len(bbox2) != 4 or item_aoi_coverage_value(item_t2) <= min_aoi_coverage:
                continue
            dt2 = parse_datetime_utc(info_t2["datetime"])

            if same_orbit_direction:
                orbit_t1 = str(info_t1["orbit_state"] or "").lower()
                orbit_t2 = str(info_t2["orbit_state"] or "").lower()
                if not orbit_t1 or not orbit_t2 or orbit_t1 != orbit_t2:
                    continue

            delta_sec = (dt2 - dt1).total_seconds()
            if delta_sec < min_delta_sec:
                continue
            if delta_sec > max_support_gap_sec:
                break

            anchor_dt = midpoint_datetime(dt1, dt2)
            pre_items, post_items = collect_anchor_window_items(
                items=sorted_items,
                aoi_geometry=aoi_geometry,
                aoi_bbox=aoi_bbox,
                anchor_dt=anchor_dt,
                window_before_days=window_before_days,
                window_after_days=window_after_days,
                min_aoi_coverage=min_aoi_coverage,
            )
            pre_scenes = summarize_unique_scenes(pre_items)
            post_scenes = summarize_unique_scenes(post_items)
            if len(pre_scenes) < min_scenes_per_window or len(post_scenes) < min_scenes_per_window:
                continue

            pre_latest_dt = max(scene["datetime"] for scene in pre_scenes)
            post_latest_dt = max(scene["datetime"] for scene in post_scenes)

            candidates.append(
                {
                    "anchor_strategy": "midpoint_pair",
                    "anchor_datetime": anchor_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "window_before_days": float(window_before_days),
                    "window_after_days": float(window_after_days),
                    "pre_scene_count": len(pre_scenes),
                    "post_scene_count": len(post_scenes),
                    "pre_scenes": pre_scenes,
                    "post_scenes": post_scenes,
                    "pre_latest_scene_datetime": pre_latest_dt,
                    "post_latest_scene_datetime": post_latest_dt,
                    "support_t1_id": info_t1["id"],
                    "support_t2_id": info_t2["id"],
                    "support_t1_datetime": info_t1["datetime"],
                    "support_t2_datetime": info_t2["datetime"],
                    "support_pair_delta_seconds": delta_sec,
                    "support_pair_delta_hours": delta_sec / 3600.0,
                    "support_pair_delta_days": delta_sec / 86400.0,
                    "support_t1_orbit_state": info_t1["orbit_state"],
                    "support_t2_orbit_state": info_t2["orbit_state"],
                    "support_t1_platform": info_t1["platform"],
                    "support_t2_platform": info_t2["platform"],
                    "aoi_bbox_coverage_t1": item_aoi_bbox_coverage_value(item_t1),
                    "aoi_bbox_coverage_t2": item_aoi_bbox_coverage_value(item_t2),
                    "aoi_coverage_t1": item_aoi_coverage_value(item_t1),
                    "aoi_coverage_t2": item_aoi_coverage_value(item_t2),
                    "aoi_union_coverage_pair": items_union_coverage([item_t1, item_t2], aoi_geometry),
                }
            )

    return sorted(candidates, key=anchor_rank_key)


def build_trainlike_anchor_manifest(
    candidate: Dict[str, Any],
    aoi_bbox: List[float],
    geojson_path: Optional[str],
    required_pols: List[str],
) -> Dict[str, Any]:
    """Tao manifest co the dua thang vao gee_trainlike_download.py."""
    anchor_dt = parse_datetime_utc(candidate["anchor_datetime"])
    pair_id = f"anchor_{anchor_dt.strftime('%Y%m%dT%H%M%S')}_pre{int(candidate['window_before_days'])}_post{int(candidate['window_after_days'])}"
    support = support_pair_later_first(candidate)
    return {
        "manifest_type": "trainlike_anchor",
        "anchor_source": "stac_midpoint_pair",
        "anchor_strategy": candidate["anchor_strategy"],
        "selection_priority": "latest_input_datetime",
        "anchor_datetime": candidate["anchor_datetime"],
        "pair_id": pair_id,
        "t1_id": support["t1_id"],
        "t2_id": support["t2_id"],
        "t1_datetime": support["t1_datetime"],
        "t2_datetime": support["t2_datetime"],
        "later_id": support["later_id"],
        "earlier_id": support["earlier_id"],
        "later_datetime": support["later_datetime"],
        "earlier_datetime": support["earlier_datetime"],
        "support_pair_semantics": {
            "t1_role": "later/posterior exact support scene",
            "t2_role": "earlier/prior exact support scene",
        },
        "model_input_semantics": {
            "s1t1_role": "post/future window relative to anchor",
            "s1t2_role": "pre/past window relative to anchor",
        },
        "t1_orbit_state": support.get("t1_orbit_state"),
        "t2_orbit_state": support.get("t2_orbit_state"),
        "aoi_bbox": aoi_bbox,
        "aoi_geojson": str(Path(geojson_path).resolve()) if geojson_path else None,
        "required_polarizations": required_pols,
        "window_before_days": candidate["window_before_days"],
        "window_after_days": candidate["window_after_days"],
        "pre_scene_count": candidate["pre_scene_count"],
        "post_scene_count": candidate["post_scene_count"],
        "pre_latest_scene_datetime": candidate["pre_latest_scene_datetime"],
        "post_latest_scene_datetime": candidate["post_latest_scene_datetime"],
        "latest_input_datetime": candidate["post_latest_scene_datetime"],
        "pre_scenes": candidate["pre_scenes"],
        "post_scenes": candidate["post_scenes"],
        "support_pair_delta_hours": candidate["support_pair_delta_hours"],
        "support_pair_delta_days": candidate["support_pair_delta_days"],
        "aoi_coverage_t1": candidate.get("aoi_coverage_t2"),
        "aoi_coverage_t2": candidate.get("aoi_coverage_t1"),
        "aoi_bbox_coverage_t1": candidate["aoi_bbox_coverage_t2"],
        "aoi_bbox_coverage_t2": candidate["aoi_bbox_coverage_t1"],
        "aoi_union_coverage_pair": candidate.get("aoi_union_coverage_pair", 0.0),
    }


def print_anchor_table(candidates: List[Dict[str, Any]], top_k: int = 10) -> None:
    """In bang de xuat anchor train-like."""
    if not candidates:
        print("[ANCHOR] Khong co anchor hop le.")
        return

    print("=" * 160)
    print(f"{'#':>3} {'pre':>5} {'post':>5} {'latest_in':<22} {'anchor':<22} {'support_t1':<40} {'support_t2':<40}")
    print("=" * 160)
    for idx, cand in enumerate(candidates[:top_k], 1):
        support = support_pair_later_first(cand)
        print(
            f"{idx:>3} {cand['pre_scene_count']:>5} {cand['post_scene_count']:>5} "
            f"{cand['post_latest_scene_datetime'][:22]:<22} {cand['anchor_datetime'][:22]:<22} "
            f"{support['t1_id'][:40]:<40} {support['t2_id'][:40]:<40}"
        )


def diagnose_no_pair(
    items: List[Dict[str, Any]],
    aoi_bbox: List[float],
    aoi_geometry: Dict[str, Any],
    strict_slice: bool,
    min_overlap: float,
    min_aoi_coverage: float,
    max_delta_days: int,
    min_delta_hours: float,
    same_orbit_direction: bool,
) -> Dict[str, Any]:
    """
    Chan doan nhanh khi khong co pair:
      - item_count
      - co cap chi khac o datatake/time hay khong
      - cap tiem nang tot nhat neu bo mot so dieu kien
    """
    sorted_items = sorted(items, key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]))
    total_pairs = max(0, len(sorted_items) * (len(sorted_items) - 1) // 2)
    same_datatake_pairs = 0
    orbit_direction_fail_pairs = 0
    min_time_pairs = 0
    max_time_pairs = 0
    spatial_pairs = 0

    min_delta_sec = max(0.0, float(min_delta_hours) * 3600.0)
    max_delta_sec = max(0.0, float(max_delta_days) * 86400.0)
    annotate_items_for_aoi(sorted_items, aoi_geometry, aoi_bbox=aoi_bbox)

    for i in range(len(sorted_items) - 1):
        info_t1 = extract_item_info(sorted_items[i])
        dt1 = parse_datetime_utc(info_t1["datetime"])
        for j in range(i + 1, len(sorted_items)):
            info_t2 = extract_item_info(sorted_items[j])
            dt2 = parse_datetime_utc(info_t2["datetime"])
            delta_sec = (dt2 - dt1).total_seconds()

            datatake_t1 = info_t1["datatake_id"]
            datatake_t2 = info_t2["datatake_id"]
            is_same_datatake = datatake_t1 is not None and datatake_t2 is not None and datatake_t1 == datatake_t2
            if is_same_datatake:
                same_datatake_pairs += 1

            if same_orbit_direction:
                orbit_t1 = str(info_t1["orbit_state"] or "").lower()
                orbit_t2 = str(info_t2["orbit_state"] or "").lower()
                if not orbit_t1 or not orbit_t2 or orbit_t1 != orbit_t2:
                    orbit_direction_fail_pairs += 1
                    continue

            if delta_sec < min_delta_sec:
                min_time_pairs += 1
                continue
            if delta_sec > max_delta_sec:
                max_time_pairs += 1
                continue

            bbox1 = info_t1["bbox"]
            bbox2 = info_t2["bbox"]
            if len(bbox1) != 4 or len(bbox2) != 4:
                continue

            overlap = bbox_overlap_ratio(bbox1, bbox2)
            cover_t1 = item_aoi_coverage_value(sorted_items[i])
            cover_t2 = item_aoi_coverage_value(sorted_items[j])
            if cover_t1 <= min_aoi_coverage or cover_t2 <= min_aoi_coverage:
                spatial_pairs += 1

    relaxed_pairs = search_pairs_sorted(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        min_overlap=0.0,
        min_aoi_coverage=0.0,
        max_delta_days=90,
        min_delta_hours=min_delta_hours,
        strict_slice=strict_slice,
        same_orbit_direction=same_orbit_direction,
    )

    if len(items) < 2:
        reason = "INSUFFICIENT_ITEMS"
    elif total_pairs > 0 and same_orbit_direction and orbit_direction_fail_pairs == total_pairs:
        reason = "NO_PAIR_WITH_ORBIT_DIRECTION"
    elif total_pairs > 0 and same_datatake_pairs == total_pairs and min_time_pairs == total_pairs:
        reason = "ONLY_SAME_DATATAKE_CANDIDATES"
    elif total_pairs > 0 and min_time_pairs == total_pairs:
        reason = "NO_PAIR_WITH_MIN_TIME_GAP"
    elif total_pairs > 0 and max_time_pairs == total_pairs:
        reason = "NO_PAIR_WITH_MAX_TIME_WINDOW"
    elif total_pairs > 0 and spatial_pairs == total_pairs:
        reason = "NO_PAIR_WITH_AOI_GEOMETRY_COVERAGE"
    else:
        reason = "NO_VALID_PAIR"

    best_relaxed = relaxed_pairs[0] if relaxed_pairs else None
    return {
        "reason": reason,
        "item_count": len(items),
        "pair_count_examined": total_pairs,
        "same_datatake_pairs": same_datatake_pairs,
        "orbit_direction_fail_pairs": orbit_direction_fail_pairs,
        "too_close_pairs": min_time_pairs,
        "too_far_pairs": max_time_pairs,
        "spatial_fail_pairs": spatial_pairs,
        "best_relaxed": best_relaxed,
    }


def ensure_geojson_file_list(
    geojson_dir: str,
    pattern: str = "*.geojson",
    include_names: Optional[List[str]] = None,
) -> List[Path]:
    """Lay danh sach geojson files tu thu muc."""
    base = Path(geojson_dir)
    if not base.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc geojson: {base}")

    paths = sorted(base.glob(pattern))
    if include_names:
        wanted = {n.strip() for n in include_names if n.strip()}
        paths = [p for p in paths if p.name in wanted or p.stem in wanted]
    return paths


def remove_path_if_exists(path: Path) -> None:
    """Xoa file/thu muc neu ton tai."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def remove_manifest_variants(manifest_dir: Path, stem: str) -> None:
    """Xoa cac manifest cu cua cung 1 AOI de tranh stale output."""
    for path in manifest_dir.glob(f"{stem}.*.json"):
        remove_path_if_exists(path)


def format_duration_human(total_seconds: float) -> str:
    """Format so giay thanh chuoi de doc."""
    total = int(round(max(0.0, float(total_seconds))))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def build_pair_run_config(args: argparse.Namespace, required_pols: List[str]) -> Dict[str, Any]:
    """Thong tin run/config de luu vao report."""
    return {
        "stac_url": args.stac_url,
        "collection": args.collection,
        "geojson_dir": args.geojson_dir,
        "glob": args.glob,
        "only": args.only,
        "datetime": args.datetime,
        "limit": args.limit,
        "required_polarizations": required_pols,
        "orbit_filter": args.orbit,
        "relative_orbit_filter": args.rel_orbit,
        "min_overlap": args.min_overlap,
        "bbox_overlap_filter_enforced": False,
        "min_aoi_coverage": args.min_aoi_coverage,
        "min_delta_hours": args.min_delta_hours,
        "max_delta_days": args.max_delta_days,
        "same_orbit_direction": args.same_orbit_direction,
        "strict_slice": args.strict_slice,
        "auto_relax": args.auto_relax,
        "pick_index": args.pick_index,
        "full_item": args.full_item,
        "dry_run": args.dry_run,
        "out_dir": args.out_dir,
        "report_dir": args.report_dir,
    }


def build_selected_pair_info(pair: Dict[str, Any], manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Tom tat thong tin cap duoc chon de dua vao summary/report."""
    canonical = exact_pair_later_first(pair)
    min_cov = min(canonical["aoi_coverage_t1"], canonical["aoi_coverage_t2"])
    min_bbox_cov = min(canonical["aoi_bbox_coverage_t1"], canonical["aoi_bbox_coverage_t2"])
    info: Dict[str, Any] = {
        "t1_id": canonical["t1_id"],
        "t2_id": canonical["t2_id"],
        "t1_datetime": canonical["t1_datetime"],
        "t2_datetime": canonical["t2_datetime"],
        "later_id": canonical["later_id"],
        "earlier_id": canonical["earlier_id"],
        "later_datetime": canonical["later_datetime"],
        "earlier_datetime": canonical["earlier_datetime"],
        "pair_semantics": {
            "t1_role": "later/posterior exact scene",
            "t2_role": "earlier/prior exact scene",
            "matches_training_semantics": True,
        },
        "latest_input_datetime": canonical["latest_input_datetime"],
        "t1_datatake_id": canonical.get("t1_datatake_id"),
        "t2_datatake_id": canonical.get("t2_datatake_id"),
        "t1_orbit_state": canonical.get("t1_orbit_state"),
        "t2_orbit_state": canonical.get("t2_orbit_state"),
        "relative_orbit": pair.get("relative_orbit"),
        "slice_number": pair.get("slice_number"),
        "delta_seconds": pair["delta_seconds"],
        "delta_hours": pair["delta_hours"],
        "delta_days": pair["delta_days"],
        "delta_human": format_duration_human(pair["delta_seconds"]),
        "bbox_overlap": pair["bbox_overlap"],
        "aoi_union_coverage_pair": canonical.get("aoi_union_coverage_pair", 0.0),
        "aoi_bbox_coverage_t1": canonical["aoi_bbox_coverage_t1"],
        "aoi_bbox_coverage_t2": canonical["aoi_bbox_coverage_t2"],
        "aoi_bbox_coverage_min": min_bbox_cov,
        "aoi_coverage_t1": canonical["aoi_coverage_t1"],
        "aoi_coverage_t2": canonical["aoi_coverage_t2"],
        "aoi_coverage_min": min_cov,
        "coverage_source_t1": canonical.get("coverage_source_t1", "none"),
        "coverage_source_t2": canonical.get("coverage_source_t2", "none"),
    }
    if manifest is not None:
        info["pair_id"] = manifest.get("pair_id")
    return info


def build_item_download_plan(item: Dict[str, Any], required_pols: List[str]) -> Optional[List[Tuple[str, str, str]]]:
    """
    Tao plan download cho 1 item:
      [(pol, asset_key, href), ...]
    """
    plan: List[Tuple[str, str, str]] = []
    for pol in required_pols:
        pick = select_asset_href(item, pol)
        if not pick:
            return None
        asset_key, href = pick
        plan.append((pol, asset_key, href))
    return plan


def download_manifest_pair(
    manifest: Dict[str, Any],
    required_pols: List[str],
    out_dir: str,
    t1_prefix: str,
    t2_prefix: str,
    subset_aoi: bool = True,
    aoi_geometry: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Download theo naming infer_production scan duoc:
      s1t1_<pair_id>_<pol>.tif
      s1t2_<pair_id>_<pol>.tif
    """
    try:
        downloader = S3Downloader()
    except RuntimeError as e:
        print(f"[Loi] {e}")
        return []
    out_paths: List[str] = []
    pair_id = manifest["pair_id"]
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pol in required_pols:
        pol_l = pol.lower()
        entry = manifest["assets"][pol]
        t1_href = entry["t1_href"]
        t2_href = entry["t2_href"]

        t1_ext = Path(urlparse(t1_href).path).suffix or ".tif"
        t2_ext = Path(urlparse(t2_href).path).suffix or ".tif"

        t1_name = f"{t1_prefix}{pair_id}_{pol_l}{t1_ext}"
        t2_name = f"{t2_prefix}{pair_id}_{pol_l}{t2_ext}"
        t1_local = str(output_dir / t1_name)
        t2_local = str(output_dir / t2_name)

        print(f"\n[DOWNLOAD] {pol} T1")
        if subset_aoi and aoi_geometry is not None:
            ok_t1 = downloader.download_aoi_subset_from_href(t1_href, t1_local, aoi_geometry)
        else:
            ok_t1 = downloader.download_from_href(t1_href, t1_local)
        if ok_t1:
            out_paths.append(t1_local)

        print(f"\n[DOWNLOAD] {pol} T2")
        if subset_aoi and aoi_geometry is not None:
            ok_t2 = downloader.download_aoi_subset_from_href(t2_href, t2_local, aoi_geometry)
        else:
            ok_t2 = downloader.download_from_href(t2_href, t2_local)
        if ok_t2:
            out_paths.append(t2_local)

    return out_paths


def cmd_list(args: argparse.Namespace) -> None:
    """Liet ke item phu hop AOI + filters."""
    client = STACClient(args.stac_url)
    required_pols = parse_required_pols(args.pols)
    items, _, _ = collect_items_with_filters(client, args, required_pols)

    if not items:
        print("[KET QUA] Khong tim thay item nao.")
        return

    print("=" * 120)
    print(f"{'ID':<62} {'datetime':<24} {'orbit':>5} {'rel':>4} {'slice':>6} {'pols':<12} {'prod':<5}")
    print("=" * 120)
    for item in items:
        info = extract_item_info(item)
        pols_str = ",".join(info["polarizations"])
        slice_str = f"{info['slice_number']}/{info['total_slices']}" if info["slice_number"] else "?"
        print(
            f"{info['id']:<62} {info['datetime'][:19]:<24} "
            f"{str(info['orbit_state'])[:4]:>5} {str(info['relative_orbit']):>4} {slice_str:>6} "
            f"{pols_str:<12} {info['product_type']:<5}"
        )

    sample_info = extract_item_info(items[0])
    print("\n--- Asset mau (item dau tien) ---")
    for key, asset in sample_info["assets"].items():
        href = asset.get("href", "N/A")
        media = asset.get("type", "N/A")
        pol = infer_asset_pol(key, asset)
        print(f"[{key}] pol={pol or '?'} type={media}")
        print(f"  href={href}")


def cmd_pair(args: argparse.Namespace) -> None:
    """Tim va in danh sach cap T1/T2 hop le."""
    client = STACClient(args.stac_url)
    required_pols = parse_required_pols(args.pols)
    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, args, required_pols)

    pairs = search_pairs_sorted(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        min_overlap=args.min_overlap,
        min_aoi_coverage=args.min_aoi_coverage,
        max_delta_days=args.max_delta_days,
        min_delta_hours=args.min_delta_hours,
        strict_slice=args.strict_slice,
        same_orbit_direction=args.same_orbit_direction,
    )
    print_pairs_table(pairs, top_k=args.top_k)


def cmd_suggest_anchor(args: argparse.Namespace) -> None:
    """De xuat anchor cho train-like GEE windows khi chi co AOI + STAC."""
    client = STACClient(args.stac_url)
    required_pols = parse_required_pols(args.pols)
    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, args, required_pols)
    if not items:
        print("[KET QUA] Khong co item hop le sau filter.")
        return

    candidates = suggest_trainlike_anchors(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        window_before_days=args.window_before_days,
        window_after_days=args.window_after_days,
        min_aoi_coverage=args.min_aoi_coverage,
        min_delta_hours=args.min_delta_hours,
        same_orbit_direction=args.same_orbit_direction,
        min_scenes_per_window=args.min_scenes_per_window,
    )
    print_anchor_table(candidates, top_k=args.top_k)
    if not candidates:
        return

    pick_index = max(1, args.pick_index)
    pick_index = min(pick_index, len(candidates))
    chosen = candidates[pick_index - 1]
    manifest = build_trainlike_anchor_manifest(
        candidate=chosen,
        aoi_bbox=aoi_bbox,
        geojson_path=getattr(args, "geojson", None),
        required_pols=required_pols,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.save_manifest:
        manifest_path = Path(args.save_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK] Da luu anchor manifest: {manifest_path}")


def cmd_prepare(args: argparse.Namespace) -> None:
    """
    Luong de xai trong production:
      1) Query + filter + rank
      2) Chon cap tot nhat
      3) Xuat manifest href
      4) (Tuy chon) probe rasterio
      5) (Tuy chon) download theo naming infer_production
    """
    client = STACClient(args.stac_url)
    required_pols = parse_required_pols(args.pols)
    items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, args, required_pols)

    if not items:
        print("[KET QUA] Khong co item hop le sau filter.")
        return

    strict_pairs = search_pairs_sorted(
        items=items,
        aoi_bbox=aoi_bbox,
        aoi_geometry=aoi_geometry,
        min_overlap=args.min_overlap,
        min_aoi_coverage=args.min_aoi_coverage,
        max_delta_days=args.max_delta_days,
        min_delta_hours=args.min_delta_hours,
        strict_slice=args.strict_slice,
        same_orbit_direction=args.same_orbit_direction,
    )
    selected_profile = "strict"
    ranked_pairs = strict_pairs

    if not ranked_pairs:
        print("[KET QUA] Khong co cap hop le voi nguong hien tai.")
        diag = diagnose_no_pair(
            items=items,
            aoi_bbox=aoi_bbox,
            aoi_geometry=aoi_geometry,
            strict_slice=args.strict_slice,
            min_overlap=args.min_overlap,
            min_aoi_coverage=args.min_aoi_coverage,
            max_delta_days=args.max_delta_days,
            min_delta_hours=args.min_delta_hours,
            same_orbit_direction=args.same_orbit_direction,
        )
        print(
            f"[DIAG] reason={diag['reason']} | item_count={diag['item_count']} | "
            f"pair_count={diag['pair_count_examined']}"
        )
        print(
            f"[DIAG] same_datatake_pairs={diag['same_datatake_pairs']} | "
            f"orbit_direction_fail_pairs={diag['orbit_direction_fail_pairs']} | "
            f"too_close_pairs={diag['too_close_pairs']} | "
            f"too_far_pairs={diag['too_far_pairs']} | "
            f"spatial_fail_pairs={diag['spatial_fail_pairs']}"
        )
        if diag["best_relaxed"] is not None:
            br = diag["best_relaxed"]
            print(
                "[DIAG] best_relaxed_with_time_rule: "
                f"{br['t1_id']} -> {br['t2_id']} | Δt={br['delta_hours']:.2f}h | overlap={br['bbox_overlap']:.1%}"
            )

        if not args.auto_relax:
            print("[DIAG] Tip: dung --auto-relax de script tu thu nguong mem hon.")
            return

        relax_profiles = [
            ("balanced", args.min_overlap, args.min_aoi_coverage, 30, args.min_delta_hours),
            ("loose", args.min_overlap, args.min_aoi_coverage, 90, args.min_delta_hours),
        ]
        for name, min_ov, min_cov, max_days, min_hours in relax_profiles:
            candidate_pairs = search_pairs_sorted(
                items=items,
                aoi_bbox=aoi_bbox,
                aoi_geometry=aoi_geometry,
                min_overlap=min_ov,
                min_aoi_coverage=min_cov,
                max_delta_days=max_days,
                min_delta_hours=min_hours,
                strict_slice=args.strict_slice,
                same_orbit_direction=args.same_orbit_direction,
            )
            if candidate_pairs:
                ranked_pairs = candidate_pairs
                selected_profile = name
                print(
                    f"[AUTO-RELAX] Chon profile '{name}' "
                    f"(min_overlap={min_ov}, min_cov={min_cov}, max_days={max_days}, min_delta_h={min_hours})"
                )
                break

        if not ranked_pairs:
            print("[KET QUA] Da thu auto-relax nhung van khong tim thay pair.")
            return

    print(f"[PAIR PROFILE] {selected_profile}")
    print_pairs_table(ranked_pairs, top_k=args.top_k)

    pair_index = max(1, args.pick_index)
    if pair_index > len(ranked_pairs):
        print(f"[Loi] pick-index={pair_index} vuot qua so cap ({len(ranked_pairs)}).")
        return

    selected_pair = ranked_pairs[pair_index - 1]
    manifest = build_manifest_for_pair(selected_pair, required_pols)
    if manifest is None:
        print("[Loi] Cap duoc chon khong du href cho cac polarization yeu cau.")
        return
    manifest["selection_profile"] = selected_profile

    print("\n[SELECTED PAIR]")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.save_manifest:
        manifest_path = Path(args.save_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK] Da luu manifest: {manifest_path}")

    if args.probe_rasterio:
        for pol in required_pols:
            entry = manifest["assets"][pol]
            probe_rasterio_href(f"T1-{pol}", entry["t1_href"])
            probe_rasterio_href(f"T2-{pol}", entry["t2_href"])

    if args.download:
        if args.geojson:
            _, aoi_geometry = load_geojson_aoi(args.geojson)
        else:
            aoi_geometry = bbox_to_geometry(aoi_bbox)

        out_paths = download_manifest_pair(
            manifest=manifest,
            required_pols=required_pols,
            out_dir=args.out_dir,
            t1_prefix=args.t1_prefix,
            t2_prefix=args.t2_prefix,
            subset_aoi=not args.full_item,
            aoi_geometry=aoi_geometry,
        )
        print("\n[DOWNLOAD RESULT]")
        if out_paths:
            for p in out_paths:
                print(f"  - {p}")
        else:
            print("  Khong tai duoc file nao.")


def cmd_download_aoi_matches(args: argparse.Namespace) -> None:
    """
    Batch download cho AOI goc:
      - Moi AOI query item intersect
      - Download item match (VV,VH) theo so luong gioi han
    """
    required_pols = parse_required_pols(args.pols)
    client = STACClient(args.stac_url)

    include_names = args.only.split(",") if args.only else None
    geojson_paths = ensure_geojson_file_list(args.geojson_dir, args.glob, include_names)
    if not geojson_paths:
        print("[KET QUA] Khong tim thay geojson nao de xu ly.")
        return

    downloader: Optional[S3Downloader] = None
    if not args.dry_run:
        try:
            downloader = S3Downloader()
        except RuntimeError as e:
            print(f"[Loi] {e}")
            return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    total_downloaded = 0

    for gj in geojson_paths:
        # skip generated AOIs unless user explicitly points to that folder
        if args.skip_generated and "generated_aoi" in gj.parts:
            continue

        qargs = argparse.Namespace(
            stac_url=args.stac_url,
            collection=args.collection,
            bbox=None,
            geojson=str(gj),
            datetime=args.datetime,
            limit=args.limit,
            orbit=args.orbit,
            rel_orbit=args.rel_orbit,
            pols=args.pols,
        )
        items, _, _ = collect_items_with_filters(client, qargs, required_pols)
        _, aoi_geometry = load_geojson_aoi(str(gj))
        items = sorted(items, key=lambda it: parse_datetime_utc(extract_item_info(it)["datetime"]), reverse=True)
        if args.max_items_per_aoi > 0:
            items = items[: args.max_items_per_aoi]

        rec: Dict[str, Any] = {
            "geojson": gj.name,
            "items_considered": len(items),
            "downloaded_files": [],
            "item_ids": [],
        }

        aoi_out = out_root / gj.stem
        aoi_out.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(items, 1):
            info = extract_item_info(item)
            rec["item_ids"].append(info["id"])
            plan = build_item_download_plan(item, required_pols)
            if not plan:
                continue

            item_tag = f"{idx:02d}_{info['id']}"
            item_dir = aoi_out / item_tag
            item_dir.mkdir(parents=True, exist_ok=True)

            for pol, _, href in plan:
                ext = Path(urlparse(href).path).suffix or ".tif"
                local_path = item_dir / f"{gj.stem}_{pol.lower()}{ext}"
                if args.dry_run:
                    rec["downloaded_files"].append(str(local_path))
                    continue
                assert downloader is not None
                if args.full_item:
                    ok = downloader.download_from_href(href, str(local_path))
                else:
                    ok = downloader.download_aoi_subset_from_href(href, str(local_path), aoi_geometry)
                if ok:
                    rec["downloaded_files"].append(str(local_path))
                    total_downloaded += 1

        summary.append(rec)
        print(
            f"[AOI MATCH] {gj.name} | items={rec['items_considered']} | files={len(rec['downloaded_files'])}"
        )

    summary_path = report_dir / "original_aoi_download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# Original AOI Download Summary",
        "",
        "| AOI | Items Considered | Downloaded Files |",
        "|---|---:|---:|",
    ]
    for rec in summary:
        report_lines.append(
            f"| {rec['geojson']} | {rec['items_considered']} | {len(rec['downloaded_files'])} |"
        )
    report_lines.append("")
    report_lines.append(f"- Summary JSON: `{summary_path}`")
    report_lines.append(f"- Output dir: `{out_root}`")
    report_lines.append(f"- Total downloaded files: {total_downloaded}")

    report_path = report_dir / "original_aoi_download_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Saved: {summary_path}")
    print(f"[OK] Saved: {report_path}")


def cmd_download_generated_pairs(args: argparse.Namespace) -> None:
    """
    Batch download theo cap cho generated AOI.
    - Thu strict truoc, neu fail co the auto-relax.
    - Download cap tot nhat cho moi AOI.
    """
    required_pols = parse_required_pols(args.pols)
    client = STACClient(args.stac_url)

    include_names = args.only.split(",") if args.only else None
    geojson_paths = ensure_geojson_file_list(args.geojson_dir, args.glob, include_names)
    if not geojson_paths:
        print("[KET QUA] Khong tim thay geojson nao de xu ly.")
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = report_dir / "generated_pair_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_config = build_pair_run_config(args, required_pols)

    print("[RUN CONFIG]")
    print(json.dumps(run_config, indent=2, ensure_ascii=False))

    summary: List[Dict[str, Any]] = []

    for gj in geojson_paths:
        _, aoi_geometry = load_geojson_aoi(str(gj))
        aoi_out = out_root / gj.stem
        remove_manifest_variants(manifest_dir, gj.stem)
        qargs = argparse.Namespace(
            stac_url=args.stac_url,
            collection=args.collection,
            bbox=None,
            geojson=str(gj),
            datetime=args.datetime,
            limit=args.limit,
            orbit=args.orbit,
            rel_orbit=args.rel_orbit,
            pols=args.pols,
            min_overlap=args.min_overlap,
            min_aoi_coverage=args.min_aoi_coverage,
            min_delta_hours=args.min_delta_hours,
            max_delta_days=args.max_delta_days,
            strict_slice=args.strict_slice,
        )
        items, aoi_bbox, aoi_geometry = collect_items_with_filters(client, qargs, required_pols)
        strict_pairs = search_pairs_sorted(
            items=items,
            aoi_bbox=aoi_bbox,
            aoi_geometry=aoi_geometry,
            min_overlap=args.min_overlap,
            min_aoi_coverage=args.min_aoi_coverage,
            max_delta_days=args.max_delta_days,
            min_delta_hours=args.min_delta_hours,
            strict_slice=args.strict_slice,
            same_orbit_direction=args.same_orbit_direction,
        )

        selected_profile = "strict"
        selected_pairs = strict_pairs
        diag: Optional[Dict[str, Any]] = None

        if not selected_pairs and args.auto_relax:
            diag = diagnose_no_pair(
                items=items,
                aoi_bbox=aoi_bbox,
                aoi_geometry=aoi_geometry,
                strict_slice=args.strict_slice,
                min_overlap=args.min_overlap,
                min_aoi_coverage=args.min_aoi_coverage,
                max_delta_days=args.max_delta_days,
                min_delta_hours=args.min_delta_hours,
                same_orbit_direction=args.same_orbit_direction,
            )
            balanced_pairs = search_pairs_sorted(
                items=items,
                aoi_bbox=aoi_bbox,
                aoi_geometry=aoi_geometry,
                min_overlap=args.min_overlap,
                min_aoi_coverage=args.min_aoi_coverage,
                max_delta_days=30,
                min_delta_hours=args.min_delta_hours,
                strict_slice=args.strict_slice,
                same_orbit_direction=args.same_orbit_direction,
            )
            if balanced_pairs:
                selected_profile = "balanced"
                selected_pairs = balanced_pairs
            else:
                loose_pairs = search_pairs_sorted(
                    items=items,
                    aoi_bbox=aoi_bbox,
                    aoi_geometry=aoi_geometry,
                    min_overlap=args.min_overlap,
                    min_aoi_coverage=args.min_aoi_coverage,
                    max_delta_days=90,
                    min_delta_hours=args.min_delta_hours,
                    strict_slice=args.strict_slice,
                    same_orbit_direction=args.same_orbit_direction,
                )
                if loose_pairs:
                    selected_profile = "loose"
                    selected_pairs = loose_pairs

        rec: Dict[str, Any] = {
            "geojson": gj.name,
            "items": len(items),
            "selected_profile": selected_profile if selected_pairs else None,
            "pair_found": bool(selected_pairs),
            "no_pair_reason": None,
            "selected_pair": None,
            "diagnostics": None,
            "downloaded_files": [],
        }
        if not selected_pairs:
            if diag is None:
                diag = diagnose_no_pair(
                    items=items,
                    aoi_bbox=aoi_bbox,
                    aoi_geometry=aoi_geometry,
                    strict_slice=args.strict_slice,
                    min_overlap=args.min_overlap,
                    min_aoi_coverage=args.min_aoi_coverage,
                    max_delta_days=args.max_delta_days,
                    min_delta_hours=args.min_delta_hours,
                    same_orbit_direction=args.same_orbit_direction,
                )
            rec["no_pair_reason"] = diag["reason"]
            rec["diagnostics"] = diag
            if not args.dry_run:
                remove_path_if_exists(aoi_out)

        if selected_pairs:
            pair_index = max(1, args.pick_index)
            pair_index = min(pair_index, len(selected_pairs))
            chosen = selected_pairs[pair_index - 1]
            manifest = build_manifest_for_pair(chosen, required_pols)
            if manifest is not None:
                manifest["selection_profile"] = selected_profile
                manifest_path = manifest_dir / f"{gj.stem}.{selected_profile}.json"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                rec["manifest_path"] = str(manifest_path)
                rec["selected_pair"] = build_selected_pair_info(chosen, manifest)

                if not args.dry_run:
                    remove_path_if_exists(aoi_out)
                    out_paths = download_manifest_pair(
                        manifest=manifest,
                        required_pols=required_pols,
                        out_dir=str(aoi_out),
                        t1_prefix=args.t1_prefix,
                        t2_prefix=args.t2_prefix,
                        subset_aoi=not args.full_item,
                        aoi_geometry=aoi_geometry,
                    )
                    rec["downloaded_files"] = out_paths

        summary.append(rec)
        if rec["selected_pair"] is not None:
            sp = rec["selected_pair"]
            print(
                f"[PAIR DETAIL] {gj.name} | pair_id={sp.get('pair_id')} | "
                f"delta={sp['delta_human']} ({sp['delta_hours']:.3f}h) | "
                f"overlap={sp['bbox_overlap']:.1%} | "
                f"aoi_min={sp['aoi_coverage_min']:.1%} | "
                f"orbit={sp.get('t1_orbit_state')}/{sp.get('t2_orbit_state')}"
            )
        print(
            f"[GENERATED PAIR] {gj.name} | items={rec['items']} | "
            f"pair={rec['pair_found']} | profile={rec['selected_profile']} | "
            f"files={len(rec['downloaded_files'])} | reason={rec['no_pair_reason'] or '-'}"
        )

    summary_path = report_dir / "generated_pair_download_summary.json"
    summary_payload = {
        "run_config": run_config,
        "results": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# Generated AOI Pair Download Summary",
        "",
        "## Run Config",
        "",
        f"- STAC URL: `{run_config['stac_url']}`",
        f"- Collection: `{run_config['collection']}`",
        f"- AOI dir: `{run_config['geojson_dir']}`",
        f"- File glob: `{run_config['glob']}`",
        f"- Only: `{run_config['only']}`",
        f"- Datetime filter: `{run_config['datetime']}`",
        f"- Limit: `{run_config['limit']}`",
        f"- Polarizations: `{','.join(run_config['required_polarizations'])}`",
        f"- Orbit filter: `{run_config['orbit_filter']}`",
        f"- Relative orbit filter: `{run_config['relative_orbit_filter']}`",
        f"- Min overlap (diagnostic only): `{run_config['min_overlap']}`",
        f"- BBox overlap enforced: `{run_config['bbox_overlap_filter_enforced']}`",
        f"- Min AOI coverage: `{run_config['min_aoi_coverage']}`",
        f"- Min delta hours: `{run_config['min_delta_hours']}`",
        f"- Max delta days: `{run_config['max_delta_days']}`",
        f"- Same orbit direction: `{run_config['same_orbit_direction']}`",
        f"- Strict slice legacy flag: `{run_config['strict_slice']}`",
        f"- Auto relax: `{run_config['auto_relax']}`",
        f"- Pick index: `{run_config['pick_index']}`",
        f"- Full item: `{run_config['full_item']}`",
        f"- Dry run: `{run_config['dry_run']}`",
        "",
        "## Overview",
        "",
        "| AOI | Items | Pair Found | Profile | Reason | Downloaded Files |",
        "|---|---:|---|---|---|---:|",
    ]
    for rec in summary:
        report_lines.append(
            f"| {rec['geojson']} | {rec['items']} | {rec['pair_found']} | "
            f"{rec['selected_profile'] or '-'} | {rec['no_pair_reason'] or '-'} | {len(rec['downloaded_files'])} |"
        )
    report_lines.append("")
    report_lines.append("## Pair Details")
    report_lines.append("")
    for rec in summary:
        report_lines.append(f"### {rec['geojson']}")
        if rec["selected_pair"] is not None:
            sp = rec["selected_pair"]
            report_lines.append(f"- Pair found: `True`")
            report_lines.append(f"- Profile: `{rec['selected_profile']}`")
            report_lines.append(f"- Pair ID: `{sp.get('pair_id')}`")
            report_lines.append(f"- T1 ID (later): `{sp['t1_id']}`")
            report_lines.append(f"- T2 ID (earlier): `{sp['t2_id']}`")
            report_lines.append(f"- T1 datetime (later): `{sp['t1_datetime']}`")
            report_lines.append(f"- T2 datetime (earlier): `{sp['t2_datetime']}`")
            report_lines.append(f"- Later exact scene: `{sp.get('later_id')}` @ `{sp.get('later_datetime')}`")
            report_lines.append(f"- Earlier exact scene: `{sp.get('earlier_id')}` @ `{sp.get('earlier_datetime')}`")
            report_lines.append(f"- Delta exact: `{sp['delta_human']}`")
            report_lines.append(f"- Delta hours: `{sp['delta_hours']:.6f}`")
            report_lines.append(f"- Delta days: `{sp['delta_days']:.6f}`")
            report_lines.append(f"- AOI geometry coverage T1: `{sp['aoi_coverage_t1']:.6f}`")
            report_lines.append(f"- AOI geometry coverage T2: `{sp['aoi_coverage_t2']:.6f}`")
            report_lines.append(f"- AOI geometry coverage min: `{sp['aoi_coverage_min']:.6f}`")
            report_lines.append(f"- AOI union coverage pair: `{sp.get('aoi_union_coverage_pair', 0.0):.6f}`")
            report_lines.append(f"- BBox overlap (diagnostic): `{sp['bbox_overlap']:.6f}`")
            report_lines.append(f"- AOI bbox coverage T1: `{sp['aoi_bbox_coverage_t1']:.6f}`")
            report_lines.append(f"- AOI bbox coverage T2: `{sp['aoi_bbox_coverage_t2']:.6f}`")
            report_lines.append(f"- AOI bbox coverage min: `{sp['aoi_bbox_coverage_min']:.6f}`")
            report_lines.append(f"- Orbit direction: `{sp.get('t1_orbit_state')}` -> `{sp.get('t2_orbit_state')}`")
            report_lines.append(f"- Relative orbit: `{sp.get('relative_orbit')}`")
            report_lines.append(f"- Slice number: `{sp.get('slice_number')}`")
            report_lines.append(f"- Datatake T1: `{sp.get('t1_datatake_id')}`")
            report_lines.append(f"- Datatake T2: `{sp.get('t2_datatake_id')}`")
            report_lines.append(f"- Manifest: `{rec.get('manifest_path')}`")
            report_lines.append(f"- Downloaded files: `{len(rec['downloaded_files'])}`")
        else:
            report_lines.append(f"- Pair found: `False`")
            report_lines.append(f"- Reason: `{rec['no_pair_reason']}`")
            diag = rec.get("diagnostics") or {}
            if diag:
                report_lines.append(f"- Pair count examined: `{diag.get('pair_count_examined')}`")
                report_lines.append(f"- Same datatake pairs: `{diag.get('same_datatake_pairs')}`")
                report_lines.append(f"- Orbit direction fail pairs: `{diag.get('orbit_direction_fail_pairs')}`")
                report_lines.append(f"- Too close pairs: `{diag.get('too_close_pairs')}`")
                report_lines.append(f"- Too far pairs: `{diag.get('too_far_pairs')}`")
                report_lines.append(f"- AOI geometry coverage fail pairs: `{diag.get('spatial_fail_pairs')}`")
        report_lines.append("")

    report_lines.append(f"- Summary JSON: `{summary_path}`")
    report_lines.append(f"- Manifest dir: `{manifest_dir}`")
    report_lines.append(f"- Output dir: `{out_root}`")

    report_path = report_dir / "generated_pair_download_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Saved: {summary_path}")
    print(f"[OK] Saved: {report_path}")


def cmd_download(args: argparse.Namespace) -> None:
    """Tai 1 cap T1/T2 cu the theo item ID."""
    client = STACClient(args.stac_url)
    print(f"[1/3] Lay item T1: {args.t1_id}")
    t1_item = client.get_item(args.collection, args.t1_id)
    print(f"[2/3] Lay item T2: {args.t2_id}")
    t2_item = client.get_item(args.collection, args.t2_id)

    if not t1_item or not t2_item:
        print("[Loi] Khong tim thay mot hoac ca hai items tren STAC.")
        return

    t1_info = extract_item_info(t1_item)
    t2_info = extract_item_info(t2_item)
    try:
        downloader = S3Downloader()
    except RuntimeError as e:
        print(f"[Loi] {e}")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_keys = args.asset_keys.split(",") if args.asset_keys else None

    print("\n[3/3] Download T1 assets")
    downloader.download_item_assets(t1_info["assets"], str(out_dir), f"{args.t1_prefix}{t1_info['id']}", asset_keys)

    print("\nDownload T2 assets")
    downloader.download_item_assets(t2_info["assets"], str(out_dir), f"{args.t2_prefix}{t2_info['id']}", asset_keys)

    print(f"\nOK. Du lieu luu tai: {out_dir.resolve()}")


def cmd_download_href(args: argparse.Namespace) -> None:
    """Tai truc tiep 1 href."""
    try:
        downloader = S3Downloader()
    except RuntimeError as e:
        print(f"[Loi] {e}")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = args.filename or Path(urlparse(args.href).path).name
    local_path = str(out_dir / filename)
    downloader.download_from_href(args.href, local_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STAC Query + Best Pair Selection + S3 Pipeline cho ISSM_SAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Lenh con")

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--stac-url", default=DEFAULT_STAC_API, help="STAC API URL (required if STAC_API_URL is not set)")
        p.add_argument("--collection", default=DEFAULT_COLLECTION, help="STAC collection ID")
        p.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
        p.add_argument("--geojson", default=None, help="AOI GeoJSON path (uu tien hon --bbox)")
        p.add_argument("--datetime", default=None, help="Khoang thoi gian, VD: 2025-01-01/2025-12-31")
        p.add_argument("--limit", type=int, default=300, help="So item toi da query")
        p.add_argument("--orbit", default=None, help="Filter item theo orbit_state neu can debug")
        p.add_argument("--rel-orbit", type=int, default=None, help="Filter item theo relative_orbit neu can debug")
        p.add_argument("--pols", default="VV,VH", help="Polarizations yeu cau, VD: VV hoac VV,VH")

    def add_pair_rules(p: argparse.ArgumentParser) -> None:
        p.add_argument("--min-overlap", type=float, default=0.0, help="Tham so legacy cho bbox overlap; hien chi dung de report/ranking")
        p.add_argument("--min-aoi-coverage", type=float, default=0.0, help="AOI geometry coverage toi thieu cho moi anh; hard gate la coverage > nguong")
        p.add_argument("--min-delta-hours", type=float, default=24.0, help="Delta t toi thieu (gio)")
        p.add_argument("--max-delta-days", type=int, default=10, help="Delta t toi da (ngay)")
        p.add_argument("--same-orbit-direction", action="store_true", help="Chi chap nhan cap co cung orbit_state (ascending/descending)")
        p.add_argument("--strict-slice", action="store_true", help="Legacy flag; pairing hien tai khong group theo slice")
        p.add_argument("--top-k", type=int, default=10, help="So cap hien thi")

    p_list = subparsers.add_parser("list", help="Liet ke item STAC phu hop")
    add_common_args(p_list)
    p_list.set_defaults(func=cmd_list)

    p_pair = subparsers.add_parser("pair", help="Tim cac cap T1/T2 hop le")
    add_common_args(p_pair)
    add_pair_rules(p_pair)
    p_pair.set_defaults(func=cmd_pair)

    p_anchor = subparsers.add_parser(
        "suggest-anchor",
        help="De xuat anchor train-like tu STAC timeline khi khong co san system_t1/system_t2",
    )
    add_common_args(p_anchor)
    add_pair_rules(p_anchor)
    p_anchor.add_argument("--window-before-days", type=float, default=30.0, help="So ngay cua window truoc anchor")
    p_anchor.add_argument("--window-after-days", type=float, default=30.0, help="So ngay cua window sau anchor")
    p_anchor.add_argument("--min-scenes-per-window", type=int, default=1, help="So scene unique toi thieu moi window")
    p_anchor.add_argument("--pick-index", type=int, default=1, help="Chon anchor thu N trong bang rank (1-based)")
    p_anchor.add_argument("--save-manifest", default=None, help="Duong dan luu train-like anchor manifest JSON")
    p_anchor.set_defaults(func=cmd_suggest_anchor)

    p_prepare = subparsers.add_parser("prepare", help="Chon cap tot nhat + manifest + tuy chon download/probe")
    add_common_args(p_prepare)
    add_pair_rules(p_prepare)
    p_prepare.add_argument("--pick-index", type=int, default=1, help="Chon cap thu N trong bang rank (1-based)")
    p_prepare.add_argument("--auto-relax", action="store_true", help="Neu strict khong co pair, tu thu profile balanced/loose")
    p_prepare.add_argument("--save-manifest", default=None, help="Duong dan luu manifest JSON")
    p_prepare.add_argument("--probe-rasterio", action="store_true", help="Thu mo href bang rasterio de check")
    p_prepare.add_argument("--download", action="store_true", help="Download cap duoc chon")
    p_prepare.add_argument(
        "--full-item",
        action="store_true",
        help="Tai toan bo item thay vi cat subset theo AOI giao item",
    )
    p_prepare.add_argument("--out-dir", default="data/input", help="Thu muc download cho prepare")
    p_prepare.add_argument("--t1-prefix", default="s1t1_", help="Prefix filename T1")
    p_prepare.add_argument("--t2-prefix", default="s1t2_", help="Prefix filename T2")
    p_prepare.set_defaults(func=cmd_prepare)

    p_batch_items = subparsers.add_parser(
        "download-aoi-matches",
        help="Batch download item matches cho nhom AOI goc",
    )
    add_common_args(p_batch_items)
    p_batch_items.add_argument("--geojson-dir", default="geojson", help="Thu muc chua AOI geojson")
    p_batch_items.add_argument("--glob", default="*.geojson", help="Mau file geojson")
    p_batch_items.add_argument("--only", default=None, help="Chi xu ly cac ten file (comma-separated)")
    p_batch_items.add_argument("--skip-generated", action="store_true", help="Bo qua thu muc generated_aoi")
    p_batch_items.add_argument("--max-items-per-aoi", type=int, default=0, help="Gioi han so items moi AOI (0=tat ca)")
    p_batch_items.add_argument(
        "--full-item",
        action="store_true",
        help="Tai toan bo item thay vi cat subset theo AOI giao item",
    )
    p_batch_items.add_argument("--out-dir", default="data/original_aoi_items", help="Thu muc tai du lieu")
    p_batch_items.add_argument("--report-dir", default="docs/geojson_scan/batch_download", help="Thu muc luu report")
    p_batch_items.add_argument("--dry-run", action="store_true", help="Chi lap ke hoach, khong tai file")
    p_batch_items.set_defaults(func=cmd_download_aoi_matches)

    p_batch_pairs = subparsers.add_parser(
        "download-generated-pairs",
        help="Batch download cap T1/T2 cho generated AOI",
    )
    add_common_args(p_batch_pairs)
    add_pair_rules(p_batch_pairs)
    p_batch_pairs.add_argument("--geojson-dir", default="geojson/generated_aoi", help="Thu muc generated AOI")
    p_batch_pairs.add_argument("--glob", default="*.geojson", help="Mau file geojson")
    p_batch_pairs.add_argument("--only", default=None, help="Chi xu ly cac ten file (comma-separated)")
    p_batch_pairs.add_argument("--auto-relax", action="store_true", help="Neu strict fail thi thu balanced/loose")
    p_batch_pairs.add_argument("--pick-index", type=int, default=1, help="Chon cap thu N trong ranking")
    p_batch_pairs.add_argument(
        "--full-item",
        action="store_true",
        help="Tai toan bo item thay vi cat subset theo AOI giao item",
    )
    p_batch_pairs.add_argument("--out-dir", default="data/generated_aoi_pairs", help="Thu muc tai du lieu")
    p_batch_pairs.add_argument("--report-dir", default="docs/geojson_scan/batch_download", help="Thu muc luu report")
    p_batch_pairs.add_argument("--t1-prefix", default="s1t1_", help="Prefix filename T1")
    p_batch_pairs.add_argument("--t2-prefix", default="s1t2_", help="Prefix filename T2")
    p_batch_pairs.add_argument("--dry-run", action="store_true", help="Chi lap ke hoach, khong tai file")
    p_batch_pairs.set_defaults(func=cmd_download_generated_pairs)

    p_dl = subparsers.add_parser("download", help="Tai cap T1/T2 bang item IDs")
    p_dl.add_argument("--stac-url", default=DEFAULT_STAC_API, help="STAC API URL (required if STAC_API_URL is not set)")
    p_dl.add_argument("--collection", default=DEFAULT_COLLECTION)
    p_dl.add_argument("--t1-id", required=True, help="STAC item ID cho T1")
    p_dl.add_argument("--t2-id", required=True, help="STAC item ID cho T2")
    p_dl.add_argument("--out-dir", default="data/input", help="Thu muc luu")
    p_dl.add_argument("--t1-prefix", default="s1t1_", help="Prefix filename T1")
    p_dl.add_argument("--t2-prefix", default="s1t2_", help="Prefix filename T2")
    p_dl.add_argument("--asset-keys", default=None, help="Chi tai cac asset keys, VD: vv,vh")
    p_dl.set_defaults(func=cmd_download)

    p_href = subparsers.add_parser("download-href", help="Tai truc tiep 1 href")
    p_href.add_argument("href", help="s3://bucket/key hoac http(s)://...")
    p_href.add_argument("--out-dir", default="data/input", help="Thu muc luu")
    p_href.add_argument("--filename", default=None, help="Ten file local")
    p_href.set_defaults(func=cmd_download_href)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

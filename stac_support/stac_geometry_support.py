from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyproj import Geod
from shapely.geometry import GeometryCollection, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from stac_support.stac_item_support import extract_item_info, summarize_unique_scenes

WGS84_GEOD = Geod(ellps="WGS84")
_BBOX_WARN_TOL = 1e-7

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

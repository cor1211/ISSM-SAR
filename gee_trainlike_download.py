#!/usr/bin/env python3
"""Helper utilities for canonical GEE representative composite runtime paths.

The old standalone pair-aligned compare CLI has been retired from the core
runtime. This module now keeps only the helpers still used by `sar_pipeline.py`
and the canonical GEE tooling.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from query_stac_download import canonical_bbox_from_geometry

try:
    import ee
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("earthengine-api is required. Install requirements first.") from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gee_trainlike")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    return dt1 + (dt2 - dt1) / 2


def to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_anchor(manifest: Dict[str, Any], strategy: str) -> datetime:
    if manifest.get("anchor_datetime") and (
        "t1_datetime" not in manifest
        or "t2_datetime" not in manifest
        or str(strategy or "").strip().lower() in {"anchor", "precomputed", "manifest"}
    ):
        return parse_utc(manifest["anchor_datetime"])
    t1 = parse_utc(manifest["t1_datetime"])
    t2 = parse_utc(manifest["t2_datetime"])
    key = str(strategy or "midpoint").strip().lower()
    if key in {"midpoint", "midpoint_pair", "stac_midpoint_pair"}:
        return midpoint_datetime(t1, t2)
    if key == "t1":
        return t1
    if key == "t2":
        return t2
    if key in {"anchor", "precomputed", "manifest"} and manifest.get("anchor_datetime"):
        return parse_utc(manifest["anchor_datetime"])
    raise ValueError(f"Unsupported anchor strategy: {strategy}")


def resolve_window(anchor: datetime, start_day: float, end_day: float) -> Tuple[datetime, datetime]:
    start = anchor + timedelta(days=float(start_day))
    end = anchor + timedelta(days=float(end_day))
    if end <= start:
        raise ValueError(f"Invalid window [{start_day}, {end_day}] around anchor {anchor}")
    return start, end


def ee_geometry_from_geojson(geometry: Dict[str, Any]) -> ee.Geometry:
    return ee.Geometry(geometry)


def rectangle_geometry(bbox: List[float]) -> ee.Geometry:
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def clip_geometry(mode: str, aoi_bbox: List[float], aoi_geometry: Dict[str, Any]) -> ee.Geometry:
    key = str(mode or "bbox").strip().lower()
    if key == "bbox":
        return rectangle_geometry(aoi_bbox)
    if key == "geometry":
        return ee_geometry_from_geojson(aoi_geometry)
    raise ValueError(f"Unsupported clip mode: {mode}")


def build_collection(
    collection_id: str,
    filter_geom: ee.Geometry,
    start_dt: datetime,
    end_dt: datetime,
    orbit_pass: str,
) -> ee.ImageCollection:
    coll = (
        ee.ImageCollection(collection_id)
        .filterBounds(filter_geom)
        .filterDate(start_dt.isoformat(), end_dt.isoformat())
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )
    if orbit_pass.upper() != "BOTH":
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass.upper()))
    return coll


def collection_summary(collection: ee.ImageCollection, max_items: int = 20) -> Dict[str, Any]:
    count = int(collection.size().getInfo())
    scenes: List[Dict[str, Any]] = []
    if count > 0:
        sorted_coll = collection.sort("system:time_start")
        items = sorted_coll.toList(min(count, max_items))
        for idx in range(min(count, max_items)):
            image = ee.Image(items.get(idx))
            ts_millis = image.date().millis().getInfo()
            dt = datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)
            scenes.append(
                {
                    "gee_id": str(image.get("system:index").getInfo()),
                    "gee_datetime": to_rfc3339(dt),
                    "orbit_pass": image.get("orbitProperties_pass").getInfo(),
                }
            )
    return {
        "count": count,
        "scenes": scenes,
        "scenes_truncated": count > max_items,
    }


def collection_scene_items(
    collection: ee.ImageCollection,
    aoi_bbox: List[float],
    max_items: int | None = None,
) -> List[Dict[str, Any]]:
    """Convert a GEE collection into STAC-like scene items for shared selection logic."""
    count = int(collection.size().getInfo())
    if count == 0:
        return []

    limit = count if max_items is None else min(count, int(max_items))
    sorted_coll = collection.sort("system:time_start")
    items = sorted_coll.toList(limit)
    scenes: List[Dict[str, Any]] = []

    for idx in range(limit):
        image = ee.Image(items.get(idx))
        ts_millis = image.date().millis().getInfo()
        dt = datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)
        system_index = str(image.get("system:index").getInfo())
        orbit_pass = image.get("orbitProperties_pass").getInfo()
        relative_orbit = image.get("relativeOrbitNumber_start").getInfo()
        slice_number = image.get("sliceNumber").getInfo()
        platform = system_index.split("_", 1)[0] if system_index else ""
        geom_info = image.geometry().getInfo()
        bbox = canonical_bbox_from_geometry(geom_info)
        scenes.append(
            {
                "id": system_index,
                "bbox": bbox,
                "geometry": geom_info,
                "properties": {
                    "datetime": to_rfc3339(dt),
                    "platform": platform,
                    "sat:orbit_state": str(orbit_pass or "").lower(),
                    "sat:relative_orbit": relative_orbit,
                    "s1:slice_number": slice_number,
                    "sar:polarizations": ["VV", "VH"],
                    "sar:instrument_mode": "IW",
                    "sar:product_type": "GRD",
                },
            }
        )
    return scenes


def build_trainlike_image(
    collection: ee.ImageCollection,
    clip_geom: ee.Geometry,
    focal_radius_m: float,
) -> ee.Image:
    image = collection.select(["VV", "VH"]).median()
    if focal_radius_m and focal_radius_m > 0:
        image = image.focal_median(float(focal_radius_m), "circle", "meters")
    return image.clip(clip_geom)


def download_gee_image(image: ee.Image, params: Dict[str, Any], out_path: Path) -> Path:
    import requests

    url = image.getDownloadURL(params)
    logger.info("Downloading %s", out_path.name)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path

#!/usr/bin/env python3
"""Download a train-like GEE Sentinel-1 pair aligned to a system-selected STAC pair.

This tool reproduces the train/test-style GEE preprocessing used by the current
model: time-window composites, median reduction, focal median smoothing, and
EPSG:3857 export, while still anchoring the temporal windows to the system pair.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from gee_compare_download import (
    build_target_grid,
    ensure_dir,
    init_gee,
    load_system_manifest,
    parse_utc,
    rewrite_with_descriptions,
    to_jsonable,
    validate_pair,
)
from query_stac_download import load_geojson_aoi

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


def build_trainlike_image(
    collection: ee.ImageCollection,
    clip_geom: ee.Geometry,
    focal_radius_m: float,
) -> ee.Image:
    image = collection.select(["VV", "VH"]).median()
    if focal_radius_m and focal_radius_m > 0:
        image = image.focal_median(float(focal_radius_m), "circle", "meters")
    return image.clip(clip_geom)


def build_export_params(
    export_name: str,
    grid: Dict[str, Any],
    band_names: List[str],
) -> Dict[str, Any]:
    return {
        "name": export_name,
        "format": "GEO_TIFF",
        "filePerBand": False,
        "bands": band_names,
        "crs": grid["crs"],
        "crs_transform": grid["crs_transform"],
        "dimensions": [grid["width"], grid["height"]],
    }


def download_gee_image(image: ee.Image, params: Dict[str, Any], out_path: Path) -> Path:
    import requests

    url = image.getDownloadURL(params)
    logger.info("Downloading %s", out_path.name)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def run_inference(run_dir: Path, pair_id: str, infer_config_path: str | Path) -> Path:
    from infer_production import SARInferencer, load_yaml

    cfg = load_yaml(infer_config_path)
    cfg["input"]["input_dir"] = str(run_dir)
    cfg["input"]["t1_prefix"] = "s1t1_"
    cfg["input"]["t2_prefix"] = "s1t2_"
    out_dir = run_dir / "infer_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_path"] = str(out_dir)
    cfg["output"]["save_name"] = f"{pair_id}_SR_x2.tif"
    inferencer = SARInferencer(cfg)
    t1_path = run_dir / f"s1t1_{pair_id}.tif"
    t2_path = run_dir / f"s1t2_{pair_id}.tif"
    return inferencer.run_pair_from_multiband_files(pair_id, t1_path, t2_path, out_path=out_dir / f"{pair_id}_SR_x2.tif")


def write_report(run_dir: Path, report: Dict[str, Any]) -> Tuple[Path, Path]:
    json_path = run_dir / "gee_trainlike_report.json"
    md_path = run_dir / "gee_trainlike_report.md"
    report_jsonable = to_jsonable(report)
    json_path.write_text(json.dumps(report_jsonable, indent=2, ensure_ascii=False), encoding="utf-8")

    sys_ref = report_jsonable["system_reference"]
    train_cfg = report_jsonable["trainlike"]
    infer = report_jsonable.get("inference")
    lines = [
        "# GEE Train-Like Compare Report",
        "",
        "## System Reference",
        "",
        f"- AOI: `{report_jsonable['aoi_geojson']}`",
        f"- Pair ID: `{sys_ref['pair_id']}`",
        f"- T1 exact datetime: `{sys_ref['t1_datetime']}`",
        f"- T2 exact datetime: `{sys_ref['t2_datetime']}`",
        f"- AOI bbox: `{sys_ref['aoi_bbox']}`",
        "",
        "## Train-Like Windows",
        "",
        f"- Anchor strategy: `{train_cfg['anchor_strategy']}`",
        f"- Anchor datetime: `{train_cfg['anchor_datetime']}`",
        f"- T1 window: `{train_cfg['t1_window']['start']}` -> `{train_cfg['t1_window']['end']}`",
        f"- T2 window: `{train_cfg['t2_window']['start']}` -> `{train_cfg['t2_window']['end']}`",
        f"- Orbit pass mode: `{train_cfg['orbit_pass']}`",
        f"- Reducer: `{train_cfg['reducer']}`",
        f"- Focal median radius (m): `{train_cfg['focal_median_radius_m']}`",
        f"- Clip mode: `{train_cfg['clip_mode']}`",
        "",
        "## Source Collections",
        "",
        f"- T1 contributing images: `{train_cfg['t1_collection']['count']}`",
        f"- T2 contributing images: `{train_cfg['t2_collection']['count']}`",
        "",
        "## Outputs",
        "",
        f"- T1 file: `{report_jsonable['output_files']['t1']}`",
        f"- T2 file: `{report_jsonable['output_files']['t2']}`",
        f"- Validation same grid: `{report_jsonable['validation']['same_grid']}`",
        f"- Validation infer scan OK: `{report_jsonable['validation']['pair_scan_ok']}`",
    ]
    if infer:
        lines.extend(
            [
                "",
                "## Inference",
                "",
                f"- Infer config: `{infer['infer_config_path']}`",
                f"- Output: `{infer['output_path']}`",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a train-like GEE Sentinel-1 pair aligned to a system-selected STAC pair")
    parser.add_argument("--geojson", required=True, help="AOI GeoJSON path")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yaml", help="Pipeline config used to resolve the system pair")
    parser.add_argument("--gee-config", default="config/gee_compare_config.yaml", help="GEE compare config path")
    parser.add_argument("--manifest", default=None, help="Optional system manifest path to lock the reference pair")
    parser.add_argument("--gee-project", default=None, help="Earth Engine project id")
    parser.add_argument("--out-dir", default=None, help="Override output root directory")
    parser.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() if initialization fails")
    parser.add_argument("--run-infer", action="store_true", help="Run infer_production.py logic on the downloaded train-like pair")
    parser.add_argument("--infer-config", default="config/infer_config.yaml", help="Inference config path when --run-infer is used")
    parser.add_argument("--anchor-strategy", default=None, help="Override trainlike.anchor_strategy")
    parser.add_argument("--t1-window-days", nargs=2, type=float, default=None, metavar=("START", "END"), help="Override trainlike T1 window days relative to anchor")
    parser.add_argument("--t2-window-days", nargs=2, type=float, default=None, metavar=("START", "END"), help="Override trainlike T2 window days relative to anchor")
    parser.add_argument("--orbit-pass", default=None, help="Override trainlike.orbit_pass (BOTH/ASCENDING/DESCENDING)")
    parser.add_argument("--focal-median-radius-m", type=float, default=None, help="Override trainlike.focal_median_radius_m")
    parser.add_argument("--clip-mode", default=None, help="Override trainlike.clip_mode (bbox/geometry)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.gee_config).read_text(encoding="utf-8")) or {}
    gee_cfg = config.get("gee", {})
    train_cfg = config.get("trainlike", {})
    pipeline_config, system_manifest, system_diag = load_system_manifest(args.geojson, args.pipeline_config, args.manifest)
    aoi_bbox, aoi_geometry = load_geojson_aoi(args.geojson)

    gee_project = args.gee_project or gee_cfg.get("project")
    if not gee_project:
        raise RuntimeError("Missing GEE project. Provide --gee-project or gee.project in config/gee_compare_config.yaml")
    init_gee(gee_project, authenticate=args.authenticate)

    target_crs = gee_cfg.get("export_crs", "EPSG:3857")
    target_scale = float(gee_cfg.get("export_scale", 10.0))
    band_names = list(gee_cfg.get("band_names", ["VV", "VH"]))
    output_descs = list(gee_cfg.get("output_band_descriptions", ["S1_VV", "S1_VH"]))

    anchor_strategy = (
        args.anchor_strategy
        or system_manifest.get("anchor_strategy")
        or train_cfg.get("anchor_strategy", "midpoint")
    )
    anchor_dt = resolve_anchor(system_manifest, anchor_strategy)
    manifest_t1_window = None
    manifest_t2_window = None
    if system_manifest.get("window_before_days") is not None and system_manifest.get("window_after_days") is not None:
        manifest_t1_window = [0.0, float(system_manifest["window_after_days"])]
        manifest_t2_window = [-float(system_manifest["window_before_days"]), 0.0]

    t1_window_days = args.t1_window_days or manifest_t1_window or train_cfg.get("t1_window_days", [0, 7])
    t2_window_days = args.t2_window_days or manifest_t2_window or train_cfg.get("t2_window_days", [-7, 0])
    orbit_pass = str(args.orbit_pass or train_cfg.get("orbit_pass", "BOTH"))
    focal_radius_m = float(args.focal_median_radius_m if args.focal_median_radius_m is not None else train_cfg.get("focal_median_radius_m", 15.0))
    clip_mode_name = args.clip_mode or train_cfg.get("clip_mode", "bbox")

    t1_start, t1_end = resolve_window(anchor_dt, float(t1_window_days[0]), float(t1_window_days[1]))
    t2_start, t2_end = resolve_window(anchor_dt, float(t2_window_days[0]), float(t2_window_days[1]))

    filter_geom = ee_geometry_from_geojson(aoi_geometry)
    clip_geom = clip_geometry(clip_mode_name, aoi_bbox, aoi_geometry)
    grid = build_target_grid(system_manifest["aoi_bbox"], target_crs, target_scale, target_scale)

    collection_id = gee_cfg.get("collection", "COPERNICUS/S1_GRD")
    t1_coll = build_collection(collection_id, filter_geom, t1_start, t1_end, orbit_pass)
    t2_coll = build_collection(collection_id, filter_geom, t2_start, t2_end, orbit_pass)
    t1_summary = collection_summary(t1_coll, max_items=int(train_cfg.get("max_scene_report", 20)))
    t2_summary = collection_summary(t2_coll, max_items=int(train_cfg.get("max_scene_report", 20)))
    if t1_summary["count"] == 0:
        raise RuntimeError("No GEE images found for train-like T1 window.")
    if t2_summary["count"] == 0:
        raise RuntimeError("No GEE images found for train-like T2 window.")

    t1_image = build_trainlike_image(t1_coll, clip_geom, focal_radius_m)
    t2_image = build_trainlike_image(t2_coll, clip_geom, focal_radius_m)

    run_root = ensure_dir(args.out_dir or config.get("output", {}).get("trainlike_root_dir", "runs/gee_trainlike_compare"))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = ensure_dir(run_root / Path(args.geojson).stem / run_id)
    manifest_copy_path = run_dir / "system_reference_manifest.json"
    manifest_copy_path.write_text(json.dumps(system_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    pair_id = system_manifest["pair_id"]
    t1_path = run_dir / f"s1t1_{pair_id}.tif"
    t2_path = run_dir / f"s1t2_{pair_id}.tif"
    download_gee_image(t1_image, build_export_params(f"s1t1_{pair_id}", grid, band_names), t1_path)
    download_gee_image(t2_image, build_export_params(f"s1t2_{pair_id}", grid, band_names), t2_path)
    rewrite_with_descriptions(t1_path, output_descs, grid)
    rewrite_with_descriptions(t2_path, output_descs, grid)
    validation = validate_pair(run_dir, pair_id, grid, output_descs)

    inference_info = None
    if args.run_infer:
        out_path = run_inference(run_dir, pair_id, args.infer_config)
        inference_info = {
            "infer_config_path": str(Path(args.infer_config).resolve()),
            "output_path": str(out_path),
        }

    report = {
        "aoi_geojson": str(Path(args.geojson).resolve()),
        "gee_project": gee_project,
        "system_reference_manifest": str(manifest_copy_path),
        "system_reference": {
            "pair_id": system_manifest["pair_id"],
            "t1_id": system_manifest.get("t1_id"),
            "t2_id": system_manifest.get("t2_id"),
            "t1_datetime": system_manifest.get("t1_datetime"),
            "t2_datetime": system_manifest.get("t2_datetime"),
            "aoi_bbox": system_manifest["aoi_bbox"],
        },
        "trainlike": {
            "anchor_strategy": anchor_strategy,
            "anchor_datetime": to_rfc3339(anchor_dt),
            "orbit_pass": orbit_pass.upper(),
            "reducer": "median",
            "focal_median_radius_m": focal_radius_m,
            "clip_mode": clip_mode_name,
            "t1_window": {"start": to_rfc3339(t1_start), "end": to_rfc3339(t1_end)},
            "t2_window": {"start": to_rfc3339(t2_start), "end": to_rfc3339(t2_end)},
            "t1_collection": t1_summary,
            "t2_collection": t2_summary,
        },
        "diagnostics": {
            "system_pair_resolution": system_diag,
        },
        "export": {
            "collection": collection_id,
            "crs": grid["crs"],
            "scale": target_scale,
            "width": grid["width"],
            "height": grid["height"],
            "transform": list(grid["transform"]),
            "band_names": band_names,
            "band_descriptions": output_descs,
        },
        "output_files": {"t1": str(t1_path), "t2": str(t2_path)},
        "validation": validation,
        "inference": inference_info,
    }
    json_path, md_path = write_report(run_dir, report)
    logger.info("Saved report JSON: %s", json_path)
    logger.info("Saved report MD: %s", md_path)


if __name__ == "__main__":
    main()

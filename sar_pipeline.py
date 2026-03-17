#!/usr/bin/env python3
"""
AOI GeoJSON -> STAC pair selection -> raw single-band download -> optional staged 2-band inputs -> SR inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import median_filter
import yaml

from query_stac_download import (
    DEFAULT_COLLECTION,
    DEFAULT_STAC_API,
    STACClient,
    S3Downloader,
    build_manifest_for_pair,
    build_trainlike_anchor_manifest,
    build_selected_pair_info,
    collect_anchor_window_items,
    collect_items_with_filters,
    diagnose_no_pair,
    download_manifest_pair,
    extract_item_info,
    format_duration_human,
    item_scene_key,
    load_geojson_aoi,
    parse_required_pols,
    select_asset_href,
    search_pairs_sorted,
    suggest_trainlike_anchors,
)


logger = logging.getLogger("sar_pipeline")


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
            window_before_days=before_days,
            window_after_days=after_days,
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 1.0)),
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
            _, href = asset_info
            local_path = out_dir / f"{dt_token}__{item_token}__{pol.lower()}.tif"
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
    out_file = write_multiband_geotiff(
        out_path,
        np.stack(bands, axis=0),
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
) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any], str, Optional[Dict[str, Any]]]:
    pair_cfg = config.get("pairing", {})
    query_args = build_query_namespace(config, geojson_path)
    required_pols = parse_required_pols(query_args.pols)
    if required_pols != ["VV", "VH"]:
        raise ValueError("Pipeline end-to-end currently requires pols=VV,VH.")

    items, aoi_bbox = collect_items_with_filters(client, query_args, required_pols)
    strict_pairs = search_pairs_sorted(
        items=items,
        aoi_bbox=aoi_bbox,
        min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 1.0)),
        max_delta_days=int(pair_cfg.get("max_delta_days", 10)),
        min_delta_hours=float(pair_cfg.get("min_delta_hours", 24.0)),
        strict_slice=bool(pair_cfg.get("strict_slice", False)),
        same_orbit_direction=bool(pair_cfg.get("same_orbit_direction", False)),
    )
    if strict_pairs:
        return items, aoi_bbox, strict_pairs[0], "strict", None

    diag = diagnose_no_pair(
        items=items,
        aoi_bbox=aoi_bbox,
        strict_slice=bool(pair_cfg.get("strict_slice", False)),
        min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 1.0)),
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
            min_overlap=float(pair_cfg.get("min_overlap", 0.0)),
            min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 1.0)),
            max_delta_days=max_days,
            min_delta_hours=float(pair_cfg.get("min_delta_hours", 24.0)),
            strict_slice=bool(pair_cfg.get("strict_slice", False)),
            same_orbit_direction=bool(pair_cfg.get("same_orbit_direction", False)),
        )
        if candidates:
            return items, aoi_bbox, candidates[0], profile_name, diag

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


def write_run_summary(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Path]:
    json_path = run_dir / "run_summary.json"
    md_path = run_dir / "run_summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pair = summary["selected_pair"]
    report_lines = [
        "# SAR Pipeline Run Summary",
        "",
        "## Inputs",
        "",
        f"- AOI: `{summary['aoi_geojson']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Selection profile: `{summary['selection_profile']}`",
        f"- Selection priority: `latest_input_datetime`",
        f"- Latest input datetime: `{pair['latest_input_datetime']}`",
        f"- Delta exact: `{pair['delta_human']}`",
        f"- Delta days: `{pair['delta_days']:.6f}`",
        f"- AOI bbox coverage T1: `{pair['aoi_bbox_coverage_t1']:.6f}`",
        f"- AOI bbox coverage T2: `{pair['aoi_bbox_coverage_t2']:.6f}`",
        f"- BBox overlap (diagnostic): `{pair['bbox_overlap']:.6f}`",
    ]
    compatibility = summary.get("compatibility")
    if compatibility:
        report_lines.extend(
            [
                "",
                "## Compatibility",
                "",
                f"- Trained input profile: `{compatibility.get('trained_input_profile')}`",
                f"- Current download profile: `{compatibility.get('current_download_profile')}`",
                f"- Domain mismatch: `{compatibility.get('domain_mismatch')}`",
                f"- Allow mismatch: `{compatibility.get('allow_domain_mismatch')}`",
            ]
        )
        if compatibility.get("message"):
            report_lines.append(f"- Note: {compatibility['message']}")
    report_lines.extend(
        [
        "",
        "## Files",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Raw dir: `{summary['raw_dir']}`",
        f"- Staging dir: `{summary['staging_dir']}`",
        f"- Output GeoTIFF: `{summary['output_tif']}`",
        "",
        "## Pair",
        "",
        f"- Pair ID: `{pair['pair_id']}`",
        f"- T1 ID: `{pair['t1_id']}`",
        f"- T2 ID: `{pair['t2_id']}`",
        f"- T1 datetime: `{pair['t1_datetime']}`",
        f"- T2 datetime: `{pair['t2_datetime']}`",
        f"- Orbit direction: `{pair.get('t1_orbit_state')}` -> `{pair.get('t2_orbit_state')}`",
        f"- Relative orbit: `{pair.get('relative_orbit')}`",
        f"- Slice number: `{pair.get('slice_number')}`",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return json_path, md_path


def write_trainlike_run_summary(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write JSON + Markdown summary for the STAC-only composite workflow."""
    json_path = run_dir / "run_summary.json"
    md_path = run_dir / "run_summary.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    anchor = summary["anchor"]
    comp = summary["composite"]
    report_lines = [
        "# STAC Train-Like Pipeline Run Summary",
        "",
        "## Inputs",
        "",
        f"- AOI: `{summary['aoi_geojson']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Workflow mode: `{summary['workflow_mode']}`",
        f"- STAC URL: `{summary['run_config']['stac_url']}`",
        f"- Collection: `{summary['run_config']['collection']}`",
        f"- Datetime filter: `{summary['run_config']['datetime']}`",
        "",
        "## Anchor",
        "",
        f"- Anchor strategy: `{anchor['anchor_strategy']}`",
        f"- Selection priority: `{anchor.get('selection_priority')}`",
        f"- Anchor datetime: `{anchor['anchor_datetime']}`",
        f"- Latest input datetime: `{anchor.get('latest_input_datetime')}`",
        f"- Window before days: `{anchor['window_before_days']}`",
        f"- Window after days: `{anchor['window_after_days']}`",
        f"- Required min scenes/window: `{anchor['required_scene_count']}`",
        f"- Support T1: `{anchor.get('support_t1_id')}`",
        f"- Support T2: `{anchor.get('support_t2_id')}`",
        f"- Support delta days: `{anchor.get('support_pair_delta_days')}`",
        f"- Pre unique scenes: `{anchor['pre_scene_count']}`",
        f"- Post unique scenes: `{anchor['post_scene_count']}`",
        "",
        "## Composite",
        "",
        f"- Target CRS: `{comp['grid']['crs']}`",
        f"- Target size: `{comp['grid']['width']}x{comp['grid']['height']}`",
        f"- Target transform: `{comp['grid']['transform']}`",
        f"- Focal median radius (m): `{summary['run_config']['focal_median_radius_m']}`",
        f"- Pre VV scenes: `{comp['pre']['scene_counts']['vv']}`",
        f"- Pre VH scenes: `{comp['pre']['scene_counts']['vh']}`",
        f"- Post VV scenes: `{comp['post']['scene_counts']['vv']}`",
        f"- Post VH scenes: `{comp['post']['scene_counts']['vh']}`",
        "",
        "## Files",
        "",
        f"- Anchor manifest: `{summary['anchor_manifest_path']}`",
        f"- Window raw dir: `{summary['window_raw_dir']}`",
        f"- Composite dir: `{summary['composite_dir']}`",
        f"- T1 composite: `{summary['t1_composite_path']}`",
        f"- T2 composite: `{summary['t2_composite_path']}`",
        f"- Output GeoTIFF: `{summary['output_tif']}`",
    ]
    compatibility = summary.get("compatibility")
    if compatibility:
        report_lines.extend(
            [
                "",
                "## Compatibility",
                "",
                f"- Trained input profile: `{compatibility.get('trained_input_profile')}`",
                f"- Current download profile: `{compatibility.get('current_download_profile')}`",
                f"- Domain mismatch: `{compatibility.get('domain_mismatch')}`",
            ]
        )
        if compatibility.get("message"):
            report_lines.append(f"- Note: {compatibility['message']}")

    report_lines.extend(["", "## Window Scenes", ""])
    for label, scenes in (("Pre / S1T2", anchor["pre_scenes"]), ("Post / S1T1", anchor["post_scenes"])):
        report_lines.append(f"### {label}")
        for scene in scenes:
            report_lines.append(
                f"- `{scene['datetime']}` | `{scene['id']}` | `{scene.get('platform')}` | "
                f"`{scene.get('orbit_state')}` | rel_orbit=`{scene.get('relative_orbit')}` | slice=`{scene.get('slice_number')}`"
            )
        report_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return json_path, md_path


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

    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    dl_cfg = config.get("download", {})
    staging_cfg = config.get("staging", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="stac_measurement_raw")

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("Pipeline end-to-end currently requires pols=VV,VH.")

    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = ensure_dir(run_root / aoi_path.stem / run_id)
    raw_dir = ensure_dir(run_dir / dl_cfg.get("raw_dir_name", "raw"))
    staging_dir = run_dir / staging_cfg.get("dir_name", "staging")
    output_dir = ensure_dir(run_dir / out_cfg.get("output_dir_name", "output"))

    client = STACClient(stac_cfg.get("url", DEFAULT_STAC_API))
    items, aoi_bbox, chosen_pair, selected_profile, diag = select_best_pair(client, config, str(aoi_path))
    manifest = build_manifest_for_pair(chosen_pair, required_pols)
    if manifest is None:
        raise RuntimeError("Selected pair is missing required VV/VH asset hrefs.")
    manifest["selection_profile"] = selected_profile
    manifest["aoi_geojson"] = str(aoi_path)
    manifest["aoi_bbox"] = aoi_bbox

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _, aoi_geometry = load_geojson_aoi(aoi_path)
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

    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    if device:
        infer_config["device"] = device

    output_tif = output_dir / f"{aoi_path.stem}__{manifest['pair_id']}_SR_x2.tif"
    cache_enabled = cache_staging or bool(staging_cfg.get("cache_aligned_inputs", False))
    inferencer = SARInferencer(infer_config)
    inferencer.run_pair_from_single_band_files(
        t1_vv=expected["t1_vv"],
        t1_vh=expected["t1_vh"],
        t2_vv=expected["t2_vv"],
        t2_vh=expected["t2_vh"],
        out_path=output_tif,
        config={
            "resampling": staging_cfg.get("resampling", "bilinear"),
            "target_crs": staging_cfg.get("target_crs"),
            "target_resolution": staging_cfg.get("target_resolution"),
        },
        cache_dir=staging_dir if cache_enabled else None,
        identifier=manifest["pair_id"],
    )

    selected_pair_info = build_selected_pair_info(chosen_pair, manifest)
    summary = {
        "workflow_mode": "exact_pair",
        "aoi_geojson": str(aoi_path),
        "run_dir": str(run_dir),
        "raw_dir": str(raw_dir),
        "staging_dir": str(staging_dir if cache_enabled else ""),
        "output_tif": str(output_tif),
        "selection_profile": selected_profile,
        "selected_pair": selected_pair_info,
        "manifest_path": str(manifest_path),
        "downloaded_files": [str(p) for p in downloaded_paths],
        "items_after_hard_filter": len(items),
        "run_config": {
            "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
            "limit": int(stac_cfg.get("limit", 300)),
            "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 1.0)),
            "min_delta_hours": float(pair_cfg.get("min_delta_hours", 24.0)),
            "max_delta_days": int(pair_cfg.get("max_delta_days", 10)),
            "selection_priority": "latest_input_datetime",
            "same_orbit_direction": bool(pair_cfg.get("same_orbit_direction", False)),
            "auto_relax": bool(pair_cfg.get("auto_relax", False)),
            "resampling": staging_cfg.get("resampling", "bilinear"),
            "target_crs": staging_cfg.get("target_crs"),
            "target_resolution": staging_cfg.get("target_resolution"),
            "cache_staging": cache_enabled,
            "device": infer_config.get("device"),
        },
    }
    if diag is not None:
        summary["initial_diagnostics"] = diag
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    summary["summary_json"] = str(run_dir / "run_summary.json")
    summary["summary_md"] = str(run_dir / "run_summary.md")
    write_run_summary(run_dir, summary)
    return summary


def run_stac_trainlike_pipeline(
    config: Dict[str, Any],
    geojson_path: str,
    output_root: Optional[str],
    cache_staging: bool,
    device: Optional[str],
) -> Dict[str, Any]:
    """AOI -> STAC timeline -> anchor -> multi-scene window download -> local composite -> inference."""
    try:
        from infer_production import SARInferencer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Please install the production inference environment before running sar_pipeline.py."
        ) from exc

    aoi_path = Path(geojson_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {aoi_path}")

    stac_cfg = config.get("stac", {})
    pair_cfg = config.get("pairing", {})
    train_cfg = config.get("trainlike", {})
    infer_cfg = config.get("inference", {})
    out_cfg = config.get("output", {})
    compatibility_info = check_domain_compatibility(config, current_profile_override="stac_trainlike_composite_db")

    required_pols = parse_required_pols(pair_cfg.get("pols", "VV,VH"))
    if required_pols != ["VV", "VH"]:
        raise ValueError("STAC train-like pipeline currently requires pols=VV,VH.")

    run_root = ensure_dir(output_root or out_cfg.get("root_dir", "runs/pipeline"))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = ensure_dir(run_root / aoi_path.stem / run_id)
    manifest_path = run_dir / "manifest.json"
    window_raw_dir = ensure_dir(run_dir / train_cfg.get("window_raw_dir_name", "window_raw"))
    composite_dir = ensure_dir(run_dir / train_cfg.get("composite_dir_name", "composite"))
    output_dir = ensure_dir(run_dir / out_cfg.get("output_dir_name", "output"))

    client = STACClient(stac_cfg.get("url", DEFAULT_STAC_API))
    query_args = build_query_namespace(config, str(aoi_path))
    items, aoi_bbox = collect_items_with_filters(client, query_args, required_pols)
    if not items:
        raise RuntimeError(f"No STAC items passed hard filters for {aoi_path}.")

    anchor_candidate, required_scene_count = choose_anchor_candidate(items, aoi_bbox, train_cfg, pair_cfg)
    manifest = build_trainlike_anchor_manifest(anchor_candidate, aoi_bbox, str(aoi_path), required_pols)
    manifest["required_scene_count"] = required_scene_count
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _, aoi_geometry = load_geojson_aoi(aoi_path)
    anchor_dt = datetime.fromisoformat(manifest["anchor_datetime"].replace("Z", "+00:00"))
    pre_items_full, post_items_full = collect_anchor_window_items(
        items=items,
        aoi_bbox=aoi_bbox,
        anchor_dt=anchor_dt,
        window_before_days=float(manifest["window_before_days"]),
        window_after_days=float(manifest["window_after_days"]),
        min_aoi_coverage=float(pair_cfg.get("min_aoi_coverage", 1.0)),
    )
    pre_items = dedupe_items_by_scene(pre_items_full)
    post_items = dedupe_items_by_scene(post_items_full)
    if len(pre_items) < required_scene_count or len(post_items) < required_scene_count:
        raise RuntimeError(
            "Selected anchor no longer satisfies the required scene count after item reconstruction."
        )

    downloader = S3Downloader()
    pre_paths = download_window_assets(pre_items, window_raw_dir / "pre", aoi_geometry, required_pols, downloader)
    post_paths = download_window_assets(post_items, window_raw_dir / "post", aoi_geometry, required_pols, downloader)

    target_crs = str(train_cfg.get("target_crs", "EPSG:3857"))
    target_resolution = float(train_cfg.get("target_resolution", 10.0))
    grid = build_target_grid(aoi_bbox, target_crs, target_resolution, target_resolution)
    resampling_name = str(train_cfg.get("resampling", config.get("staging", {}).get("resampling", "bilinear")))
    focal_radius_m = float(train_cfg.get("focal_median_radius_m", 15.0))

    pair_id = manifest["pair_id"]
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

    infer_config = load_yaml(infer_cfg.get("config_path", "config/infer_config.yaml"))
    if device:
        infer_config["device"] = device
    inferencer = SARInferencer(infer_config)
    output_tif = output_dir / f"{aoi_path.stem}__{pair_id}_SR_x2.tif"
    inferencer.run_pair_from_multiband_files(
        identifier=pair_id,
        t1_path=t1_composite_path,
        t2_path=t2_composite_path,
        out_path=output_tif,
    )

    summary = {
        "workflow_mode": "stac_trainlike_composite",
        "aoi_geojson": str(aoi_path),
        "run_dir": str(run_dir),
        "anchor_manifest_path": str(manifest_path),
        "window_raw_dir": str(window_raw_dir),
        "composite_dir": str(composite_dir),
        "t1_composite_path": str(t1_composite_path),
        "t2_composite_path": str(t2_composite_path),
        "output_tif": str(output_tif),
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
            "support_pair_delta_hours": manifest.get("support_pair_delta_hours"),
            "support_pair_delta_days": manifest.get("support_pair_delta_days"),
            "pre_scene_count": len(pre_items),
            "post_scene_count": len(post_items),
            "pre_scenes": manifest.get("pre_scenes", []),
            "post_scenes": manifest.get("post_scenes", []),
        },
        "composite": {
            "grid": post_meta["grid"],
            "pre": pre_meta,
            "post": post_meta,
        },
        "downloaded_files": {
            "pre": {pol: [str(p) for p in paths] for pol, paths in pre_paths.items()},
            "post": {pol: [str(p) for p in paths] for pol, paths in post_paths.items()},
        },
        "run_config": {
            "stac_url": stac_cfg.get("url", DEFAULT_STAC_API),
            "collection": stac_cfg.get("collection", DEFAULT_COLLECTION),
            "datetime": stac_cfg.get("datetime"),
            "limit": int(stac_cfg.get("limit", 300)),
            "min_aoi_coverage": float(pair_cfg.get("min_aoi_coverage", 1.0)),
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
        },
    }
    if compatibility_info is not None:
        summary["compatibility"] = compatibility_info
    summary["summary_json"] = str(run_dir / "run_summary.json")
    summary["summary_md"] = str(run_dir / "run_summary.md")
    write_trainlike_run_summary(run_dir, summary)
    return summary


def run_pipeline(config: Dict[str, Any], geojson_path: str, output_root: Optional[str], cache_staging: bool, device: Optional[str]) -> Dict[str, Any]:
    workflow_cfg = config.get("workflow", {})
    mode = str(workflow_cfg.get("mode", "exact_pair")).strip().lower()
    if mode == "exact_pair":
        return run_exact_pair_pipeline(config, geojson_path, output_root, cache_staging, device)
    if mode == "stac_trainlike_composite":
        return run_stac_trainlike_pipeline(config, geojson_path, output_root, cache_staging, device)
    raise ValueError(f"Unsupported workflow.mode: {workflow_cfg.get('mode')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AOI -> STAC -> preprocess -> ISSM-SAR inference pipeline")
    parser.add_argument("--geojson", required=True, help="Path to input AOI GeoJSON")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Path to pipeline config")
    parser.add_argument("--mode", default=None, help="Override workflow mode: exact_pair or stac_trainlike_composite")
    parser.add_argument("--datetime", default=None, help="Override STAC datetime range")
    parser.add_argument("--min-delta-hours", type=float, default=None, help="Override minimum time delta")
    parser.add_argument("--max-delta-days", type=int, default=None, help="Override maximum time delta")
    parser.add_argument("--min-aoi-coverage", type=float, default=None, help="Override minimum AOI bbox coverage")
    parser.add_argument("--same-orbit-direction", action="store_true", help="Require same orbit direction")
    parser.add_argument("--auto-relax", action="store_true", help="Enable balanced/loose time window fallback")
    parser.add_argument("--window-before-days", type=float, default=None, help="Override STAC train-like pre-window length")
    parser.add_argument("--window-after-days", type=float, default=None, help="Override STAC train-like post-window length")
    parser.add_argument("--min-scenes-per-window", type=int, default=None, help="Override minimum unique scenes per STAC window")
    parser.add_argument("--target-crs", default=None, help="Override train-like target CRS, e.g. EPSG:3857")
    parser.add_argument("--target-resolution", type=float, default=None, help="Override train-like target pixel size in meters")
    parser.add_argument("--focal-median-radius-m", type=float, default=None, help="Override train-like focal median radius in meters")
    parser.add_argument("--device", default=None, help="Override inference device")
    parser.add_argument("--output-dir", default=None, help="Override pipeline run root directory")
    parser.add_argument("--cache-staging", action="store_true", help="Persist aligned 2-band inputs in staging dir")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config.setdefault("workflow", {})
    config.setdefault("stac", {})
    config.setdefault("pairing", {})
    config.setdefault("trainlike", {})
    if args.mode is not None:
        config["workflow"]["mode"] = args.mode
    if args.datetime is not None:
        config["stac"]["datetime"] = args.datetime
    if args.min_delta_hours is not None:
        config["pairing"]["min_delta_hours"] = args.min_delta_hours
    if args.max_delta_days is not None:
        config["pairing"]["max_delta_days"] = args.max_delta_days
    if args.min_aoi_coverage is not None:
        config["pairing"]["min_aoi_coverage"] = args.min_aoi_coverage
    if args.same_orbit_direction:
        config["pairing"]["same_orbit_direction"] = True
        config["trainlike"]["same_orbit_direction"] = True
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

    try:
        summary = run_pipeline(
            config=config,
            geojson_path=args.geojson,
            output_root=args.output_dir,
            cache_staging=args.cache_staging,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("[PIPELINE] interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"[PIPELINE] failed: {exc}")
        sys.exit(1)

    print("[PIPELINE] completed")
    print(f"  Mode: {summary['workflow_mode']}")
    print(f"  AOI: {summary['aoi_geojson']}")
    if summary["workflow_mode"] == "exact_pair":
        pair = summary["selected_pair"]
        print(f"  Pair: {pair['pair_id']}")
        print(f"  Latest input: {pair['latest_input_datetime']}")
        print(f"  Delta: {format_duration_human(pair['delta_seconds'])}")
        print(f"  AOI bbox coverage min: {pair['aoi_bbox_coverage_min']:.3f}")
    else:
        anchor = summary["anchor"]
        print(f"  Anchor: {anchor['anchor_datetime']}")
        print(f"  Latest input: {anchor.get('latest_input_datetime')}")
        print(f"  Window: -{anchor['window_before_days']}d / +{anchor['window_after_days']}d")
        print(f"  Scenes: pre={anchor['pre_scene_count']} post={anchor['post_scene_count']}")
    print(f"  Output: {summary['output_tif']}")
    print(f"  Summary: {summary['summary_json']}")


if __name__ == "__main__":
    main()

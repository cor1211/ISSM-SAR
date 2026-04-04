#!/usr/bin/env python3
"""Helper utilities shared by canonical GEE composite runtime paths.

The old standalone exact-pair compare workflow has been retired from the core
runtime. This module intentionally keeps only the helper functions still used by
`sar_pipeline.py`, tools, and other composite-oriented code paths.
"""

from __future__ import annotations

from collections import defaultdict
import importlib.metadata as importlib_metadata_std
import logging
import math
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

import requests
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS


def _patch_importlib_metadata_compat() -> None:
    """Patch Python 3.9 stdlib importlib.metadata for newer Google libs."""
    if hasattr(importlib_metadata_std, "packages_distributions"):
        return
    try:
        import importlib_metadata as importlib_metadata_backport
    except Exception:
        importlib_metadata_backport = None
    if importlib_metadata_backport and hasattr(importlib_metadata_backport, "packages_distributions"):
        importlib_metadata_std.packages_distributions = importlib_metadata_backport.packages_distributions
        return

    def _fallback_packages_distributions() -> Dict[str, List[str]]:
        mapping: DefaultDict[str, List[str]] = defaultdict(list)
        for dist in importlib_metadata_std.distributions():
            dist_name = None
            try:
                dist_name = dist.metadata.get("Name")
            except Exception:
                dist_name = None
            if not dist_name:
                continue

            top_level = []
            try:
                top_level_text = dist.read_text("top_level.txt")
            except Exception:
                top_level_text = None
            if top_level_text:
                top_level = [line.strip() for line in top_level_text.splitlines() if line.strip()]

            if not top_level:
                files = getattr(dist, "files", None) or []
                for file_ref in files:
                    parts = str(file_ref).split("/")
                    if not parts:
                        continue
                    root = parts[0].strip()
                    if not root or root.endswith(".dist-info") or root.endswith(".data") or "." in root:
                        continue
                    top_level.append(root)

            for pkg_name in sorted(set(top_level)):
                if dist_name not in mapping[pkg_name]:
                    mapping[pkg_name].append(dist_name)
        return dict(mapping)

    importlib_metadata_std.packages_distributions = _fallback_packages_distributions


_patch_importlib_metadata_compat()

try:
    import ee
except ImportError as exc:  # pragma: no cover - dependency issue is runtime-only.
    raise RuntimeError("earthengine-api is required. Install requirements first.") from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gee_compare")


def build_target_grid(aoi_bbox: List[float], target_crs: str, xres: float, yres: float) -> Dict[str, Any]:
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
        "xres": xres,
        "yres": yres,
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "width": width,
        "height": height,
        "transform": transform,
        "crs_transform": [xres, 0.0, left, 0.0, -yres, top],
    }


def init_gee(project: str, authenticate: bool) -> None:
    try:
        ee.Initialize(project=project)
        return
    except Exception as first_exc:
        if not authenticate:
            raise RuntimeError(
                "Failed to initialize Earth Engine. Run `earthengine authenticate` or rerun this tool "
                "with `--authenticate` and provide `--gee-project` (or gee.project in config). "
                f"Original error: {first_exc}"
            ) from first_exc
    logger.info("Earth Engine auth is required. Starting interactive authentication...")
    ee.Authenticate()
    ee.Initialize(project=project)


def build_export_params(
    export_name: str,
    grid: Dict[str, Any],
    band_names: List[str],
) -> Dict[str, Any]:
    crs_transform = grid.get("crs_transform")
    if crs_transform is None:
        transform = grid.get("transform")
        if transform is None:
            raise KeyError("crs_transform")
        crs_transform = list(transform)[:6]
    return {
        "name": export_name,
        "format": "GEO_TIFF",
        "filePerBand": False,
        "bands": band_names,
        "crs": grid["crs"],
        "crs_transform": crs_transform,
        "dimensions": [grid["width"], grid["height"]],
    }


def rewrite_with_descriptions(path: Path, descriptions: List[str], expected_grid: Dict[str, Any]) -> None:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
        crs=CRS.from_user_input(expected_grid["crs"]),
        transform=expected_grid["transform"],
        width=expected_grid["width"],
        height=expected_grid["height"],
        compress="DEFLATE",
        tiled=True,
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        for idx, desc in enumerate(descriptions, start=1):
            dst.set_band_description(idx, desc)
    tmp_path.replace(path)


def validate_pair(out_dir: Path, pair_id: str, expected_grid: Dict[str, Any], expected_descs: List[str]) -> Dict[str, Any]:
    t1_path = out_dir / f"s1t1_{pair_id}.tif"
    t2_path = out_dir / f"s1t2_{pair_id}.tif"
    validation: Dict[str, Any] = {
        "t1_path": str(t1_path),
        "t2_path": str(t2_path),
        "pair_scan_ok": False,
    }

    with rasterio.open(t1_path) as t1, rasterio.open(t2_path) as t2:
        validation["t1_count"] = t1.count
        validation["t2_count"] = t2.count
        validation["t1_crs"] = str(t1.crs)
        validation["t2_crs"] = str(t2.crs)
        validation["t1_transform"] = list(t1.transform)[:6]
        validation["t2_transform"] = list(t2.transform)[:6]
        validation["t1_shape"] = [t1.height, t1.width]
        validation["t2_shape"] = [t2.height, t2.width]
        validation["t1_descriptions"] = list(t1.descriptions)
        validation["t2_descriptions"] = list(t2.descriptions)
        validation["same_grid"] = (
            t1.crs == t2.crs
            and t1.transform.almost_equals(t2.transform)
            and t1.width == t2.width
            and t1.height == t2.height
        )
        validation["matches_expected_grid"] = (
            str(t1.crs) == expected_grid["crs"]
            and str(t2.crs) == expected_grid["crs"]
            and t1.transform.almost_equals(expected_grid["transform"])
            and t2.transform.almost_equals(expected_grid["transform"])
            and t1.width == expected_grid["width"]
            and t2.width == expected_grid["width"]
            and t1.height == expected_grid["height"]
            and t2.height == expected_grid["height"]
        )
        validation["matches_expected_descriptions"] = list(t1.descriptions) == expected_descs and list(t2.descriptions) == expected_descs

    from infer_production import SARInferencer

    jobs = SARInferencer._scan_input_dir(out_dir, "s1t1_", "s1t2_")
    validation["scan_jobs"] = jobs
    validation["pair_scan_ok"] = len(jobs) == 1 and jobs[0]["mode"] == "multiband" and jobs[0]["identifier"] == pair_id
    return validation


def download_gee_image(image: ee.Image, params: Dict[str, Any], out_path: Path) -> Path:
    url = image.getDownloadURL(params)
    logger.info("Downloading %s", out_path.name)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path

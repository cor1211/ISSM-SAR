#!/usr/bin/env python3
"""Download a GEE Sentinel-1 comparison pair for a system-selected AOI pair.

This tool keeps the ESA/STAC pipeline unchanged and provides a separate
comparison path using GEE Sentinel-1 dB imagery exported as two 2-band
GeoTIFFs compatible with infer_production.py.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import importlib.metadata as importlib_metadata_std
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import requests
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.geometry import shape
import yaml

from query_stac_download import (
    DEFAULT_STAC_API,
    build_manifest_for_pair,
    canonical_bbox_from_geometry,
    compute_item_aoi_geometry_metrics,
    load_geojson_aoi,
)
from sar_pipeline import load_yaml as load_pipeline_yaml, select_best_pair


def _patch_importlib_metadata_compat() -> None:
    """Patch Python 3.9 stdlib importlib.metadata for newer Google libs.

    Some Earth Engine / google-api-core code paths expect
    importlib.metadata.packages_distributions(), which is absent in the
    stdlib module on the Python 3.9 build used by the `issm` environment.
    We bridge it from the backport package when available.
    """
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


@dataclass
class GEECandidate:
    full_id: str
    system_index: str
    datetime_utc: datetime
    orbit_pass: Optional[str]
    coverage_ratio: float
    exact_id_match: bool
    orbit_match: bool
    delta_to_target_seconds: float
    geometry_geojson: Dict[str, Any]


def _gee_id_matches_target(system_index: str, target_system_id: str) -> bool:
    """Match STAC item ids to GEE system:index values conservatively.

    GEE ids for COPERNICUS/S1_GRD may append a product suffix after the
    Sentinel-1 scene id. We therefore accept an exact match or a prefix
    match on the scene id stem.
    """
    if not system_index or not target_system_id:
        return False
    if system_index == target_system_id:
        return True
    return system_index.startswith(f"{target_system_id}_")


def ensure_dir(path: str | Path) -> Path:
    dst = Path(path)
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def to_jsonable(value: Any) -> Any:
    """Recursively convert common runtime objects into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def normalize_manifest_model_order(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize public T1/T2 fields into canonical model order:
      - T1 = later/posterior
      - T2 = earlier/prior

    Older manifests may still store public T1/T2 as earlier/later.
    """
    if "t1_datetime" not in manifest or "t2_datetime" not in manifest:
        return manifest

    out = copy.deepcopy(manifest)
    t1_dt = parse_utc(out["t1_datetime"])
    t2_dt = parse_utc(out["t2_datetime"])

    if t1_dt >= t2_dt:
        out.setdefault("later_id", out.get("t1_id"))
        out.setdefault("earlier_id", out.get("t2_id"))
        out.setdefault("later_datetime", out.get("t1_datetime"))
        out.setdefault("earlier_datetime", out.get("t2_datetime"))
        return out

    out["later_id"] = out.get("t2_id")
    out["earlier_id"] = out.get("t1_id")
    out["later_datetime"] = out.get("t2_datetime")
    out["earlier_datetime"] = out.get("t1_datetime")

    for left, right in [
        ("t1_id", "t2_id"),
        ("t1_datetime", "t2_datetime"),
        ("t1_datatake_id", "t2_datatake_id"),
        ("t1_orbit_state", "t2_orbit_state"),
        ("aoi_bbox_coverage_t1", "aoi_bbox_coverage_t2"),
        ("aoi_coverage_t1", "aoi_coverage_t2"),
    ]:
        if left in out or right in out:
            out[left], out[right] = out.get(right), out.get(left)

    assets = out.get("assets")
    if isinstance(assets, dict):
        for pol, entry in assets.items():
            if not isinstance(entry, dict):
                continue
            for left, right in [("t1_asset_key", "t2_asset_key"), ("t1_href", "t2_href")]:
                if left in entry or right in entry:
                    entry[left], entry[right] = entry.get(right), entry.get(left)
            assets[pol] = entry

    semantics = out.get("pair_semantics")
    if isinstance(semantics, dict):
        semantics["t1_role"] = "later/posterior exact scene"
        semantics["t2_role"] = "earlier/prior exact scene"
        semantics["matches_training_semantics"] = True
        out["pair_semantics"] = semantics
    return out


def normalize_orbit_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip().lower()
    if v == "ascending":
        return "ASCENDING"
    if v == "descending":
        return "DESCENDING"
    return None


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


def load_system_manifest(
    geojson_path: str | Path,
    pipeline_config_path: str | Path,
    manifest_path: Optional[str | Path],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    pipeline_config = load_pipeline_yaml(pipeline_config_path)
    aoi_geojson = str(Path(geojson_path).resolve())
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest = normalize_manifest_model_order(manifest)
        manifest.setdefault("aoi_geojson", aoi_geojson)
        if "aoi_bbox" not in manifest:
            raise ValueError("Manifest must contain aoi_bbox.")
        return pipeline_config, manifest, None

    from query_stac_download import STACClient

    client = STACClient(pipeline_config.get("stac", {}).get("url", DEFAULT_STAC_API))
    _, aoi_bbox, _, pair, selection_profile, diagnostics = select_best_pair(client, pipeline_config, aoi_geojson)
    manifest = build_manifest_for_pair(pair, ["VV", "VH"])
    if manifest is None:
        raise RuntimeError("Failed to build system manifest for selected pair.")
    manifest = normalize_manifest_model_order(manifest)
    manifest["selection_profile"] = selection_profile
    manifest["aoi_geojson"] = aoi_geojson
    manifest["aoi_bbox"] = aoi_bbox
    return pipeline_config, manifest, diagnostics


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


def collection_base(config: Dict[str, Any], aoi_bbox_geom: ee.Geometry, target_dt: datetime, window_minutes: int):
    gee_cfg = config["gee"]
    start = target_dt - timedelta(minutes=window_minutes)
    end = target_dt + timedelta(minutes=window_minutes)
    return (
        ee.ImageCollection(gee_cfg["collection"])
        .filterBounds(aoi_bbox_geom)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )


def image_to_candidate(
    image: ee.Image,
    target_dt: datetime,
    target_system_id: str,
    target_orbit_pass: Optional[str],
    aoi_geometry_geojson: Optional[Dict[str, Any]],
    aoi_bbox: List[float],
) -> GEECandidate:
    full_id = image.id().getInfo()
    system_index = full_id.split("/")[-1] if full_id else str(image.get("system:index").getInfo())
    ts = image.date().millis().getInfo()
    dt_utc = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    orbit_pass = image.get("orbitProperties_pass").getInfo()
    geom_info = image.geometry().getInfo()
    coverage_metrics = compute_item_aoi_geometry_metrics(
        {
            "geometry": geom_info,
            "bbox": canonical_bbox_from_geometry(geom_info),
            "properties": {},
        },
        aoi_geometry_geojson or {"type": "Polygon", "coordinates": [[
            [aoi_bbox[0], aoi_bbox[1]],
            [aoi_bbox[2], aoi_bbox[1]],
            [aoi_bbox[2], aoi_bbox[3]],
            [aoi_bbox[0], aoi_bbox[3]],
            [aoi_bbox[0], aoi_bbox[1]],
        ]]},
        aoi_bbox=aoi_bbox,
    )
    return GEECandidate(
        full_id=full_id,
        system_index=system_index,
        datetime_utc=dt_utc,
        orbit_pass=orbit_pass,
        coverage_ratio=float(coverage_metrics["aoi_coverage"]),
        exact_id_match=_gee_id_matches_target(system_index, target_system_id),
        orbit_match=(target_orbit_pass is not None and orbit_pass == target_orbit_pass),
        delta_to_target_seconds=abs((dt_utc - target_dt).total_seconds()),
        geometry_geojson=geom_info,
    )


def select_gee_candidate(
    config: Dict[str, Any],
    aoi_bbox: List[float],
    aoi_geometry_geojson: Optional[Dict[str, Any]],
    target_dt_str: str,
    target_system_id: str,
    target_orbit_state: Optional[str],
) -> Tuple[Optional[GEECandidate], Dict[str, Any]]:
    gee_cfg = config["gee"]
    window_minutes = int(gee_cfg.get("match_window_minutes", 30))
    min_aoi_coverage = float(gee_cfg.get("min_aoi_coverage", 0.0))
    target_dt = parse_utc(target_dt_str)
    aoi_geom = ee.Geometry.Rectangle(aoi_bbox, proj="EPSG:4326", geodesic=False)

    base_coll = collection_base(config, aoi_geom, target_dt, window_minutes)
    base_count = int(base_coll.size().getInfo())
    if base_count == 0:
        return None, {
            "reason": "NO_GEE_IMAGE_IN_TIME_WINDOW",
            "window_minutes": window_minutes,
            "raw_candidate_count": 0,
            "polarized_candidate_count": 0,
            "covered_candidate_count": 0,
        }

    pol_coll = (
        base_coll
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    pol_count = int(pol_coll.size().getInfo())
    if pol_count == 0:
        return None, {
            "reason": "MISSING_VV_VH",
            "window_minutes": window_minutes,
            "raw_candidate_count": base_count,
            "polarized_candidate_count": 0,
            "covered_candidate_count": 0,
        }

    target_orbit_pass = normalize_orbit_state(target_orbit_state)
    images = pol_coll.toList(pol_count)
    covered_candidates: List[GEECandidate] = []
    all_candidates: List[Dict[str, Any]] = []
    for idx in range(pol_count):
        candidate = image_to_candidate(
            ee.Image(images.get(idx)),
            target_dt=target_dt,
            target_system_id=target_system_id,
            target_orbit_pass=target_orbit_pass,
            aoi_geometry_geojson=aoi_geometry_geojson,
            aoi_bbox=aoi_bbox,
        )
        all_candidates.append(
            {
                "gee_id": candidate.system_index,
                "gee_full_id": candidate.full_id,
                "gee_datetime": candidate.datetime_utc.isoformat().replace("+00:00", "Z"),
                "orbit_pass": candidate.orbit_pass,
                "coverage_ratio": candidate.coverage_ratio,
                "exact_id_match": candidate.exact_id_match,
                "orbit_match": candidate.orbit_match,
                "delta_to_target_seconds": candidate.delta_to_target_seconds,
            }
        )
        if candidate.coverage_ratio > min_aoi_coverage:
            covered_candidates.append(candidate)

    if not covered_candidates:
        return None, {
            "reason": "AOI_GEOMETRY_COVERAGE_BELOW_THRESHOLD",
            "window_minutes": window_minutes,
            "raw_candidate_count": base_count,
            "polarized_candidate_count": pol_count,
            "covered_candidate_count": 0,
            "min_aoi_coverage": min_aoi_coverage,
            "candidates": all_candidates,
        }

    covered_candidates.sort(
        key=lambda c: (
            0 if c.exact_id_match else 1,
            c.delta_to_target_seconds,
            0 if c.orbit_match else 1,
            c.system_index,
        )
    )
    chosen = covered_candidates[0]
    return chosen, {
        "reason": "OK",
        "window_minutes": window_minutes,
        "min_aoi_coverage": min_aoi_coverage,
        "raw_candidate_count": base_count,
        "polarized_candidate_count": pol_count,
        "covered_candidate_count": len(covered_candidates),
        "candidates": all_candidates,
    }


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


def download_gee_image(image: ee.Image, params: Dict[str, Any], out_path: Path) -> Path:
    url = image.getDownloadURL(params)
    logger.info("Downloading %s", out_path.name)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


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


def write_report(run_dir: Path, report: Dict[str, Any]) -> Tuple[Path, Path]:
    json_path = run_dir / "gee_compare_report.json"
    md_path = run_dir / "gee_compare_report.md"
    report_jsonable = to_jsonable(report)
    json_path.write_text(json.dumps(report_jsonable, indent=2, ensure_ascii=False), encoding="utf-8")

    system_pair = report_jsonable["system_reference"]
    gee_pair = report_jsonable.get("gee_match", {})
    lines = [
        "# GEE Compare Report",
        "",
        "## System Reference",
        "",
        f"- AOI: `{report_jsonable['aoi_geojson']}`",
        f"- System manifest: `{report_jsonable['system_reference_manifest']}`",
        f"- Pair ID: `{system_pair['pair_id']}`",
        f"- T1 ID (later): `{system_pair['t1_id']}`",
        f"- T2 ID (earlier): `{system_pair['t2_id']}`",
        f"- T1 datetime (later): `{system_pair['t1_datetime']}`",
        f"- T2 datetime (earlier): `{system_pair['t2_datetime']}`",
        f"- Chronology provenance: earlier=`{system_pair.get('earlier_id')}` @ `{system_pair.get('earlier_datetime')}` -> later=`{system_pair.get('later_id')}` @ `{system_pair.get('later_datetime')}`",
        f"- Delta: `{system_pair['delta_human']}`",
        f"- AOI bbox: `{system_pair['aoi_bbox']}`",
        f"- AOI geometry coverage T1: `{system_pair.get('aoi_coverage_t1')}`",
        f"- AOI geometry coverage T2: `{system_pair.get('aoi_coverage_t2')}`",
        f"- AOI bbox coverage T1: `{system_pair.get('aoi_bbox_coverage_t1')}`",
        f"- AOI bbox coverage T2: `{system_pair.get('aoi_bbox_coverage_t2')}`",
        "",
        "## GEE Match",
        "",
        f"- T1 GEE ID (later): `{gee_pair.get('t1', {}).get('gee_id')}`",
        f"- T2 GEE ID (earlier): `{gee_pair.get('t2', {}).get('gee_id')}`",
        f"- T1 GEE datetime (later): `{gee_pair.get('t1', {}).get('gee_datetime')}`",
        f"- T2 GEE datetime (earlier): `{gee_pair.get('t2', {}).get('gee_datetime')}`",
        f"- T1 orbit pass: `{gee_pair.get('t1', {}).get('orbit_pass')}`",
        f"- T2 orbit pass: `{gee_pair.get('t2', {}).get('orbit_pass')}`",
        f"- T1 exact id match: `{gee_pair.get('t1', {}).get('exact_id_match')}`",
        f"- T2 exact id match: `{gee_pair.get('t2', {}).get('exact_id_match')}`",
        f"- T1 delta to target (s): `{gee_pair.get('t1', {}).get('delta_to_target_seconds')}`",
        f"- T2 delta to target (s): `{gee_pair.get('t2', {}).get('delta_to_target_seconds')}`",
        f"- T1 AOI coverage ratio: `{gee_pair.get('t1', {}).get('coverage_ratio')}`",
        f"- T2 AOI coverage ratio: `{gee_pair.get('t2', {}).get('coverage_ratio')}`",
        "",
        "## Selection Diagnostics",
        "",
        f"- T1 selection reason: `{report_jsonable['diagnostics']['t1_selection'].get('reason')}`",
        f"- T2 selection reason: `{report_jsonable['diagnostics']['t2_selection'].get('reason')}`",
        f"- Match window minutes: `{report_jsonable['diagnostics']['t1_selection'].get('window_minutes')}`",
        "",
        "## Files",
        "",
        f"- Output dir: `{report_jsonable['output_dir']}`",
        f"- T1 file: `{report_jsonable['output_files']['t1']}`",
        f"- T2 file: `{report_jsonable['output_files']['t2']}`",
        "",
        "## Export Grid",
        "",
        f"- Collection: `{report_jsonable['export']['collection']}`",
        f"- CRS: `{report_jsonable['export']['crs']}`",
        f"- Scale: `{report_jsonable['export']['scale']}`",
        f"- Width: `{report_jsonable['export']['width']}`",
        f"- Height: `{report_jsonable['export']['height']}`",
        f"- Band names: `{report_jsonable['export']['band_names']}`",
        f"- Band descriptions: `{report_jsonable['export']['band_descriptions']}`",
        "",
        "## Validation",
        "",
        f"- Same grid: `{report_jsonable['validation']['same_grid']}`",
        f"- Matches expected grid: `{report_jsonable['validation']['matches_expected_grid']}`",
        f"- Matches expected descriptions: `{report_jsonable['validation']['matches_expected_descriptions']}`",
        f"- Infer scan OK: `{report_jsonable['validation']['pair_scan_ok']}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def candidate_to_dict(candidate: GEECandidate) -> Dict[str, Any]:
    return {
        "gee_id": candidate.system_index,
        "gee_full_id": candidate.full_id,
        "gee_datetime": candidate.datetime_utc.isoformat().replace("+00:00", "Z"),
        "orbit_pass": candidate.orbit_pass,
        "coverage_ratio": candidate.coverage_ratio,
        "exact_id_match": candidate.exact_id_match,
        "orbit_match": candidate.orbit_match,
        "delta_to_target_seconds": candidate.delta_to_target_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a GEE comparison pair aligned to a system-selected STAC pair")
    parser.add_argument("--geojson", required=True, help="AOI GeoJSON path")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yaml", help="Pipeline config used to resolve the system pair")
    parser.add_argument("--gee-config", default="config/gee_compare_config.yaml", help="GEE compare config path")
    parser.add_argument("--manifest", default=None, help="Optional system manifest path to lock the reference pair")
    parser.add_argument("--gee-project", default=None, help="Earth Engine project id")
    parser.add_argument("--out-dir", default=None, help="Override output root directory")
    parser.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() if initialization fails")
    args = parser.parse_args()

    gee_config = yaml.safe_load(Path(args.gee_config).read_text(encoding="utf-8")) or {}
    pipeline_config, system_manifest, system_diag = load_system_manifest(args.geojson, args.pipeline_config, args.manifest)

    gee_project = args.gee_project or gee_config.get("gee", {}).get("project")
    if not gee_project:
        raise RuntimeError("Missing GEE project. Provide --gee-project or gee.project in config/gee_compare_config.yaml")
    init_gee(gee_project, authenticate=args.authenticate)

    gee_cfg = gee_config["gee"]
    target_crs = gee_cfg.get("export_crs", "EPSG:3857")
    target_scale = float(gee_cfg.get("export_scale", 10.0))
    output_descs = list(gee_cfg.get("output_band_descriptions", ["S1_VV", "S1_VH"]))
    band_names = list(gee_cfg.get("band_names", ["VV", "VH"]))
    grid = build_target_grid(system_manifest["aoi_bbox"], target_crs, target_scale, target_scale)
    _, aoi_geometry = load_geojson_aoi(args.geojson)

    run_root = ensure_dir(args.out_dir or gee_config.get("output", {}).get("root_dir", "runs/gee_compare"))
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = ensure_dir(run_root / Path(args.geojson).stem / run_id)
    manifest_copy_path = run_dir / "system_reference_manifest.json"
    manifest_copy_path.write_text(json.dumps(system_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    t1_candidate, t1_diag = select_gee_candidate(
        gee_config,
        aoi_bbox=system_manifest["aoi_bbox"],
        aoi_geometry_geojson=aoi_geometry,
        target_dt_str=system_manifest["t1_datetime"],
        target_system_id=system_manifest["t1_id"],
        target_orbit_state=system_manifest.get("t1_orbit_state"),
    )
    if t1_candidate is None:
        raise RuntimeError(f"GEE T1 selection failed: {t1_diag['reason']}")

    t2_candidate, t2_diag = select_gee_candidate(
        gee_config,
        aoi_bbox=system_manifest["aoi_bbox"],
        aoi_geometry_geojson=aoi_geometry,
        target_dt_str=system_manifest["t2_datetime"],
        target_system_id=system_manifest["t2_id"],
        target_orbit_state=system_manifest.get("t2_orbit_state"),
    )
    if t2_candidate is None:
        raise RuntimeError(f"GEE T2 selection failed: {t2_diag['reason']}")

    collection = gee_cfg["collection"]
    aoi_geom = ee.Geometry.Rectangle(system_manifest["aoi_bbox"], proj="EPSG:4326", geodesic=False)
    t1_image = ee.Image(f"{collection}/{t1_candidate.system_index}").select(band_names).clip(aoi_geom)
    t2_image = ee.Image(f"{collection}/{t2_candidate.system_index}").select(band_names).clip(aoi_geom)

    pair_id = system_manifest["pair_id"]
    t1_path = run_dir / f"s1t1_{pair_id}.tif"
    t2_path = run_dir / f"s1t2_{pair_id}.tif"
    download_gee_image(
        t1_image,
        build_export_params(f"s1t1_{pair_id}", grid, band_names),
        t1_path,
    )
    download_gee_image(
        t2_image,
        build_export_params(f"s1t2_{pair_id}", grid, band_names),
        t2_path,
    )

    rewrite_with_descriptions(t1_path, output_descs, grid)
    rewrite_with_descriptions(t2_path, output_descs, grid)
    validation = validate_pair(run_dir, pair_id, grid, output_descs)

    report = {
        "aoi_geojson": str(Path(args.geojson).resolve()),
        "gee_project": gee_project,
        "system_reference_manifest": str(manifest_copy_path),
        "output_dir": str(run_dir),
        "system_reference": {
            "pair_id": system_manifest["pair_id"],
            "t1_id": system_manifest["t1_id"],
            "t2_id": system_manifest["t2_id"],
            "t1_datetime": system_manifest["t1_datetime"],
            "t2_datetime": system_manifest["t2_datetime"],
            "delta_hours": system_manifest["delta_hours"],
            "delta_days": system_manifest["delta_days"],
            "delta_human": f"{system_manifest['delta_days']:.6f} days / {system_manifest['delta_hours']:.6f} hours",
            "aoi_bbox": system_manifest["aoi_bbox"],
            "aoi_coverage_t1": system_manifest.get("aoi_coverage_t1"),
            "aoi_coverage_t2": system_manifest.get("aoi_coverage_t2"),
            "aoi_bbox_coverage_t1": system_manifest.get("aoi_bbox_coverage_t1"),
            "aoi_bbox_coverage_t2": system_manifest.get("aoi_bbox_coverage_t2"),
        },
        "gee_match": {
            "t1": candidate_to_dict(t1_candidate),
            "t2": candidate_to_dict(t2_candidate),
        },
        "diagnostics": {
            "system_pair_resolution": system_diag,
            "t1_selection": t1_diag,
            "t2_selection": t2_diag,
        },
        "export": {
            "collection": collection,
            "crs": grid["crs"],
            "scale": target_scale,
            "width": grid["width"],
            "height": grid["height"],
            "transform": list(grid["transform"]),
            "band_names": band_names,
            "band_descriptions": output_descs,
        },
        "output_files": {
            "t1": str(t1_path),
            "t2": str(t2_path),
        },
        "validation": validation,
    }
    json_path, md_path = write_report(run_dir, report)
    logger.info("Saved report JSON: %s", json_path)
    logger.info("Saved report MD: %s", md_path)


if __name__ == "__main__":
    main()

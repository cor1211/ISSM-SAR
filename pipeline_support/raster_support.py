from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from pipeline_support.json_support import compact_jsonable, to_jsonable
from pipeline_support.runtime_support import ensure_dir
from query_stac_download import extract_item_info, geodesic_area_wgs84, item_scene_key, normalize_polygonal_geojson_geometry
from runtime_logging import emit_runtime_log


def emit_pipeline_log(level: int, message: str, **fields: Any) -> None:
    emit_runtime_log("sar_pipeline", level, message, **fields)


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

def align_single_band_to_grid(
    path: str | Path,
    grid: Dict[str, Any],
    resampling: Resampling,
    *,
    valid_min_db: Optional[float] = None,
    valid_max_db: Optional[float] = None,
) -> np.ndarray:
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
        vmin = float(valid_min_db) if valid_min_db is not None else None
        vmax = float(valid_max_db) if valid_max_db is not None else None
        if vmin is not None and vmax is not None and vmin > vmax:
            raise ValueError(f"Invalid valid dB range: min={vmin} > max={vmax}")
        if vmin is not None:
            dst[dst < vmin] = np.nan
        if vmax is not None:
            dst[dst > vmax] = np.nan
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

__all__ = [
    "align_single_band_to_grid",
    "apply_focal_median_db",
    "apply_geometry_mask_to_multiband",
    "build_circular_footprint",
    "build_grid_meta",
    "build_sr_grid_from_input_grid",
    "build_target_grid",
    "dedupe_items_by_scene",
    "export_masked_sr_band_cogs",
    "geometry_mask_for_grid",
    "mosaic_component_sr_multibands_to_parent",
    "nanmedian_stack",
    "resolve_resampling",
    "write_geojson",
    "write_geometry_mask_like_raster",
    "write_geometry_mask_tif",
    "write_multiband_geotiff",
]

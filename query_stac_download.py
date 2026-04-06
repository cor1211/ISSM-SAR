#!/usr/bin/env python3
"""
=============================================================
 STAC Query -> Runtime Support -> S3/Rasterio Utilities
=============================================================

Muc dich:
  - Doc AOI tu .geojson hoac --bbox
  - Query STAC Item Sentinel-1 GRD
  - Loc theo cac tieu chi model dau vao (IW, GRD, polarization)
  - Cung cap helper cho representative monthly pipeline
  - Lay href asset tren S3, kiem tra bang rasterio, tuy chon download
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

import stac_support.s3_download_support as _s3_download_support
from stac_support.representative_selection_support import (
    build_representative_period_manifest,
    collect_period_half_items,
    normalize_representative_pool_mode,
    select_representative_scene_pools,
    select_witness_support_pair,
)
from runtime_logging import emit_runtime_log
from stac_support.s3_download_support import (
    build_rasterio_env_kwargs,
    href_to_rasterio_path,
    probe_rasterio_href,
)
from stac_support.stac_client_support import STACClient
from stac_support.stac_filter_support import (
    apply_hard_filters,
    collect_items_with_filters,
    parse_required_pols,
    resolve_spatial_filter,
)
from stac_support.stac_geometry_support import (
    annotate_items_for_aoi,
    bbox_area,
    bbox_intersection,
    bbox_intersection_bounds,
    bbox_overlap_ratio,
    bbox_to_geometry,
    build_seed_intersection_region_candidates,
    canonical_bbox_from_geometry,
    collect_items_covering_region,
    compute_item_aoi_geometry_metrics,
    compute_item_region_coverage_metrics,
    coverage_ratio,
    geodesic_area_wgs84,
    item_aoi_bbox_coverage_value,
    item_aoi_coverage_value,
    item_coverage_source,
    items_union_coverage,
    load_geojson_aoi,
    normalize_polygonal_geojson_geometry,
    normalize_polygonal_shapely_geometry,
)
from stac_support.stac_item_support import (
    extract_item_info,
    infer_asset_pol,
    is_raster_asset,
    item_scene_key,
    scene_datetime_values,
    select_asset_href,
    summarize_unique_scenes,
    unique_datetime_count_from_items,
)
from stac_support.stac_time_support import (
    add_month_utc,
    expand_month_periods,
    floor_month_utc,
    midpoint_datetime,
    normalize_datetime_range,
    parse_datetime_utc,
    parse_finite_datetime_range,
)

load_dotenv()

DEFAULT_STAC_API = (os.getenv("STAC_API_URL") or "").strip() or None
DEFAULT_COLLECTION = "sentinel-1-grd"
logger = logging.getLogger("query_stac_download")
boto3 = _s3_download_support.boto3


class S3Downloader(_s3_download_support.S3Downloader):
    """Compatibility facade that keeps `query_stac_download.boto3` patchable in tests."""

    def __init__(self, *args, **kwargs):
        _s3_download_support.boto3 = boto3
        super().__init__(*args, **kwargs)

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STAC Query + Runtime Support + S3 Utilities cho ISSM_SAR",
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
    p_list = subparsers.add_parser("list", help="Liet ke item STAC phu hop")
    add_common_args(p_list)
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_STAC_API",
    "S3Downloader",
    "STACClient",
    "add_month_utc",
    "annotate_items_for_aoi",
    "apply_hard_filters",
    "bbox_area",
    "bbox_intersection",
    "bbox_intersection_bounds",
    "bbox_overlap_ratio",
    "bbox_to_geometry",
    "build_rasterio_env_kwargs",
    "build_representative_period_manifest",
    "build_seed_intersection_region_candidates",
    "canonical_bbox_from_geometry",
    "collect_items_covering_region",
    "collect_items_with_filters",
    "collect_period_half_items",
    "compute_item_aoi_geometry_metrics",
    "compute_item_region_coverage_metrics",
    "coverage_ratio",
    "expand_month_periods",
    "extract_item_info",
    "floor_month_utc",
    "geodesic_area_wgs84",
    "href_to_rasterio_path",
    "infer_asset_pol",
    "is_raster_asset",
    "item_aoi_bbox_coverage_value",
    "item_aoi_coverage_value",
    "item_coverage_source",
    "item_scene_key",
    "items_union_coverage",
    "load_geojson_aoi",
    "main",
    "midpoint_datetime",
    "normalize_datetime_range",
    "normalize_polygonal_geojson_geometry",
    "normalize_polygonal_shapely_geometry",
    "normalize_representative_pool_mode",
    "parse_datetime_utc",
    "parse_finite_datetime_range",
    "parse_required_pols",
    "probe_rasterio_href",
    "resolve_spatial_filter",
    "scene_datetime_values",
    "select_asset_href",
    "select_representative_scene_pools",
    "select_witness_support_pair",
    "summarize_unique_scenes",
    "unique_datetime_count_from_items",
]

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from runtime_logging import emit_runtime_log
from stac_support.stac_client_support import STACClient
from stac_support.stac_geometry_support import annotate_items_for_aoi, bbox_to_geometry, load_geojson_aoi
from stac_support.stac_item_support import extract_item_info, select_asset_href

logger = logging.getLogger("query_stac_download")

def parse_required_pols(raw: Optional[str]) -> List[str]:
    """Parse --pols thanh danh sach polarization upper-case."""
    if not raw:
        return ["VV"]
    pols = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return pols or ["VV"]

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

def resolve_spatial_filter(args: argparse.Namespace) -> Tuple[Optional[List[float]], Optional[Dict[str, Any]]]:
    """
    Uu tien --geojson.
    Neu co geojson, su dung bbox + intersects cho query.
    """
    if getattr(args, "geojson", None):
        bbox, geometry = load_geojson_aoi(args.geojson)
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
        "STAC query completed",
        collection=args.collection,
        datetime=args.datetime,
        matched_items=len(items),
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
        "STAC query result summary",
        matched_items=len(items),
        hard_filtered_items=len(filtered),
    )
    return filtered, bbox, aoi_geometry

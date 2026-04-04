from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from stac_support.stac_time_support import parse_datetime_utc

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

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasterio

from pipeline_support.json_support import to_jsonable
from pipeline_support.raster_support import _pixel_size_meters
from pipeline_support.runtime_support import resolve_workflow_backend
from query_stac_download import canonical_bbox_from_geometry, extract_item_info


def _utc_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _local_href(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _join_url(base: str, *parts: str) -> str:
    cleaned = [str(base).rstrip("/")]
    cleaned.extend(str(part).strip("/") for part in parts if str(part).strip("/"))
    return "/".join(cleaned)


def _infer_sr_collection_name(summary: Dict[str, Any]) -> str:
    if summary.get("period") is not None and summary.get("component") is not None:
        return "issm-sar-sr-x2-monthly-component"
    if summary.get("period") is not None:
        return "issm-sar-sr-x2-monthly"
    return "issm-sar-sr-x2"


def _resolve_sr_collection_name(summary: Dict[str, Any]) -> str:
    inferred = _infer_sr_collection_name(summary)
    if summary.get("period") is not None and summary.get("component") is not None:
        return os.getenv("SR_COLLECTION_ID_MONTHLY_COMPONENT", inferred)
    if summary.get("period") is not None:
        return os.getenv("SR_COLLECTION_ID_MONTHLY", inferred)
    return os.getenv("SR_COLLECTION_ID_DEFAULT", inferred)


def _summary_aoi_id(summary: Dict[str, Any]) -> Optional[str]:
    explicit = str(summary.get("aoi_id") or "").strip()
    if explicit:
        return explicit

    for candidate_key in ("period_dir", "run_dir"):
        candidate = summary.get(candidate_key)
        if not candidate:
            continue
        try:
            path = Path(str(candidate)).resolve()
        except Exception:
            path = Path(str(candidate))
        parts = list(path.parts)
        if "aois" in parts:
            idx = parts.index("aois")
            if idx + 1 < len(parts):
                derived = str(parts[idx + 1]).strip()
                if derived:
                    return derived

    aoi_geojson = summary.get("aoi_geojson")
    if not aoi_geojson:
        return None
    try:
        stem = Path(str(aoi_geojson)).stem
        if stem and stem != "input_aoi":
            return stem
        return None
    except Exception:
        return None


def _summary_period_token(summary: Dict[str, Any]) -> Optional[str]:
    if summary.get("period"):
        return summary["period"].get("period_id")
    return None


def _normalize_public_token(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    chars: List[str] = []
    last_was_sep = False
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_")


def _whole_monthly_sr_item_id(aoi_id: str, period_id: str) -> str:
    normalized_aoi = _normalize_public_token(aoi_id)
    normalized_period = _normalize_public_token(period_id)
    if not normalized_aoi or not normalized_period:
        raise ValueError("Whole-monthly SR item id requires non-empty AOI id and period id.")
    collection_stem = _normalize_public_token(os.getenv("SR_COLLECTION_ID_MONTHLY", "sentinel-1sr-5m-monthly"))
    if not collection_stem:
        collection_stem = "sentinel-1sr-5m-monthly"
    return f"{collection_stem}_{normalized_period}_{normalized_aoi}"


def _summary_whole_monthly_item_id(summary: Dict[str, Any]) -> Optional[str]:
    period = summary.get("period") or {}
    if not period or summary.get("component") is not None:
        return None
    aoi_id = _summary_aoi_id(summary)
    period_id = period.get("period_id")
    if not aoi_id or not period_id:
        return None
    return _whole_monthly_sr_item_id(aoi_id, period_id)


def _summary_period_year_month(summary: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    period_id = _summary_period_token(summary)
    if not period_id:
        return None, None
    text = str(period_id).strip()
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        return text[:4], text[5:7]
    return None, None


def _resolve_sr_item_id(summary: Dict[str, Any], fallback: Optional[str | Path] = None) -> str:
    explicit = summary.get("public_item_id") or summary.get("item_id")
    if explicit:
        return str(explicit)
    monthly_item_id = _summary_whole_monthly_item_id(summary)
    if monthly_item_id:
        return monthly_item_id
    if fallback is not None:
        return Path(str(fallback)).stem
    output_tif = summary.get("output_tif")
    if output_tif:
        return Path(str(output_tif)).stem
    raise ValueError("Unable to resolve SR item id.")


def _resolve_sr_s3_prefix(summary: Dict[str, Any]) -> str:
    if summary.get("period") is not None and summary.get("component") is not None:
        return os.getenv("SR_S3_PREFIX_MONTHLY_COMPONENT", "issm-sar-sr-x2/monthly-component")
    if summary.get("period") is not None:
        return os.getenv("SR_S3_PREFIX_MONTHLY", "issm-sar-sr-x2/monthly")
    return os.getenv("SR_S3_PREFIX_DEFAULT", "issm-sar-sr-x2")


def _resolve_sr_object_key(summary: Dict[str, Any], item_id: str, filename: str) -> str:
    prefix = _resolve_sr_s3_prefix(summary).strip("/")
    aoi_id = _summary_aoi_id(summary) or "unknown-aoi"
    period_token = _summary_period_token(summary)
    parts = [prefix, aoi_id]
    year, month = _summary_period_year_month(summary)
    if summary.get("period") is not None and summary.get("component") is None and year and month:
        parts.extend([year, month])
    elif period_token:
        parts.append(str(period_token))
    parts.append(item_id)
    parts.append(filename)
    return "/".join(part.strip("/") for part in parts if str(part).strip("/"))


def _resolve_sr_href(
    summary: Dict[str, Any],
    item_id: str,
    filename: str | Path,
    *,
    mode: str,
    published_filename: Optional[str] = None,
) -> str:
    normalized_mode = str(mode or "local").strip().lower()
    if normalized_mode == "local":
        return _local_href(filename)
    object_key = _resolve_sr_object_key(summary, item_id, published_filename or Path(filename).name)
    if normalized_mode == "s3":
        bucket = os.getenv("SR_S3_BUCKET")
        if not bucket:
            raise ValueError("SR_S3_BUCKET is required when SR href mode is 's3'.")
        return f"s3://{bucket}/{object_key}"
    if normalized_mode == "http":
        base_url = os.getenv("SR_PUBLIC_BASE_URL")
        if not base_url:
            raise ValueError("SR_PUBLIC_BASE_URL is required when SR href mode is 'http'.")
        return _join_url(base_url, object_key)
    if normalized_mode == "stac_api":
        root_url = _resolve_sr_stac_root_url()
        collection_name = _resolve_sr_collection_name(summary)
        if not root_url:
            raise ValueError("SR_STAC_ROOT_URL or STAC_API_URL is required when item href mode is 'stac_api'.")
        return _join_url(root_url, "collections", collection_name, "items", item_id)
    raise ValueError(f"Unsupported SR href mode: {mode}")


def _resolve_sr_root_links(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    root_url = _resolve_sr_stac_root_url()
    if not root_url:
        return []
    collection_name = _resolve_sr_collection_name(summary)
    collection_href = _join_url(root_url, "collections", collection_name)
    return [
        {"rel": "collection", "type": "application/json", "href": collection_href},
        {"rel": "parent", "type": "application/json", "href": collection_href},
        {"rel": "root", "type": "application/json", "href": root_url.rstrip("/") + "/"},
    ]


def _resolve_sr_stac_root_url() -> str:
    return str(os.getenv("SR_STAC_ROOT_URL") or os.getenv("STAC_API_URL") or "").strip()


def _summarize_source_items(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        info = extract_item_info(item)
        item_id = str(info.get("id") or "")
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        self_href = None
        for link in item.get("links", []):
            if str(link.get("rel") or "").lower() == "self" and link.get("href"):
                self_href = str(link["href"])
                break
        summaries.append(
            {
                "id": item_id,
                "datetime": info.get("datetime"),
                "platform": info.get("platform"),
                "orbit_state": info.get("orbit_state"),
                "relative_orbit": info.get("relative_orbit"),
                "slice_number": info.get("slice_number"),
                "aoi_geometry_coverage": item.get("_aoi_coverage"),
                "aoi_bbox_coverage": item.get("_aoi_bbox_coverage"),
                "coverage_source": item.get("_coverage_source"),
                "self_href": self_href,
            }
        )
    return summaries


def _raster_asset_type(src: rasterio.io.DatasetReader) -> str:
    layout = str(src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT", "")).upper()
    if layout == "COG":
        return "image/tiff; application=geotiff; profile=cloud-optimized"
    return "image/tiff; application=geotiff"


def _raster_asset_metadata(
    path: str | Path,
    *,
    title: str,
    roles: List[str],
    description: Optional[str] = None,
    eo_bands: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    asset_path = Path(path)
    with rasterio.open(asset_path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        gsd = _pixel_size_meters(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
        )
        raster_bands: List[Dict[str, Any]] = []
        for band_index in range(src.count):
            band_meta: Dict[str, Any] = {
                "nodata": src.nodatavals[band_index],
                "data_type": src.dtypes[band_index],
            }
            if gsd is not None:
                band_meta["spatial_resolution"] = gsd
            raster_bands.append(to_jsonable(band_meta))

        left, bottom, right, top = src.bounds
        asset: Dict[str, Any] = {
            "href": _local_href(asset_path),
            "type": _raster_asset_type(src),
            "roles": roles,
            "title": title,
            "proj:bbox": [float(left), float(bottom), float(right), float(top)],
            "proj:shape": [int(src.height), int(src.width)],
            "proj:transform": list(src.transform)[:6],
            "raster:bands": raster_bands,
        }
        if description:
            asset["description"] = description
        if gsd is not None:
            asset["gsd"] = gsd
        if epsg is not None:
            asset["proj:epsg"] = epsg
        if eo_bands:
            asset["eo:bands"] = eo_bands
    return asset


def _json_asset_metadata(path: str | Path, *, title: str, roles: List[str], asset_type: str) -> Dict[str, Any]:
    asset_path = Path(path)
    return {
        "href": _local_href(asset_path),
        "type": asset_type,
        "roles": roles,
        "title": title,
    }


def _strip_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none_values(v) for v in value]
    return value


def write_sr_output_geojson(
    *,
    out_path: str | Path,
    summary: Dict[str, Any],
    geometry_wgs84: Dict[str, Any],
    infer_config: Dict[str, Any],
    source_t1_items: Optional[List[Dict[str, Any]]] = None,
    source_t2_items: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sr_vv_path = summary.get("output_sr_vv_tif")
    sr_vh_path = summary.get("output_sr_vh_tif")
    if not sr_vv_path or not sr_vh_path:
        raise ValueError("SR GeoJSON requires SR_VV and SR_VH outputs.")

    source_t1 = _summarize_source_items(source_t1_items)
    source_t2 = _summarize_source_items(source_t2_items)

    nominal_datetime = None
    start_datetime = None
    end_datetime = None
    pair_id = None
    if summary.get("period"):
        period = summary["period"]
        nominal_datetime = period.get("period_anchor_datetime")
        start_datetime = period.get("period_start")
        end_datetime = period.get("period_end")
        pair_id = summary.get("component", {}).get("pair_id") or f"period_{period.get('period_id')}"
    item_id = _resolve_sr_item_id(summary, fallback=out_path)
    collection_name = _resolve_sr_collection_name(summary)
    asset_href_mode = os.getenv("SR_ASSET_HREF_MODE", "s3")
    item_href_mode = os.getenv("SR_ITEM_SELF_MODE", "stac_api")
    include_local_source_paths = _env_flag("SR_INCLUDE_LOCAL_SOURCE_PATHS", default=False)
    whole_monthly_public = summary.get("period") is not None and summary.get("component") is None
    sr_vv_publish_name = f"{item_id}_vv.tif" if whole_monthly_public else Path(str(sr_vv_path)).name
    sr_vh_publish_name = f"{item_id}_vh.tif" if whole_monthly_public else Path(str(sr_vh_path)).name

    sr_vv_asset = _raster_asset_metadata(
        sr_vv_path,
        title="SR VV (x2)",
        roles=["data"],
        description="Super-resolved VV backscatter output masked by the valid geometry mask.",
        eo_bands=[{"name": "VV", "description": "Super-resolved VV backscatter"}],
    )
    sr_vh_asset = _raster_asset_metadata(
        sr_vh_path,
        title="SR VH (x2)",
        roles=["data"],
        description="Super-resolved VH backscatter output masked by the valid geometry mask.",
        eo_bands=[{"name": "VH", "description": "Super-resolved VH backscatter"}],
    )
    sr_vv_asset["href"] = _resolve_sr_href(
        summary,
        item_id,
        sr_vv_path,
        mode=asset_href_mode,
        published_filename=sr_vv_publish_name,
    )
    sr_vh_asset["href"] = _resolve_sr_href(
        summary,
        item_id,
        sr_vh_path,
        mode=asset_href_mode,
        published_filename=sr_vh_publish_name,
    )
    assets: Dict[str, Any] = {
        "sr_vv": sr_vv_asset,
        "sr_vh": sr_vh_asset,
    }

    gsd = sr_vv_asset.get("gsd") or sr_vh_asset.get("gsd")
    proj_epsg = sr_vv_asset.get("proj:epsg") or sr_vh_asset.get("proj:epsg")
    input_semantics = summary.get("inference_input_semantics") or {}

    common_properties: Dict[str, Any] = {
        "created": _utc_rfc3339(datetime.now(timezone.utc)),
        "datetime": nominal_datetime,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "gsd": gsd,
        "proj:epsg": proj_epsg,
        "constellation": "sentinel-1",
        "instruments": ["C-SAR"],
        "sar:instrument_mode": "IW",
        "sar:frequency_band": "C",
        "sar:product_type": "GRD",
        "sar:polarizations": ["VV", "VH"],
        "sr:method": "ISSM-SAR",
        "sr:model": "ISSM-SAR x2 dual-polarization",
        "sr:scale_factor": 2,
        "sr:product_version": os.getenv("SR_PRODUCT_VERSION", "v1"),
        "sr:publisher": os.getenv("SR_PUBLISHER", "EOV"),
        "sr:license": os.getenv("SR_LICENSE", "proprietary"),
    }
    if whole_monthly_public:
        properties: Dict[str, Any] = {
            **common_properties,
            "source:aoi_id": _summary_aoi_id(summary),
            "source:s1t1_items": source_t1,
            "source:s1t2_items": source_t2,
            "source:s1t1_scene_ids": [item["id"] for item in source_t1],
            "source:s1t2_scene_ids": [item["id"] for item in source_t2],
        }
    else:
        properties = {
            **common_properties,
            "sr:model_vv_checkpoint": Path(str(infer_config.get("ckpt_path_vv", ""))).name or None,
            "sr:model_vh_checkpoint": Path(str(infer_config.get("ckpt_path_vh", ""))).name or None,
            "source:bands_used": ["VV", "VH"],
            "source:workflow_mode": summary.get("workflow_mode"),
            "source:selection_strategy": summary.get("selection_strategy"),
            "source:pair_id": pair_id,
            "source:aoi_id": _summary_aoi_id(summary),
            "source:aoi_geojson": str(Path(summary["aoi_geojson"]).resolve()) if include_local_source_paths and summary.get("aoi_geojson") else None,
            "source:backend": resolve_workflow_backend(summary.get("workflow_mode")),
            "source:input_semantics": input_semantics,
            "source:s1t1_label": input_semantics.get("t1_label"),
            "source:s1t2_label": input_semantics.get("t2_label"),
            "source:s1t1_role": input_semantics.get("t1_role"),
            "source:s1t2_role": input_semantics.get("t2_role"),
            "source:s1t1_scene_count": len(source_t1),
            "source:s1t2_scene_count": len(source_t2),
            "source:s1t1_items": source_t1,
            "source:s1t2_items": source_t2,
            "source:s1t1_scene_ids": [item["id"] for item in source_t1],
            "source:s1t2_scene_ids": [item["id"] for item in source_t2],
        }
    if summary.get("period"):
        period = summary["period"]
        properties.update(
            {
                "source:period_id": period.get("period_id"),
                "source:period_mode": period.get("period_mode"),
                "source:period_start": period.get("period_start"),
                "source:period_end": period.get("period_end"),
                "source:period_anchor_datetime": period.get("period_anchor_datetime"),
            }
        )
    if summary.get("selection") and not whole_monthly_public:
        selection = summary["selection"]
        properties.update(
            {
                "source:selected_relaxation_level": selection.get("selected_relaxation_level"),
                "source:selected_relaxation_name": selection.get("selected_relaxation_name"),
                "source:required_scene_count": selection.get("required_scene_count"),
                "source:scene_signature_mode": selection.get("scene_signature_mode"),
                "source:scene_signature_value": selection.get("scene_signature_value"),
                "source:latest_input_datetime": selection.get("latest_input_datetime"),
            }
        )
    if summary.get("component"):
        component = summary["component"]
        properties.update(
            {
                "source:component_id": component.get("component_id"),
                "source:component_pair_id": component.get("pair_id"),
                "source:component_area_m2": component.get("area_m2"),
                "source:component_area_ratio_vs_parent": component.get("area_ratio_vs_parent"),
                "source:component_seed_item_ids": component.get("seed_item_ids"),
            }
        )
    links: List[Dict[str, Any]] = _resolve_sr_root_links(summary)
    links.append(
        {
            "rel": "self",
            "type": "application/json" if whole_monthly_public else "application/geo+json",
            "href": _resolve_sr_href(
                summary,
                item_id,
                out_path,
                mode=item_href_mode,
                published_filename=f"{item_id}.json" if whole_monthly_public else Path(out_path).name,
            ),
        }
    )
    seen_source_hrefs: set[str] = set()
    for item in source_t1 + source_t2:
        href = item.get("self_href")
        if not href or href in seen_source_hrefs:
            continue
        seen_source_hrefs.add(href)
        links.append(
            {
                "rel": "derived_from",
                "type": "application/geo+json",
                "href": href,
                "title": item.get("id"),
            }
        )

    feature = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/sar/v1.0.0/schema.json",
        ],
        "id": item_id,
        "collection": collection_name,
        "geometry": geometry_wgs84,
        "bbox": canonical_bbox_from_geometry(geometry_wgs84),
        "properties": _strip_none_values(properties),
        "assets": _strip_none_values(assets),
        "links": links,
    }
    out_path.write_text(json.dumps(to_jsonable(feature), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return out_path


def attach_sr_output_geojson(
    *,
    summary: Dict[str, Any],
    geometry_wgs84: Dict[str, Any],
    infer_config: Dict[str, Any],
    source_t1_items: Optional[List[Dict[str, Any]]] = None,
    source_t2_items: Optional[List[Dict[str, Any]]] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    output_tif = summary.get("output_tif")
    if out_path is None:
        preferred_source = summary.get("output_sr_vv_tif") or summary.get("output_sr_vh_tif") or output_tif
        if not preferred_source:
            raise ValueError("SR metadata path requires at least one SR output band or explicit out_path.")
        item_id = _resolve_sr_item_id(summary, fallback=preferred_source)
        preferred_dir = Path(str(preferred_source)).parent
        out_path = preferred_dir / f"{item_id}.json"
    geojson_path = write_sr_output_geojson(
        out_path=out_path,
        summary=summary,
        geometry_wgs84=geometry_wgs84,
        infer_config=infer_config,
        source_t1_items=source_t1_items,
        source_t2_items=source_t2_items,
    )
    summary["output_sr_geojson_path"] = str(geojson_path)
    return geojson_path

__all__ = [
    "_summary_aoi_id",
    "_summary_period_token",
    "_resolve_sr_href",
    "_resolve_sr_root_links",
    "_whole_monthly_sr_item_id",
    "attach_sr_output_geojson",
    "write_sr_output_geojson",
]

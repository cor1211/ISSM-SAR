from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from runtime_logging import emit_runtime_log, safe_text_snippet
from stac_support.stac_time_support import normalize_datetime_range

class STACClient:
    """Lightweight STAC API client su dung requests."""

    def __init__(self, api_url: str):
        self.api_url = str(api_url or "").strip().rstrip("/")
        if not self.api_url:
            raise ValueError("STAC API URL is required. Set STAC_API_URL or pass --stac-url.")
        self.session = requests.Session()
        emit_runtime_log("query_stac_download", logging.INFO, "Initialized STAC client", stac_url=self.api_url)

    def search_items(
        self,
        collection: str,
        bbox: Optional[List[float]] = None,
        intersects: Optional[Dict[str, Any]] = None,
        datetime_range: Optional[str] = None,
        limit: int = 200,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query STAC /search, then fallback to GET /search, then collection items."""
        datetime_range = normalize_datetime_range(datetime_range)
        search_url = f"{self.api_url}/searchh"
        base_payload: Dict[str, Any] = {
            "collections": [collection],
            "limit": max(1, int(limit)),
        }
        if datetime_range:
            base_payload["datetime"] = datetime_range
        if query:
            base_payload["query"] = query

        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Starting STAC search",
            collection=collection,
            datetime=datetime_range,
            limit=max(1, int(limit)),
            has_bbox=bool(bbox),
            has_intersects=bool(intersects),
            has_query=bool(query),
        )

        def _try_post(payload: Dict[str, Any], tag: str) -> Optional[List[Dict[str, Any]]]:
            request_started = datetime.now()
            try:
                resp = self.session.post(search_url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
                emit_runtime_log(
                    "query_stac_download",
                    logging.INFO,
                    "STAC POST /search succeeded",
                    search_mode=tag,
                    item_count=len(features),
                    duration_ms=duration_ms,
                    http_status=resp.status_code,
                )
                return features
            except Exception as e:
                status_code = None
                err_text = ""
                try:
                    if "resp" in locals():
                        status_code = resp.status_code
                        err_text = safe_text_snippet(resp.text, limit=300)
                except Exception:
                    err_text = ""
                emit_runtime_log(
                    "query_stac_download",
                    logging.WARNING,
                    "STAC POST /search failed",
                    search_mode=tag,
                    http_status=status_code,
                    error=e,
                    response_body=err_text or None,
                )
                return None

        def _build_search_get_params(
            *,
            use_bbox: bool,
            use_intersects: bool,
        ) -> Dict[str, Any]:
            params: Dict[str, Any] = {
                "collections": collection,
                "limit": max(1, int(limit)),
            }
            if datetime_range:
                params["datetime"] = datetime_range
            if use_bbox and bbox:
                params["bbox"] = ",".join(str(v) for v in bbox)
            if use_intersects and intersects:
                params["intersects"] = json.dumps(intersects, separators=(",", ":"))
            if query:
                # GET /search expects a JSON string payload in the query parameter.
                params["query"] = json.dumps(query, separators=(",", ":"))
            return params

        def _try_get_search(params: Dict[str, Any], tag: str) -> Optional[List[Dict[str, Any]]]:
            request_started = datetime.now()
            try:
                resp = self.session.get(search_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
                emit_runtime_log(
                    "query_stac_download",
                    logging.INFO,
                    "STAC GET /search succeeded",
                    search_mode=tag,
                    item_count=len(features),
                    duration_ms=duration_ms,
                    http_status=resp.status_code,
                )
                return features
            except Exception as e:
                status_code = None
                err_text = ""
                try:
                    if "resp" in locals():
                        status_code = resp.status_code
                        err_text = safe_text_snippet(resp.text, limit=300)
                except Exception:
                    err_text = ""
                emit_runtime_log(
                    "query_stac_download",
                    logging.WARNING,
                    "STAC GET /search failed",
                    search_mode=tag,
                    http_status=status_code,
                    error=e,
                    response_body=err_text or None,
                )
                return None

        # Luu y: nhieu STAC backend khong chap nhan gui dong thoi bbox + intersects.
        # Vi vay chi gui 1 trong 2 moi lan thu.
        if intersects:
            payload_intersects = dict(base_payload)
            payload_intersects["intersects"] = intersects
            out = _try_post(payload_intersects, "intersects")
            if out is not None:
                return out

            if bbox:
                payload_bbox = dict(base_payload)
                payload_bbox["bbox"] = bbox
                out = _try_post(payload_bbox, "bbox-fallback")
                if out is not None:
                    return out
        else:
            payload_bbox = dict(base_payload)
            if bbox:
                payload_bbox["bbox"] = bbox
            out = _try_post(payload_bbox, "bbox")
            if out is not None:
                return out

        if intersects:
            out = _try_get_search(_build_search_get_params(use_bbox=False, use_intersects=True), "intersects")
            if out is not None:
                return out
            if bbox:
                out = _try_get_search(_build_search_get_params(use_bbox=True, use_intersects=False), "bbox-fallback")
                if out is not None:
                    return out
        else:
            out = _try_get_search(_build_search_get_params(use_bbox=bool(bbox), use_intersects=False), "bbox")
            if out is not None:
                return out

        emit_runtime_log(
            "query_stac_download",
            logging.WARNING,
            "Falling back to STAC collection items after /search failed",
            collection=collection,
            datetime=datetime_range,
            has_bbox=bool(bbox),
            has_intersects=bool(intersects),
            has_query=bool(query),
        )

        items_url = f"{self.api_url}/collections/{collection}/items"
        params: Dict[str, Any] = {"limit": max(1, int(limit))}
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        if datetime_range:
            params["datetime"] = datetime_range
        if query:
            # Fallback GET khong ho tro query day du tren moi STAC implementation.
            # Van gui de tan dung neu server co support.
            params["query"] = json.dumps(query)

        try:
            request_started = datetime.now()
            resp = self.session.get(items_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "STAC GET /collections/{collection}/items succeeded",
                item_count=len(features),
                duration_ms=duration_ms,
                http_status=resp.status_code,
            )
            return features
        except Exception as e:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "Unable to query STAC API",
                error=e,
                collection=collection,
                datetime=datetime_range,
                has_bbox=bool(bbox),
            )
            return []

    def get_item(self, collection: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Lay item theo ID, uu tien endpoint chuan /collections/{id}/items/{item_id}."""
        direct_url = f"{self.api_url}/collections/{collection}/items/{item_id}"
        try:
            resp = self.session.get(direct_url, timeout=30)
            if resp.status_code == 200:
                emit_runtime_log(
                    "query_stac_download",
                    logging.DEBUG,
                    "Fetched STAC item by direct endpoint",
                    collection=collection,
                    item_id=item_id,
                    http_status=resp.status_code,
                )
                return resp.json()
        except Exception:
            pass

        # Fallback 1: STAC ids extension
        try:
            payload = {"collections": [collection], "ids": [item_id], "limit": 1}
            resp = self.session.post(f"{self.api_url}/search", json=payload, timeout=30)
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    emit_runtime_log(
                        "query_stac_download",
                        logging.DEBUG,
                        "Fetched STAC item by ids extension fallback",
                        collection=collection,
                        item_id=item_id,
                    )
                    return features[0]
        except Exception:
            pass

        # Fallback 2: GET /search ids form, matching this eoAPI OpenAPI contract.
        try:
            params = {"collections": collection, "ids": item_id, "limit": 1}
            resp = self.session.get(f"{self.api_url}/search", params=params, timeout=30)
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    emit_runtime_log(
                        "query_stac_download",
                        logging.DEBUG,
                        "Fetched STAC item by GET /search ids fallback",
                        collection=collection,
                        item_id=item_id,
                    )
                    return features[0]
        except Exception:
            pass

        # Fallback 3: query id eq
        emit_runtime_log(
            "query_stac_download",
            logging.DEBUG,
            "Falling back to STAC id query",
            collection=collection,
            item_id=item_id,
        )
        features = self.search_items(
            collection=collection,
            limit=1,
            query={"id": {"eq": item_id}},
        )
        return features[0] if features else None

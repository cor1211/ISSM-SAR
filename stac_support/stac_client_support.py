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
        """Query STAC items via canonical POST /search (eoAPI-compatible payload)."""
        datetime_range = normalize_datetime_range(datetime_range)
        search_url = f"{self.api_url}/search"
        payload: Dict[str, Any] = {
            "collections": [collection],
            "limit": max(1, int(limit)),
        }
        if datetime_range:
            payload["datetime"] = datetime_range
        if query:
            payload["query"] = query
        # eoAPI supports both bbox and intersects; prefer intersects when available.
        if intersects:
            payload["intersects"] = intersects
        elif bbox:
            payload["bbox"] = bbox

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

        try:
            request_started = datetime.now()
            resp = self.session.post(search_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "STAC POST /search succeeded",
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
                error=e,
                http_status=status_code,
                response_body=err_text or None,
                collection=collection,
                datetime=datetime_range,
                has_bbox=bool(bbox),
                has_intersects=bool(intersects),
            )

        get_params: Dict[str, Any] = {
            "collections": collection,
            "limit": max(1, int(limit)),
        }
        if datetime_range:
            get_params["datetime"] = datetime_range
        if intersects:
            get_params["intersects"] = json.dumps(intersects, separators=(",", ":"))
        elif bbox:
            get_params["bbox"] = ",".join(str(v) for v in bbox)
        if query:
            get_params["query"] = json.dumps(query, separators=(",", ":"))

        try:
            request_started = datetime.now()
            resp = self.session.get(search_url, params=get_params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            duration_ms = int((datetime.now() - request_started).total_seconds() * 1000)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "STAC GET /search fallback succeeded",
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
                logging.ERROR,
                "STAC GET /search fallback failed",
                error=e,
                http_status=status_code,
                response_body=err_text or None,
                collection=collection,
                datetime=datetime_range,
                has_bbox=bool(bbox),
                has_intersects=bool(intersects),
            )
            return []

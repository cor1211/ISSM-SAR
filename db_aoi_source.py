from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from runtime_logging import emit_runtime_log


logger = logging.getLogger("db_aoi_source")


def _load_dotenv_values(env_path: str | Path = ".env") -> Dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}
    try:
        from dotenv import dotenv_values
    except ModuleNotFoundError:
        return {}
    return {str(k): str(v) for k, v in (dotenv_values(path) or {}).items() if v is not None}


def inspect_database_settings(env_path: str | Path = ".env") -> Dict[str, Any]:
    dotenv_values = _load_dotenv_values(env_path)
    resolved: Dict[str, Dict[str, Any]] = {}
    missing_keys: List[str] = []
    for key in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        env_value = os.getenv(key)
        dotenv_value = dotenv_values.get(key)
        value = env_value or dotenv_value
        present = bool(value)
        if not present:
            missing_keys.append(key)
        resolved[key] = {
            "present": present,
            "source": "env" if env_value else ("dotenv" if dotenv_value else None),
            "value": None if key == "PGPASSWORD" else value,
        }
    return {
        "env_path": str(Path(env_path)),
        "resolved": resolved,
        "missing_keys": missing_keys,
    }


def resolve_database_settings(env_path: str | Path = ".env") -> Dict[str, str]:
    inspected = inspect_database_settings(env_path)
    settings: Dict[str, str] = {}
    for key in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        value = os.getenv(key) or inspected["resolved"][key]["value"]
        if key == "PGPASSWORD" and not value:
            value = os.getenv(key) or _load_dotenv_values(env_path).get(key)
        if not value:
            raise RuntimeError(
                f"Missing database setting `{key}`. Export it in the environment or provide it in {env_path}."
            )
        settings[key] = value
    emit_runtime_log(
        "db_aoi_source",
        logging.DEBUG,
        "Resolved database settings",
        env_path=inspected["env_path"],
        pghost=inspected["resolved"]["PGHOST"]["value"],
        pgport=inspected["resolved"]["PGPORT"]["value"],
        pguser=inspected["resolved"]["PGUSER"]["value"],
        pgdatabase=inspected["resolved"]["PGDATABASE"]["value"],
        pgpassword_present=inspected["resolved"]["PGPASSWORD"]["present"],
    )
    return settings


def _import_pg8000() -> Any:
    try:
        import pg8000
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Database-backed AOI mode requires `pg8000`. Install dependencies first or keep using --geojson."
        ) from exc
    return pg8000


def normalize_aoi_uuid(aoi_id: str) -> str:
    return str(UUID(str(aoi_id).strip()))


def fetch_active_aois_from_database(
    *,
    aoi_id: Optional[str] = None,
    limit: Optional[int] = None,
    env_path: str | Path = ".env",
) -> List[Dict[str, Any]]:
    inspected = inspect_database_settings(env_path)
    emit_runtime_log(
        "db_aoi_source",
        logging.INFO,
        "Preparing ACTIVE AOI database query",
        env_path=inspected["env_path"],
        requested_aoi_id=(normalize_aoi_uuid(aoi_id) if aoi_id is not None else None),
        limit=limit,
        pghost=inspected["resolved"]["PGHOST"]["value"],
        pgport=inspected["resolved"]["PGPORT"]["value"],
        pguser=inspected["resolved"]["PGUSER"]["value"],
        pgdatabase=inspected["resolved"]["PGDATABASE"]["value"],
        pghost_present=inspected["resolved"]["PGHOST"]["present"],
        pgport_present=inspected["resolved"]["PGPORT"]["present"],
        pguser_present=inspected["resolved"]["PGUSER"]["present"],
        pgdatabase_present=inspected["resolved"]["PGDATABASE"]["present"],
        pgpassword_present=inspected["resolved"]["PGPASSWORD"]["present"],
        missing_keys=inspected["missing_keys"],
    )
    if inspected["missing_keys"]:
        emit_runtime_log(
            "db_aoi_source",
            logging.WARNING,
            "Database configuration is incomplete",
            env_path=inspected["env_path"],
            missing_keys=inspected["missing_keys"],
        )

    settings = resolve_database_settings(env_path)
    pg8000 = _import_pg8000()

    sql = """
        SELECT
            id::text,
            COALESCE(name, '') AS name,
            status::text,
            ST_AsGeoJSON(
                CASE
                    WHEN ST_SRID(geom) = 4326 THEN geom
                    ELSE ST_Transform(geom, 4326)
                END
            )::text AS geom_geojson,
            ST_GeometryType(geom) AS geom_type,
            ST_SRID(geom) AS geom_srid,
            ST_IsValid(geom) AS geom_is_valid,
            CASE
                WHEN NOT ST_IsValid(geom) THEN ST_IsValidReason(geom)
                ELSE NULL
            END AS geom_invalid_reason
        FROM public.aois
        WHERE status = 'ACTIVE'
    """
    params: List[Any] = []
    if aoi_id is not None:
        sql += " AND id = CAST(%s AS uuid)"
        params.append(normalize_aoi_uuid(aoi_id))
    sql += " ORDER BY created_at DESC NULLS LAST, id"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    conn = pg8000.connect(
        host=settings["PGHOST"],
        port=int(settings["PGPORT"]),
        user=settings["PGUSER"],
        password=settings["PGPASSWORD"],
        database=settings["PGDATABASE"],
        timeout=8,
    )
    emit_runtime_log(
        "db_aoi_source",
        logging.INFO,
        "Connected to database for AOI query",
        host=settings["PGHOST"],
        port=settings["PGPORT"],
        user=settings["PGUSER"],
        database=settings["PGDATABASE"],
        query_mode=("db_single" if aoi_id is not None else "db_batch"),
    )
    try:
        cursor = conn.cursor()
        emit_runtime_log("db_aoi_source", logging.INFO, "Issuing read-only AOI transaction", phase="BEGIN READ ONLY")
        cursor.execute("BEGIN READ ONLY")
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        emit_runtime_log(
            "db_aoi_source",
            logging.INFO,
            "AOI query returned rows",
            row_count=len(rows),
            requested_aoi_id=(normalize_aoi_uuid(aoi_id) if aoi_id is not None else None),
            limit=limit,
        )
        cursor.execute("ROLLBACK")
        emit_runtime_log("db_aoi_source", logging.INFO, "Read-only AOI transaction closed", phase="ROLLBACK")
    finally:
        conn.close()
        emit_runtime_log("db_aoi_source", logging.DEBUG, "Database connection closed")

    records: List[Dict[str, Any]] = []
    for row in rows:
        (
            row_id,
            name,
            status,
            geom_geojson,
            geom_type,
            geom_srid,
            geom_is_valid,
            geom_invalid_reason,
        ) = row
        records.append(
            {
                "id": str(row_id),
                "name": str(name or ""),
                "status": str(status),
                "geometry": json.loads(str(geom_geojson)),
                "geometry_type": str(geom_type),
                "geometry_srid": int(geom_srid),
                "geometry_is_valid": bool(geom_is_valid),
                "geometry_invalid_reason": None if geom_invalid_reason is None else str(geom_invalid_reason),
            }
        )
    return records


def materialize_database_aoi_geojson(
    record: Dict[str, Any],
    output_dir: str | Path,
    filename: Optional[str] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (filename or f"{record['id']}.geojson")
    feature = {
        "type": "Feature",
        "geometry": record["geometry"],
        "properties": {
            "id": record["id"],
            "status": record["status"],
        },
    }
    if record.get("name"):
        feature["properties"]["name"] = record["name"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feature, f, indent=2, ensure_ascii=False)
    emit_runtime_log(
        "db_aoi_source",
        logging.DEBUG,
        "Materialized temporary AOI GeoJSON for pipeline handoff",
        aoi_id=record.get("id"),
        output_path=out_path,
        geometry_type=record.get("geometry_type"),
        geometry_srid=record.get("geometry_srid"),
    )
    return out_path

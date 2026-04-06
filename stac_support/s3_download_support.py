from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import rasterio
from rasterio.features import bounds as geometry_bounds
from rasterio.session import AWSSession
from rasterio.warp import transform_geom
from rasterio.windows import Window, from_bounds

from runtime_logging import detect_s3_credential_source, emit_runtime_log
from stac_support.stac_geometry_support import bbox_intersection_bounds, normalize_polygonal_geojson_geometry

try:
    import boto3
except ImportError:
    boto3 = None

class S3Downloader:
    """Download asset tu S3 bang boto3."""

    def __init__(self):
        if boto3 is None:
            raise RuntimeError("Chua cai boto3. Vui long cai 'pip install boto3' de dung tinh nang download.")
        access_key = os.getenv("S3_ACCESS_KEY")
        secret_key = os.getenv("S3_SECRET_KEY")
        client_kwargs: Dict[str, Any] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        s3_endpoint = os.getenv("S3_ENDPOINT")
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint
        credential_source = detect_s3_credential_source()
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Initialized S3 downloader",
            s3_endpoint=s3_endpoint,
            credential_source=credential_source,
            s3_access_key_present=bool(access_key),
            s3_secret_key_present=bool(secret_key),
            explicit_env_present=bool(access_key and secret_key),
        )
        if credential_source == "none":
            emit_runtime_log(
                "query_stac_download",
                logging.WARNING,
                "S3 credentials are not explicitly configured",
                credential_source=credential_source,
                note="The pipeline will continue and only fail if S3 access is required later.",
            )
        self.client = boto3.client("s3", **client_kwargs)

    @staticmethod
    def parse_href_to_bucket_key(href: str) -> Optional[Tuple[str, str]]:
        """Parse href (s3:// hoac http(s)) -> (bucket, key)."""
        parsed = urlparse(href)
        if parsed.scheme == "s3":
            return parsed.netloc, parsed.path.lstrip("/")

        if parsed.scheme in ("http", "https"):
            path = parsed.path.lstrip("/")
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

            # Virtual-host style: https://bucket.endpoint/key
            host = parsed.netloc.split(":")[0]
            host_parts = host.split(".")
            if len(host_parts) >= 2 and host_parts[0] not in {"s3", "minio", "storage"} and path:
                return host_parts[0], path
            return None
        return None

    @staticmethod
    def _key_tail(key: Optional[str], parts: int = 3) -> Optional[str]:
        if not key:
            return None
        pieces = [piece for piece in str(key).split("/") if piece]
        if not pieces:
            return None
        return "/".join(pieces[-max(1, int(parts)):])

    def download_aoi_subset_from_href(
        self,
        href: str,
        local_path: str,
        aoi_geometry_wgs84: Dict[str, Any],
    ) -> bool:
        """
        Doc remote raster va chi ghi ra bbox giao nhau giua AOI va item bounds.
        Khong tai full item ve local.
        """
        raster_path = href_to_rasterio_path(href)
        env_kwargs = build_rasterio_env_kwargs()

        try:
            access_key = os.getenv("S3_ACCESS_KEY")
            secret_key = os.getenv("S3_SECRET_KEY")
            aws_session = None
            if boto3 is not None and access_key and secret_key:
                b3_session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                aws_session = AWSSession(b3_session)

            env = rasterio.Env(session=aws_session, **env_kwargs) if aws_session is not None else rasterio.Env(**env_kwargs)
            with env:
                with rasterio.open(raster_path, "r") as src:
                    if src.crs is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because raster has no CRS",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    clip_geometry_wgs84 = normalize_polygonal_geojson_geometry(aoi_geometry_wgs84)
                    if clip_geometry_wgs84 is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because clip geometry has no polygonal area",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    aoi_in_src = transform_geom("EPSG:4326", src.crs, clip_geometry_wgs84, antimeridian_cutting=True, precision=15)
                    aoi_in_src = normalize_polygonal_geojson_geometry(aoi_in_src)
                    if aoi_in_src is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because transformed clip geometry has no polygonal area",
                            href=href,
                            local_path=local_path,
                            raster_crs=src.crs,
                        )
                        return False
                    aoi_bounds_src = list(geometry_bounds(aoi_in_src))
                    src_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    clip_bbox = bbox_intersection_bounds(aoi_bounds_src, src_bounds)
                    if clip_bbox is None:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.WARNING,
                            "Subset skipped because AOI does not intersect raster bounds",
                            href=href,
                            local_path=local_path,
                            raster_crs=src.crs,
                        )
                        return False

                    # Snap outward theo pixel grid:
                    # - floor offsets (col/row start)
                    # - ceil max extents (col/row end)
                    # de dam bao subset bao kin AOI bounds, tranh bi "hut" 1 phan pixel o goc/canh.
                    raw_win = from_bounds(*clip_bbox, transform=src.transform)
                    col_off = int(math.floor(raw_win.col_off))
                    row_off = int(math.floor(raw_win.row_off))
                    col_max = int(math.ceil(raw_win.col_off + raw_win.width))
                    row_max = int(math.ceil(raw_win.row_off + raw_win.height))

                    col_off = max(0, col_off)
                    row_off = max(0, row_off)
                    col_max = min(src.width, col_max)
                    row_max = min(src.height, row_max)

                    win = Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=max(0, col_max - col_off),
                        height=max(0, row_max - row_off),
                    )
                    if win.width < 1 or win.height < 1:
                        emit_runtime_log(
                            "query_stac_download",
                            logging.ERROR,
                            "Subset failed because clipped raster window is empty",
                            href=href,
                            local_path=local_path,
                        )
                        return False

                    data = src.read(window=win)
                    out_transform = src.window_transform(win)

                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=int(win.width),
                        height=int(win.height),
                        transform=out_transform,
                    )
                    # Tranh loi tile size khi cua so subset nho hon block size goc.
                    profile.pop("tiled", None)
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)

                    out_file = Path(local_path)
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_file, "w", **profile) as dst:
                        dst.write(data)
            parsed = self.parse_href_to_bucket_key(href)
            emit_runtime_log(
                "query_stac_download",
                logging.DEBUG,
                "Completed AOI subset download",
                bucket=(parsed[0] if parsed else None),
                key_tail=(self._key_tail(parsed[1]) if parsed else None),
                local_name=Path(local_path).name,
                raster_name=Path(raster_path).name,
            )
            return True
        except Exception as e:
            parsed = self.parse_href_to_bucket_key(href)
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "AOI subset download failed",
                href=href,
                bucket=(parsed[0] if parsed else None),
                key=(parsed[1] if parsed else None),
                local_path=local_path,
                error=e,
            )
            return False

def href_to_rasterio_path(href: str) -> str:
    """Chuyen href thanh duong dan GDAL VSI neu can."""
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}/{parsed.path.lstrip('/')}"
    if parsed.scheme in ("http", "https"):
        return f"/vsicurl/{href}"
    return href

def build_rasterio_env_kwargs() -> Dict[str, Any]:
    """Build cac env cho rasterio de doc duoc MinIO/S3."""
    env: Dict[str, Any] = {}
    endpoint = os.getenv("S3_ENDPOINT")

    if endpoint:
        p = urlparse(endpoint)
        if p.scheme:
            env["AWS_HTTPS"] = "YES" if p.scheme == "https" else "NO"
            env["AWS_S3_ENDPOINT"] = p.netloc
        else:
            env["AWS_S3_ENDPOINT"] = endpoint
        env["AWS_VIRTUAL_HOSTING"] = "FALSE"

    return env

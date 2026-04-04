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

    def download_from_href(self, href: str, local_path: str) -> bool:
        """Tai file tu href ve local_path."""
        parsed = self.parse_href_to_bucket_key(href)
        if not parsed:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "Failed to parse asset href for download",
                href=href,
                local_path=local_path,
            )
            return False
        bucket, key = parsed

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        emit_runtime_log(
            "query_stac_download",
            logging.INFO,
            "Starting S3 asset download",
            href=href,
            bucket=bucket,
            key=key,
            local_path=local_path,
        )

        try:
            head = self.client.head_object(Bucket=bucket, Key=key)
            size_mb = head["ContentLength"] / (1024 * 1024)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "Resolved S3 object metadata",
                bucket=bucket,
                key=key,
                size_mb=f"{size_mb:.1f}",
            )

            local_file = Path(local_path)
            if local_file.exists() and local_file.stat().st_size == int(head["ContentLength"]):
                emit_runtime_log(
                    "query_stac_download",
                    logging.INFO,
                    "Skipping download because local file already matches remote size",
                    bucket=bucket,
                    key=key,
                    local_path=local_path,
                )
                return True

            downloaded = [0]
            total_size = max(1, int(head["ContentLength"]))
            progress_state = {"last_print": -5}

            def progress(bytes_amount: int) -> None:
                downloaded[0] += bytes_amount
                pct = downloaded[0] / total_size * 100
                bucket_pct = int(pct // 5) * 5
                if bucket_pct > progress_state["last_print"]:
                    progress_state["last_print"] = bucket_pct
                    emit_runtime_log(
                        "query_stac_download",
                        logging.DEBUG,
                        "S3 download progress",
                        bucket=bucket,
                        key=key,
                        progress_pct=f"{pct:.1f}",
                    )

            self.client.download_file(bucket, key, local_path, Callback=progress)
            emit_runtime_log(
                "query_stac_download",
                logging.INFO,
                "Completed S3 asset download",
                bucket=bucket,
                key=key,
                local_path=local_path,
            )
            return True
        except Exception as e:
            emit_runtime_log(
                "query_stac_download",
                logging.ERROR,
                "S3 asset download failed",
                href=href,
                bucket=bucket,
                key=key,
                local_path=local_path,
                error=e,
            )
            return False

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
            elif boto3 is None and access_key and secret_key:
                env_kwargs["AWS_ACCESS_KEY_ID"] = access_key
                env_kwargs["AWS_SECRET_ACCESS_KEY"] = secret_key

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

    def download_item_assets(
        self,
        assets: Dict[str, Any],
        out_dir: str,
        item_id: str,
        asset_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Tai cac asset key can thiet cua 1 item."""
        downloaded_paths: List[str] = []
        keys = asset_keys or list(assets.keys())
        for key in keys:
            if key not in assets:
                emit_runtime_log(
                    "query_stac_download",
                    logging.WARNING,
                    "Requested asset key is missing from STAC item",
                    asset_key=key,
                    item_id=item_id,
                )
                continue
            asset = assets[key]
            href = str(asset.get("href", ""))
            if not href:
                continue

            ext = Path(urlparse(href).path).suffix or ".tif"
            local_path = str(Path(out_dir) / f"{item_id}_{key}{ext}")
            if self.download_from_href(href, local_path):
                downloaded_paths.append(local_path)
        return downloaded_paths

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

def probe_rasterio_href(label: str, href: str) -> bool:
    """Thu mo href bang rasterio va in metadata chinh."""
    parsed = urlparse(href)
    if parsed.scheme == "s3" and boto3 is None:
        print(f"[RASTERIO] {label} -> Loi: can cai boto3 de doc authenticated s3:// href")
        return False

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
        elif boto3 is None and access_key and secret_key:
            # Fallback cho moi truong chua co boto3 package.
            env_kwargs["AWS_ACCESS_KEY_ID"] = access_key
            env_kwargs["AWS_SECRET_ACCESS_KEY"] = secret_key

        env = rasterio.Env(session=aws_session, **env_kwargs) if aws_session is not None else rasterio.Env(**env_kwargs)
        with env:
            with rasterio.open(raster_path, "r") as src:
                print(f"[RASTERIO] {label}")
                print(f"  path: {raster_path}")
                print(f"  size: {src.width}x{src.height}, bands={src.count}, dtype={src.dtypes[0]}")
                print(f"  crs: {src.crs}")
                print(f"  transform: {src.transform}")
        return True
    except Exception as e:
        print(f"[RASTERIO] {label} -> Loi: {e}")
        return False

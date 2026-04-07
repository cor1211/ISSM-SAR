from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv


class PublishError(RuntimeError):
    pass


def _resolve_env_value(values: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        raw = values.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


def _utc_rfc3339_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _join_url(base: str, *parts: str) -> str:
    current = str(base or "").rstrip("/")
    for part in parts:
        current = f"{current}/{str(part).strip('/')}"
    return current


def _normalize_url(url: str) -> str:
    return str(url or "").rstrip("/")


def _find_link(item: Dict[str, Any], rel: str) -> Optional[str]:
    for link in item.get("links", []):
        if str(link.get("rel") or "").lower() == rel.lower() and link.get("href"):
            return str(link["href"])
    return None


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise PublishError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _head_object_exists(s3_client: Any, bucket: str, key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return True, response
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", "")).lower()
        if code in {"404", "nosuchkey", "notfound"}:
            return False, None
        raise


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise PublishError(message)


@dataclass
class PublishArtifact:
    role: str
    local_path: Path
    bucket: str
    key: str
    href: str
    content_type: str

    def to_report(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "local_path": str(self.local_path),
            "bucket": self.bucket,
            "key": self.key,
            "href": self.href,
            "content_type": self.content_type,
            "size_bytes": self.local_path.stat().st_size if self.local_path.exists() else None,
        }


@dataclass
class PublishPlan:
    item_json_path: Path
    item_id: str
    collection_id: str
    stac_root_url: str
    collection_url: str
    item_url: str
    item_payload: Dict[str, Any]
    artifacts: List[PublishArtifact]
    expected_s3_bucket: Optional[str]

    def to_report(self) -> Dict[str, Any]:
        return {
            "item_json_path": str(self.item_json_path),
            "item_id": self.item_id,
            "collection_id": self.collection_id,
            "expected_s3_bucket": self.expected_s3_bucket,
            "stac_root_url": self.stac_root_url,
            "collection_url": self.collection_url,
            "item_url": self.item_url,
            "artifacts": [artifact.to_report() for artifact in self.artifacts],
        }


def build_s3_client_from_env(env: Optional[Dict[str, str]] = None) -> Any:
    import boto3

    values = dict(os.environ)
    if env:
        values.update({k: str(v) for k, v in env.items() if v is not None})
    endpoint_url = _resolve_env_value(values, "SR_S3_ENDPOINT", "S3_ENDPOINT")
    access_key = _resolve_env_value(values, "SR_S3_ACCESS_KEY", "S3_ACCESS_KEY")
    secret_key = _resolve_env_value(values, "SR_S3_SECRET_KEY", "S3_SECRET_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def build_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def _derive_json_artifact(item_json_path: Path, item_id: str, vv_bucket: str, vv_key: str) -> PublishArtifact:
    directory = str(Path(vv_key).parent).replace("\\", "/")
    json_key = f"{directory}/{item_id}.json"
    return PublishArtifact(
        role="item_json",
        local_path=item_json_path,
        bucket=vv_bucket,
        key=json_key,
        href=f"s3://{vv_bucket}/{json_key}",
        content_type="application/json",
    )


def _validate_item_links(
    item: Dict[str, Any],
    *,
    collection_id: str,
    item_id: str,
    env: Dict[str, str],
) -> Tuple[str, str, str]:
    root_href = _find_link(item, "root")
    collection_href = _find_link(item, "collection")
    self_href = _find_link(item, "self")

    _ensure(root_href is not None, "Item JSON is missing the STAC root link.")
    _ensure(collection_href is not None, "Item JSON is missing the STAC collection link.")
    _ensure(self_href is not None, "Item JSON is missing the STAC self link.")

    env_root = _resolve_env_value(env, "SR_STAC_ROOT_URL", "STAC_API_URL")
    if env_root:
        _ensure(
            _normalize_url(root_href) == _normalize_url(env_root),
            f"Item root link {_normalize_url(root_href)} does not match configured STAC root {_normalize_url(env_root)}.",
        )

    expected_collection_url = _join_url(root_href, "collections", collection_id)
    expected_item_url = _join_url(expected_collection_url, "items", item_id)
    _ensure(
        _normalize_url(collection_href) == _normalize_url(expected_collection_url),
        f"Item collection link {collection_href} does not match expected {expected_collection_url}.",
    )
    _ensure(
        _normalize_url(self_href) == _normalize_url(expected_item_url),
        f"Item self link {self_href} does not match expected {expected_item_url}.",
    )
    return root_href, collection_href, self_href


def build_publish_plan(
    *,
    item_json_path: str | Path,
    env: Optional[Dict[str, str]] = None,
) -> PublishPlan:
    values = dict(os.environ)
    if env:
        values.update({k: str(v) for k, v in env.items() if v is not None})

    item_json = Path(item_json_path)
    _ensure(item_json.exists(), f"Item JSON not found: {item_json}")
    item = json.loads(item_json.read_text(encoding="utf-8"))

    item_id = str(item.get("id") or "").strip()
    collection_id = str(item.get("collection") or "").strip()
    _ensure(item_id, "Item JSON is missing 'id'.")
    _ensure(collection_id, "Item JSON is missing 'collection'.")
    _ensure(item_json.stem == item_id, f"Item JSON filename stem '{item_json.stem}' does not match item id '{item_id}'.")

    assets = item.get("assets") or {}
    asset_keys = set(assets.keys())
    _ensure(asset_keys == {"vv", "vh"}, f"Expected exactly vv and vh assets, got: {sorted(asset_keys)}")

    vv_local = item_json.parent / f"{item_id}_vv.tif"
    vh_local = item_json.parent / f"{item_id}_vh.tif"
    _ensure(vv_local.exists(), f"Local VV COG not found next to item JSON: {vv_local}")
    _ensure(vh_local.exists(), f"Local VH COG not found next to item JSON: {vh_local}")

    vv_href = str(assets["vv"].get("href") or "").strip()
    vh_href = str(assets["vh"].get("href") or "").strip()
    vv_bucket, vv_key = _parse_s3_uri(vv_href)
    vh_bucket, vh_key = _parse_s3_uri(vh_href)

    _ensure(vv_bucket == vh_bucket, "VV and VH assets point to different S3 buckets.")
    _ensure(Path(vv_key).name == f"{item_id}_vv.tif", f"VV asset filename must be {item_id}_vv.tif.")
    _ensure(Path(vh_key).name == f"{item_id}_vh.tif", f"VH asset filename must be {item_id}_vh.tif.")
    _ensure(Path(vv_key).parent == Path(vh_key).parent, "VV and VH assets must live in the same S3 directory.")

    expected_bucket = str(values.get("SR_S3_BUCKET") or "").strip() or None
    if expected_bucket:
        _ensure(vv_bucket == expected_bucket, f"Item assets point to bucket '{vv_bucket}', expected '{expected_bucket}'.")

    root_href, collection_href, self_href = _validate_item_links(
        item,
        collection_id=collection_id,
        item_id=item_id,
        env=values,
    )

    artifacts = [
        PublishArtifact(
            role="sr_vv",
            local_path=vv_local,
            bucket=vv_bucket,
            key=vv_key,
            href=vv_href,
            content_type="image/tiff; application=geotiff; profile=cloud-optimized",
        ),
        PublishArtifact(
            role="sr_vh",
            local_path=vh_local,
            bucket=vh_bucket,
            key=vh_key,
            href=vh_href,
            content_type="image/tiff; application=geotiff; profile=cloud-optimized",
        ),
    ]
    artifacts.append(_derive_json_artifact(item_json, item_id, vv_bucket, vv_key))

    return PublishPlan(
        item_json_path=item_json,
        item_id=item_id,
        collection_id=collection_id,
        expected_s3_bucket=expected_bucket,
        stac_root_url=root_href,
        collection_url=collection_href,
        item_url=self_href,
        item_payload=item,
        artifacts=artifacts,
    )


def run_preflight(
    *,
    plan: PublishPlan,
    session: Any,
    s3_client: Any,
    overwrite: bool = False,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": _utc_rfc3339_now(),
        "mode": "preflight",
        "overwrite": overwrite,
        "plan": plan.to_report(),
    }

    collection_response = session.get(plan.collection_url, timeout=timeout_seconds)
    collection_check: Dict[str, Any] = {
        "url": plan.collection_url,
        "status_code": collection_response.status_code,
        "exists": collection_response.ok,
    }
    if collection_response.ok:
        try:
            payload = collection_response.json()
            collection_check["title"] = payload.get("title")
            collection_check["description"] = payload.get("description")
        except Exception:
            pass
    report["collection_check"] = collection_check
    _ensure(collection_response.ok, f"STAC collection does not exist: {plan.collection_id}")

    item_response = session.get(plan.item_url, timeout=timeout_seconds)
    item_exists = item_response.status_code == 200
    report["item_check"] = {
        "url": plan.item_url,
        "status_code": item_response.status_code,
        "exists": item_exists,
    }
    if item_response.status_code not in {200, 404}:
        raise PublishError(f"Unexpected STAC item check status: {item_response.status_code}")
    if item_exists and not overwrite:
        raise PublishError(f"STAC item already exists and overwrite is disabled: {plan.item_id}")

    s3_checks: List[Dict[str, Any]] = []
    for artifact in plan.artifacts:
        exists, head = _head_object_exists(s3_client, artifact.bucket, artifact.key)
        entry = artifact.to_report()
        entry["exists_remote"] = exists
        if head:
            entry["remote_size_bytes"] = head.get("ContentLength")
            entry["remote_content_type"] = head.get("ContentType")
            entry["remote_etag"] = head.get("ETag")
        s3_checks.append(entry)
        if exists and not overwrite:
            raise PublishError(f"S3 object already exists and overwrite is disabled: s3://{artifact.bucket}/{artifact.key}")
    report["s3_checks"] = s3_checks
    report["ready_to_publish"] = True
    return report


def execute_publish(
    *,
    plan: PublishPlan,
    session: Any,
    s3_client: Any,
    overwrite: bool = False,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    report = run_preflight(
        plan=plan,
        session=session,
        s3_client=s3_client,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
    )
    report["mode"] = "execute"

    uploads: List[Dict[str, Any]] = []
    for artifact in plan.artifacts:
        s3_client.upload_file(
            str(artifact.local_path),
            artifact.bucket,
            artifact.key,
            ExtraArgs={"ContentType": artifact.content_type},
        )
        exists, head = _head_object_exists(s3_client, artifact.bucket, artifact.key)
        _ensure(exists and head is not None, f"Uploaded artifact could not be verified on S3: s3://{artifact.bucket}/{artifact.key}")
        uploads.append(
            {
                **artifact.to_report(),
                "verified_remote_size_bytes": head.get("ContentLength"),
                "verified_remote_content_type": head.get("ContentType"),
            }
        )
    report["uploads"] = uploads

    if report["item_check"]["exists"]:
        response = session.put(plan.item_url, json=plan.item_payload, timeout=timeout_seconds)
        operation = "put"
    else:
        response = session.post(_join_url(plan.collection_url, "items"), json=plan.item_payload, timeout=timeout_seconds)
        operation = "post"
    report["stac_write"] = {
        "operation": operation,
        "status_code": response.status_code,
        "url": response.url,
    }
    _ensure(response.status_code in {200, 201}, f"STAC item write failed with status {response.status_code}")

    verify_response = session.get(plan.item_url, timeout=timeout_seconds)
    _ensure(verify_response.ok, f"Published STAC item could not be reloaded: {plan.item_url}")
    remote_item = verify_response.json()
    report["verification"] = {
        "item_url": plan.item_url,
        "status_code": verify_response.status_code,
        "item_id_matches": remote_item.get("id") == plan.item_id,
        "item_collection_matches": remote_item.get("collection") == plan.collection_id,
        "asset_hrefs_match": {
            artifact.role: (((remote_item.get("assets") or {}).get(artifact.role) or {}).get("href") == artifact.href)
            for artifact in plan.artifacts
            if artifact.role in {"sr_vv", "sr_vh"}
        },
    }
    _ensure(report["verification"]["item_id_matches"], "Published STAC item id mismatch.")
    _ensure(report["verification"]["item_collection_matches"], "Published STAC collection mismatch.")
    _ensure(all(report["verification"]["asset_hrefs_match"].values()), "Published STAC asset href mismatch.")
    report["published"] = True
    return report


def write_publish_report(report_path: str | Path, report: Dict[str, Any]) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely publish SR outputs to S3 and STAC with strict preflight checks.")
    parser.add_argument("--item-json", required=True, help="Local SR metadata JSON path produced by the pipeline.")
    parser.add_argument(
        "--report",
        default=None,
        help="Where to write publish_report.json. Defaults next to the item JSON.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually upload to S3 and register the STAC item. Without this flag the command only runs preflight checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing S3 objects and updating an existing STAC item.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout in seconds for STAC requests.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    report_path = Path(args.report) if args.report else Path(args.item_json).with_name("publish_report.json")

    session = build_requests_session()
    s3_client = build_s3_client_from_env()

    try:
        plan = build_publish_plan(
            item_json_path=args.item_json,
        )
        if args.execute:
            report = execute_publish(
                plan=plan,
                session=session,
                s3_client=s3_client,
                overwrite=args.overwrite,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            report = run_preflight(
                plan=plan,
                session=session,
                s3_client=s3_client,
                overwrite=args.overwrite,
                timeout_seconds=args.timeout_seconds,
            )
            report["published"] = False
        report["status"] = "ok"
        write_publish_report(report_path, report)
        print(f"Publish report written to {report_path}")
        return 0
    except Exception as exc:
        failure_report = {
            "timestamp": _utc_rfc3339_now(),
            "status": "failed",
            "error": str(exc),
            "item_json_path": str(args.item_json),
            "mode": "execute" if args.execute else "preflight",
            "overwrite": args.overwrite,
        }
        write_publish_report(report_path, failure_report)
        print(f"Publish failed. Report written to {report_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

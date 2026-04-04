from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_support.json_support import compact_jsonable
from pipeline_support.runtime_support import aoi_log_path

DEFAULT_MAX_RETRY_ATTEMPTS = 3


def classify_skip_reason_code(skip_reason: Optional[str]) -> str:
    text = str(skip_reason or "").strip().lower()
    if not text:
        return "SKIPPED_BY_PIPELINE_RULE"
    if "no valid child aoi intersections remained" in text:
        return "NO_VALID_CHILDREN"
    if "no valid in-period representative scene pools" in text:
        return "NO_VALID_REPRESENTATIVE_POOLS"
    if "scene-count" in text or "scene count" in text:
        return "SCENE_COUNT_BELOW_MIN"
    if "no items" in text:
        return "NO_ITEMS_IN_PERIOD"
    return "SKIPPED_BY_PIPELINE_RULE"


def classify_failure_reason(error_message: Optional[str]) -> Tuple[str, bool]:
    text = str(error_message or "").strip().lower()
    if not text:
        return "PIPELINE_EXECUTION_ERROR", False
    retryable_markers = [
        "503",
        "service unavailable",
        "429",
        "502",
        "504",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    ]
    if any(marker in text for marker in retryable_markers):
        return "BACKEND_SERVICE_UNAVAILABLE", True
    if "inference dependencies are missing" in text or "module not found" in text:
        return "DEPENDENCY_ERROR", False
    if "config" in text and "missing" in text:
        return "INVALID_CONFIG", False
    if "unsupported" in text or "requires" in text:
        return "INVALID_CONFIG", False
    if "geojson not found" in text or "invalid_aoi" in text:
        return "INVALID_AOI_GEOMETRY", False
    if "download" in text:
        return "DOWNLOAD_ERROR", True
    if "infer" in text or "checkpoint" in text or "cuda" in text:
        return "MODEL_INFERENCE_ERROR", False
    if "write" in text or "cog" in text or "tif" in text:
        return "ARTIFACT_WRITE_ERROR", False
    return "PIPELINE_EXECUTION_ERROR", False


def infer_completion_class(summary: Dict[str, Any]) -> str:
    raw_status = str(summary.get("status", "")).strip().lower()
    if raw_status in {"skipped", "failed"}:
        return "none"

    period_counts = summary.get("period_counts")
    if isinstance(period_counts, dict):
        total = int(period_counts.get("total", 0) or 0)
        completed = int(period_counts.get("completed", 0) or 0)
        skipped = int(period_counts.get("skipped", 0) or 0)
        failed = int(period_counts.get("failed", 0) or 0)
        if completed > 0 and skipped == 0 and failed == 0 and completed == total:
            return "full"
        if completed > 0:
            return "partial"
        return "none"

    componentization = summary.get("componentization")
    if isinstance(componentization, dict):
        completed = int(componentization.get("completed_component_count", 0) or 0)
        rejected = int(componentization.get("rejected_component_count", 0) or 0)
        try:
            supported_area_ratio = float(componentization.get("parent_supported_area_ratio"))
        except Exception:
            supported_area_ratio = None
        if completed > 0 and rejected == 0:
            if supported_area_ratio is not None and supported_area_ratio < 0.999999:
                return "partial"
            return "full"
        if completed > 0:
            return "partial"
        return "none"

    if raw_status == "completed":
        return "full"
    return "none"


def has_only_suppressed_component_rejections(summary: Dict[str, Any]) -> bool:
    componentization = summary.get("componentization")
    rejected_candidates = summary.get("rejected_component_candidates")
    if not isinstance(componentization, dict) or not isinstance(rejected_candidates, list):
        return False

    completed = int(componentization.get("completed_component_count", 0) or 0)
    rejected = int(componentization.get("rejected_component_count", 0) or 0)
    if completed <= 0 or rejected <= 0 or len(rejected_candidates) != rejected:
        return False

    return all(str(item.get("status", "")).strip().lower() == "suppressed" for item in rejected_candidates)


def attach_run_log_path(summary: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = summary.get("run_dir")
    if run_dir and not summary.get("run_log_path"):
        summary["run_log_path"] = str(aoi_log_path(run_dir))
    return summary


def apply_execution_contract(
    summary: Dict[str, Any],
    *,
    default_reason_code: Optional[str] = None,
    default_reason_message: Optional[str] = None,
    retry_attempt: int = 1,
    max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS,
) -> Dict[str, Any]:
    attach_run_log_path(summary)
    raw_status = str(summary.get("status", "")).strip().lower()
    if not raw_status:
        period_counts = summary.get("period_counts") or {}
        completed = int(period_counts.get("completed", 0) or 0)
        skipped = int(period_counts.get("skipped", 0) or 0)
        failed = int(period_counts.get("failed", 0) or 0)
        if completed > 0:
            raw_status = "completed"
        elif skipped > 0 and failed == 0:
            raw_status = "skipped"
        elif failed > 0:
            raw_status = "failed"
        else:
            raw_status = "failed"
        summary["status"] = raw_status

    if raw_status == "completed":
        final_status = "pass"
        completion_class = infer_completion_class(summary)
        if default_reason_code:
            reason_code = default_reason_code
        elif completion_class == "partial" and has_only_suppressed_component_rejections(summary):
            reason_code = "COMPLETED_WITH_COMPONENT_SUPPRESSION"
        elif completion_class == "partial":
            reason_code = "COMPLETED_WITH_PARTIAL_SKIPS"
        else:
            reason_code = "COMPLETED_WITH_OUTPUT"

        if default_reason_message:
            reason_message = default_reason_message
        elif reason_code == "COMPLETED_WITH_OUTPUT":
            reason_message = "Execution finished and produced valid output artifacts."
        elif reason_code == "COMPLETED_WITH_COMPONENT_SUPPRESSION":
            reason_message = (
                "Execution finished with valid outputs after intentionally suppressing nested child "
                "components that were fully covered by larger regions."
            )
        else:
            reason_message = "Execution finished with valid outputs, but some periods or child components were skipped or rejected."
        retryable = False
    elif raw_status == "skipped":
        final_status = "skip"
        reason_message = default_reason_message or str(summary.get("skip_reason") or "Pipeline skipped this run because no valid data path remained.")
        reason_code = default_reason_code or classify_skip_reason_code(reason_message)
        retryable = False
    else:
        final_status = "fail"
        failure_text = default_reason_message or str(summary.get("error_message") or summary.get("error") or "Pipeline execution failed.")
        reason_code, retryable = classify_failure_reason(failure_text)
        if default_reason_code:
            reason_code = default_reason_code
        reason_message = failure_text

    retry_state = {
        "attempt": int(summary.get("retry", {}).get("attempt", retry_attempt) or retry_attempt),
        "max_attempts": int(summary.get("retry", {}).get("max_attempts", max_retry_attempts) or max_retry_attempts),
        "retryable": bool(summary.get("retry", {}).get("retryable", retryable)),
        "next_retry_at": summary.get("retry", {}).get("next_retry_at"),
    }
    retry_state["remaining_attempts"] = max(0, retry_state["max_attempts"] - retry_state["attempt"])

    summary["lifecycle_status"] = "finished"
    summary["final_status"] = final_status
    summary["completion_class"] = infer_completion_class(summary)
    summary["reason_code"] = reason_code
    summary["reason_message"] = reason_message
    summary["retry"] = retry_state
    summary["status_contract_version"] = "2026-03-24.v1"
    return summary


def build_period_manifest_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    period = summary.get("period") or {}
    manifest_type = "representative_calendar_period"
    if summary.get("component_results") is not None and "selection" not in summary:
        manifest_type = "representative_calendar_period_parent"
    manifest = {
        "manifest_type": manifest_type,
        "workflow_mode": summary.get("workflow_mode"),
        "selection_strategy": summary.get("selection_strategy"),
        "aoi_geojson": summary.get("aoi_geojson"),
        "period_id": period.get("period_id"),
        "period_mode": period.get("period_mode"),
        "period_start": period.get("period_start"),
        "period_end": period.get("period_end"),
        "period_anchor_datetime": period.get("period_anchor_datetime"),
        "period_split_policy": period.get("period_split_policy"),
        "status": summary.get("status"),
        "lifecycle_status": summary.get("lifecycle_status"),
        "final_status": summary.get("final_status"),
        "completion_class": summary.get("completion_class"),
        "reason_code": summary.get("reason_code"),
        "reason_message": summary.get("reason_message"),
        "retry": summary.get("retry"),
        "summary_json": summary.get("summary_json"),
        "summary_md": summary.get("summary_md"),
        "run_log_path": summary.get("run_log_path"),
        "output_sr_geojson_path": summary.get("output_sr_geojson_path"),
    }
    if summary.get("skip_reason"):
        manifest["skip_reason"] = summary.get("skip_reason")
    if summary.get("items_in_period") is not None:
        manifest["items_in_period"] = summary.get("items_in_period")
    if summary.get("componentization") is not None:
        manifest["componentization"] = summary.get("componentization")
    if summary.get("rejected_component_candidates") is not None:
        manifest["rejected_component_candidates"] = summary.get("rejected_component_candidates")
    if summary.get("component_results") is not None:
        manifest["component_results"] = summary.get("component_results")
    if summary.get("selection") is not None:
        manifest["selection"] = summary.get("selection")
    return manifest


def normalize_runtime_record(record: Dict[str, Any], *, scope: str) -> Dict[str, Any]:
    normalized = copy.deepcopy(record)
    if "run_log_path" in normalized:
        normalized["log_path"] = normalized.get("run_log_path")
    elif "job_log_path" in normalized:
        normalized["log_path"] = normalized.get("job_log_path")

    for key in (
        "summary_json",
        "summary_md",
        "manifest_path",
        "manifest_type",
        "record_type",
        "run_log_path",
        "job_log_path",
    ):
        normalized.pop(key, None)

    normalized["scope"] = scope
    return compact_jsonable(normalized)


def write_period_manifest_from_summary(summary: Dict[str, Any]) -> Optional[Path]:
    manifest_path_value = summary.get("manifest_path")
    if not manifest_path_value:
        return None
    manifest_path = Path(manifest_path_value)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(build_period_record(summary), f, indent=2, ensure_ascii=False)
    return manifest_path


def build_run_manifest(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_type": "representative_calendar_run",
        "workflow_mode": summary.get("workflow_mode"),
        "selection_strategy": summary.get("selection_strategy"),
        "aoi_geojson": summary.get("aoi_geojson"),
        "run_dir": summary.get("run_dir"),
        "periods_dir": summary.get("periods_dir"),
        "status": summary.get("status"),
        "lifecycle_status": summary.get("lifecycle_status"),
        "final_status": summary.get("final_status"),
        "completion_class": summary.get("completion_class"),
        "reason_code": summary.get("reason_code"),
        "reason_message": summary.get("reason_message"),
        "retry": summary.get("retry"),
        "period_counts": summary.get("period_counts"),
        "period_results": summary.get("period_results"),
        "run_config": summary.get("run_config"),
        "summary_json": summary.get("summary_json"),
        "summary_md": summary.get("summary_md"),
        "run_log_path": summary.get("run_log_path"),
        "output_sr_geojson_path": summary.get("output_sr_geojson_path"),
    }


def build_period_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    record = build_period_manifest_from_summary(summary)
    record.update(copy.deepcopy(summary))
    return normalize_runtime_record(record, scope="period")


def build_aoi_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    record = build_run_manifest(summary)
    record.update(copy.deepcopy(summary))
    return normalize_runtime_record(record, scope="aoi")


__all__ = [
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "apply_execution_contract",
    "attach_run_log_path",
    "build_aoi_record",
    "build_period_manifest_from_summary",
    "build_period_record",
    "build_run_manifest",
    "classify_failure_reason",
    "classify_skip_reason_code",
    "has_only_suppressed_component_rejections",
    "infer_completion_class",
    "normalize_runtime_record",
    "write_period_manifest_from_summary",
]

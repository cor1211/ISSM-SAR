from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


DEFAULT_LOG_FORMAT = "%(asctime)s | %(level_badge)-10s | %(name)-18s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"
LEVEL_ICONS = {
    "DEBUG": "·",
    "INFO": "ℹ",
    "WARNING": "⚠",
    "ERROR": "✖",
    "CRITICAL": "✖",
}
_SECRET_MARKERS = (
    "secret",
    "password",
    "passwd",
    "pwd",
    "token",
    "api_key",
    "access_key",
    "private_key",
    "credentials",
)


def normalize_log_level_name(value: Optional[str], default: str = DEFAULT_LOG_LEVEL) -> str:
    candidate = str(value or default).strip().upper()
    if candidate == "WARN":
        candidate = "WARNING"
    if candidate not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        return str(default).strip().upper()
    return candidate


def resolve_runtime_log_level(
    *,
    cli_level: Optional[str],
    config: Optional[Mapping[str, Any]],
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str]:
    values = dict(os.environ if env is None else env)
    if cli_level:
        return normalize_log_level_name(cli_level), "cli"
    env_level = values.get("PIPELINE_LOG_LEVEL")
    if env_level:
        return normalize_log_level_name(env_level), "env:PIPELINE_LOG_LEVEL"
    config_level = None
    if isinstance(config, Mapping):
        config_level = ((config.get("logging") or {}) if isinstance(config.get("logging"), Mapping) else {}).get("level")
    if config_level:
        return normalize_log_level_name(str(config_level)), "config.logging.level"
    return DEFAULT_LOG_LEVEL, "default"


def level_icon(level_name: str) -> str:
    return LEVEL_ICONS.get(normalize_log_level_name(level_name), "·")


def level_badge(level_name: str) -> str:
    normalized = normalize_log_level_name(level_name)
    return f"{normalized:<7} {level_icon(normalized)}"


class RuntimeLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.level_badge = level_badge(record.levelname)
        return super().format(record)


def configure_root_logging(level_name: str, *, force: bool = True) -> str:
    normalized = normalize_log_level_name(level_name)
    logging.basicConfig(level=getattr(logging, normalized, logging.INFO), force=force)
    formatter = RuntimeLogFormatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    return normalized


def ensure_root_logging(level_name: str = DEFAULT_LOG_LEVEL) -> str:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return normalize_log_level_name(level_name)
    return configure_root_logging(level_name, force=False)


def is_sensitive_key(key: Optional[str]) -> bool:
    lowered = str(key or "").strip().lower()
    if lowered.endswith("_present") or lowered.endswith("_source"):
        return False
    return any(marker in lowered for marker in _SECRET_MARKERS)


def safe_text_snippet(value: Any, limit: int = 300) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max(1, int(limit)):
        return text
    return f"{text[: max(1, int(limit))]}..."


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def sanitize_log_value(
    key: Optional[str],
    value: Any,
    *,
    redact_secrets: bool = True,
    limit: Optional[int] = None,
) -> str:
    if redact_secrets and is_sensitive_key(key):
        return "<redacted>"
    text = _stringify_value(value)
    if limit is not None:
        text = safe_text_snippet(text, limit=max(1, int(limit)))
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in {'"', "|", "="}):
        return json.dumps(text, ensure_ascii=False)
    return text


def format_log_message(
    message: str,
    *,
    redact_secrets: bool = True,
    snippet_limit: Optional[int] = None,
    **fields: Any,
) -> str:
    base = str(message)
    if not fields:
        return base
    parts = [
        f"{key}={sanitize_log_value(key, value, redact_secrets=redact_secrets, limit=snippet_limit)}"
        for key, value in fields.items()
        if value is not None
    ]
    if not parts:
        return base
    if len(parts) == 1:
        return f"{base} | {parts[0]}"
    detail_block = "\n".join(f"    {part}" for part in parts)
    return f"{base}\n{detail_block}"


def emit_runtime_log(
    logger_name: str,
    level: int,
    message: str,
    *,
    redact_secrets: bool = True,
    snippet_limit: Optional[int] = None,
    **fields: Any,
) -> None:
    rendered = format_log_message(
        message,
        redact_secrets=redact_secrets,
        snippet_limit=snippet_limit,
        **fields,
    )
    root_logger = logging.getLogger()
    if root_logger.handlers:
        logging.getLogger(logger_name).log(level, rendered)
        return

    timestamp = datetime.now().strftime(DEFAULT_DATE_FORMAT)
    level_name = logging.getLevelName(level)
    print(f"{timestamp} | {level_badge(level_name):<10} | {logger_name:<18} | {rendered}", file=sys.stderr)


def detect_s3_credential_source(env: Optional[Mapping[str, str]] = None) -> str:
    values = dict(os.environ if env is None else env)
    if values.get("S3_ACCESS_KEY") and values.get("S3_SECRET_KEY"):
        return "explicit_env"
    return "none"


def env_presence_map(keys: Iterable[str], env: Optional[Mapping[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    values = dict(os.environ if env is None else env)
    return {
        str(key): {
            "present": bool(values.get(key)),
            "value": None if is_sensitive_key(key) else values.get(key),
        }
        for key in keys
    }

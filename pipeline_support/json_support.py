from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Affine):
        return list(value)[:6]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def compact_jsonable(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, dict):
        return {k: compact_jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [compact_jsonable(v) for v in value if v is not None]
    return value


__all__ = ["compact_jsonable", "to_jsonable"]

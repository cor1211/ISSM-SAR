from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional


OverrideRecord = Dict[str, Any]


def _ensure_mapping_path(root: MutableMapping[str, Any], dotted_path: str) -> MutableMapping[str, Any]:
    current: MutableMapping[str, Any] = root
    parts = [part for part in str(dotted_path or "").split(".") if part]
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    return current


def _get_mapping_value(root: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = root
    for part in [segment for segment in str(dotted_path or "").split(".") if segment]:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _set_mapping_value(root: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    target = _ensure_mapping_path(root, dotted_path)
    final_key = [part for part in str(dotted_path or "").split(".") if part][-1]
    target[final_key] = value


def _parse_bool(raw: str) -> bool:
    normalized = str(raw or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got: {raw!r}")


def _parse_int(raw: str) -> int:
    return int(str(raw).strip())


def _parse_float(raw: str) -> float:
    return float(str(raw).strip())


def _parse_text(raw: str) -> str:
    return str(raw).strip()


def _apply_overrides(
    config: MutableMapping[str, Any],
    *,
    env: Optional[Mapping[str, str]],
    specs: Mapping[str, tuple[str, Callable[[str], Any]]],
) -> List[OverrideRecord]:
    values = dict(os.environ if env is None else env)
    overrides: List[OverrideRecord] = []
    for env_key, (target_path, parser) in specs.items():
        raw_value = values.get(env_key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and raw_value.strip() == "":
            continue
        try:
            parsed = parser(raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid value for {env_key}: {raw_value!r}") from exc
        _set_mapping_value(config, target_path, parsed)
        overrides.append(
            {
                "target": target_path,
                "source": f"env:{env_key}",
            }
        )
    return overrides


PIPELINE_ENV_SPECS: Dict[str, tuple[str, Callable[[str], Any]]] = {
    "PIPELINE_STAC_LIMIT": ("stac.limit", _parse_int),
    "PIPELINE_REPRESENTATIVE_POOL_MODE": ("trainlike.representative_pool_mode", _parse_text),
    "PIPELINE_MIN_SCENES_PER_HALF": ("trainlike.min_scenes_per_half", _parse_int),
    "PIPELINE_COMPONENT_ITEM_MIN_COVERAGE": ("trainlike.component_item_min_coverage", _parse_float),
    "PIPELINE_COMPONENT_MIN_AREA_RATIO": ("trainlike.component_min_area_ratio", _parse_float),
    "PIPELINE_TARGET_CRS": ("trainlike.target_crs", _parse_text),
    "PIPELINE_TARGET_RESOLUTION": ("trainlike.target_resolution", _parse_float),
    "PIPELINE_FOCAL_MEDIAN_RADIUS_M": ("trainlike.focal_median_radius_m", _parse_float),
    "PIPELINE_SAVE_DEBUG_DATA": ("output.save_debug_artifacts", _parse_bool),
}


INFERENCE_ENV_SPECS: Dict[str, tuple[str, Callable[[str], Any]]] = {
    "INFER_DEVICE": ("device", _parse_text),
    "INFER_PATCH_SIZE": ("inference.patch_size", _parse_int),
    "INFER_OVERLAP": ("inference.overlap", _parse_float),
    "INFER_BATCH_SIZE": ("inference.batch_size", _parse_int),
    "INFER_USE_AMP": ("inference.use_amp", _parse_bool),
    "INFER_GAUSSIAN_BLEND": ("inference.gaussian_blend", _parse_bool),
}


def apply_pipeline_env_overrides(
    config: MutableMapping[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> List[OverrideRecord]:
    return _apply_overrides(config, env=env, specs=PIPELINE_ENV_SPECS)


def apply_inference_env_overrides(
    config: MutableMapping[str, Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> List[OverrideRecord]:
    return _apply_overrides(config, env=env, specs=INFERENCE_ENV_SPECS)

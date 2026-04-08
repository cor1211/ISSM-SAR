from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio


_NOISE_FLOOR_DB = -55


def _band_floor_ratio(path: Path) -> float:
    with rasterio.open(path, "r") as src:
        arr = src.read(1).astype(np.float32)
    return float(np.mean(arr <= _NOISE_FLOOR_DB))


def filter_grouped_paths_by_scene_quality(
    grouped_paths: Dict[str, List[Path]],
    *,
    floor_ratio_threshold: float,
    required_scene_count: int,
    expected_polarizations: List[str],
) -> Dict[str, List[Path]]:
    pols = [str(pol).lower() for pol in expected_polarizations]
    if not pols:
        return {}
    threshold = float(floor_ratio_threshold)
    min_required = max(1, int(required_scene_count))
    filtered: Dict[str, List[Path]] = {}

    for pol in pols:
        input_paths = [Path(path) for path in grouped_paths.get(pol, [])]
        if not input_paths:
            filtered[pol] = []
            continue
        kept = [path for path in input_paths if _band_floor_ratio(path) < threshold]
        if kept and len(kept) < min_required:
            raise RuntimeError(
                "Scene quality filtering removed too many scenes "
                f"for band={pol} (kept={len(kept)} required={min_required})."
            )
        if not kept:
            raise RuntimeError(f"Scene quality filtering removed all scenes for band={pol}.")
        filtered[pol] = kept

    return filtered


__all__ = [
    "filter_grouped_paths_by_scene_quality",
]

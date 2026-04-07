#!/usr/bin/env python3
"""
=============================================================
 ISSM_SAR — Production Inference Script
 Super-Resolution for Multi-temporal SAR Imagery (GeoTIFF)
=============================================================

Features:
  • Reads 2-band (VV, VH) GeoTIFF via rasterio, preserves CRS & transform
  • SAR dB normalisation: clip → [0,1] → [-1,1]
  • Independent VV / VH model instances with separate checkpoints
  • Logs temporal semantics of the current pair (`T1/T2` vs `S1T1/S1T2`)
  • Sliding-window inference with configurable overlap
  • 2D Gaussian blending to eliminate grid artefacts
  • Batch-wise GPU inference with optional AMP (FP16)
  • Denormalize back to dB, write 2-band GeoTIFF (×2 resolution)

Usage:
  python infer_production.py --config config/infer_config.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
import torch
import yaml
from runtime_logging import DEFAULT_LOG_LEVEL, ensure_root_logging, format_log_message
from runtime_env_overrides import apply_inference_env_overrides

# ── project imports ──────────────────────────────────────────
# Ensure src/ is importable directly to bypass __init__.py (and avoid PyTorch Lightning dependence)
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import ISSM_SAR

# ── logging ──────────────────────────────────────────────────
ensure_root_logging(DEFAULT_LOG_LEVEL)
logger = logging.getLogger("infer_production")


def emit_infer_log(level: int, message: str, **fields: Any) -> None:
    logger.log(level, format_log_message(message, **fields))


def log_inference_env_overrides(config: Dict[str, Any]) -> None:
    for override in (config.get("_runtime", {}) or {}).get("env_overrides", []) or []:
        emit_infer_log(
            logging.DEBUG,
            "Applied inference environment override",
            target=override.get("target"),
            source=override.get("source"),
        )


# ==============================================================
#  Helper: Configuration loader
# ==============================================================
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def default_input_semantics() -> Dict[str, Any]:
    """Canonical semantic convention expected by the trained model."""
    return {
        "t1_label": "S1T1",
        "t2_label": "S1T2",
        "t1_role": "post/future window or later model input",
        "t2_role": "pre/past window or earlier model input",
        "matches_training_semantics": True,
        "note": "Current checkpoints were trained with model(S1T1_post_future, S1T2_pre_past).",
    }


# ==============================================================
#  SARInferencer
# ==============================================================
class SARInferencer:
    """Production-grade sliding-window inference for ISSM_SAR.

    The inferencer handles:
      1. Loading two independent model instances (VV & VH)
      2. Reading large multi-band GeoTIFFs via rasterio
      3. SAR dB ↔ [-1,1] normalisation / denormalisation
      4. Sliding-window patch extraction with Gaussian blending
      5. Writing the SR output as a geo-referenced GeoTIFF
    """

    # ----------------------------------------------------------
    #  Initialisation
    # ----------------------------------------------------------
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Parsed YAML config dictionary.
        """
        self.config = config
        log_inference_env_overrides(config)

        # ── device ──
        device_name: str = config.get("device", "cuda")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            emit_infer_log(
                logging.WARNING,
                "CUDA unavailable, falling back to CPU",
                requested_device=device_name,
                active_device="cpu",
            )
            device_name = "cpu"
        self.device = torch.device(device_name)
        emit_infer_log(logging.DEBUG, "Inference device", device=str(self.device))
        self._log_device_overview()

        # ── normalisation params ──
        norm_cfg = config["normalization"]
        self.v_min: float = float(norm_cfg["v_min"])
        self.v_max: float = float(norm_cfg["v_max"])
        emit_infer_log(logging.DEBUG, "Normalization setup", db_clip_min=self.v_min, db_clip_max=self.v_max)

        # ── inference params ──
        inf_cfg = config["inference"]
        self.patch_size: int = int(inf_cfg.get("patch_size", 128))
        self.overlap_frac: float = float(inf_cfg.get("overlap", 0.25))
        self.batch_size: int = int(inf_cfg.get("batch_size", 16))
        self.use_amp: bool = bool(inf_cfg.get("use_amp", True))
        self.gaussian_blend: bool = bool(inf_cfg.get("gaussian_blend", True))

        self.overlap_px: int = int(self.patch_size * self.overlap_frac)
        self.stride: int = self.patch_size - self.overlap_px
        emit_infer_log(
            logging.DEBUG,
            "Inference tiling",
            patch_size=self.patch_size,
            overlap_px=self.overlap_px,
            stride=self.stride,
            batch_size=self.batch_size,
            amp=self.use_amp,
            blending=self.gaussian_blend,
        )

        # ── load model architecture config ──
        model_config_path = Path(config["model_config_path"])
        if not model_config_path.is_absolute():
            model_config_path = _PROJECT_ROOT / model_config_path
        arch_cfg = load_yaml(model_config_path)
        model_cfg: Dict[str, Any] = arch_cfg["model"]

        # ── initialise two models ──
        emit_infer_log(logging.DEBUG, "Initializing model", band="VV")
        self.model_vv = self._load_model(model_cfg, config["ckpt_path_vv"])
        self._log_model_footprint("VV", self.model_vv)
        self._log_vram("After loading VV model")
        emit_infer_log(logging.DEBUG, "Initializing model", band="VH")
        self.model_vh = self._load_model(model_cfg, config["ckpt_path_vh"])
        self._log_model_footprint("VH", self.model_vh)
        self._log_vram("After loading VH model")

        # ── pre-compute blending window (on SR output scale = 2×patch) ──
        sr_patch = self.patch_size * 2
        if self.gaussian_blend:
            self.blend_window = self._create_gaussian_window(sr_patch).to(self.device)
            emit_infer_log(logging.DEBUG, "Blending window", mode="gaussian", width=sr_patch, height=sr_patch)
        else:
            self.blend_window = torch.ones((sr_patch, sr_patch), dtype=torch.float32, device=self.device)
            emit_infer_log(logging.DEBUG, "Blending window", mode="average", width=sr_patch, height=sr_patch)

    # ----------------------------------------------------------
    #  Model loading
    # ----------------------------------------------------------
    def _load_model(self, model_cfg: Dict[str, Any], ckpt_path: str) -> ISSM_SAR:
        """Instantiate ISSM_SAR, load checkpoint, set eval mode.

        Handles both raw state_dict and PyTorch Lightning checkpoint
        formats (keys prefixed with ``model.``).
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = _PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = ISSM_SAR(config=model_cfg)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ── extract state_dict from various checkpoint layouts ──
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            # PyTorch Lightning — strip 'model.' prefix
            raw = checkpoint["state_dict"]
            state_dict = {}
            for k, v in raw.items():
                if k.startswith("model."):
                    state_dict[k[len("model."):]] = v
            if not state_dict:
                # Fallback: maybe keys don't have the prefix
                state_dict = raw
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        emit_infer_log(logging.DEBUG, "Loaded checkpoint", checkpoint=ckpt_path.name)
        return model

    # ----------------------------------------------------------
    #  VRAM Logging
    # ----------------------------------------------------------
    @staticmethod
    def _bytes_to_mb(num_bytes: float) -> float:
        return float(num_bytes) / (1024 ** 2)

    def _cuda_device_index(self) -> int:
        """Return a concrete CUDA device index even when config uses plain 'cuda'."""
        if self.device.type != "cuda":
            raise ValueError("CUDA device index requested while running on non-CUDA device.")
        if self.device.index is not None:
            return int(self.device.index)
        return int(torch.cuda.current_device())

    def _snapshot_vram(self) -> Optional[Dict[str, float]]:
        """Capture a detailed CUDA memory snapshot for the active device."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None

        device_index = self._cuda_device_index()
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)

        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        max_alloc = torch.cuda.max_memory_allocated(device_index)
        max_reserved = torch.cuda.max_memory_reserved(device_index)
        used_driver = total_bytes - free_bytes
        return {
            "total_mb": self._bytes_to_mb(total_bytes),
            "free_mb": self._bytes_to_mb(free_bytes),
            "used_driver_mb": self._bytes_to_mb(used_driver),
            "allocated_mb": self._bytes_to_mb(allocated),
            "reserved_mb": self._bytes_to_mb(reserved),
            "peak_allocated_mb": self._bytes_to_mb(max_alloc),
            "peak_reserved_mb": self._bytes_to_mb(max_reserved),
        }

    def _log_device_overview(self) -> None:
        """Log static GPU information once so VRAM planning is easier."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            emit_infer_log(logging.DEBUG, "CUDA VRAM logging disabled", reason="cpu_inference")
            return

        device_index = self._cuda_device_index()
        props = torch.cuda.get_device_properties(device_index)
        total_mb = self._bytes_to_mb(props.total_memory)
        emit_infer_log(
            logging.DEBUG,
            "CUDA device",
            name=props.name,
            capability=f"{props.major}.{props.minor}",
            total_vram_mb=round(total_mb, 1),
            sms=props.multi_processor_count,
        )
        self._log_vram("Startup")

    def _log_model_footprint(self, label: str, model: torch.nn.Module) -> None:
        """Log parameter/buffer footprint per model to estimate persistent VRAM."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_bytes = param_bytes + buffer_bytes
        emit_infer_log(
            logging.DEBUG,
            "Model footprint",
            band=label,
            params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            param_mb=round(self._bytes_to_mb(param_bytes), 1),
            buffer_mb=round(self._bytes_to_mb(buffer_bytes), 1),
            total_mb=round(self._bytes_to_mb(total_bytes), 1),
        )

    def _log_vram(self, stage: str, baseline: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
        """Log current GPU memory usage using the minimal runtime metrics we care about."""
        _ = baseline
        stats = self._snapshot_vram()
        if stats is None:
            return None

        payload = {
            "stage": stage,
            "free_mb": round(stats["free_mb"], 1),
            "allocated_mb": round(stats["allocated_mb"], 1),
            "peak_allocated_mb": round(stats["peak_allocated_mb"], 1),
        }
        emit_infer_log(logging.DEBUG, "VRAM", **payload)
        return stats

    def _estimate_band_inference_memory(
        self,
        padded_h: int,
        padded_w: int,
        batch_len: int,
        sr_patch_size: int,
    ) -> None:
        """Log rough memory sizing hints for one band inference pass."""
        input_batch_bytes = 2 * batch_len * self.patch_size * self.patch_size * np.dtype(np.float32).itemsize
        sr_batch_bytes = batch_len * sr_patch_size * sr_patch_size * np.dtype(np.float32).itemsize
        cpu_acc_bytes = 2 * (padded_h * 2) * (padded_w * 2) * np.dtype(np.float32).itemsize
        emit_infer_log(
            logging.DEBUG,
            "Memory hints",
            batch_input_mb=round(self._bytes_to_mb(input_batch_bytes), 1),
            sr_batch_output_mb=round(self._bytes_to_mb(sr_batch_bytes), 1),
            cpu_accumulator_mb=round(self._bytes_to_mb(cpu_acc_bytes), 1),
        )

    # ----------------------------------------------------------
    #  Normalisation (dB ↔ [-1, 1])
    # ----------------------------------------------------------
    @staticmethod
    def _normalize_db_static(
        data: np.ndarray, v_min: float, v_max: float
    ) -> np.ndarray:
        """Normalise SAR dB values to [-1, 1].

        Steps:
          1. Replace NaN/Inf with v_min/v_max bounds
          2. Clip to [v_min, v_max]
          3. Scale to [0, 1]:  (x - v_min) / (v_max - v_min)
          4. Scale to [-1, 1]: x * 2 - 1
        """
        # Critical Fix: Any NaN in the input will corrupt the entire Convolution
        # receptive field, causing huge "white holes" (NoData) in the final output.
        # We replace NaNs with the lowest backscatter (v_min - representing shadow).
        # np.clip automatically handles Inf and -Inf values to the [v_min, v_max] range.
        safe_data = np.nan_to_num(data, nan=v_min)
        
        clipped = np.clip(safe_data, v_min, v_max)
        scaled_01 = (clipped - v_min) / (v_max - v_min)
        scaled_11 = scaled_01 * 2.0 - 1.0
        return scaled_11.astype(np.float32)

    @staticmethod
    def _denormalize_db_static(
        data: np.ndarray, v_min: float, v_max: float
    ) -> np.ndarray:
        """Reverse normalisation: [-1, 1] → dB [v_min, v_max].

        Steps:
          1. [-1, 1] → [0, 1]:  (x + 1) / 2
          2. [0, 1] → [v_min, v_max]:  x * (v_max - v_min) + v_min
        """
        scaled_01 = (data + 1.0) / 2.0
        db = scaled_01 * (v_max - v_min) + v_min
        return db.astype(np.float32)

    def normalize_db(self, data: np.ndarray) -> np.ndarray:
        return self._normalize_db_static(data, self.v_min, self.v_max)

    def denormalize_db(self, data: np.ndarray) -> np.ndarray:
        return self._denormalize_db_static(data, self.v_min, self.v_max)

    # ----------------------------------------------------------
    #  Gaussian blending window
    # ----------------------------------------------------------
    @staticmethod
    def _create_gaussian_window(size: int, sigma: Optional[float] = None) -> torch.Tensor:
        """Create a 2D Gaussian weighting window for seamless blending.

        Args:
            size: Window width/height (square).
            sigma: Gaussian sigma.  Default = size / 6  (covers ~3σ).

        Returns:
            Tensor of shape (size, size) with values in (0, 1].
        """
        if sigma is None:
            sigma = size / 6.0

        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)  # outer product

        # Normalise peak to 1.0
        gauss_2d = gauss_2d / gauss_2d.max()
        return gauss_2d

    # ----------------------------------------------------------
    #  Image padding
    # ----------------------------------------------------------
    def _pad_image(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Pad image so that (H - patch_size) and (W - patch_size) are
        divisible by stride.  Uses reflect padding.

        Args:
            img: 2D array (H, W).

        Returns:
            (padded_img, pad_h, pad_w) — padded result and amount added.
        """
        h, w = img.shape

        # Minimum size must be at least patch_size
        pad_h = 0
        pad_w = 0

        if h < self.patch_size:
            pad_h = self.patch_size - h
        elif (h - self.patch_size) % self.stride != 0:
            pad_h = self.stride - ((h - self.patch_size) % self.stride)

        if w < self.patch_size:
            pad_w = self.patch_size - w
        elif (w - self.patch_size) % self.stride != 0:
            pad_w = self.stride - ((w - self.patch_size) % self.stride)

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

        return img, pad_h, pad_w

    # ----------------------------------------------------------
    #  Patch coordinate extraction
    # ----------------------------------------------------------
    def _get_patch_coords(self, h: int, w: int) -> List[Tuple[int, int]]:
        """Generate top-left (row, col) coordinates for sliding window.

        Args:
            h, w: Padded image dimensions.

        Returns:
            List of (row, col) tuples.
        """
        coords: List[Tuple[int, int]] = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords

    # ----------------------------------------------------------
    #  Band-level inference (sliding window + blending)
    # ----------------------------------------------------------
    @torch.no_grad()
    def _infer_band(
        self,
        model: ISSM_SAR,
        band_t1: np.ndarray,
        band_t2: np.ndarray,
        label: str,
    ) -> np.ndarray:
        """Run sliding-window inference for a single polarisation band.

        Args:
            model: The ISSM_SAR model instance (already on device, eval mode).
            band_t1: Normalised 2D array (H, W) in [-1, 1].
            band_t2: Normalised 2D array (H, W) in [-1, 1].

        Returns:
            SR result as 2D numpy array (2H, 2W) in [-1, 1].
        """
        orig_h, orig_w = band_t1.shape

        # ── pad ──
        t1_pad, pad_h, pad_w = self._pad_image(band_t1)
        t2_pad, _, _ = self._pad_image(band_t2)
        padded_h, padded_w = t1_pad.shape

        # ── output accumulators (SR space = 2×) ──
        # FIX OOM: Move giant accumulators (can be up to 10-20GB for large TIFs) to CPU RAM instantly.
        sr_h, sr_w = padded_h * 2, padded_w * 2
        output_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device="cpu")
        weight_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device="cpu")

        # ── generate patch coordinates ──
        coords = self._get_patch_coords(padded_h, padded_w)
        total_patches = len(coords)
        emit_infer_log(logging.DEBUG, "Generated inference patches", label=label, total_patches=total_patches)

        sr_patch_size = self.patch_size * 2
        blend_win = self.blend_window  # (sr_patch_size, sr_patch_size)
        first_batch_len = min(self.batch_size, total_patches)
        self._estimate_band_inference_memory(padded_h, padded_w, first_batch_len, sr_patch_size)
        self._log_vram(f"{label} before patch loop")

        # ── batch inference ──
        first_batch_logged = False
        for batch_start in range(0, total_patches, self.batch_size):
            batch_coords = coords[batch_start : batch_start + self.batch_size]
            batch_t1: List[torch.Tensor] = []
            batch_t2: List[torch.Tensor] = []

            for (y, x) in batch_coords:
                patch_t1 = t1_pad[y : y + self.patch_size, x : x + self.patch_size]
                patch_t2 = t2_pad[y : y + self.patch_size, x : x + self.patch_size]
                # → (1, 1, H_patch, W_patch)
                batch_t1.append(
                    torch.from_numpy(patch_t1).unsqueeze(0).unsqueeze(0)
                )
                batch_t2.append(
                    torch.from_numpy(patch_t2).unsqueeze(0).unsqueeze(0)
                )

            # Stack into batch tensors → (B, 1, H_patch, W_patch)
            inp_t1 = torch.cat(batch_t1, dim=0).to(self.device)
            inp_t2 = torch.cat(batch_t2, dim=0).to(self.device)
            if not first_batch_logged:
                emit_infer_log(
                    logging.DEBUG,
                    "First inference batch tensors",
                    label=label,
                    t1_shape=list(inp_t1.shape),
                    t2_shape=list(inp_t2.shape),
                    dtype=str(inp_t1.dtype),
                )
                self._log_vram(f"{label} after uploading first batch")

            # ── forward pass ──
            if self.use_amp and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    sr_up, sr_down = model(inp_t1, inp_t2)
            else:
                sr_up, sr_down = model(inp_t1, inp_t2)
            if not first_batch_logged:
                self._log_vram(f"{label} after first forward")
                first_batch_logged = True

            # sr_fusion: average of last elements → (B, 1, 2*H_patch, 2*W_patch)
            sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]
            
            # Move to CPU before accumulating to avoid massive VRAM usage
            sr_fusion = sr_fusion.cpu()
            blend_win_cpu = blend_win.cpu()

            # ── accumulate with gaussian weighting ──
            for idx, (y, x) in enumerate(batch_coords):
                sr_patch = sr_fusion[idx, 0, :, :]  # (sr_patch_size, sr_patch_size)
                out_y = y * 2
                out_x = x * 2
                output_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += (
                    sr_patch * blend_win_cpu
                )
                weight_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += blend_win_cpu

            # Free GPU memory for this batch
            del inp_t1, inp_t2, sr_up, sr_down, sr_fusion

        self._log_vram(f"{label} after patch loop")

        # ── normalise by accumulated weights ──
        # Avoid division by zero (shouldn't happen with proper tiling)
        weight_acc = torch.clamp(weight_acc, min=1e-8)
        result = output_acc / weight_acc

        # ── crop back to original SR dimensions (remove padding) ──
        sr_orig_h = orig_h * 2
        sr_orig_w = orig_w * 2
        result = result[:sr_orig_h, :sr_orig_w]

        return result.cpu().numpy()

    # ----------------------------------------------------------
    #  Read GeoTIFF
    # ----------------------------------------------------------
    @staticmethod
    def _read_geotiff(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read a multi-band GeoTIFF and return data + metadata.

        Args:
            path: Path to .tif file.

        Returns:
            (data, meta) where data has shape (bands, H, W) as float32,
            and meta is rasterio profile dict.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input GeoTIFF not found: {path}")

        with rasterio.open(path, "r") as src:
            data = src.read().astype(np.float32)  # (bands, H, W)
            meta = src.profile.copy()
            meta["descriptions"] = src.descriptions
            emit_infer_log(
                logging.DEBUG,
                "Read raster",
                path=path.name,
                shape=list(data.shape),
                crs=str(src.crs),
                dtype=src.dtypes[0],
            )
        return data, meta

    @staticmethod
    def _read_single_band_geotiff(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read a single-band GeoTIFF and return 2D data + metadata."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input GeoTIFF not found: {path}")

        with rasterio.open(path, "r") as src:
            if src.count < 1:
                raise ValueError(f"Raster has no readable bands: {path}")
            data = src.read(1).astype(np.float32)
            meta = src.profile.copy()
            meta["descriptions"] = src.descriptions
            meta["transform"] = src.transform
            meta["crs"] = src.crs
            meta["width"] = src.width
            meta["height"] = src.height
            emit_infer_log(
                logging.DEBUG,
                "Read single-band raster",
                path=path.name,
                shape=list(data.shape),
                crs=str(src.crs),
                dtype=src.dtypes[0],
            )
        return data, meta

    @staticmethod
    def _resolve_resampling(name: str) -> Resampling:
        """Map config string to rasterio Resampling enum."""
        key = str(name or "bilinear").strip().lower()
        mapping = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "average": Resampling.average,
            "lanczos": Resampling.lanczos,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported resampling mode: {name}")
        return mapping[key]

    @staticmethod
    def _same_grid(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
        """Check if two rasters share identical grid metadata."""
        crs_a = meta_a.get("crs")
        crs_b = meta_b.get("crs")
        transform_a = meta_a.get("transform")
        transform_b = meta_b.get("transform")
        return (
            crs_a == crs_b
            and transform_a is not None
            and transform_b is not None
            and transform_a.almost_equals(transform_b)
            and int(meta_a.get("width", 0)) == int(meta_b.get("width", 0))
            and int(meta_a.get("height", 0)) == int(meta_b.get("height", 0))
        )

    def _align_single_band_to_reference(
        self,
        path: str | Path,
        reference_meta: Dict[str, Any],
        resampling: Resampling,
    ) -> np.ndarray:
        """Align a 1-band GeoTIFF to the reference grid."""
        path = Path(path)
        with rasterio.open(path, "r") as src:
            if src.count < 1:
                raise ValueError(f"Raster has no readable bands: {path}")
            if src.crs is None:
                raise ValueError(f"Raster missing CRS and cannot be aligned: {path}")

            ref_crs = reference_meta.get("crs")
            ref_transform = reference_meta.get("transform")
            ref_width = int(reference_meta.get("width", 0))
            ref_height = int(reference_meta.get("height", 0))
            if ref_crs is None or ref_transform is None or ref_width < 1 or ref_height < 1:
                raise ValueError("Reference grid is incomplete for alignment.")

            if (
                src.crs == ref_crs
                and src.transform.almost_equals(ref_transform)
                and src.width == ref_width
                and src.height == ref_height
            ):
                return src.read(1).astype(np.float32)

            dst_nodata = src.nodata if src.nodata is not None else np.nan
            destination = np.full((ref_height, ref_width), dst_nodata, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                dst_nodata=dst_nodata,
                resampling=resampling,
            )
            emit_infer_log(
                logging.DEBUG,
                "Aligned raster to reference grid",
                path=path.name,
                width=ref_width,
                height=ref_height,
                crs=str(ref_crs),
                resampling=resampling.name,
            )
            return destination

    @staticmethod
    def _parse_target_resolution(value: Any) -> Optional[Tuple[float, float]]:
        """Parse target resolution config into positive (xres, yres)."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            res = float(value)
            if res <= 0:
                raise ValueError(f"target_resolution must be > 0, got {value}")
            return (res, res)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            xres = float(value[0])
            yres = float(value[1])
            if xres <= 0 or yres <= 0:
                raise ValueError(f"target_resolution values must be > 0, got {value}")
            return (xres, yres)
        raise ValueError(
            "target_resolution must be null, a positive number, or a 2-item [xres, yres] list"
        )

    def _build_reference_meta_for_single_band_pair(
        self,
        ref_path: str | Path,
        runtime_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the canonical reference grid for 4x1-band inference.

        By default we preserve the native T1_VV grid. When target CRS / resolution
        are provided, we reproject all four rasters onto that fixed grid instead.
        """
        ref_path = Path(ref_path)
        with rasterio.open(ref_path, "r") as src:
            if src.count < 1:
                raise ValueError(f"Raster has no readable bands: {ref_path}")
            if src.crs is None:
                raise ValueError(f"Raster missing CRS and cannot define reference grid: {ref_path}")

            native_meta = src.profile.copy()
            native_meta["descriptions"] = ("VV", "VH")
            native_meta["count"] = 2
            native_meta["transform"] = src.transform
            native_meta["crs"] = src.crs
            native_meta["width"] = src.width
            native_meta["height"] = src.height

            target_crs_value = runtime_cfg.get("target_crs")
            target_res_value = runtime_cfg.get("target_resolution")
            if target_crs_value in (None, "", "native") and target_res_value is None:
                return native_meta

            target_crs = rasterio.crs.CRS.from_user_input(target_crs_value) if target_crs_value not in (None, "", "native") else src.crs
            target_res = self._parse_target_resolution(target_res_value)
            if target_res is None:
                src_xres, src_yres = src.res
                target_res = (abs(float(src_xres)), abs(float(src_yres)))

            if target_crs == src.crs:
                left, bottom, right, top = src.bounds
            else:
                left, bottom, right, top = transform_bounds(
                    src.crs,
                    target_crs,
                    *src.bounds,
                    densify_pts=21,
                )

            xres, yres = target_res
            left = np.floor(left / xres) * xres
            bottom = np.floor(bottom / yres) * yres
            right = np.ceil(right / xres) * xres
            top = np.ceil(top / yres) * yres
            width = max(1, int(np.ceil((right - left) / xres)))
            height = max(1, int(np.ceil((top - bottom) / yres)))
            transform = Affine(xres, 0.0, left, 0.0, -yres, top)

            reference_meta = native_meta.copy()
            reference_meta["crs"] = target_crs
            reference_meta["transform"] = transform
            reference_meta["width"] = width
            reference_meta["height"] = height
            emit_infer_log(
                logging.DEBUG,
                "Canonical grid override",
                crs=str(target_crs),
                xres=round(float(xres), 6),
                yres=round(float(yres), 6),
                width=width,
                height=height,
            )
            return reference_meta

    @staticmethod
    def _find_band_idx(
        meta: Dict[str, Any],
        n_bands: int,
        target_pol: str,
        fallback_idx: int,
        path: Path,
    ) -> Optional[int]:
        """Find band index by checking descriptions, else fall back safely."""
        descs = meta.get("descriptions", [])

        for i, desc in enumerate(descs):
            if desc and target_pol.upper() in desc.upper():
                return i

        has_any_desc = any(d is not None for d in descs)
        if not has_any_desc and n_bands >= 2 and fallback_idx < n_bands:
            return fallback_idx

        if not has_any_desc and n_bands == 1:
            fname = path.name.lower()
            if target_pol.lower() in fname:
                return 0

        return None

    def _resolve_output_path(
        self,
        identifier: str,
        explicit_path: Optional[str | Path] = None,
    ) -> Path:
        """Resolve final output path for a logical pair identifier."""
        if explicit_path is not None:
            out_path = Path(explicit_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            return out_path

        cfg_out = self.config["output"]
        save_dir = Path(cfg_out["save_path"])
        save_dir.mkdir(parents=True, exist_ok=True)

        save_name = cfg_out.get("save_name")
        if save_name:
            return save_dir / str(save_name)

        blend_suffix = "" if self.gaussian_blend else "_no_blend"
        return save_dir / f"{identifier}_SR_x2{blend_suffix}.tif"

    def _resolve_input_semantics(self, runtime_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve semantic labels for the current pair without changing behaviour."""
        cfg_input = self.config.get("input", {})
        semantics = default_input_semantics()
        merged = dict(semantics)
        for source in (cfg_input, runtime_cfg or {}):
            if not isinstance(source, dict):
                continue
            for key in ("t1_label", "t2_label", "t1_role", "t2_role", "note"):
                value = source.get(key)
                if value not in (None, ""):
                    merged[key] = value
            if "matches_training_semantics" in source:
                merged["matches_training_semantics"] = bool(source["matches_training_semantics"])
        return merged

    def _log_input_semantics(self, identifier: str, semantics: Dict[str, Any]) -> None:
        """Emit a clear, one-place log of temporal semantics for the current inference pair."""
        emit_infer_log(
            logging.DEBUG,
            "Input semantics",
            pair=identifier,
            t1_label=semantics.get("t1_label", "T1"),
            t1_role=semantics.get("t1_role"),
            t2_label=semantics.get("t2_label", "T2"),
            t2_role=semantics.get("t2_role"),
            matches_training=semantics.get("matches_training_semantics"),
        )
        note = semantics.get("note")
        if note:
            emit_infer_log(logging.DEBUG, "Input semantics note", pair=identifier, note=note)
        if not bool(semantics.get("matches_training_semantics", True)):
            logger.warning(
                "  Pair '%s' does not follow the canonical training semantics. "
                "This is acceptable for standalone pair benchmarking/debugging, but not the preferred production input convention.",
                identifier,
            )

    def _write_optional_staged_inputs(
        self,
        cache_dir: Optional[str | Path],
        t1_stack: np.ndarray,
        t2_stack: np.ndarray,
        meta: Dict[str, Any],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Optionally persist aligned 2-band inputs for debugging/reuse."""
        if cache_dir is None:
            return None, None

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        t1_path = cache_root / "t1_input.tif"
        t2_path = cache_root / "t2_input.tif"
        meta_cache = meta.copy()
        meta_cache["descriptions"] = ("VV", "VH")
        meta_cache["count"] = 2
        self._write_geotiff(t1_path, t1_stack, meta_cache, meta_cache["transform"])
        self._write_geotiff(t2_path, t2_stack, meta_cache, meta_cache["transform"])
        return t1_path, t2_path

    def _infer_pair_arrays(
        self,
        identifier: str,
        data_t1: np.ndarray,
        meta_t1: Dict[str, Any],
        data_t2: np.ndarray,
        meta_t2: Dict[str, Any],
        out_path: str | Path,
        require_both_polarizations: bool = False,
    ) -> Path:
        """Infer a single logical pair from already aligned multi-band arrays."""
        if data_t1.shape[1:] != data_t2.shape[1:]:
            raise ValueError(
                f"Spatial shape mismatch for '{identifier}': {data_t1.shape[1:]} vs {data_t2.shape[1:]}"
            )
        if not self._same_grid(meta_t1, meta_t2):
            raise ValueError(f"Input grids differ for '{identifier}'; align before inference.")

        n_bands_t1, orig_h, orig_w = data_t1.shape
        n_bands_t2 = data_t2.shape[0]
        emit_infer_log(
            logging.INFO,
            "Starting pair inference",
            identifier=identifier,
            width=orig_w,
            height=orig_h,
            t1_band_count=n_bands_t1,
            t2_band_count=n_bands_t2,
        )

        vv_idx_t1 = self._find_band_idx(meta_t1, n_bands_t1, "VV", 0, Path(f"{identifier}_t1.tif"))
        vh_idx_t1 = self._find_band_idx(meta_t1, n_bands_t1, "VH", 1, Path(f"{identifier}_t1.tif"))
        vv_idx_t2 = self._find_band_idx(meta_t2, n_bands_t2, "VV", 0, Path(f"{identifier}_t2.tif"))
        vh_idx_t2 = self._find_band_idx(meta_t2, n_bands_t2, "VH", 1, Path(f"{identifier}_t2.tif"))

        has_vv = vv_idx_t1 is not None and vv_idx_t2 is not None
        has_vh = vh_idx_t1 is not None and vh_idx_t2 is not None

        if require_both_polarizations and (not has_vv or not has_vh):
            raise ValueError(f"Pair '{identifier}' is missing required VV/VH bands after alignment.")
        if not has_vv and not has_vh:
            raise ValueError(f"Pair '{identifier}' has no common valid bands (VV/VH).")

        out_bands = []
        out_descs = []

        if has_vv:
            emit_infer_log(logging.INFO, "Running band inference", identifier=identifier, band="VV")
            vv_t1 = data_t1[vv_idx_t1]
            vv_t2 = data_t2[vv_idx_t2]
            vv_t1_norm = self.normalize_db(vv_t1)
            vv_t2_norm = self.normalize_db(vv_t2)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            vv_baseline = self._log_vram("Before VV Inference")
            sr_vv = self._infer_band(self.model_vv, vv_t1_norm, vv_t2_norm, label="VV")
            self._log_vram("After VV Inference", baseline=vv_baseline)
            sr_vv_db = self.denormalize_db(sr_vv)
            emit_infer_log(
                logging.DEBUG,
                "Band inference output range",
                identifier=identifier,
                band="VV",
                min_db=round(float(sr_vv_db.min()), 2),
                max_db=round(float(sr_vv_db.max()), 2),
            )
            out_bands.append(sr_vv_db)
            out_descs.append("SR_VV")
        else:
            emit_infer_log(logging.WARNING, "Skipping band inference", identifier=identifier, band="VV", reason="band_missing")

        if has_vh:
            emit_infer_log(logging.INFO, "Running band inference", identifier=identifier, band="VH")
            vh_t1 = data_t1[vh_idx_t1]
            vh_t2 = data_t2[vh_idx_t2]
            vh_t1_norm = self.normalize_db(vh_t1)
            vh_t2_norm = self.normalize_db(vh_t2)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            vh_baseline = self._log_vram("Before VH Inference")
            sr_vh = self._infer_band(self.model_vh, vh_t1_norm, vh_t2_norm, label="VH")
            self._log_vram("After VH Inference", baseline=vh_baseline)
            sr_vh_db = self.denormalize_db(sr_vh)
            emit_infer_log(
                logging.DEBUG,
                "Band inference output range",
                identifier=identifier,
                band="VH",
                min_db=round(float(sr_vh_db.min()), 2),
                max_db=round(float(sr_vh_db.max()), 2),
            )
            out_bands.append(sr_vh_db)
            out_descs.append("SR_VH")
        else:
            emit_infer_log(logging.WARNING, "Skipping band inference", identifier=identifier, band="VH", reason="band_missing")

        sr_output = np.stack(out_bands, axis=0)
        src_transform: Affine = meta_t1["transform"]
        sr_transform = Affine(
            src_transform.a / 2.0,
            src_transform.b,
            src_transform.c,
            src_transform.d,
            src_transform.e / 2.0,
            src_transform.f,
        )

        meta_out = meta_t1.copy()
        meta_out["descriptions"] = tuple(out_descs)
        meta_out["count"] = len(out_bands)
        out_path = self._resolve_output_path(identifier, explicit_path=out_path)
        cfg_out = self.config["output"]
        self._write_geotiff(
            out_path,
            sr_output,
            meta_out,
            sr_transform,
            compression=cfg_out.get("compression", "DEFLATE"),
            tiled=cfg_out.get("tiled", True),
            blockxsize=cfg_out.get("blockxsize", 256),
            blockysize=cfg_out.get("blockysize", 256),
        )
        self._log_vram("After writing output")
        emit_infer_log(
            logging.INFO,
            "Completed pair inference",
            identifier=identifier,
            output_path=str(out_path),
            width=int(sr_output.shape[2]),
            height=int(sr_output.shape[1]),
            band_count=int(sr_output.shape[0]),
        )
        return out_path

    # ----------------------------------------------------------
    #  Write GeoTIFF
    # ----------------------------------------------------------
    @staticmethod
    def _write_geotiff(
        path: str | Path,
        data: np.ndarray,
        meta: Dict[str, Any],
        transform: Affine,
        compression: str = "DEFLATE",
        tiled: bool = True,
        blockxsize: int = 256,
        blockysize: int = 256,
    ) -> None:
        """Write a multi-band GeoTIFF with updated transform.

        Args:
            path: Output file path.
            data: Array of shape (bands, H, W).
            meta: Original rasterio profile (CRS will be reused).
            transform: Updated Affine transform for the SR output.
            compression: Compression codec.
            tiled: Whether to write a tiled TIFF.
            blockxsize, blockysize: Tile dimensions.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bands, h, w = data.shape
        write_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": bands,
            "height": h,
            "width": w,
            "crs": meta.get("crs"),
            "transform": transform,
            "compress": compression,
            "tiled": tiled,
        }
        if tiled:
            write_profile["blockxsize"] = blockxsize
            write_profile["blockysize"] = blockysize

        with rasterio.open(path, "w", **write_profile) as dst:
            for b in range(bands):
                dst.write(data[b], b + 1)
            
            # Copy band descriptons if present in meta
            descs = meta.get("descriptions")
            if descs:
                for b in range(bands):
                    if b < len(descs) and descs[b]:
                        dst.set_band_description(b + 1, descs[b])

        emit_infer_log(logging.DEBUG, "Saved GeoTIFF", path=str(path), band_count=bands, height=h, width=w)

    # ----------------------------------------------------------
    #  Directory Scanning
    # ----------------------------------------------------------
    @staticmethod
    def _scan_input_dir(
        input_dir: Path, t1_prefix: str, t2_prefix: str
    ) -> List[Dict[str, Any]]:
        """Scan directory and return logical jobs for both 2-band and 4x1-band layouts."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        all_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        legacy_t1: Dict[str, Path] = {}
        legacy_t2: Dict[str, Path] = {}
        single_band_groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
        valid_pols = {"vv", "vh", "hh", "hv"}

        for f in all_files:
            name = f.stem
            if name.startswith(t1_prefix):
                side = "t1"
                iden = name[len(t1_prefix):]
            elif name.startswith(t2_prefix):
                side = "t2"
                iden = name[len(t2_prefix):]
            else:
                continue

            parts = iden.rsplit("_", 1)
            if len(parts) == 2 and parts[1].lower() in valid_pols:
                pair_id, pol = parts[0], parts[1].lower()
                single_band_groups[pair_id][f"{side}_{pol}"] = f
            else:
                if side == "t1":
                    legacy_t1[iden] = f
                else:
                    legacy_t2[iden] = f

        jobs: List[Dict[str, Any]] = []
        for iden, t1_path in legacy_t1.items():
            t2_path = legacy_t2.get(iden)
            if t2_path is None:
                logger.warning(f"  [!] Missing T2: Found {t1_path.name} but no matching T2 for identifier '{iden}'")
                continue
            jobs.append(
                {
                    "mode": "multiband",
                    "identifier": iden,
                    "t1_path": t1_path,
                    "t2_path": t2_path,
                }
            )
        for iden, t2_path in legacy_t2.items():
            if iden not in legacy_t1:
                logger.warning(f"  [!] Missing T1: Found {t2_path.name} but no matching T1 for identifier '{iden}'")

        required_keys = {"t1_vv", "t1_vh", "t2_vv", "t2_vh"}
        for iden, band_group in single_band_groups.items():
            missing = sorted(required_keys - set(band_group.keys()))
            if missing:
                logger.warning(
                    f"  [!] Incomplete 1-band pair for '{iden}': missing {', '.join(missing)}"
                )
                continue
            jobs.append(
                {
                    "mode": "single-band",
                    "identifier": iden,
                    "t1_vv": band_group["t1_vv"],
                    "t1_vh": band_group["t1_vh"],
                    "t2_vv": band_group["t2_vv"],
                    "t2_vh": band_group["t2_vh"],
                }
            )

        jobs.sort(key=lambda x: (x["identifier"], x["mode"]))
        return jobs

    def run_pair_from_multiband_files(
        self,
        identifier: str,
        t1_path: str | Path,
        t2_path: str | Path,
        out_path: Optional[str | Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Run inference for one logical 2-band pair.

        `config` is runtime-only and may include semantic labels such as:
          - t1_label / t2_label
          - t1_role / t2_role
          - matches_training_semantics
        """
        self._log_input_semantics(identifier, self._resolve_input_semantics(config))
        data_t1, meta_t1 = self._read_geotiff(t1_path)
        data_t2, meta_t2 = self._read_geotiff(t2_path)
        return self._infer_pair_arrays(identifier, data_t1, meta_t1, data_t2, meta_t2, out_path or None)

    def run_pair_from_single_band_files(
        self,
        t1_vv: str | Path,
        t1_vh: str | Path,
        t2_vv: str | Path,
        t2_vh: str | Path,
        out_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str | Path] = None,
        identifier: Optional[str] = None,
    ) -> Path:
        """Align 4 single-band GeoTIFFs, optionally cache 2-band inputs, and infer one SR output."""
        pair_id = identifier or Path(out_path).stem.replace("_SR_x2", "")
        runtime_cfg = config or {}
        self._log_input_semantics(pair_id, self._resolve_input_semantics(runtime_cfg))
        resampling_name = runtime_cfg.get("resampling", "bilinear")
        resampling = self._resolve_resampling(resampling_name)

        ref_meta = self._build_reference_meta_for_single_band_pair(t1_vv, runtime_cfg)
        t1_vv_arr = self._align_single_band_to_reference(t1_vv, ref_meta, resampling)
        t1_vh_arr = self._align_single_band_to_reference(t1_vh, ref_meta, resampling)
        t2_vv_arr = self._align_single_band_to_reference(t2_vv, ref_meta, resampling)
        t2_vh_arr = self._align_single_band_to_reference(t2_vh, ref_meta, resampling)

        t1_stack = np.stack([t1_vv_arr, t1_vh_arr], axis=0).astype(np.float32)
        t2_stack = np.stack([t2_vv_arr, t2_vh_arr], axis=0).astype(np.float32)
        self._write_optional_staged_inputs(cache_dir, t1_stack, t2_stack, ref_meta)
        return self._infer_pair_arrays(
            pair_id,
            t1_stack,
            ref_meta,
            t2_stack,
            ref_meta.copy(),
            out_path,
            require_both_polarizations=True,
        )

    # ----------------------------------------------------------
    #  Main entry point
    # ----------------------------------------------------------
    def run(self) -> None:
        """Execute the full production inference pipeline for a directory.

        Steps:
          1. Scan input directory for valid T1/T2 pairs
          2. Loop over each pair:
             a. Read GeoTIFFs (2 bands each: VV, VH)
             b. Normalise (dB → [-1,1])
             c. Infer VV and VH sequentially 
             d. Denormalize ([-1,1] → dB)
             e. Write Output (PhiênHiệu_SR_x2.tif)
        """
        t_global_start = time.time()

        # ── paths ──
        cfg_in = self.config["input"]
        cfg_out = self.config["output"]
        
        input_dir = Path(cfg_in.get("input_dir", "data/input"))
        t1_prefix = cfg_in.get("t1_prefix", "s1t1_")
        t2_prefix = cfg_in.get("t2_prefix", "s1t2_")
        
        save_dir = Path(cfg_out["save_path"])
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Scan inputs ──
        logger.info("=" * 60)
        logger.info(f"Scanning directory: {input_dir}")
        logger.info(f"Looking for prefixes: T1='{t1_prefix}', T2='{t2_prefix}'")
        logger.info("=" * 60)

        jobs = self._scan_input_dir(input_dir, t1_prefix, t2_prefix)
        if not jobs:
            logger.error("No valid logical pairs found. Please check filenames and directory.")
            return

        logger.info(f"==> Found {len(jobs)} logical pairs for inference.")
        self._log_vram("Before processing directory")
        cache_aligned_inputs = bool(cfg_in.get("cache_aligned_inputs", False))
        cache_root = cfg_in.get("cache_dir")
        if cfg_out.get("save_name") and len(jobs) > 1:
            logger.warning("output.save_name is set but multiple jobs were found; per-job default names will be used.")

        ok_count = 0
        for idx, job in enumerate(jobs, 1):
            t_start = time.time()
            iden = job["identifier"]
            logger.info("=" * 60)
            logger.info(f"Processing Pair {idx}/{len(jobs)}  |  Identifier: '{iden}'")
            logger.info("=" * 60)
            try:
                explicit_out = None
                if cfg_out.get("save_name") and len(jobs) == 1:
                    explicit_out = save_dir / str(cfg_out["save_name"])

                if job["mode"] == "multiband":
                    out_path = self.run_pair_from_multiband_files(
                        iden,
                        job["t1_path"],
                        job["t2_path"],
                        out_path=explicit_out,
                    )
                else:
                    pair_cache_dir = None
                    if cache_aligned_inputs:
                        base_cache_root = Path(cache_root) if cache_root else save_dir / "_aligned_inputs"
                        pair_cache_dir = base_cache_root / iden
                    out_path = self.run_pair_from_single_band_files(
                        t1_vv=job["t1_vv"],
                        t1_vh=job["t1_vh"],
                        t2_vv=job["t2_vv"],
                        t2_vh=job["t2_vh"],
                        out_path=explicit_out or self._resolve_output_path(iden),
                        cache_dir=pair_cache_dir,
                        identifier=iden,
                    )
                ok_count += 1
                elapsed = time.time() - t_start
                self._log_vram(f"After finishing '{iden}'")
                logger.info(f"  ✓ Finished '{iden}' in {elapsed:.1f}s -> {out_path}")
            except Exception as exc:
                logger.error(f"  [!] Skipped '{iden}': {exc}")

        total_time = time.time() - t_global_start
        logger.info("=" * 60)
        logger.info(f"Processed {ok_count}/{len(jobs)} pairs successfully in {total_time:.1f}s!")
        logger.info("=" * 60)


# ==============================================================
#  CLI entry point
# ==============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ISSM_SAR Production Inference — SAR Super-Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python infer_production.py --config config/infer_config.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/infer_config.yaml",
        help="Path to the production inference YAML config.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Override the output filename (e.g. 'E_48_44_B_a_4_SR_x2.tif')",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    env_overrides = apply_inference_env_overrides(config)
    if env_overrides:
        config.setdefault("_runtime", {})["env_overrides"] = env_overrides
    
    if args.out_name:
        if "output" not in config:
            config["output"] = {}
        config["output"]["save_name"] = args.out_name

    try:
        inferencer = SARInferencer(config)
        inferencer.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Inference failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

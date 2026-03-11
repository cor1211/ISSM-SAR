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
  • Sliding-window inference with configurable overlap
  • 2D Gaussian blending to eliminate grid artefacts
  • Batch-wise GPU inference with optional AMP (FP16)
  • Denormalize back to dB, write 2-band GeoTIFF (×2 resolution)

Usage:
  python infer_production.py --config config/infer_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
import yaml
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────
# Ensure src/ is importable directly to bypass __init__.py (and avoid PyTorch Lightning dependence)
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import ISSM_SAR

# ── logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SARInferencer")


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

        # ── device ──
        device_name: str = config.get("device", "cuda")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            device_name = "cpu"
        self.device = torch.device(device_name)
        logger.info(f"Device: {self.device}")

        # ── normalisation params ──
        norm_cfg = config["normalization"]
        self.v_min: float = float(norm_cfg["v_min"])
        self.v_max: float = float(norm_cfg["v_max"])
        logger.info(f"dB clip range: [{self.v_min}, {self.v_max}]")

        # ── inference params ──
        inf_cfg = config["inference"]
        self.patch_size: int = int(inf_cfg.get("patch_size", 128))
        self.overlap_frac: float = float(inf_cfg.get("overlap", 0.25))
        self.batch_size: int = int(inf_cfg.get("batch_size", 16))
        self.use_amp: bool = bool(inf_cfg.get("use_amp", True))

        self.overlap_px: int = int(self.patch_size * self.overlap_frac)
        self.stride: int = self.patch_size - self.overlap_px
        logger.info(
            f"Patch={self.patch_size}, overlap={self.overlap_px}px, "
            f"stride={self.stride}, batch={self.batch_size}, AMP={self.use_amp}"
        )

        # ── load model architecture config ──
        model_config_path = Path(config["model_config_path"])
        if not model_config_path.is_absolute():
            model_config_path = _PROJECT_ROOT / model_config_path
        arch_cfg = load_yaml(model_config_path)
        model_cfg: Dict[str, Any] = arch_cfg["model"]

        # ── initialise two models ──
        logger.info("Initialising VV model …")
        self.model_vv = self._load_model(model_cfg, config["ckpt_path_vv"])
        logger.info("Initialising VH model …")
        self.model_vh = self._load_model(model_cfg, config["ckpt_path_vh"])

        # ── pre-compute gaussian window (on SR output scale = 2×patch) ──
        sr_patch = self.patch_size * 2
        self.gaussian_window = self._create_gaussian_window(sr_patch).to(self.device)
        logger.info(f"Gaussian blending window: {sr_patch}×{sr_patch}")

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
        logger.info(f"  ✓ Loaded checkpoint: {ckpt_path.name}")
        return model

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
        sr_h, sr_w = padded_h * 2, padded_w * 2
        output_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device=self.device)
        weight_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device=self.device)

        # ── generate patch coordinates ──
        coords = self._get_patch_coords(padded_h, padded_w)
        total_patches = len(coords)
        logger.info(f"  Total patches: {total_patches}")

        sr_patch_size = self.patch_size * 2
        gauss_win = self.gaussian_window  # (sr_patch_size, sr_patch_size)

        # ── batch inference ──
        for batch_start in tqdm(
            range(0, total_patches, self.batch_size),
            desc="  Inferring",
            leave=False,
        ):
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

            # ── forward pass ──
            if self.use_amp and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    sr_up, sr_down = model(inp_t1, inp_t2)
            else:
                sr_up, sr_down = model(inp_t1, inp_t2)

            # sr_fusion: average of last elements → (B, 1, 2*H_patch, 2*W_patch)
            sr_fusion = 0.5 * sr_up[-1] + 0.5 * sr_down[-1]

            # ── accumulate with gaussian weighting ──
            for idx, (y, x) in enumerate(batch_coords):
                sr_patch = sr_fusion[idx, 0, :, :]  # (sr_patch_size, sr_patch_size)
                out_y = y * 2
                out_x = x * 2
                output_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += (
                    sr_patch * gauss_win
                )
                weight_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += gauss_win

            # Free GPU memory for this batch
            del inp_t1, inp_t2, sr_up, sr_down, sr_fusion

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
            logger.info(
                f"  Read: {path.name}  |  shape={data.shape}  |  "
                f"CRS={src.crs}  |  dtype={src.dtypes[0]}"
            )
        return data, meta

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

        logger.info(f"  ✓ Saved: {path}  |  shape=({bands}, {h}, {w})")

    # ----------------------------------------------------------
    #  Main entry point
    # ----------------------------------------------------------
    def run(self) -> None:
        """Execute the full production inference pipeline.

        Steps:
          1. Read input GeoTIFFs (T1, T2) — 2 bands each (VV, VH)
          2. Normalise each band (dB → [-1,1])
          3. Run sliding-window inference per band
          4. Denormalize ([-1,1] → dB)
          5. Stack VV + VH → 2-band output
          6. Write output GeoTIFF with halved pixel size
        """
        t_start = time.time()

        # ── paths ──
        cfg_in = self.config["input"]
        cfg_out = self.config["output"]
        t1_path = Path(cfg_in["s1t1_path"])
        t2_path = Path(cfg_in["s1t2_path"])
        save_dir = Path(cfg_out["save_path"])
        save_dir.mkdir(parents=True, exist_ok=True)

        compression = cfg_out.get("compression", "DEFLATE")
        tiled = cfg_out.get("tiled", True)
        blockxsize = cfg_out.get("blockxsize", 256)
        blockysize = cfg_out.get("blockysize", 256)

        # ── 1. Read inputs ──
        logger.info("=" * 60)
        logger.info("Step 1/5 — Reading input GeoTIFFs")
        logger.info("=" * 60)
        data_t1, meta_t1 = self._read_geotiff(t1_path)
        data_t2, meta_t2 = self._read_geotiff(t2_path)

        # Validate shapes
        if data_t1.shape != data_t2.shape:
            raise ValueError(
                f"T1 and T2 shape mismatch: {data_t1.shape} vs {data_t2.shape}"
            )
        if data_t1.shape[0] < 2:
            raise ValueError(
                f"Expected at least 2 bands (VV, VH), got {data_t1.shape[0]}"
            )

        n_bands, orig_h, orig_w = data_t1.shape
        logger.info(f"  Image size: {orig_w}×{orig_h}, bands={n_bands}")

        # Extract bands: Band 1 = VV, Band 2 = VH
        vv_t1 = data_t1[0]  # (H, W)
        vh_t1 = data_t1[1]  # (H, W)
        vv_t2 = data_t2[0]
        vh_t2 = data_t2[1]

        # ── 2. Normalise dB → [-1, 1] ──
        logger.info("=" * 60)
        logger.info("Step 2/5 — Normalising (dB → [-1, 1])")
        logger.info("=" * 60)
        vv_t1_norm = self.normalize_db(vv_t1)
        vh_t1_norm = self.normalize_db(vh_t1)
        vv_t2_norm = self.normalize_db(vv_t2)
        vh_t2_norm = self.normalize_db(vh_t2)
        logger.info(
            f"  VV range after norm: [{vv_t1_norm.min():.4f}, {vv_t1_norm.max():.4f}]"
        )
        logger.info(
            f"  VH range after norm: [{vh_t1_norm.min():.4f}, {vh_t1_norm.max():.4f}]"
        )

        # ── 3. Infer VV ──
        logger.info("=" * 60)
        logger.info("Step 3/5 — Inference: VV polarisation")
        logger.info("=" * 60)
        sr_vv = self._infer_band(self.model_vv, vv_t1_norm, vv_t2_norm)

        # ── 4. Infer VH ──
        logger.info("=" * 60)
        logger.info("Step 4/5 — Inference: VH polarisation")
        logger.info("=" * 60)
        sr_vh = self._infer_band(self.model_vh, vh_t1_norm, vh_t2_norm)

        # ── 5. Denormalize & write output ──
        logger.info("=" * 60)
        logger.info("Step 5/5 — Denormalising & writing output")
        logger.info("=" * 60)

        sr_vv_db = self.denormalize_db(sr_vv)
        sr_vh_db = self.denormalize_db(sr_vh)
        logger.info(
            f"  VV SR dB range: [{sr_vv_db.min():.2f}, {sr_vv_db.max():.2f}]"
        )
        logger.info(
            f"  VH SR dB range: [{sr_vh_db.min():.2f}, {sr_vh_db.max():.2f}]"
        )

        # Stack → (2, 2H, 2W)
        sr_output = np.stack([sr_vv_db, sr_vh_db], axis=0)

        # ── Update geospatial transform (pixel size ÷ 2) ──
        src_transform: Affine = meta_t1["transform"]
        sr_transform = Affine(
            src_transform.a / 2.0,   # pixel width  (halved)
            src_transform.b,
            src_transform.c,         # top-left X
            src_transform.d,
            src_transform.e / 2.0,   # pixel height (halved, negative)
            src_transform.f,         # top-left Y
        )

        # ── output filename ──
        stem = t1_path.stem
        out_name = f"{stem}_SR_x2.tif"
        out_path = save_dir / out_name

        self._write_geotiff(
            out_path,
            sr_output,
            meta_t1,
            sr_transform,
            compression=compression,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,
        )

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info(f"Done!  Total time: {elapsed:.1f}s")
        logger.info(
            f"Output: {out_path}  "
            f"({sr_output.shape[2]}×{sr_output.shape[1]}, {sr_output.shape[0]} bands)"
        )
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
    args = parser.parse_args()

    config = load_yaml(args.config)

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

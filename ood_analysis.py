#!/usr/bin/env python3
"""
OOD (Out-of-Distribution) Stability Analysis
=============================================
Evaluates a trained Tustin-Mamba model across four data regimes and
measures the *Stability Gap* — the degradation of Tustin denominator
condition number before accuracy even drops.

Scenarios
---------
  ID         : Path-X 128 test set  (the training distribution)
  Near-OOD   : Path-X with Gaussian blur (σ=2) + intensity inversion
  Far-OOD    : LRA-Image (CIFAR-10) upscaled 32×32→128×128  (archive present)
  Noise-OOD  : Pure Gaussian noise clamped to [0, 255]

Key metrics
-----------
  accuracy              – binary classification accuracy (ID / Near-OOD only)
  predictive_entropy    – H = -Σ p_i log p_i  (normalised to [0, 1])
  spectral_radius       – max |eigenvalue| of Ā  (block 0)
  tustin_cond           – condition number of (I − Δ/2·A)  (block 0)
  latent_variance       – variance of last-block hidden activations
  stability_gap         – (ood_cond − id_cond) / id_cond × 100 %

Usage
-----
  python ood_analysis.py \\
      --checkpoint_path checkpoints_research/best_checkpoint.pt \\
      [--data_dir archive] \\
      [--batch_size 4] \\
      [--num_batches 50] \\
      [--output_dir results/ood] \\
      [--device cuda]
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba_pytorch import MambaLM
from research_metrics import StabilityAnalyzer

# ── scipy is used only for Gaussian blur; imported lazily per function ─────────
try:
    from scipy.ndimage import gaussian_filter
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    warnings.warn("scipy not found – Gaussian blur will use a box filter fallback.")

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD Stability Analysis for Tustin-Mamba Path-X model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="archive",
        help="Directory containing the LRA pickle files.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Primary batch size. Falls back to 1 on OOM.",
    )
    p.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to evaluate per scenario.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/ood",
        help="Directory to write JSON results.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override ('cuda', 'cpu'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--id_cond_baseline",
        type=float,
        default=4.12,
        help="Known ID condition number baseline for Stability Gap reference.",
    )
    return p.parse_args()

# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint(path: str) -> Dict:
    """
    Load checkpoint safely.

    Tries weights_only=True first (security best practice).  Falls back to
    weights_only=False with a clear warning if the checkpoint contains
    non-tensor Python objects (optimizer state dicts often do).
    """
    log.info(f"Loading checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        log.info("  Loaded with weights_only=True  ✓")
    except Exception as e_safe:
        log.warning(
            f"  weights_only=True failed ({type(e_safe).__name__}: {e_safe}). "
            "Falling back to weights_only=False. "
            "Only load checkpoints you trust."
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        log.info("  Loaded with weights_only=False  (trusted source)")
    return ckpt


def log_checkpoint_metadata(ckpt: Dict) -> None:
    """Print all useful metadata so we can verify the correct checkpoint."""
    log.info("=" * 70)
    log.info("CHECKPOINT METADATA")
    log.info("=" * 70)
    log.info(f"  Step          : {ckpt.get('step', 'N/A')}")
    log.info(f"  Timestamp     : {ckpt.get('timestamp', 'N/A')}")
    wc = ckpt.get("wall_clock_time", 0)
    log.info(f"  Wall-clock    : {wc / 3600:.2f} h")

    metrics = ckpt.get("metrics", {})
    for k, v in metrics.items():
        log.info(f"  {k:<20}: {v}")

    model_cfg = ckpt.get("config", {}).get("model", {})
    log.info("  Model config  :")
    for k, v in model_cfg.items():
        log.info(f"    {k:<18}: {v}")

    dv = ckpt.get("dataset_version", {})
    if dv:
        log.info("  Dataset version:")
        for k, v in dv.items():
            log.info(f"    {k}: {v}")
    log.info("=" * 70)

# =============================================================================
# Model construction and loading
# =============================================================================

def build_model(ckpt: Dict, device: torch.device) -> MambaLM:
    """Reconstruct MambaLM from the config embedded in the checkpoint."""
    cfg = ckpt["config"]["model"]
    model = MambaLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        mode=cfg["mode"],
    )
    # Strip _orig_mod. prefix if checkpoint came from a compiled model
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        log.info("  Stripping _orig_mod. prefix from state dict (compiled checkpoint).")
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        log.warning(f"  Missing keys in state dict: {missing}")
    if unexpected:
        log.warning(f"  Unexpected keys in state dict: {unexpected}")

    model = model.to(device)
    model.eval()
    log.info(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

# =============================================================================
# Dataset classes
# =============================================================================

class PathXPickle(Dataset):
    """Loads a Path-X128 LRA pickle (train / dev / test)."""

    def __init__(self, path: str):
        import pickle
        log.info(f"  Loading: {path}")
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self.data = raw
        log.info(f"  Loaded {len(raw):,} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.data[idx]
        x = torch.from_numpy(ex["input_ids_0"].astype(np.int64))
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        return x, y


class NearOODDataset(Dataset):
    """
    Path-X with two transformations applied:
      1. Gaussian blur (σ=2) on the 128×128 pixel grid
      2. Intensity inversion (255 − pixel)

    Both transformations preserve the token domain [0, 255] and keep the
    same sequence length (16 384) so the model receives a well-formed input.
    """

    def __init__(self, base_dataset: PathXPickle, sigma: float = 2.0):
        self.base = base_dataset
        self.sigma = sigma
        self.side = int(math.sqrt(len(base_dataset[0][0])))   # 128

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        arr = x.numpy().astype(np.float32).reshape(self.side, self.side)

        # 1. Gaussian blur
        if _SCIPY_OK:
            arr = gaussian_filter(arr, sigma=self.sigma)
        else:
            # Box-filter fallback: average with a 5×5 uniform kernel
            kernel = torch.ones(1, 1, 5, 5) / 25.0
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            arr = F.conv2d(t, kernel, padding=2).squeeze().numpy()

        # 2. Intensity inversion
        arr = 255.0 - arr

        arr = np.clip(arr, 0, 255).astype(np.int64).flatten()
        return torch.from_numpy(arr), y


class FarOODDataset(Dataset):
    """
    LRA-Image (CIFAR-10) upscaled from 32×32 → 128×128 using bilinear
    interpolation (scipy zoom, factor 4), then flattened to 16 384 tokens.

    Labels are NOT compatible with the binary Path-X task; accuracy is
    therefore not reported for this scenario.
    """

    def __init__(self, path: str):
        import pickle
        log.info(f"  Loading Far-OOD (LRA-Image): {path}")
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self.data = raw
        self.side_src = 32
        self.side_dst = 128
        log.info(f"  Loaded {len(raw):,} LRA-Image examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.data[idx]
        arr = ex["input_ids_0"].astype(np.float32).reshape(
            self.side_src, self.side_src
        )
        # Bilinear upscale 32 → 128
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,32,32)
        t = F.interpolate(
            t,
            size=(self.side_dst, self.side_dst),
            mode="bilinear",
            align_corners=False,
        )
        arr = t.squeeze().numpy()
        arr = np.clip(arr, 0, 255).astype(np.int64).flatten()   # (16384,)
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        return torch.from_numpy(arr), y


class NoiseOODDataset(Dataset):
    """
    Pure Gaussian noise samples in pixel space.

    Noise is drawn from N(128, 60²), clamped to [0, 255], and rounded to
    integer tokens — matching the PathX vocabulary without using any real
    image structure.
    """

    def __init__(self, seq_len: int = 16384, num_samples: int = 10000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Fix seed for reproducibility
        rng = np.random.default_rng(42)
        self._data = rng.normal(128.0, 60.0, size=(num_samples, seq_len))
        self._data = np.clip(self._data, 0, 255).astype(np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._data[idx])
        y = torch.tensor(-1, dtype=torch.long)   # no valid label
        return x, y

# =============================================================================
# Metric computation helpers
# =============================================================================

def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """
    H = -Σ p_i log p_i for binary classification logits.

    Follows the training convention: pool over sequence dimension, take
    the first two vocabulary entries as class logits.

    Returns normalised entropy in [0, 1]  (H / log 2).
    """
    pooled = logits.mean(dim=1)[:, :2]          # (B, 2)
    probs = F.softmax(pooled, dim=-1)            # (B, 2)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)   # (B,)
    return (entropy.mean() / math.log(2)).item()


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[float]:
    """Binary classification accuracy.  Returns None if labels are invalid."""
    if (labels == -1).all():
        return None
    pooled = logits.mean(dim=1)[:, :2]
    preds = pooled.argmax(dim=-1)
    mask = labels != -1
    if not mask.any():
        return None
    return (preds[mask] == labels[mask]).float().mean().item()


def compute_latent_variance(hidden: torch.Tensor) -> float:
    """
    Variance of the final Mamba block's output activations.

    hidden : (B, L, D) — captured by the forward hook on blocks[-1].

    We compute variance across the D (feature) dimension at each (B, L)
    position, then average over B and L.  Higher variance → richer / more
    uncertain representation.
    """
    return hidden.float().var(dim=-1).mean().item()

# =============================================================================
# Core evaluation loop
# =============================================================================

class OODAnalyzer:
    """Orchestrates evaluation across all OOD scenarios."""

    def __init__(
        self,
        model: MambaLM,
        device: torch.device,
        id_cond_baseline: float,
    ):
        self.model = model
        self.device = device
        self.id_cond_baseline = id_cond_baseline
        self._hook_handle = None
        self._hook_buffer: List[torch.Tensor] = []

    # ── forward hook ──────────────────────────────────────────────────────────

    def _register_last_block_hook(self) -> None:
        """Attach a forward hook to the last Mamba block."""

        def _hook(module, inp, output):
            # MambaBlock returns (y, diagnostics) only when return_diagnostics=True.
            # The last block is always called WITHOUT return_diagnostics, so
            # output is a plain tensor (B, L, D).
            if isinstance(output, tuple):
                self._hook_buffer.append(output[0].detach().cpu())
            else:
                self._hook_buffer.append(output.detach().cpu())

        self._hook_handle = self.model.blocks[-1].register_forward_hook(_hook)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ── single-batch evaluation ───────────────────────────────────────────────

    def _eval_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, Optional[float]]:
        """
        Run one forward pass and return all metrics.

        Uses return_diagnostics=True so block-0 exposes A_bar, A, delta
        for the Tustin stability analysis.
        """
        self._hook_buffer.clear()
        x = x.to(self.device)
        y_cpu = y

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                logits, diagnostics = self.model(x, return_diagnostics=True)

        # Stability (block 0)
        stability = StabilityAnalyzer.analyze_all(
            A_bar=diagnostics["A_bar"],
            A=diagnostics["A"],
            delta=diagnostics["delta"],
        )

        # Latent variance (last block, via hook)
        if self._hook_buffer:
            hidden = self._hook_buffer[-1]           # (B, L, D)
            lv = compute_latent_variance(hidden)
        else:
            lv = float("nan")

        # Move logits to CPU for entropy / accuracy
        logits_cpu = logits.float().cpu()

        return {
            "accuracy"          : compute_accuracy(logits_cpu, y_cpu),
            "predictive_entropy": compute_predictive_entropy(logits_cpu),
            "spectral_radius"   : stability["spectral_radius"],
            "tustin_cond"       : stability["tustin_denominator_condition"],
            "erf"               : stability["effective_receptive_field"],
            "latent_variance"   : lv,
        }

    # ── scenario evaluation with OOM fallback ─────────────────────────────────

    def _eval_scenario(
        self,
        loader: DataLoader,
        scenario_name: str,
        num_batches: int,
        fallback_batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate `num_batches` batches from `loader`.

        If a CUDA OOM error occurs the batch is retried sample-by-sample
        (effective batch size = 1) to keep the evaluation alive.
        """
        aggregated: Dict[str, List[float]] = defaultdict(list)
        batches_done = 0

        self._register_last_block_hook()
        try:
            for x_batch, y_batch in loader:
                if batches_done >= num_batches:
                    break

                # Attempt full batch first
                try:
                    metrics = self._eval_batch(x_batch, y_batch)
                    for k, v in metrics.items():
                        if v is not None:
                            aggregated[k].append(v)
                    batches_done += 1

                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    log.warning(
                        f"  [{scenario_name}] VRAM OOM on batch of size "
                        f"{x_batch.shape[0]}; retrying sample-by-sample."
                    )
                    torch.cuda.empty_cache() if self.device.type == "cuda" else None
                    gc.collect()

                    for i in range(x_batch.shape[0]):
                        try:
                            m = self._eval_batch(
                                x_batch[i : i + 1],
                                y_batch[i : i + 1],
                            )
                            for k, v in m.items():
                                if v is not None:
                                    aggregated[k].append(v)
                        except RuntimeError:
                            log.warning(
                                f"  [{scenario_name}] OOM on single sample {i}; skipping."
                            )
                            torch.cuda.empty_cache() if self.device.type == "cuda" else None
                    batches_done += 1

        finally:
            self._remove_hook()

        # Aggregate: mean ± std
        results: Dict[str, float] = {}
        for k, vals in aggregated.items():
            arr = np.array(vals, dtype=np.float64)
            results[k] = float(arr.mean())
            results[f"{k}_std"] = float(arr.std())

        return results

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        scenarios: Dict[str, DataLoader],
        num_batches: int,
    ) -> Dict[str, Dict[str, float]]:
        """Run all scenarios and return aggregated metrics."""
        all_results: Dict[str, Dict[str, float]] = {}

        for name, loader in scenarios.items():
            log.info(f"\n{'─'*70}")
            log.info(f"  Scenario: {name.upper()}")
            log.info(f"{'─'*70}")
            t0 = time.time()
            res = self._eval_scenario(loader, name, num_batches)
            elapsed = time.time() - t0
            log.info(f"  Done in {elapsed:.1f}s  ({num_batches} batches)")
            for k, v in res.items():
                if not k.endswith("_std"):
                    log.info(f"    {k:<28}: {v:.6f}")
            all_results[name] = res

            # Release VRAM between scenarios
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return all_results

# =============================================================================
# Results reporting
# =============================================================================

_COL_W = 14   # column width for the table


def _fmt(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "N/A".center(_COL_W)
    return f"{val:.{decimals}f}".center(_COL_W)


def print_stability_gap_table(
    results: Dict[str, Dict[str, float]],
    id_cond_baseline: float,
) -> None:
    """Print a formatted research table and stability gap summary."""
    scenarios = list(results.keys())
    col_header = "".join(s[:_COL_W].center(_COL_W) for s in scenarios)

    sep = "═" * (22 + _COL_W * len(scenarios))

    log.info("")
    log.info(sep)
    log.info("  STABILITY GAP ANALYSIS — RESULTS")
    log.info(sep)
    log.info(f"  {'Metric':<22}" + col_header)
    log.info("─" * (22 + _COL_W * len(scenarios)))

    rows = [
        ("accuracy (%)",        "accuracy",           lambda v: f"{v*100:.2f}%" if v is not None else "N/A"),
        ("entropy (norm.)",     "predictive_entropy", lambda v: f"{v:.4f}"),
        ("spectral_radius",     "spectral_radius",    lambda v: f"{v:.4f}"),
        ("tustin_cond",         "tustin_cond",        lambda v: f"{v:.4f}"),
        ("ERF (steps)",         "erf",                lambda v: f"{int(v)}"),
        ("latent_variance",     "latent_variance",    lambda v: f"{v:.6f}"),
    ]

    for label, key, fmt in rows:
        cells = ""
        for sc in scenarios:
            v = results[sc].get(key)
            cells += fmt(v).center(_COL_W) if v is not None else "N/A".center(_COL_W)
        log.info(f"  {label:<22}" + cells)

    log.info(sep)

    # Stability gap per OOD scenario relative to ID baseline
    id_cond = results.get("ID", {}).get("tustin_cond", id_cond_baseline)
    log.info("")
    log.info("  STABILITY GAP  (Δcond / ID_cond × 100%)")
    log.info("  Reference ID condition number : "
             f"{id_cond:.4f}  (research baseline: {id_cond_baseline:.2f})")
    log.info("")
    for sc in scenarios:
        if sc == "ID":
            continue
        ood_cond = results[sc].get("tustin_cond")
        if ood_cond is not None:
            gap = (ood_cond - id_cond) / max(id_cond, 1e-9) * 100.0
            sign = "+" if gap >= 0 else ""
            log.info(f"  {sc:<20}: {sign}{gap:.2f}%  "
                     f"(cond {id_cond:.4f} → {ood_cond:.4f})")
    log.info(sep)


def save_results_json(
    results: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
    ckpt: Dict,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"ood_results_{timestamp}.json")

    payload = {
        "checkpoint": {
            "path"        : args.checkpoint_path,
            "step"        : ckpt.get("step"),
            "timestamp"   : ckpt.get("timestamp"),
            "wall_clock_h": round(ckpt.get("wall_clock_time", 0) / 3600, 3),
            "metrics"     : ckpt.get("metrics", {}),
            "model_config": ckpt.get("config", {}).get("model", {}),
        },
        "eval_config": {
            "num_batches"     : args.num_batches,
            "batch_size"      : args.batch_size,
            "id_cond_baseline": args.id_cond_baseline,
        },
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    log.info(f"\n  Results saved → {out_path}")
    return out_path

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Auto-detected device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Running on CPU (slow for 16k sequences).")

    # ── checkpoint ────────────────────────────────────────────────────────────
    ckpt = load_checkpoint(args.checkpoint_path)
    log_checkpoint_metadata(ckpt)

    # ── model ─────────────────────────────────────────────────────────────────
    log.info("\nBuilding model from checkpoint config …")
    model = build_model(ckpt, device)

    # ── datasets ──────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    pathx_test_path = os.path.join(
        data_dir,
        "lra-pathfinder128-curv_contour_length_14.test.pickle",
    )
    lra_image_path = os.path.join(data_dir, "lra-image.test.pickle")

    log.info("\nLoading datasets …")

    pathx_test = PathXPickle(pathx_test_path)
    near_ood_ds = NearOODDataset(pathx_test, sigma=2.0)
    noise_ds = NoiseOODDataset(seq_len=16384, num_samples=min(10000, len(pathx_test)))

    _dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,           # keep 0 to avoid multiprocessing issues with large seqs
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    scenarios: Dict[str, DataLoader] = {
        "ID"        : DataLoader(pathx_test,  **_dl_kwargs),
        "Near-OOD"  : DataLoader(near_ood_ds, **_dl_kwargs),
        "Noise-OOD" : DataLoader(noise_ds,    **_dl_kwargs),
    }

    # Far-OOD only if LRA-Image is present
    if os.path.exists(lra_image_path):
        far_ood_ds = FarOODDataset(lra_image_path)
        scenarios["Far-OOD"] = DataLoader(far_ood_ds, **_dl_kwargs)
    else:
        log.warning(
            f"Far-OOD dataset not found at {lra_image_path}. "
            "Skipping Far-OOD scenario. "
            "Provide the LRA-Image pickle to include it."
        )

    # ── run analysis ──────────────────────────────────────────────────────────
    analyzer = OODAnalyzer(
        model=model,
        device=device,
        id_cond_baseline=args.id_cond_baseline,
    )

    log.info(f"\nEvaluating {len(scenarios)} scenarios × {args.num_batches} batches …")
    results = analyzer.run(scenarios, num_batches=args.num_batches)

    # ── report ────────────────────────────────────────────────────────────────
    print_stability_gap_table(results, id_cond_baseline=args.id_cond_baseline)
    save_results_json(results, args, ckpt, args.output_dir)


if __name__ == "__main__":
    main()

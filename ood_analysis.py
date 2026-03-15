"""
OOD Stability Analysis for ZOH-Discretized Mamba on Path-X
===========================================================

Hypothesis: When the model sees OOD data, the numerical stability
(Condition Number) degrades *before* the accuracy even drops.

The script:
  1. Loads a checkpoint (full state dict + optimizer + metadata)
  2. Evaluates ID, Near-OOD, Far-OOD, and Noise-OOD scenarios
  3. Extracts: spectral_radius, tustin_denominator_condition (ZOH mode),
               predictive entropy, latent variance of last Mamba block
  4. Reports the Stability Gap: Δκ between each OOD scenario and ID

Usage:
    python ood_analysis.py                        # auto-detects best/latest checkpoint
    python ood_analysis.py --checkpoint_path checkpoints_research/best_checkpoint.pt
    python ood_analysis.py --checkpoint_path <path> --batch_size 1 --no_far_ood
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Project imports (model + research metrics)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from mamba_pytorch import MambaLM
from research_metrics.stability_metrics import StabilityAnalyzer


# ===========================================================================
# CONSTANTS  –  must match train_pathx_research.py
# ===========================================================================
VOCAB_SIZE = 256
MAX_SEQ_LEN = 16384
D_MODEL = 128
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2
MODE = "zoh"

_ARCHIVE = os.path.join(_ROOT, "archive")
_PATHX_DEV = os.path.join(_ARCHIVE, "lra-pathfinder128-curv_contour_length_14.dev.pickle")
_PATHX_TEST = os.path.join(_ARCHIVE, "lra-pathfinder128-curv_contour_length_14.test.pickle")

# Known ID condition number from research (for Stability Gap reporting)
ID_CONDITION_REF = 4.12


# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ood_analysis")


# ===========================================================================
# DATASETS
# ===========================================================================

class PathXDataset(Dataset):
    """Original Path-X (Long Range Arena) dataset."""

    def __init__(self, data_path: str) -> None:
        log.info(f"  Loading Path-X: {data_path}")
        with open(data_path, "rb") as fh:
            self.data = pickle.load(fh)
        log.info(f"  → {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.data[idx]
        x = torch.from_numpy(ex["input_ids_0"]).long()   # (16384,)
        y = torch.tensor(ex["label"], dtype=torch.long)
        return x, y


class NearOODDataset(Dataset):
    """
    Near-OOD: Path-X samples with Gaussian blur + colour inversion.

    Transformation pipeline (in float pixel space [0, 255]):
      1. 1-D reflected Gaussian blur (kernel σ=2)
      2. Colour inversion: pixel ← 255 − pixel
      3. Re-quantize to integer vocab [0, 255]
    """

    def __init__(self, base: PathXDataset, sigma: float = 2.0) -> None:
        self.base = base
        self.sigma = sigma
        k = max(3, int(6 * sigma + 1) | 1)   # odd kernel size
        half = k // 2
        t = torch.arange(k, dtype=torch.float32) - half
        kern = torch.exp(-t ** 2 / (2 * sigma ** 2))
        kern /= kern.sum()
        # Store as (1, 1, k) for F.conv1d
        self.register_kernel = kern.view(1, 1, -1)
        self.pad = half

    def _blur(self, x_f: torch.Tensor) -> torch.Tensor:
        """1-D reflected-pad Gaussian blur on a (L,) float tensor."""
        kern = self.register_kernel
        x_pad = F.pad(x_f.view(1, 1, -1), (self.pad, self.pad), mode="reflect")
        return F.conv1d(x_pad, kern).squeeze()   # (L,)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        x_f = x.float()               # [0, 255]
        x_f = self._blur(x_f)         # Gaussian blur
        x_f = 255.0 - x_f             # colour inversion
        return x_f.clamp(0, 255).long(), y


class FarOODDataset(Dataset):
    """
    Far-OOD: CIFAR-10 images flattened and zero-padded to MAX_SEQ_LEN tokens.

    CIFAR-10: 3 × 32 × 32 = 3 072 tokens → pad to 16 384.
    Labels mapped to binary (label % 2) to match Path-X output head.
    """

    def __init__(self, max_samples: int = 1_000) -> None:
        try:
            import torchvision
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for the CIFAR-10 Far-OOD scenario. "
                "Install with: pip install torchvision"
            ) from exc

        self._ds = torchvision.datasets.CIFAR10(
            root="/tmp/cifar10_ood",
            train=False,
            download=True,
            transform=T.ToTensor(),
        )
        self.n = min(max_samples, len(self._ds))
        log.info(f"  CIFAR-10 Far-OOD: {self.n} samples (padded 3072 → {MAX_SEQ_LEN})")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self._ds[idx]             # img: (3, 32, 32) float [0, 1]
        tokens = (img.flatten() * 255).long()  # (3072,)
        tokens = F.pad(tokens, (0, MAX_SEQ_LEN - len(tokens)), value=0)  # (16384,)
        return tokens, torch.tensor(label % 2, dtype=torch.long)


class NoiseOODDataset(Dataset):
    """
    Noise-OOD: Pure Gaussian noise quantized to integer vocab [0, 255].

    Each sample has an independent, reproducible seed (idx + 42).
    """

    def __init__(
        self,
        n_samples: int = 500,
        seq_len: int = MAX_SEQ_LEN,
        mean: float = 127.5,
        std: float = 50.0,
    ) -> None:
        self.n = n_samples
        self.seq_len = seq_len
        self.mean = mean
        self.std = std
        log.info(f"  Noise-OOD: {n_samples} synthetic Gaussian noise samples (L={seq_len})")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(idx + 42)
        x = torch.normal(mean=self.mean, std=self.std, size=(self.seq_len,), generator=g)
        return x.clamp(0, 255).long(), torch.tensor(idx % 2, dtype=torch.long)


# ===========================================================================
# CHECKPOINT LOADING
# ===========================================================================

def load_checkpoint(
    path: str,
    model: MambaLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """
    Load a full training checkpoint.

    - Uses weights_only=True for security; falls back with a warning if the
      checkpoint contains non-tensor objects that require pickle.
    - Strips _orig_mod. prefix produced by torch.compile.
    - Restores optimizer state if an optimizer is supplied.
    - Logs step, wall-clock time, and best val accuracy to the console.
    """
    log.info(f"\nLoading checkpoint: {path}")

    ckpt: Dict
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        warnings.warn(
            f"weights_only=True failed ({e}). Retrying with weights_only=False. "
            "Only load checkpoints from trusted sources.",
            stacklevel=2,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ── Model state dict ────────────────────────────────────────────────────
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        log.warning(f"  Missing keys: {missing}")
        log.warning(f"  Unexpected keys: {unexpected}")
    else:
        log.info("  Model state dict: loaded (all keys matched)")

    # ── Optimizer state dict ─────────────────────────────────────────────────
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        log.info("  Optimizer state dict: loaded")

    # ── Metadata ─────────────────────────────────────────────────────────────
    step        = ckpt.get("step", "unknown")
    timestamp   = ckpt.get("timestamp", "N/A")
    wall_clock  = ckpt.get("wall_clock_time", 0)
    metrics     = ckpt.get("metrics", {})
    config      = ckpt.get("config", {})

    val_acc  = metrics.get("val_accuracy",  metrics.get("val_acc",  "N/A"))
    val_loss = metrics.get("val_loss", "N/A")
    train_loss = metrics.get("train_loss", "N/A")

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║             CHECKPOINT METADATA (verification)              ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info(f"║  Step / Iteration   : {str(step):<40s}║")
    log.info(f"║  Saved at (UTC)     : {str(timestamp):<40s}║")
    log.info(f"║  Wall-clock elapsed : {wall_clock/3600:.2f} h{'':<36s}║")
    log.info(f"║  Train Loss         : {str(train_loss):<40s}║")
    log.info(f"║  Val   Loss         : {str(val_loss):<40s}║")
    log.info(f"║  Best Val Accuracy  : {str(val_acc):<40s}║")
    log.info(f"║  Mode               : {config.get('mode', MODE):<40s}║")
    log.info(f"║  D_model / N_layers : {config.get('d_model', D_MODEL)} / {config.get('n_layers', N_LAYERS)}"
             f"{'':<35s}║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    return ckpt


# ===========================================================================
# STABILITY METRIC EXTRACTION  (wraps StabilityAnalyzer)
# ===========================================================================

def extract_stability_metrics(diagnostics: Dict) -> Dict[str, float]:
    """
    Compute all relevant stability metrics from the S6-layer diagnostics dict.

    diagnostics keys (from MambaLM.forward with return_diagnostics=True):
        A_bar : (B, L, D, N)  – discretised state matrix
        A     : (D, N)         – continuous A (all negative)
        delta : (B, L, D)      – step sizes (all positive)

    Returns dict with keys:
        stability/spectral_radius
        stability/tustin_denominator_condition   ← primary stability metric
        stability/zoh_exponent_condition         ← ZOH-specific dynamic range
        stability/effective_receptive_field
    """
    A_bar = diagnostics["A_bar"]
    A     = diagnostics["A"]
    delta = diagnostics["delta"]

    spectral_radius  = StabilityAnalyzer.compute_spectral_radius(A_bar)
    zoh_condition    = StabilityAnalyzer.compute_zoh_exponent_condition(A, delta)
    # Tustin denominator condition is computed even in ZOH mode as a comparative
    # signal: it represents what the Tustin denominator would be with these A/Δ.
    tustin_condition = StabilityAnalyzer.compute_tustin_denominator_condition(A, delta)
    erf              = StabilityAnalyzer.compute_effective_receptive_field(A_bar)

    return {
        "stability/spectral_radius":              spectral_radius,
        "stability/tustin_denominator_condition": tustin_condition,
        "stability/zoh_exponent_condition":       zoh_condition,
        "stability/effective_receptive_field":    float(erf),
    }


# ===========================================================================
# PREDICTIVE ENTROPY
# ===========================================================================

def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """
    H = -Σ p_i log(p_i), averaged over the batch.

    For binary Path-X: pool logits[:, :, :2] over the sequence dimension,
    then compute entropy of the resulting B-dimensional softmax distribution.

    Args:
        logits: (B, L, V) raw logits

    Returns:
        Mean entropy (nats) over the batch.
    """
    # Use first 2 logit positions (Path-X binary classification head)
    binary_logits = logits[:, :, :2].float()          # (B, L, 2)
    pooled_logits = binary_logits.mean(dim=1)         # (B, 2)  – mean-pool over L
    probs     = F.softmax(pooled_logits, dim=-1)      # (B, 2)
    log_probs = torch.log(probs + 1e-10)
    entropy   = -(probs * log_probs).sum(dim=-1)      # (B,)
    return float(entropy.mean().item())


# ===========================================================================
# LATENT VARIANCE  –  last Mamba block hidden states
# ===========================================================================

def _register_last_block_hook(model: MambaLM) -> Tuple[Dict, object]:
    """
    Register a forward hook on the LAST MambaBlock to capture its output
    tensor (shape: B, L, D) before the final RMSNorm.

    Returns:
        captured : mutable dict; populated with 'hidden' (B, L, D) after each fwd
        handle   : hook handle (call .remove() when done)
    """
    captured: Dict = {}

    def _hook(_module: nn.Module, _inp, output):
        # MambaBlock.forward returns either a tensor or (tensor, diagnostics)
        h = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = h.detach()

    handle = model.blocks[-1].register_forward_hook(_hook)
    return captured, handle


def compute_latent_variance(hidden: torch.Tensor) -> float:
    """
    Mean per-sample variance of the last-block hidden states.

    Args:
        hidden: (B, L, D) last Mamba block output

    Returns:
        Mean variance across D, averaged over the batch.
    """
    # var over time L → (B, D); then mean over (B, D)
    return float(hidden.float().var(dim=1).mean().item())


# ===========================================================================
# SINGLE-SCENARIO EVALUATION
# ===========================================================================

@torch.no_grad()
def evaluate_scenario(
    model: MambaLM,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> Dict[str, float]:
    """
    Run the evaluation pipeline for one OOD scenario.

    For each batch:
      1. Forward pass with return_diagnostics=True (gets first-block A_bar/A/Δ)
      2. Extract stability metrics via StabilityAnalyzer
      3. Compute predictive entropy from logits
      4. Capture latent variance from last Mamba block (via forward hook)
      5. Compute accuracy (binary classification)

    VRAM safety: catches RuntimeError/OOM; automatically retries each offending
    batch sample-by-sample (batch-size-1 fallback) and concatenates results.

    Returns:
        Dict with mean values over all processed batches.
    """
    model.eval()

    agg_stability: List[Dict[str, float]] = []
    agg_entropy:   List[float] = []
    agg_lat_var:   List[float] = []
    n_correct = 0
    n_total   = 0

    captured, hook_handle = _register_last_block_hook(model)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        logits, diagnostics, hidden = _forward_with_fallback(model, x, captured, device, label)

        # ── Stability metrics ────────────────────────────────────────────────
        agg_stability.append(extract_stability_metrics(diagnostics))

        # ── Predictive entropy ───────────────────────────────────────────────
        agg_entropy.append(compute_predictive_entropy(logits))

        # ── Latent variance ──────────────────────────────────────────────────
        if hidden is not None:
            agg_lat_var.append(compute_latent_variance(hidden))

        # ── Accuracy ─────────────────────────────────────────────────────────
        binary_mean = logits[:, :, :2].float().mean(dim=1)   # (B, 2)
        preds = binary_mean.argmax(dim=-1)
        n_correct += (preds == y).sum().item()
        n_total   += len(y)

        if (batch_idx + 1) % 10 == 0:
            log.info(f"  [{label}] {batch_idx + 1} batches | {n_total} samples")

    hook_handle.remove()

    if not agg_stability:
        log.warning(f"  [{label}] No batches processed — empty loader?")
        return {}

    # ── Aggregate ────────────────────────────────────────────────────────────
    metric_keys = list(agg_stability[0].keys())
    mean_stability = {k: float(np.mean([s[k] for s in agg_stability])) for k in metric_keys}

    return {
        **mean_stability,
        "predictive_entropy": float(np.mean(agg_entropy)),
        "latent_variance":    float(np.mean(agg_lat_var)) if agg_lat_var else math.nan,
        "accuracy":           n_correct / max(n_total, 1),
        "n_samples":          n_total,
    }


def _forward_with_fallback(
    model: MambaLM,
    x: torch.Tensor,
    captured: Dict,
    device: torch.device,
    label: str,
) -> Tuple[torch.Tensor, Dict, Optional[torch.Tensor]]:
    """
    Run forward pass; on OOM retries sample-by-sample (batch-size-1 fallback).

    Returns: (logits, diagnostics, hidden_states)
      hidden_states is None if the hook didn't fire (shouldn't happen normally).
    """
    try:
        logits, diagnostics = model(x, return_diagnostics=True)
        hidden = captured.get("hidden")
        return logits, diagnostics, hidden

    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise

        log.warning(
            f"  [OOM] {label} – batch size {x.shape[0]} caused VRAM spike. "
            "Switching to batch-size-1 fallback for this batch."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logits_parts:   List[torch.Tensor] = []
        hidden_parts:   List[torch.Tensor] = []
        last_diag:      Dict = {}

        for i in range(x.shape[0]):
            xi = x[i : i + 1]
            li, di = model(xi, return_diagnostics=True)
            logits_parts.append(li)
            last_diag = di
            hi = captured.get("hidden")
            if hi is not None:
                hidden_parts.append(hi)

        logits_cat = torch.cat(logits_parts, dim=0)
        hidden_cat = torch.cat(hidden_parts, dim=0) if hidden_parts else None
        return logits_cat, last_diag, hidden_cat


# ===========================================================================
# STABILITY GAP REPORT
# ===========================================================================

_GAP_METRICS = [
    ("stability/tustin_denominator_condition", "Tustin κ (primary)"),
    ("stability/spectral_radius",              "Spectral radius ρ"),
    ("stability/zoh_exponent_condition",       "ZOH exponent cond."),
    ("stability/effective_receptive_field",    "Effective receptive field"),
    ("predictive_entropy",                     "Predictive entropy H"),
    ("latent_variance",                        "Latent variance"),
    ("accuracy",                               "Accuracy"),
]


def report_stability_gap(
    id_res: Dict,
    ood_res: Dict,
    scenario: str,
    id_cond_ref: float = ID_CONDITION_REF,
) -> None:
    log.info(f"\n┌─ Stability Gap: ID vs {scenario} {'─' * max(0, 40 - len(scenario))}┐")
    for key, display in _GAP_METRICS:
        id_v  = id_res.get(key, math.nan)
        ood_v = ood_res.get(key, math.nan)
        gap   = ood_v - id_v
        pct   = (gap / (abs(id_v) + 1e-10)) * 100
        marker = " ↑" if gap > 0 else (" ↓" if gap < 0 else "  ")
        log.info(
            f"│  {display:<35s}: ID={id_v:>9.4f}  OOD={ood_v:>9.4f}  "
            f"Δ={gap:>+9.4f} ({pct:>+6.1f}%){marker}"
        )

    ood_kappa = ood_res.get("stability/tustin_denominator_condition", math.nan)
    gap_vs_known = ood_kappa - id_cond_ref
    log.info(f"│")
    log.info(f"│  [Research] Known ID κ ≈ {id_cond_ref:.2f}   OOD κ = {ood_kappa:.4f}   "
             f"Gap = {gap_vs_known:+.4f} ({(gap_vs_known / id_cond_ref) * 100:+.1f}%)")
    if not math.isnan(ood_kappa):
        if ood_kappa > id_cond_ref * 1.1:
            log.info(
                "│  ★ FINDING: Numerical stability DEGRADES under OOD inputs "
                "(κ rose before accuracy dropped)."
            )
        elif ood_kappa < id_cond_ref * 0.9:
            log.info("│  ★ FINDING: Numerical stability is BETTER under OOD (unusual).")
        else:
            log.info("│  ★ FINDING: Numerical stability is comparable to ID baseline.")
    log.info(f"└{'─' * 80}┘")


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================

_TABLE_METRICS: List[Tuple[str, str]] = [
    ("accuracy",                               "Accuracy"),
    ("predictive_entropy",                     "Predictive entropy H"),
    ("latent_variance",                        "Latent variance"),
    ("stability/spectral_radius",              "Spectral radius ρ"),
    ("stability/tustin_denominator_condition", "Tustin denom. cond. κ"),
    ("stability/zoh_exponent_condition",       "ZOH exponent cond."),
    ("stability/effective_receptive_field",    "ERF (steps)"),
    ("n_samples",                              "N samples"),
]


def print_summary_table(all_results: Dict[str, Dict]) -> None:
    scenarios = list(all_results.keys())
    col_w = 20
    name_w = 32

    header_line = f"{'Metric':<{name_w}}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    sep = "─" * len(header_line)

    log.info("")
    log.info("═" * len(header_line))
    log.info("OOD STABILITY ANALYSIS  –  SUMMARY TABLE")
    log.info("═" * len(header_line))
    log.info(header_line)
    log.info(sep)

    for key, display in _TABLE_METRICS:
        row = f"{display:<{name_w}}"
        for s in scenarios:
            v = all_results[s].get(key, math.nan)
            if math.isnan(v):
                row += f"{'N/A':>{col_w}}"
            elif key == "n_samples":
                row += f"{int(v):>{col_w}}"
            elif key == "stability/effective_receptive_field":
                row += f"{int(v):>{col_w}}"
            else:
                row += f"{v:>{col_w}.4f}"
        log.info(row)

    log.info(sep)
    log.info("")
    log.info("LEGEND")
    log.info("  Spectral radius ρ  < 1.0            → BIBO stable")
    log.info("  Tustin cond. κ     ≫ 1              → near-singular Tustin denominators")
    log.info("  ZOH exponent cond. ≫ 1              → broad time-scale spread")
    log.info("  Predictive entropy max = log(2)≈0.693 → maximum binary uncertainty")
    log.info("  Latent variance    ↑ under OOD      → hidden-state activation instability")
    log.info("  Stability Gap      = OOD κ − ID κ   → core research signal")
    log.info("═" * len(header_line))


# ===========================================================================
# CLI + DEVICE
# ===========================================================================

def _get_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(dev)
        vram = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        log.info(f"Device: {dev}  ({name}, {vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        log.info(f"Device: {dev}  (Apple Silicon MPS)")
    else:
        dev = torch.device("cpu")
        log.info(f"Device: {dev}  (CPU — inference may be slow for L=16384)")
    return dev


def _build_loader(dataset: Dataset, batch_size: int, max_samples: Optional[int]) -> DataLoader:
    if max_samples and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD Stability Analysis for ZOH-Discretized Mamba on Path-X",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint_path", default=None,
        help="Path to .pt checkpoint file. If omitted, auto-selects best_checkpoint.pt "
             "then the latest step checkpoint in checkpoints_research/.",
    )
    p.add_argument("--batch_size", type=int, default=4,
                   help="Eval batch size. Use 1 if VRAM is limited.")
    p.add_argument("--max_id_samples", type=int, default=200,
                   help="Cap on ID evaluation samples.")
    p.add_argument("--max_ood_samples", type=int, default=200,
                   help="Cap on OOD evaluation samples per scenario.")
    p.add_argument("--n_noise_samples", type=int, default=200,
                   help="Number of Gaussian noise samples for Noise-OOD.")
    p.add_argument("--pathx_dev", default=_PATHX_DEV,
                   help="Path to Path-X dev split pickle.")
    p.add_argument("--pathx_test", default=_PATHX_TEST,
                   help="Path to Path-X test split pickle (fallback if dev missing).")
    p.add_argument("--id_condition_ref", type=float, default=ID_CONDITION_REF,
                   help="Known ID condition number (κ) from training logs.")
    p.add_argument("--no_far_ood", action="store_true",
                   help="Skip CIFAR-10 Far-OOD (use if torchvision is unavailable).")
    p.add_argument("--device", default=None,
                   help="Force device: cuda / cpu / mps.")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    args = _parse_args()
    device = _get_device(args.device)

    # ── Build model ──────────────────────────────────────────────────────────
    log.info("\nInstantiating MambaLM …")
    model = MambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        mode=MODE,
        gradient_checkpointing=False,   # not needed for eval
    )

    # ── Resolve checkpoint path ──────────────────────────────────────────────
    ckpt_dir = os.path.join(_ROOT, "checkpoints_research")
    if args.checkpoint_path is None:
        # Priority 1: best_checkpoint.pt
        best = os.path.join(ckpt_dir, "best_checkpoint.pt")
        if os.path.isfile(best):
            args.checkpoint_path = best
            log.info(f"Auto-selected: {best}")
        else:
            # Priority 2: latest step checkpoint by modification time
            import glob as _glob
            candidates = sorted(
                _glob.glob(os.path.join(ckpt_dir, "checkpoint_step*.pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            if candidates:
                args.checkpoint_path = candidates[0]
                log.info(f"Auto-selected (latest step): {candidates[0]}")
            else:
                log.error(
                    f"No checkpoint found. Pass --checkpoint_path or place a "
                    f"checkpoint in {ckpt_dir}/"
                )
                sys.exit(1)

    if not os.path.isfile(args.checkpoint_path):
        log.error(f"Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    load_checkpoint(args.checkpoint_path, model)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel parameters: {total_params:,}")

    # ── Build datasets ───────────────────────────────────────────────────────
    log.info("\nBuilding datasets …")

    id_path = args.pathx_dev if os.path.isfile(args.pathx_dev) else args.pathx_test
    if not os.path.isfile(id_path):
        log.error(f"No Path-X data found at {args.pathx_dev} or {args.pathx_test}")
        sys.exit(1)

    id_base = PathXDataset(id_path)

    id_loader       = _build_loader(id_base,                    args.batch_size, args.max_id_samples)
    near_ood_loader = _build_loader(NearOODDataset(id_base),    args.batch_size, args.max_ood_samples)
    noise_loader    = _build_loader(
        NoiseOODDataset(n_samples=args.n_noise_samples),
        args.batch_size, None,
    )

    far_ood_loader: Optional[DataLoader] = None
    if not args.no_far_ood:
        try:
            far_ood_loader = _build_loader(
                FarOODDataset(max_samples=args.max_ood_samples),
                args.batch_size, None,
            )
        except (ImportError, Exception) as exc:
            log.warning(f"  Far-OOD (CIFAR-10) skipped: {exc}")

    # ── Run evaluation scenarios ─────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    # 1 · ID
    log.info("\n" + "━" * 64)
    log.info("SCENARIO 1/4 — ID  (Path-X dev/test set)")
    log.info("━" * 64)
    all_results["ID (Path-X)"] = evaluate_scenario(model, id_loader, device, "ID-PathX")

    # 2 · Near-OOD
    log.info("\n" + "━" * 64)
    log.info("SCENARIO 2/4 — Near-OOD  (Gaussian blur + colour inversion)")
    log.info("━" * 64)
    all_results["Near-OOD (Transform)"] = evaluate_scenario(
        model, near_ood_loader, device, "Near-OOD"
    )

    # 3 · Far-OOD
    if far_ood_loader is not None:
        log.info("\n" + "━" * 64)
        log.info("SCENARIO 3/4 — Far-OOD  (CIFAR-10 → 16384 tokens)")
        log.info("━" * 64)
        all_results["Far-OOD (CIFAR-10)"] = evaluate_scenario(
            model, far_ood_loader, device, "Far-OOD"
        )
    else:
        log.info("\nSCENARIO 3/4 — Far-OOD: SKIPPED")

    # 4 · Noise-OOD
    log.info("\n" + "━" * 64)
    log.info("SCENARIO 4/4 — Noise-OOD  (Pure Gaussian noise)")
    log.info("━" * 64)
    all_results["Noise-OOD (Gaussian)"] = evaluate_scenario(
        model, noise_loader, device, "Noise-OOD"
    )

    # ── Stability Gap reports ────────────────────────────────────────────────
    log.info("\n\n" + "=" * 80)
    log.info("STABILITY GAP ANALYSIS  (relative to ID baseline)")
    log.info("=" * 80)

    id_res = all_results["ID (Path-X)"]
    for name, res in all_results.items():
        if name != "ID (Path-X)":
            report_stability_gap(id_res, res, name, args.id_condition_ref)

    # ── Summary table ────────────────────────────────────────────────────────
    print_summary_table(all_results)

    log.info("OOD analysis complete.\n")


if __name__ == "__main__":
    main()

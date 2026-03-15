"""
Numerical Stability Metrics
============================
Analyzes spectral properties and condition numbers of discretized SSMs.

All metrics are computed over ALL batch items and time steps (not a single
sample) to give statistically representative values suitable for a research
paper.  See docstrings for exact mathematical definitions.
"""

import math
import torch
import numpy as np
from typing import Dict


class StabilityAnalyzer:
    """
    Analyzes numerical stability of discretized State Space Models.

    Metrics
    -------
    spectral_radius
        Max absolute eigenvalue of A_bar over ALL (B, L, D, N) entries.
        For a diagonal SSM the eigenvalues ARE the diagonal elements, so
        spectral_radius = max_{b,t,d,n} |A_bar_{b,t,d,n}|.
        Stability requires spectral_radius ≤ 1.

    discretization_condition  (mode-dependent)
        ZOH  : dynamic range of ZOH exponents, max|ΔA| / min|ΔA|.
               Represents the spread of time-scales across SSM channels.
        Tustin: condition number of (I − Δ/2·A), the matrix inverted in
                Tustin discretization.  Large values indicate near-singular
                Tustin denominators.

    effective_receptive_field (ERF)
        Number of past time steps whose contribution to the current state
        is above `threshold` (default 1 %).  Computed from the channel
        with the slowest mean decay: ERF = floor(log(threshold) /
        log(max_{d,n} mean_{b,t}(|A_bar_{b,t,d,n}|))).
    """

    @staticmethod
    def compute_spectral_radius(A_bar: torch.Tensor) -> float:
        """
        Spectral radius = max |A_bar| over ALL (B, L, D, N) entries.

        For a diagonal SSM, each entry is an eigenvalue of the per-step
        state-transition operator.  The spectral radius of the full
        sequential process is the maximum over all entries and all time
        steps.

        Args:
            A_bar: Discretized state matrix, shape (B, L, D, N)

        Returns:
            Spectral radius (float).  Values ≤ 1 imply BIBO stability.
        """
        return float(A_bar.detach().abs().max().item())

    @staticmethod
    def compute_tustin_denominator_condition(
        A: torch.Tensor,
        delta: torch.Tensor,
    ) -> float:
        """
        Condition number of (I − Δ/2·A) for Tustin discretization.

        For a diagonal system the condition number reduces to
            κ = max_{d,n} |1 − Δ̄_d/2·A_{d,n}|
              / min_{d,n} |1 − Δ̄_d/2·A_{d,n}|
        where Δ̄_d is the mean step size for channel d averaged over ALL
        batch items and time steps.

        Args:
            A    : Continuous state matrix, shape (D, N)
            delta: Step sizes, shape (B, L, D)

        Returns:
            Condition number κ (float).  κ ≫ 1 indicates near-singular
            denominators and potential numerical instability.
        """
        # Mean step size per channel over ALL batch items and time steps.
        delta_mean = delta.mean(dim=(0, 1)).detach().float().cpu().numpy()  # (D,)
        A_sample   = A.detach().float().cpu().numpy()                       # (D, N)

        # |1 − Δ̄_d / 2 · A_{d,n}| for every (d, n) entry.
        denom = np.abs(1.0 - (delta_mean[:, None] / 2.0) * A_sample)      # (D, N)

        max_val = float(denom.max())
        # Avoid division by zero for near-singular entries.
        nonzero = denom[denom > 1e-10]
        min_val = float(nonzero.min()) if len(nonzero) > 0 else 1e-10

        return max_val / min_val

    @staticmethod
    def compute_zoh_exponent_condition(
        A: torch.Tensor,
        delta: torch.Tensor,
    ) -> float:
        """
        Dynamic range of ZOH exponents: max|Δ·A| / min|Δ·A|.

        For ZOH discretization (A_bar = exp(Δ·A)), the exponent Δ·A is
        always negative (A < 0, Δ > 0).  Its absolute value |Δ·A| is the
        per-channel decay rate.  This metric reports the ratio of the
        fastest-decaying channel to the slowest, quantifying the spread
        of learned time-scales.

        The mean step size is averaged over ALL batch items and time steps
        for a statistically representative estimate.

        Interpretation
        --------------
        ≈ 1   : all channels decay at the same rate (degenerate)
        ≫ 1   : broad range of time-scales (desirable for long-range tasks,
                 but very large values can cause bfloat16 precision loss)

        Args:
            A    : Continuous state matrix, shape (D, N).  All entries < 0.
            delta: Step sizes, shape (B, L, D).  All entries > 0.

        Returns:
            Dynamic-range ratio (float).
        """
        # Mean step size per channel (D,) — averaged over ALL B and L.
        delta_mean = delta.mean(dim=(0, 1)).detach().float()   # (D,)
        A_sample   = A.detach().float()                        # (D, N)

        # |Δ̄_d · A_{d,n}| — always ≥ 0 since A < 0 and Δ > 0.
        deltaA_abs = (delta_mean.unsqueeze(-1) * A_sample).abs()  # (D, N)

        max_val = float(deltaA_abs.max().item())
        min_val = float(deltaA_abs.clamp(min=1e-10).min().item())

        return max_val / min_val

    @staticmethod
    def compute_effective_receptive_field(
        A_bar: torch.Tensor,
        threshold: float = 0.01,
    ) -> int:
        """
        Effective Receptive Field (ERF).

        ERF is the number of past time steps whose contribution to the
        current hidden state is still above `threshold` (default 1 %).

        For the channel with the slowest decay (largest mean |A_bar|):
            ERF = floor(log(threshold) / log(max_channel_mean_decay))
        where max_channel_mean_decay = max_{d,n} mean_{b,t}(|A_bar_{b,t,d,n}|).

        Using the mean over (B, L) gives a stable, representative estimate
        rather than a single-sample snapshot.

        Args:
            A_bar    : Discretized state matrix, shape (B, L, D, N)
            threshold: Decay threshold (default 0.01 = 1 %)

        Returns:
            ERF in number of time steps (int).
        """
        # Mean over (B, L) → per-channel summary (D, N).
        A_mean    = A_bar.detach().abs().mean(dim=(0, 1)).cpu()  # (D, N)
        max_abs_A = float(A_mean.max().item())

        if max_abs_A >= 1.0:
            return 10000   # Unstable / non-decaying — effectively infinite ERF.
        elif max_abs_A < 1e-10:
            return 1       # Immediate decay — ERF of one step.
        else:
            erf = int(np.log(threshold) / np.log(max_abs_A))
            return max(1, min(erf, 100000))

    @staticmethod
    def analyze_all(
        A_bar: torch.Tensor,
        A: torch.Tensor,
        delta: torch.Tensor,
        mode: str = "tustin",
    ) -> Dict[str, float]:
        """
        Compute all stability metrics at once.

        The `discretization_condition` key is mode-dependent:
          - "tustin" / "vanilla" : condition number of (I − Δ/2·A)
          - "zoh"               : dynamic range max|Δ·A| / min|Δ·A|

        All metrics are computed over ALL batch items and time steps.

        Args:
            A_bar: Discretized A, shape (B, L, D, N)
            A    : Continuous A, shape (D, N)
            delta: Step sizes, shape (B, L, D)
            mode : Discretization mode ("tustin", "zoh", or "vanilla")

        Returns:
            Dictionary with keys:
              spectral_radius, discretization_condition,
              effective_receptive_field
        """
        if mode == "zoh":
            disc_cond = StabilityAnalyzer.compute_zoh_exponent_condition(A, delta)
        else:
            disc_cond = StabilityAnalyzer.compute_tustin_denominator_condition(A, delta)

        return {
            'spectral_radius':           StabilityAnalyzer.compute_spectral_radius(A_bar),
            'discretization_condition':  disc_cond,
            'effective_receptive_field': StabilityAnalyzer.compute_effective_receptive_field(A_bar),
        }

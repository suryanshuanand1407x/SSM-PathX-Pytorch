"""
Numerical Stability Metrics
============================
Analyzes spectral properties and condition numbers of discretized SSMs.
"""

import torch
import numpy as np
from typing import Dict


class StabilityAnalyzer:
    """
    Analyzes numerical stability of discretized State Space Models.

    Metrics:
    - Spectral radius (max eigenvalue magnitude)
    - Condition number of (I - Δ/2 * A) for Tustin discretization
    - Effective Receptive Field (ERF)
    """

    @staticmethod
    def compute_spectral_radius(A_bar: torch.Tensor) -> float:
        """
        Compute spectral radius (max absolute eigenvalue) of discretized A.

        For stability, we want spectral_radius ≤ 1.0 (unit circle).
        Values > 1 indicate potential instability.

        Args:
            A_bar: Discretized state matrix, shape (B, L, D, N)

        Returns:
            Spectral radius (float)
        """
        # Sample from batch to avoid computing for all elements
        A_sample = A_bar[0, 0].detach().cpu()  # (D, N)

        # For diagonal/elementwise A (Mamba case), spectral radius = max(|A_ii|)
        # For general matrices, would need eigenvalue decomposition
        if A_sample.ndim == 2 and A_sample.shape[0] > 1:
            # Compute eigenvalues for a representative row
            try:
                # Take first channel's state matrix
                eigenvalues = torch.linalg.eigvals(A_sample.float())
                spectral_radius = torch.max(torch.abs(eigenvalues)).item()
            except:
                # Fallback: just take max absolute value
                spectral_radius = torch.max(torch.abs(A_sample)).item()
        else:
            # Element-wise case
            spectral_radius = torch.max(torch.abs(A_sample)).item()

        return float(spectral_radius)

    @staticmethod
    def compute_tustin_denominator_condition(
        A: torch.Tensor,
        delta: torch.Tensor,
    ) -> float:
        """
        Compute condition number of (I - Δ/2 * A).

        This is the denominator in Tustin discretization:
            A_bar = (I + Δ/2*A) / (I - Δ/2*A)

        High condition numbers indicate numerical instability in the inversion.

        Args:
            A: Continuous state matrix, shape (D, N)
            delta: Step sizes, shape (B, L, D)

        Returns:
            Condition number (float)
        """
        # Sample delta from batch
        delta_sample = delta[0, 0, 0].item()  # Scalar

        # Compute I - Δ/2 * A for a sample
        A_sample = A.detach().cpu().numpy()  # (D, N)

        # For diagonal A, this is element-wise
        denom = 1.0 - (delta_sample / 2.0) * A_sample

        # Condition number = max/min of absolute values
        abs_denom = np.abs(denom)
        if np.min(abs_denom) > 1e-10:
            cond = np.max(abs_denom) / np.min(abs_denom)
        else:
            # Near-singular
            cond = 1e10

        return float(cond)

    @staticmethod
    def compute_effective_receptive_field(
        A_bar: torch.Tensor,
        threshold: float = 0.01,
    ) -> int:
        """
        Compute Effective Receptive Field (ERF).

        ERF measures how far back in the sequence the model can "see" effectively.
        We compute this by analyzing the decay of A_bar^t as t increases.

        The ERF is the number of steps until the state contribution falls below
        the threshold (default 1%).

        Args:
            A_bar: Discretized state matrix, shape (B, L, D, N)
            threshold: Decay threshold (default 0.01 = 1%)

        Returns:
            Effective receptive field length (int)
        """
        # Sample from batch
        A_sample = A_bar[0, 0].detach().cpu()  # (D, N)

        # For diagonal A, ERF can be computed from max decay rate
        # A_bar^t = exp(t * log(A_bar))
        # We want: |A_bar^t| < threshold
        # So: t > log(threshold) / log(|A_bar|)

        # Get maximum absolute value (slowest decay)
        max_abs_A = torch.max(torch.abs(A_sample)).item()

        if max_abs_A >= 1.0:
            # Unstable or non-decaying
            erf = 10000  # Effectively infinite
        elif max_abs_A < 1e-10:
            # Immediate decay
            erf = 1
        else:
            # Compute ERF
            erf = int(np.log(threshold) / np.log(max_abs_A))
            erf = max(1, min(erf, 100000))  # Clamp to reasonable range

        return erf

    @staticmethod
    def analyze_all(
        A_bar: torch.Tensor,
        A: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all stability metrics at once.

        Args:
            A_bar: Discretized A, shape (B, L, D, N)
            A: Continuous A, shape (D, N)
            delta: Step sizes, shape (B, L, D)

        Returns:
            Dictionary of all stability metrics
        """
        return {
            'spectral_radius': StabilityAnalyzer.compute_spectral_radius(A_bar),
            'tustin_denominator_condition': StabilityAnalyzer.compute_tustin_denominator_condition(A, delta),
            'effective_receptive_field': StabilityAnalyzer.compute_effective_receptive_field(A_bar),
        }

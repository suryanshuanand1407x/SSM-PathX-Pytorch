"""
Discretization Error Analysis
==============================
Compares the discrete ZOH SSM output against a ground-truth continuous ODE
solved with 4th-order Runge-Kutta (RK4).

Two solver variants are provided:

  solve_continuous_ssm           — time-invariant B, C (original, kept for
                                   reference / non-selective modes)
  solve_continuous_ssm_selective — time-varying B_t, C_t (correct for
                                   selective SSMs such as Mamba/ZOH)

For selective SSMs the inputs B_t and C_t are treated as piecewise-constant
over each discretization interval [t, t+Δ_t), which is exactly the ZOH
assumption.  The RK4 integration under this assumption gives the "true"
continuous solution against which the ZOH discrete formula is compared.

Single-channel analysis (d = 0 of the inner SSM dimension) is used for
tractability; the result is representative of the full model because all
channels share the same A matrix (just different scalings via Δ).
"""

import torch
import numpy as np
from typing import Dict, Tuple


class DiscretizationErrorAnalyzer:
    """
    Analyzes discretization error by comparing against ground-truth RK4.
    """

    # ------------------------------------------------------------------
    # RK4 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rk4_step(
        h: torch.Tensor,   # (N,) current state
        A: torch.Tensor,   # (N,) diagonal continuous A
        B: torch.Tensor,   # (N,) continuous input matrix (one channel)
        x: torch.Tensor,   # scalar input at current time
        dt: float,
    ) -> torch.Tensor:
        """
        Single RK4 step for continuous diagonal SSM: dh/dt = A☉h + B·x.

        The system is scalar-per-state-dimension (diagonal A), so all
        operations are element-wise on vectors of length N.

        Args:
            h : Current state, shape (N,)
            A : Diagonal A, shape (N,)
            B : Input vector for this time step, shape (N,)
            x : Scalar input value
            dt: Step size

        Returns:
            h_next: Next state, shape (N,)
        """
        k1 = A * h           + B * x
        k2 = A * (h + dt/2 * k1) + B * x
        k3 = A * (h + dt/2 * k2) + B * x
        k4 = A * (h + dt    * k3) + B * x
        return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ------------------------------------------------------------------
    # Time-invariant solver (reference / non-selective modes)
    # ------------------------------------------------------------------

    @staticmethod
    def solve_continuous_ssm(
        A: torch.Tensor,       # (D, N)
        B: torch.Tensor,       # (D, N)  — time-invariant
        C: torch.Tensor,       # (D, N)  — time-invariant
        x_seq: torch.Tensor,   # (L, D)
        delta: torch.Tensor,   # (L,) scalar per step
        num_substeps: int = 10,
    ) -> torch.Tensor:
        """
        Solve continuous SSM using RK4 with time-invariant B, C.

        System: dh/dt = A☉h + B·x(t),   y = C☉h (summed over N)

        Args:
            A          : Diagonal continuous A, shape (D, N)
            B          : Constant input matrix, shape (D, N)
            C          : Constant output matrix, shape (D, N)
            x_seq      : Input sequence, shape (L, D)
            delta      : Step sizes, shape (L,)
            num_substeps: RK4 sub-steps per discrete interval

        Returns:
            y_seq: Output sequence, shape (L, D)
        """
        L, D = x_seq.shape
        N    = A.shape[1]

        A     = A.to(torch.float32)
        B     = B.to(torch.float32)
        C     = C.to(torch.float32)
        x_seq = x_seq.to(torch.float32)

        h       = torch.zeros((D, N), dtype=torch.float32)
        outputs = []

        for t in range(L):
            dt     = delta[t].item() if delta.ndim > 0 else delta.item()
            dt_sub = dt / num_substeps
            x_t    = x_seq[t]       # (D,)

            for _ in range(num_substeps):
                # Process each channel independently (diagonal A).
                x_exp  = x_t.unsqueeze(-1)   # (D, 1)
                k1 = A * h              + B * x_exp
                k2 = A * (h + dt_sub/2 * k1) + B * x_exp
                k3 = A * (h + dt_sub/2 * k2) + B * x_exp
                k4 = A * (h + dt_sub   * k3) + B * x_exp
                h  = h + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            y_t = torch.sum(C * h, dim=-1)   # (D,)
            outputs.append(y_t)

        return torch.stack(outputs, dim=0)   # (L, D)

    # ------------------------------------------------------------------
    # Time-varying (selective) solver — correct for Mamba / ZOH
    # ------------------------------------------------------------------

    @staticmethod
    def solve_continuous_ssm_selective(
        A: torch.Tensor,       # (N,)   diagonal continuous A, single channel
        B_seq: torch.Tensor,   # (L, N) time-varying selective B
        C_seq: torch.Tensor,   # (L, N) time-varying selective C
        x_seq: torch.Tensor,   # (L,)   scalar input per step, single channel
        delta: torch.Tensor,   # (L,)   step sizes
        num_substeps: int = 10,
    ) -> torch.Tensor:
        """
        Solve the continuous SSM for one SSM channel with time-varying B, C.

        B_t and C_t are treated as piecewise-constant over [t, t+Δ_t),
        which is exactly the ZOH assumption.  RK4 integration under this
        assumption gives the "true" continuous solution that ZOH should
        approximate.

        System (single channel, per state dimension n):
            dh_n/dt = A_n · h_n + B_t_n · x_t
            y_t     = Σ_n C_t_n · h_n(t)

        Args:
            A        : Diagonal A for one channel, shape (N,)
            B_seq    : Time-varying B for one channel, shape (L, N)
            C_seq    : Time-varying C for one channel, shape (L, N)
            x_seq    : Scalar inputs for one channel, shape (L,)
            delta    : Step sizes, shape (L,)
            num_substeps: RK4 sub-steps per discrete interval

        Returns:
            y_cont: Continuous output, shape (L,)
        """
        L = len(x_seq)
        N = len(A)

        A     = A.float()
        B_seq = B_seq.float()
        C_seq = C_seq.float()
        x_seq = x_seq.float()
        delta = delta.float()

        h      = torch.zeros(N, dtype=torch.float32)
        y_cont = []

        for t in range(L):
            dt     = delta[t].item()
            x_t    = x_seq[t]          # scalar
            B_t    = B_seq[t]          # (N,) — piecewise constant over interval
            dt_sub = dt / num_substeps

            for _ in range(num_substeps):
                h = DiscretizationErrorAnalyzer.rk4_step(h, A, B_t, x_t, dt_sub)

            y_cont.append((C_seq[t] * h).sum().item())

        return torch.tensor(y_cont, dtype=torch.float32)   # (L,)

    # ------------------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_l2_error(
        y_discrete: torch.Tensor,
        y_continuous: torch.Tensor,
    ) -> float:
        """
        Relative L2 error: ‖y_disc − y_cont‖₂ / ‖y_cont‖₂.

        Args:
            y_discrete  : Output from discrete SSM, shape (L,) or (L, D)
            y_continuous: RK4 ground truth,          shape (L,) or (L, D)

        Returns:
            Relative L2 error (float).  0 = perfect match.
        """
        y_discrete   = y_discrete.float()
        y_continuous = y_continuous.float()

        error_norm = torch.norm(y_discrete - y_continuous).item()
        gt_norm    = torch.norm(y_continuous).item()

        return float(error_norm / gt_norm) if gt_norm > 1e-10 else 0.0

    @staticmethod
    def compute_max_error(
        y_discrete: torch.Tensor,
        y_continuous: torch.Tensor,
    ) -> float:
        """
        Maximum absolute pointwise error: max_t |y_disc_t − y_cont_t|.

        Args:
            y_discrete  : Output from discrete SSM
            y_continuous: RK4 ground truth

        Returns:
            Max absolute error (float).
        """
        return float(
            torch.max(torch.abs(y_discrete.float() - y_continuous.float())).item()
        )

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_discretization_error(
        y_discrete: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x_seq: torch.Tensor,
        delta: torch.Tensor,
        num_substeps: int = 10,
    ) -> Dict[str, float]:
        """
        Compute discretization error against RK4 (time-invariant B, C).

        Kept for backward compatibility with non-selective modes.
        For selective SSMs use analyze_discretization_error_selective().

        Args:
            y_discrete: Discrete SSM output, shape (L,) or (L, D)
            A         : Diagonal continuous A, shape (D, N) or (N,)
            B         : Constant input matrix, shape (D, N) or (N,)
            C         : Constant output matrix, shape (D, N) or (N,)
            x_seq     : Input sequence, shape (L, D) or (L,)
            delta     : Step sizes, shape (L,)
            num_substeps: RK4 sub-steps per interval

        Returns:
            {'l2_error': float, 'max_error': float}
        """
        # Ensure 2-D tensors for the multi-channel solver.
        if A.ndim == 1:
            A     = A.unsqueeze(0)
            B     = B.unsqueeze(0)
            C     = C.unsqueeze(0)
        if x_seq.ndim == 1:
            x_seq = x_seq.unsqueeze(-1)

        y_continuous = DiscretizationErrorAnalyzer.solve_continuous_ssm(
            A, B, C, x_seq, delta, num_substeps
        )
        return {
            'l2_error':  DiscretizationErrorAnalyzer.compute_l2_error(y_discrete, y_continuous),
            'max_error': DiscretizationErrorAnalyzer.compute_max_error(y_discrete, y_continuous),
        }

    @staticmethod
    def analyze_discretization_error_selective(
        y_discrete: torch.Tensor,
        A: torch.Tensor,
        B_seq: torch.Tensor,
        C_seq: torch.Tensor,
        x_seq: torch.Tensor,
        delta: torch.Tensor,
        num_substeps: int = 10,
    ) -> Dict[str, float]:
        """
        Compute discretization error for a SELECTIVE SSM (time-varying B, C).

        This is the correct comparison for Mamba-style models where B and C
        are input-dependent (selective).  B_t / C_t are treated as
        piecewise-constant over [t, t+Δ_t) — the ZOH assumption — so the
        RK4 result is exactly the continuous solution the ZOH formula
        approximates.

        Args:
            y_discrete: Discrete ZOH output for one channel, shape (L,)
            A         : Diagonal A for one channel, shape (N,)
            B_seq     : Time-varying B for one channel, shape (L, N)
            C_seq     : Time-varying C for one channel, shape (L, N)
            x_seq     : Scalar input for one channel, shape (L,)
            delta     : Step sizes, shape (L,)
            num_substeps: RK4 sub-steps per interval

        Returns:
            {'l2_error': float, 'max_error': float}
        """
        y_continuous = DiscretizationErrorAnalyzer.solve_continuous_ssm_selective(
            A, B_seq, C_seq, x_seq, delta, num_substeps
        )
        return {
            'l2_error':  DiscretizationErrorAnalyzer.compute_l2_error(y_discrete, y_continuous),
            'max_error': DiscretizationErrorAnalyzer.compute_max_error(y_discrete, y_continuous),
        }

"""
Discretization Error Analysis
==============================
Compares discretized SSM against ground truth ODE solver (RK4).
"""

import torch
import numpy as np
from typing import Tuple


class DiscretizationErrorAnalyzer:
    """
    Analyzes discretization error by comparing against ground truth.

    Uses 4th-order Runge-Kutta (RK4) as ground truth ODE solver.
    """

    @staticmethod
    def rk4_step(
        h: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        x: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Single RK4 step for continuous SSM: dh/dt = A*h + B*x

        Args:
            h: Current state, shape (D, N)
            A: Continuous state matrix, shape (D, N)
            B: Continuous input matrix, shape (D, N)
            x: Input at current time, shape (D,)
            dt: Step size

        Returns:
            h_next: Next state, shape (D, N)
        """
        x_expanded = x.unsqueeze(-1)  # (D, 1)

        # k1 = f(h, x) = A*h + B*x
        k1 = A * h + B * x_expanded

        # k2 = f(h + dt/2*k1, x)
        k2 = A * (h + dt/2 * k1) + B * x_expanded

        # k3 = f(h + dt/2*k2, x)
        k3 = A * (h + dt/2 * k2) + B * x_expanded

        # k4 = f(h + dt*k3, x)
        k4 = A * (h + dt * k3) + B * x_expanded

        # h_next = h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        h_next = h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return h_next

    @staticmethod
    def solve_continuous_ssm(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x_seq: torch.Tensor,
        delta: torch.Tensor,
        num_substeps: int = 10,
    ) -> torch.Tensor:
        """
        Solve continuous SSM using RK4 integration.

        System:
            dh/dt = A*h + B*x(t)
            y = C*h

        Args:
            A: Continuous state matrix, shape (D, N)
            B: Continuous input matrix, shape (D, N)
            C: Output matrix, shape (D, N)
            x_seq: Input sequence, shape (L, D)
            delta: Step size, scalar or shape (L,)
            num_substeps: Number of RK4 substeps per discrete step

        Returns:
            y_seq: Output sequence, shape (L, D)
        """
        L, D = x_seq.shape
        N = A.shape[1]

        # Initialize state
        h = torch.zeros((D, N), device=x_seq.device, dtype=torch.float32)

        # Convert to float32 for stability
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
        x_seq = x_seq.to(torch.float32)

        outputs = []

        for t in range(L):
            # Get delta for this step
            dt = delta[t].item() if delta.ndim > 0 else delta.item()
            dt_sub = dt / num_substeps

            # Current input
            x_t = x_seq[t]

            # RK4 integration over [t, t+dt]
            for _ in range(num_substeps):
                h = DiscretizationErrorAnalyzer.rk4_step(h, A, B, x_t, dt_sub)

            # Output: y = C * h (sum over N dimension)
            y_t = torch.sum(C * h, dim=-1)  # (D,)
            outputs.append(y_t)

        y_seq = torch.stack(outputs, dim=0)  # (L, D)
        return y_seq

    @staticmethod
    def compute_l2_error(
        y_discrete: torch.Tensor,
        y_continuous: torch.Tensor,
    ) -> float:
        """
        Compute relative L2 error: ||y_discrete - y_continuous|| / ||y_continuous||

        Args:
            y_discrete: Output from discrete SSM, shape (L, D)
            y_continuous: Output from RK4 ground truth, shape (L, D)

        Returns:
            Relative L2 error (float)
        """
        error_norm = torch.norm(y_discrete - y_continuous).item()
        gt_norm = torch.norm(y_continuous).item()

        if gt_norm < 1e-10:
            return 0.0

        relative_error = error_norm / gt_norm
        return float(relative_error)

    @staticmethod
    def compute_max_error(
        y_discrete: torch.Tensor,
        y_continuous: torch.Tensor,
    ) -> float:
        """
        Compute maximum pointwise error.

        Args:
            y_discrete: Output from discrete SSM
            y_continuous: Output from RK4 ground truth

        Returns:
            Max absolute error (float)
        """
        max_error = torch.max(torch.abs(y_discrete - y_continuous)).item()
        return float(max_error)

    @staticmethod
    def analyze_discretization_error(
        y_discrete: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x_seq: torch.Tensor,
        delta: torch.Tensor,
        num_substeps: int = 10,
    ) -> dict:
        """
        Compute all discretization error metrics.

        Args:
            y_discrete: Output from discretized SSM, shape (L, D)
            A: Continuous A matrix
            B: Continuous B matrix
            C: Continuous C matrix
            x_seq: Input sequence
            delta: Step sizes
            num_substeps: RK4 substeps

        Returns:
            Dictionary of error metrics
        """
        # Solve ground truth using RK4
        y_continuous = DiscretizationErrorAnalyzer.solve_continuous_ssm(
            A, B, C, x_seq, delta, num_substeps
        )

        # Compute errors
        l2_error = DiscretizationErrorAnalyzer.compute_l2_error(
            y_discrete, y_continuous
        )
        max_error = DiscretizationErrorAnalyzer.compute_max_error(
            y_discrete, y_continuous
        )

        return {
            'l2_error': l2_error,
            'max_error': max_error,
        }

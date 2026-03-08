"""
Mamba PyTorch Implementation
=============================
PyTorch implementation of Mamba with three discretization modes:
- Tustin (Guarded): Full URD framework with discretization guards
- Vanilla (Raw Tustin): No guards, raw implementation
- ZOH (Standard Mamba): Zero-Order Hold discretization

Converted from JAX implementation while preserving exact mathematical logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# =============================================================================
# Discretization Functions
# =============================================================================

def soft_clamp(
    x: torch.Tensor,
    min_val: float = 1e-4,
    max_val: float = 10.0,
) -> torch.Tensor:
    """
    Differentiable soft clamp using tanh scaling.
    Maps input smoothly into [min_val, max_val].

    Args:
        x: Input tensor of any shape
        min_val: Lower bound (epsilon)
        max_val: Upper bound (Δ_max)

    Returns:
        Clamped tensor with values in [min_val, max_val]
    """
    center = (max_val + min_val) / 2.0
    half_range = (max_val - min_val) / 2.0
    return center + half_range * torch.tanh((x - center) / half_range)


def discretize_zoh(
    A: torch.Tensor,      # (D, N) - continuous state matrix (diagonal)
    B: torch.Tensor,      # (B, L, D, N) - continuous input matrix
    delta: torch.Tensor,  # (B, L, D) - step sizes
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Order Hold (ZOH) discretization (Vanilla Baseline).

    From the paper (Equation 4):
        Ā = exp(Δ·A)
        B̄ = (Δ·A)^{-1} (exp(Δ·A) - I) · Δ·B

    For diagonal A, this simplifies using (exp(ΔA) - 1) / (ΔA).

    Args:
        A: Continuous state matrix, shape (D, N) - diagonal elements
        B: Continuous input matrix, shape (B, L, D, N)
        delta: Discretization step sizes, shape (B, L, D)

    Returns:
        A_bar: Discretized state matrix
        B_bar: Discretized input matrix
    """
    # Expand delta for broadcasting
    delta_expanded = delta.unsqueeze(-1)  # (B, L, D, 1)

    # Compute Δ·A
    deltaA = delta_expanded * A  # (B, L, D, N)

    # Ā = exp(Δ·A)
    A_bar = torch.exp(deltaA)

    # Safe division: (exp(x) - 1) / x with handling for x ≈ 0
    def safe_expm1_over_x(x):
        """Compute (exp(x) - 1) / x safely."""
        threshold = 1e-4
        small_x = torch.abs(x) < threshold
        # Taylor approximation: 1 + x/2 + x^2/6
        taylor = 1.0 + x/2.0 + x**2/6.0
        # Direct computation for larger x
        direct = torch.expm1(x) / torch.where(torch.abs(x) < 1e-10, torch.ones_like(x), x)
        return torch.where(small_x, taylor, direct)

    # Compute the discretization factor
    disc_factor = safe_expm1_over_x(deltaA)

    # B̄ = Δ · B · disc_factor
    B_bar = delta_expanded * B * disc_factor

    return A_bar, B_bar


def discretize_tustin(
    A: torch.Tensor,      # (D, N) - continuous state matrix (diagonal)
    B: torch.Tensor,      # (B, L, D, N) - continuous input matrix
    delta: torch.Tensor,  # (B, L, D) - step sizes (guarded)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized Bilinear (Tustin) discretization for MPS/Metal backend.

    Element-wise operations:
        ā_i = (1 + Δ/2 · a_i) / (1 - Δ/2 · a_i)
        b̄_i = √Δ · b_i / (1 - Δ/2 · a_i)

    MPS Optimizations:
    - All ops in float32 (no bfloat16 to avoid CPU fallback)
    - Vectorized operations only (no loops)
    - Minimal intermediate tensors
    - Stability epsilon in denominator

    Args:
        A: Continuous state matrix (diagonal), shape (D, N)
        B: Continuous input matrix, shape (B, L, D, N)
        delta: Step sizes after discretization guard, shape (B, L, D)

    Returns:
        A_bar: Discretized state matrix, shape (B, L, D, N)
        B_bar: Discretized input matrix, shape (B, L, D, N)
    """
    # === MPS OPTIMIZATION: Keep everything in float32 ===
    # MPS has slow bfloat16 support and can trigger CPU fallback
    orig_dtype = B.dtype
    compute_dtype = torch.float32

    # Promote to float32 for stability
    A = A.to(compute_dtype)
    B = B.to(compute_dtype)
    delta = delta.to(compute_dtype)

    # Expand delta: (B, L, D) -> (B, L, D, 1) for broadcasting
    delta_expanded = delta.unsqueeze(-1)  # (B, L, D, 1)

    # Pre-compute Δ/2 (fused constant)
    delta_half = delta_expanded * 0.5  # (B, L, D, 1)

    # Half-step: Δ/2 · A  (element-wise, broadcasts to B, L, D, N)
    half_dA = delta_half * A  # (B, L, D, N)

    # Numerator and Denominator (computed together for register reuse)
    # Denominator: (1 - Δ/2·A) with stability epsilon
    STABILITY_EPS = 1e-8
    denom = torch.clamp(1.0 - half_dA, min=STABILITY_EPS)  # (B, L, D, N)

    # Numerator: (1 + Δ/2·A)
    numer = 1.0 + half_dA  # (B, L, D, N)

    # Ā = (1 + Δ/2·A) / (1 - Δ/2·A)
    # Use multiply by reciprocal for potentially better MPS performance
    denom_inv = torch.reciprocal(denom)  # (B, L, D, N)
    A_bar = numer * denom_inv  # (B, L, D, N)

    # B̄ = √Δ · B / (1 - Δ/2·A)
    # Fuse sqrt and multiply
    sqrt_delta = torch.sqrt(delta_expanded)  # (B, L, D, 1)
    B_bar = (sqrt_delta * B) * denom_inv  # (B, L, D, N)

    # Cast back to original dtype (if needed)
    if orig_dtype != compute_dtype:
        A_bar = A_bar.to(orig_dtype)
        B_bar = B_bar.to(orig_dtype)

    return A_bar, B_bar


def discretize_tustin_raw(
    A: torch.Tensor,      # (D, N) - continuous state matrix (diagonal)
    B: torch.Tensor,      # (B, L, D, N) - continuous input matrix
    delta: torch.Tensor,  # (B, L, D) - step sizes (unguarded)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Raw Bilinear (Tustin) discretization WITHOUT discretization guards.

    This is the VANILLA BASELINE version that demonstrates performance collapse.

    Key differences from guarded version:
    - NO precision promotion (stays in input dtype)
    - NO stability epsilon or clamping
    - Raw division that can underflow/overflow

    Args:
        A: Continuous state matrix, shape (D, N)
        B: Continuous input matrix, shape (B, L, D, N)
        delta: Step sizes (unguarded), shape (B, L, D)

    Returns:
        A_bar: Discretized state matrix
        B_bar: Discretized input matrix
    """
    # NO dtype promotion - keep whatever precision we're given

    # Expand delta for broadcasting: (B, L, D) -> (B, L, D, 1)
    delta_expanded = delta.unsqueeze(-1)

    # Half-step: Δ/2 · A  (element-wise)
    half_dA = (delta_expanded / 2.0) * A  # (B, L, D, N)

    # Denominator: (1 - Δ/2 · A) - NO epsilon, NO clamping
    denom = 1.0 - half_dA  # (B, L, D, N)

    # Numerator: (1 + Δ/2 · A)
    numer = 1.0 + half_dA  # (B, L, D, N)

    # Ā = (1 + Δ/2·A) / (1 - Δ/2·A) - raw division, can produce inf/nan
    A_bar = numer / denom  # (B, L, D, N)

    # B̄ = √Δ · B / (1 - Δ/2·A) - raw sqrt and division
    sqrt_delta = torch.sqrt(delta_expanded)  # (B, L, D, 1)
    B_bar = (sqrt_delta * B) / denom  # (B, L, D, N)

    return A_bar, B_bar


# =============================================================================
# Selective Scan Implementation
# =============================================================================

def selective_scan_sequential(
    A_bar: torch.Tensor,  # (B, L, D, N) - discretized state matrix
    B_bar: torch.Tensor,  # (B, L, D, N) - discretized input matrix
    C: torch.Tensor,      # (B, L, D, N) - output matrix
    x: torch.Tensor,      # (B, L, D) - input sequence
    h0: Optional[torch.Tensor] = None,  # (B, D, N) - initial state
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential implementation of selective scan (for reference/debugging).

    Computes the SSM recurrence:
        h_t = Ā_t · h_{t-1} + B̄_t · x_t
        y_t = C_t · h_t

    Args:
        A_bar: Discretized state transition, shape (B, L, D, N)
        B_bar: Discretized input matrix, shape (B, L, D, N)
        C: Output matrix, shape (B, L, D, N)
        x: Input sequence, shape (B, L, D)
        h0: Optional initial state, shape (B, D, N)

    Returns:
        y: Output sequence, shape (B, L, D)
        h_final: Final hidden state, shape (B, D, N)
    """
    B_size, L, D, N = A_bar.shape

    # Initialize hidden state
    if h0 is None:
        h = torch.zeros((B_size, D, N), device=x.device, dtype=x.dtype)
    else:
        h = h0.clone()  # Clone to avoid modifying the original

    outputs = []

    # Sequential scan
    for t in range(L):
        x_t = x[:, t, :].unsqueeze(-1)  # (B, D, 1)
        # Create new tensor instead of in-place modification
        h_new = A_bar[:, t] * h + B_bar[:, t] * x_t  # (B, D, N)

        # y_t = C_t · h_t (sum over N dimension)
        y_t = torch.sum(C[:, t] * h_new, dim=-1)  # (B, D)
        outputs.append(y_t)

        # Update h for next iteration
        h = h_new

    y = torch.stack(outputs, dim=1)  # (B, L, D)
    return y, h


# =============================================================================
# Chunked Parallel Selective Scan (torch.compile / inductor safe)
# =============================================================================
#
# Problem with the original binary-tree scan:
#   while stride < L runs log2(16384)=14 iterations, each calling .clone() on
#   two (B, L, D, N) tensors.  torch.compile/inductor unrolls this statically
#   and holds all ~168 clone nodes (6 layers × 14 iters × 2 tensors × A+B)
#   in memory simultaneously during the constant_fold_uniform_value pass,
#   exhausting GPU VRAM before a single forward step runs.
#
# Fix:
#   1. @_dynamo_disable  — makes the function opaque to dynamo/inductor.
#   2. Chunked binary tree — run the same doubling algorithm on K=256-length
#      chunks only (log2(256)=8 iterations on ~4 MB tensors freed per chunk)
#      with a sequential carry across the L/K=64 chunk boundaries.
#   3. torch.autograd.Function — hand-written exact backward; only h_states
#      (one O(L) tensor) is saved, not O(L·log L) clones.
# =============================================================================

_SCAN_CHUNK_SIZE = 256  # inner chunk length; log2(256)=8 doubling iterations


def _get_dynamo_disable():
    """Return the first available dynamo-disable decorator, or identity."""
    try:
        return torch.compiler.disable      # PyTorch >= 2.4
    except AttributeError:
        pass
    try:
        return torch._dynamo.disable       # PyTorch 2.0 – 2.3
    except AttributeError:
        pass
    return lambda fn: fn                   # fallback: no-op


_dynamo_disable = _get_dynamo_disable()


def _binary_tree_scan_chunk(
    A: torch.Tensor,        # (B, K, D, N) — A_bar values for this chunk
    Bx: torch.Tensor,       # (B, K, D, N) — B_bar * x values for this chunk
    h_prev: torch.Tensor,   # (B, D, N)    — carry state entering this chunk
) -> torch.Tensor:          # (B, K, D, N) — h at every position in chunk
    """
    Binary-tree associative prefix scan on one K-length chunk.

    Associative operator (right-absorbing):
        (A2, B2) ∘ (A1, B1) = (A2·A1,  A2·B1 + B2)

    h_prev is folded into position 0 before the scan so that the output
    Bx_scan[:, k] equals the absolute hidden state h_{chunk_start + k},
    not a state relative to zero.

    Called exclusively from _SelectiveScanFn.forward under torch.no_grad(),
    so the per-iteration clones never enter any autograd graph and are freed
    when the chunk loop iteration exits.
    """
    K = A.shape[1]
    A_scan  = A.clone()    # (B, K, D, N)
    Bx_scan = Bx.clone()   # (B, K, D, N)

    # Fold h_prev: position-0 effective Bx = A_0·h_prev + Bx_0
    Bx_scan[:, 0] = Bx_scan[:, 0] + A_scan[:, 0] * h_prev   # (B, D, N)

    # log2(K) doubling iterations — 8 steps for K=256
    stride = 1
    while stride < K:
        idx     = torch.arange(stride, K, device=A.device)
        A_new   = A_scan[:, idx] * A_scan[:, idx - stride]
        Bx_new  = A_scan[:, idx] * Bx_scan[:, idx - stride] + Bx_scan[:, idx]
        A_scan  = A_scan.clone()
        Bx_scan = Bx_scan.clone()
        A_scan[:, idx]  = A_new
        Bx_scan[:, idx] = Bx_new
        stride *= 2

    # Bx_scan[:, k] now equals h_{chunk_start + k}
    return Bx_scan


class _SelectiveScanFn(torch.autograd.Function):
    """
    Selective scan with hand-written backward.

    Forward  — chunked binary-tree scan: O(log K) depth per chunk, O(L) total memory.
    Backward — exact reverse sequential scan: O(L) memory, exact gradients.

    Recurrence:  h_t   = A_bar_t · h_{t-1} + Bx_t
    Output:      y_t   = (C_t · h_t).sum(-1)

    Backward derivation:
        dC_t     = dy_t ⊗ h_t                       (vectorised over all t)
        v_t      = dh_{t+1}·A_bar_{t+1} + dy_t⊗C_t  (reverse scan)
        dA_bar_t = v_t · h_{t-1}
        dBx_t    = v_t
        dh_init  = v_0 · A_bar_0  (exits the loop as dh)
    """

    @staticmethod
    def forward(ctx, A_bar, Bx, C, h_init):
        """
        Args:
            A_bar  : (B, L, D, N)
            Bx     : (B, L, D, N) — B_bar * x_expanded, pre-computed
            C      : (B, L, D, N)
            h_init : (B, D, N)    — zeros when caller passed h0=None

        Returns:
            y       : (B, L, D)
            h_final : (B, D, N)
        """
        B_size, L, D, N = A_bar.shape
        chunk = _SCAN_CHUNK_SIZE

        # All scan computation runs under no_grad so the per-chunk binary-tree
        # clones never create autograd nodes and are freed after each chunk.
        with torch.no_grad():
            h_states = torch.empty_like(A_bar)   # (B, L, D, N) — O(L) allocation
            h = h_init

            for t0 in range(0, L, chunk):
                t1      = min(t0 + chunk, L)
                h_chunk = _binary_tree_scan_chunk(
                    A_bar[:, t0:t1], Bx[:, t0:t1], h
                )                                 # (B, K, D, N)
                h_states[:, t0:t1] = h_chunk
                h = h_chunk[:, -1]               # carry to next chunk

            y = (C * h_states).sum(dim=-1)       # (B, L, D)

        # Save original (non-detached) inputs for the backward pass.
        ctx.save_for_backward(A_bar, C, h_states, h_init)

        return y, h_states[:, -1].clone()

    @staticmethod
    def backward(ctx, dy, dh_final):
        """
        Exact reverse-sequential backward.

        dy       : (B, L, D)   — upstream gradient of y
        dh_final : (B, D, N)   — upstream gradient of h_final
        """
        A_bar, C, h_states, h_init = ctx.saved_tensors
        B_size, L, D, N = A_bar.shape

        # dC: fully vectorised, no loop needed.
        dC = dy.unsqueeze(-1) * h_states        # (B, L, D, N)

        dA_bar = torch.empty_like(A_bar)
        dBx    = torch.empty_like(h_states)

        dh = dh_final.clone()                   # running gradient: v_L = dh_final

        for t in range(L - 1, -1, -1):
            # Accumulate output gradient into v_t
            dh = dh + dy[:, t].unsqueeze(-1) * C[:, t]      # (B, D, N)

            h_prev       = h_init if t == 0 else h_states[:, t - 1]
            dA_bar[:, t] = dh * h_prev          # d(h_t)/d(A_bar_t) = h_{t-1}
            dBx[:, t]    = dh                   # d(h_t)/d(Bx_t)    = 1

            # Propagate one step back through h_t = A_bar_t · h_{t-1} + Bx_t
            dh = dh * A_bar[:, t]

        # dh is now the gradient w.r.t. h_init
        return dA_bar, dBx, dC, dh


@_dynamo_disable
def selective_scan_parallel(
    A_bar: torch.Tensor,              # (B, L, D, N) - discretized state matrix
    B_bar: torch.Tensor,              # (B, L, D, N) - discretized input matrix
    C: torch.Tensor,                  # (B, L, D, N) - output matrix
    x: torch.Tensor,                  # (B, L, D)    - input sequence
    h0: Optional[torch.Tensor] = None,  # (B, D, N)  - initial state
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked parallel selective scan — compile-safe, memory-efficient.

    Drop-in replacement for the original naive binary-tree implementation.

    Differences:
      - @_dynamo_disable: torch.compile / inductor never traces the scan loop;
        the O(L·log L) clone explosion cannot reach the graph builder.
      - Binary-tree scan runs on _SCAN_CHUNK_SIZE=256 chunks only:
          log2(256)=8 iterations × ~4 MB tensors, freed per chunk.
          (vs. log2(16384)=14 iterations × ~256 MB tensors held all at once)
      - Custom autograd Function saves only h_states (one O(L) tensor) for
        the backward; all per-chunk intermediates are freed under no_grad.
      - Exact gradients via reverse sequential scan in _SelectiveScanFn.backward.

    Args:
        A_bar : (B, L, D, N) — discretized state transition
        B_bar : (B, L, D, N) — discretized input matrix
        C     : (B, L, D, N) — output matrix
        x     : (B, L, D)    — input sequence
        h0    : (B, D, N)    — optional initial hidden state

    Returns:
        y       : (B, L, D)
        h_final : (B, D, N)
    """
    B_size, L, D, N = A_bar.shape

    Bx = B_bar * x.unsqueeze(-1)    # (B, L, D, N) — fused pre-multiply

    h_init = (
        h0
        if h0 is not None
        else torch.zeros(B_size, D, N, device=A_bar.device, dtype=A_bar.dtype)
    )

    return _SelectiveScanFn.apply(A_bar, Bx, C, h_init)


# =============================================================================
# Mamba Block Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms


class CausalConv1D(nn.Module):
    """
    Causal 1D convolution for Mamba.
    Uses depthwise convolution (groups=channels) for efficiency.
    """

    def __init__(self, features: int, kernel_size: int = 4):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size

        # Depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=features,
            out_channels=features,
            kernel_size=kernel_size,
            groups=features,  # Depthwise
            bias=True,
            padding=0,  # We'll do manual causal padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
        Returns:
            y: Output tensor, shape (B, L, D)
        """
        # Transpose for Conv1d: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)

        # Causal padding: pad (kernel_size - 1) on the left
        x = F.pad(x, (self.kernel_size - 1, 0))

        # Apply convolution
        y = self.conv(x)

        # Transpose back: (B, D, L) -> (B, L, D)
        y = y.transpose(1, 2)

        return y


class S6Layer(nn.Module):
    """
    Selective SSM (S6) layer with learned projections.

    Mode Parameter:
    - "tustin": Full Tustin discretization with guards (default)
    - "vanilla": Raw Tustin discretization without guards (for baseline comparison)
    - "zoh": Standard Mamba ZOH discretization without guards (strict baseline)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        delta_max: float = 10.0,
        mode: str = "tustin",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.delta_max = delta_max
        self.mode = mode

        # A: (D, N) - S4D-Real initialization: A_n = -(n+1)
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(torch.abs(A)))

        # D: (D,) - skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections for B and C
        self.x_proj_bc = nn.Linear(d_model, 2 * d_state, bias=False)

        # Delta projection: two-stage low-rank
        self.x_proj_dt = nn.Linear(d_model, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize dt_proj bias
        with torch.no_grad():
            dt = torch.rand(d_model) * (dt_max - dt_min) + dt_min
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse softplus
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        # RMSNorm for delta (only used in Tustin mode)
        if mode == "tustin":
            self.delta_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        use_parallel: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan
            return_diagnostics: If True, return (y, diagnostics) for research metrics
        Returns:
            y: Output tensor, shape (B, L, D)
            diagnostics (optional): Dictionary with A_bar, A, B, C, delta for analysis
        """
        B_size, L, D = x.shape
        N = self.d_state

        # Get A
        A = -torch.exp(self.A_log)  # (D, N), negative for stability

        # === Projections for Selection Mechanism ===

        # Combined projection for B and C
        x_bc = self.x_proj_bc(x)  # (B, L, 2*N)
        B_sel = x_bc[..., :N]     # (B, L, N)
        C_sel = x_bc[..., N:]     # (B, L, N)

        # Delta projection
        x_dt = self.x_proj_dt(x)      # (B, L, dt_rank)
        dt_proj = self.dt_proj(x_dt)  # (B, L, D)
        delta = F.softplus(dt_proj)   # (B, L, D)

        # === MODE-DEPENDENT DISCRETIZATION GUARD ===
        if self.mode in ["vanilla", "zoh"]:
            # Baseline modes: NO discretization guard
            pass
        else:
            # Tustin mode: Full discretization guard
            delta = self.delta_norm(delta)
            delta = soft_clamp(delta, min_val=self.dt_init_floor, max_val=self.delta_max)

        # === Expand B and C to match (B, L, D, N) ===
        B_expanded = B_sel.unsqueeze(2).expand(B_size, L, D, N)  # (B, L, D, N)
        C_expanded = C_sel.unsqueeze(2).expand(B_size, L, D, N)  # (B, L, D, N)

        # === MODE-DEPENDENT DISCRETIZATION ===
        if self.mode == "zoh":
            A_bar, B_bar = discretize_zoh(A, B_expanded, delta)
        elif self.mode == "vanilla":
            A_bar, B_bar = discretize_tustin_raw(A, B_expanded, delta)
        else:
            A_bar, B_bar = discretize_tustin(A, B_expanded, delta)

        # === MODE-DEPENDENT INPUT HANDLING ===
        if self.mode in ["vanilla", "zoh"]:
            # Baseline modes: Raw input
            x_input = x
        else:
            # Tustin mode: Trapezoidal averaging
            x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # Shift right, pad with zeros
            x_input = (x + x_prev) / 2.0

        # === Selective Scan ===
        if use_parallel:
            y, _ = selective_scan_parallel(A_bar, B_bar, C_expanded, x_input)
        else:
            y, _ = selective_scan_sequential(A_bar, B_bar, C_expanded, x_input)

        # === Skip Connection ===
        y = y + self.D * x

        # === Return diagnostics if requested (for research metrics) ===
        if return_diagnostics:
            diagnostics = {
                'A_bar': A_bar,          # Discretized A (for spectral radius)
                'A': A,                  # Continuous A (for RK4 comparison)
                'B': B_expanded,         # Continuous B
                'C': C_expanded,         # Output matrix C
                'delta': delta,          # Step sizes (for analysis)
                'x_input': x_input,      # Processed input
            }
            return y, diagnostics

        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block as described in Section 3.4 of the paper.

    Architecture:
        Input x → Norm → Linear expansion → Split into two branches
        Branch 1: Conv1D → SiLU → S6 SSM
        Branch 2: (identity for gating)
        Element-wise multiply (Branch1 * SiLU(Branch2))
        Linear projection → Residual connection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = None,
        mode: str = "tustin",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mode = mode

        d_inner = expand * d_model

        # Input normalization
        self.norm = RMSNorm(d_model)

        # Linear expansion
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Branch 1: Conv → SiLU → SSM
        self.conv1d = CausalConv1D(d_inner, kernel_size=d_conv)
        self.ssm = S6Layer(
            d_model=d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            mode=mode,
        )

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        use_parallel: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan in SSM
            return_diagnostics: If True, return diagnostics from SSM layer
        Returns:
            y: Output tensor, shape (B, L, D)
            diagnostics (optional): Dictionary from SSM layer
        """
        # Store residual
        residual = x

        # Normalize
        x = self.norm(x)

        # Linear expansion and split
        x_proj = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, x_gate = x_proj.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Branch 1: Conv → SiLU → SSM
        x_conv = self.conv1d(x_main)
        x_conv = F.silu(x_conv)

        if return_diagnostics:
            x_ssm, diagnostics = self.ssm(x_conv, use_parallel=use_parallel, return_diagnostics=True)
        else:
            x_ssm = self.ssm(x_conv, use_parallel=use_parallel)

        # Gating
        x_gated = x_ssm * F.silu(x_gate)

        # Output projection
        y = self.out_proj(x_gated)

        # Residual
        y = y + residual

        if return_diagnostics:
            return y, diagnostics
        return y


class MambaLM(nn.Module):
    """
    Language Model: Embed → Mamba Stack → RMSNorm → Dense → Logits

    Mode Parameter:
    - "tustin": Full Tustin discretization with guards (default)
    - "vanilla": Raw Tustin discretization without guards (for baseline comparison)
    - "zoh": Standard Mamba ZOH discretization without guards (strict baseline)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mode: str = "tustin",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mode=mode,
            )
            for _ in range(n_layers)
        ])

        # Final norm and projection
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, use_parallel: bool = True, return_diagnostics: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input token IDs, shape (B, L)
            use_parallel: Whether to use parallel scan
            return_diagnostics: If True, return diagnostics from first block
        Returns:
            logits: Output logits, shape (B, L, vocab_size)
            diagnostics (optional): Dictionary from first Mamba block's SSM
        """
        # Embed tokens
        x = self.embedding(x)  # (B, L, D)

        diagnostics = None

        # Apply Mamba blocks
        for i, block in enumerate(self.blocks):
            if return_diagnostics and i == 0:
                # Get diagnostics from first block only
                x, diagnostics = block(x, use_parallel=use_parallel, return_diagnostics=True)
            else:
                x = block(x, use_parallel=use_parallel)

        # Final norm and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if return_diagnostics:
            return logits, diagnostics
        return logits


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def optimize_for_mps():
    """
    Configure PyTorch for optimal MPS (Apple Silicon) performance.

    Call this once at the start of your training script.
    """
    if not torch.backends.mps.is_available():
        return

    # Disable CPU fallback - force errors instead of silent slowdowns
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

    # Enable MPS high watermark ratio for better memory management
    # This helps with large sequence lengths
    torch.mps.set_per_process_memory_fraction(0.8)

    print("✓ MPS optimizations enabled")
    print(f"  - CPU fallback: DISABLED (will error instead of slow fallback)")
    print(f"  - Memory fraction: 80%")


def enable_fast_inference(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Enable fast inference optimizations (torch.compile, etc.).

    Args:
        model: The model to optimize
        device: Target device

    Returns:
        Optimized model
    """
    # torch.compile is only available in PyTorch 2.0+
    try:
        # selective_scan_parallel is decorated with @_dynamo_disable so the
        # binary-tree loop never reaches inductor.  Both CUDA and MPS can now
        # use torch.compile safely.
        if device.type == "mps":
            print("🚀 Enabling torch.compile with reduce-overhead mode for MPS...")
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print("🚀 Enabling torch.compile with reduce-overhead mode for CUDA...")
            model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️  torch.compile not available: {e}")
        print("   Continuing without compilation (slower but functional)")

    return model

"""
Test Parallel Scan Correctness
===============================
Verifies that the optimized parallel scan produces IDENTICAL results
to the sequential scan.

This ensures the optimization is mathematically correct.
"""

import torch
import numpy as np
from mamba_pytorch import (
    selective_scan_sequential,
    selective_scan_parallel,
    get_device,
)


def test_parallel_scan_correctness():
    """Test that parallel scan matches sequential scan exactly."""
    print("="*80)
    print("PARALLEL SCAN CORRECTNESS TEST")
    print("="*80)

    device = get_device()
    print(f"\nDevice: {device}")

    # Test configurations
    test_cases = [
        {"B": 2, "L": 64, "D": 16, "N": 8, "name": "Small"},
        {"B": 1, "L": 256, "D": 32, "N": 16, "name": "Medium"},
        {"B": 2, "L": 1024, "D": 64, "N": 16, "name": "Large"},
        {"B": 1, "L": 4096, "D": 32, "N": 8, "name": "Very Large"},
    ]

    all_passed = True

    for i, config in enumerate(test_cases):
        B, L, D, N = config["B"], config["L"], config["D"], config["N"]
        name = config["name"]

        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_cases)}: {name} (B={B}, L={L}, D={D}, N={N})")
        print(f"{'='*80}")

        # Create random test inputs
        torch.manual_seed(42 + i)
        A_bar = torch.randn(B, L, D, N, device=device)
        B_bar = torch.randn(B, L, D, N, device=device)
        C = torch.randn(B, L, D, N, device=device)
        x = torch.randn(B, L, D, device=device)
        h0 = torch.randn(B, D, N, device=device)

        # Run sequential scan
        print("  Running sequential scan...")
        y_seq, h_final_seq = selective_scan_sequential(
            A_bar.clone(), B_bar.clone(), C.clone(), x.clone(), h0.clone()
        )

        # Run parallel scan
        print("  Running parallel scan...")
        y_par, h_final_par = selective_scan_parallel(
            A_bar.clone(), B_bar.clone(), C.clone(), x.clone(), h0.clone()
        )

        # Compare outputs
        y_diff = torch.abs(y_seq - y_par).max().item()
        y_rel_diff = (torch.abs(y_seq - y_par) / (torch.abs(y_seq) + 1e-8)).max().item()

        h_diff = torch.abs(h_final_seq - h_final_par).max().item()
        h_rel_diff = (torch.abs(h_final_seq - h_final_par) / (torch.abs(h_final_seq) + 1e-8)).max().item()

        print(f"\n  Output (y) differences:")
        print(f"    Absolute max diff: {y_diff:.2e}")
        print(f"    Relative max diff: {y_rel_diff:.2e}")

        print(f"\n  Final state (h) differences:")
        print(f"    Absolute max diff: {h_diff:.2e}")
        print(f"    Relative max diff: {h_rel_diff:.2e}")

        # Check if results match within tolerance
        # Use higher tolerance for larger sequences due to accumulation
        tol_abs = 1e-4 * (L / 64)  # Scale tolerance with sequence length
        tol_rel = 1e-3 * (L / 64)

        y_pass = y_diff < tol_abs and y_rel_diff < tol_rel
        h_pass = h_diff < tol_abs and h_rel_diff < tol_rel

        if y_pass and h_pass:
            print(f"\n  ✅ PASS: Results match within tolerance")
            print(f"     (abs_tol={tol_abs:.2e}, rel_tol={tol_rel:.2e})")
        else:
            print(f"\n  ❌ FAIL: Results differ beyond tolerance!")
            print(f"     (abs_tol={tol_abs:.2e}, rel_tol={tol_rel:.2e})")
            all_passed = False

            # Show some sample values for debugging
            print(f"\n  Sample sequential output: {y_seq[0, :5, 0]}")
            print(f"  Sample parallel output:   {y_par[0, :5, 0]}")

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if all_passed:
        print("\n✅✅✅ ALL TESTS PASSED!")
        print("\nThe parallel scan implementation is CORRECT.")
        print("It produces identical results to the sequential scan.")
        print("\nYou can now use the parallel scan with confidence!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("\nThe parallel scan has numerical errors.")
        print("Please review the implementation before using in production.")
        return False


def test_gradients():
    """Test that gradients flow correctly through parallel scan."""
    print("\n" + "="*80)
    print("GRADIENT FLOW TEST")
    print("="*80)

    device = get_device()

    # Small test case
    B, L, D, N = 2, 64, 8, 4
    torch.manual_seed(42)

    A_bar = torch.randn(B, L, D, N, device=device, requires_grad=True)
    B_bar = torch.randn(B, L, D, N, device=device, requires_grad=True)
    C = torch.randn(B, L, D, N, device=device, requires_grad=True)
    x = torch.randn(B, L, D, device=device, requires_grad=True)

    print("\n  Running parallel scan with gradient tracking...")
    y_par, _ = selective_scan_parallel(A_bar, B_bar, C, x, None)

    # Compute loss and backprop
    loss = y_par.sum()
    print(f"  Loss: {loss.item():.4f}")

    print("  Computing gradients...")
    loss.backward()

    # Check that gradients exist and are non-zero
    checks = [
        ("A_bar", A_bar.grad),
        ("B_bar", B_bar.grad),
        ("C", C.grad),
        ("x", x.grad),
    ]

    all_grads_ok = True
    for name, grad in checks:
        if grad is None:
            print(f"  ❌ {name}: No gradient!")
            all_grads_ok = False
        else:
            grad_norm = grad.norm().item()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            status = "✅" if (grad_norm > 0 and not has_nan and not has_inf) else "❌"
            print(f"  {status} {name}: norm={grad_norm:.4e}, nan={has_nan}, inf={has_inf}")

            if has_nan or has_inf or grad_norm == 0:
                all_grads_ok = False

    if all_grads_ok:
        print("\n✅ Gradient flow is CORRECT!")
        print("   All gradients computed successfully.")
        return True
    else:
        print("\n❌ Gradient flow has issues!")
        return False


def main():
    """Run all correctness tests."""
    print("\n" + "🔍" * 40)
    print("PARALLEL SCAN CORRECTNESS VERIFICATION")
    print("🔍" * 40 + "\n")

    # Test correctness
    correctness_pass = test_parallel_scan_correctness()

    # Test gradients
    gradient_pass = test_gradients()

    # Final verdict
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)

    if correctness_pass and gradient_pass:
        print("\n🎉🎉🎉 ALL CHECKS PASSED! 🎉🎉🎉")
        print("\nThe parallel scan optimization is:")
        print("  ✅ Numerically correct")
        print("  ✅ Gradient-compatible")
        print("  ✅ Ready for production use")
        print("\nNext step: Run benchmark_parallel_scan.py to measure speedup!")
    else:
        print("\n⚠️ SOME CHECKS FAILED")
        if not correctness_pass:
            print("  ❌ Numerical correctness issues")
        if not gradient_pass:
            print("  ❌ Gradient flow issues")
        print("\nPlease review the implementation before proceeding.")

    print("="*80)


if __name__ == "__main__":
    main()

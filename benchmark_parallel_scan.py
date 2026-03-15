"""
Benchmark Parallel Scan Performance
====================================
Quick benchmark to verify the parallel associative scan optimization.

Compares:
- Sequential scan (original)
- Parallel scan (optimized with binary tree algorithm)

Expected speedup: 5-10x on MPS for L=16384
"""

import torch
import time
import sys
from mamba_pytorch import MambaLM, count_parameters, get_device, optimize_for_mps

# Test configuration
VOCAB_SIZE = 256
MAX_SEQ_LEN = 16384
D_MODEL = 64
N_LAYERS = 1  # Use 1 layer for isolated benchmark
D_STATE = 16
BATCH_SIZE = 2
NUM_WARMUP = 3
NUM_TRIALS = 10

def benchmark_model(model, device, use_parallel=True):
    """Benchmark a single forward pass."""
    model.eval()

    # Create dummy input
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN), device=device)

    # Warmup
    print(f"  Warming up ({NUM_WARMUP} iterations)...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(x, use_parallel=use_parallel)

    # Benchmark
    print(f"  Benchmarking ({NUM_TRIALS} trials)...")
    times = []

    with torch.no_grad():
        for trial in range(NUM_TRIALS):
            # Synchronize before timing (important for MPS)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            _ = model(x, use_parallel=use_parallel)

            # Synchronize after computation
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

            print(f"    Trial {trial+1}/{NUM_TRIALS}: {elapsed:.3f}s "
                  f"({MAX_SEQ_LEN * BATCH_SIZE / elapsed / 1000:.1f}k tok/s)")

    return times


def main():
    print("="*80)
    print("PARALLEL SCAN PERFORMANCE BENCHMARK")
    print("="*80)

    # Get device
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Sequence length: {MAX_SEQ_LEN:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model dimension: {D_MODEL}")
    print(f"State dimension: {D_STATE}")

    # Configure MPS if available
    if device.type == 'mps':
        print("\nConfiguring MPS optimizations...")
        optimize_for_mps()

    # Create model
    print("\nCreating model...")
    model = MambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        mode="tustin",  # Use Tustin mode
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Benchmark parallel scan
    print("\n" + "="*80)
    print("PARALLEL SCAN (OPTIMIZED)")
    print("="*80)
    parallel_times = benchmark_model(model, device, use_parallel=True)

    # Benchmark sequential scan (for comparison)
    print("\n" + "="*80)
    print("SEQUENTIAL SCAN (ORIGINAL - for comparison)")
    print("="*80)
    sequential_times = benchmark_model(model, device, use_parallel=False)

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    parallel_mean = sum(parallel_times) / len(parallel_times)
    parallel_std = (sum((t - parallel_mean)**2 for t in parallel_times) / len(parallel_times))**0.5
    sequential_mean = sum(sequential_times) / len(sequential_times)
    sequential_std = (sum((t - sequential_mean)**2 for t in sequential_times) / len(sequential_times))**0.5

    speedup = sequential_mean / parallel_mean
    parallel_throughput = MAX_SEQ_LEN * BATCH_SIZE / parallel_mean / 1000

    print(f"\nParallel Scan:")
    print(f"  Time: {parallel_mean:.3f} ± {parallel_std:.3f}s")
    print(f"  Throughput: {parallel_throughput:.1f}k tok/s")

    print(f"\nSequential Scan:")
    print(f"  Time: {sequential_mean:.3f} ± {sequential_std:.3f}s")
    print(f"  Throughput: {MAX_SEQ_LEN * BATCH_SIZE / sequential_mean / 1000:.1f}k tok/s")

    print(f"\n🚀 SPEEDUP: {speedup:.2f}x")

    if speedup < 2.0:
        print("\n⚠️  WARNING: Speedup is less than 2x!")
        print("   This may indicate the parallel scan is not working correctly.")
    elif speedup < 5.0:
        print("\n✓ Good speedup, but could be better.")
        print("  Expected 5-10x for L=16384 on MPS.")
    else:
        print("\n✓✓✓ EXCELLENT speedup! Parallel scan is working as expected.")

    print("="*80)

    # Save results
    results = {
        'device': str(device),
        'sequence_length': MAX_SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'parallel_mean': parallel_mean,
        'parallel_std': parallel_std,
        'sequential_mean': sequential_mean,
        'sequential_std': sequential_std,
        'speedup': speedup,
        'parallel_throughput': parallel_throughput,
    }

    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()

"""
Test Research Training System
==============================
Verifies all components of the research training system.
"""

import torch
import numpy as np
import sys

print("="*80)
print("TESTING RESEARCH TRAINING SYSTEM")
print("="*80)

# Test 1: Import all modules
print("\n[TEST 1] Importing research modules...")
try:
    from research_metrics import (
        StabilityAnalyzer,
        DiscretizationErrorAnalyzer,
        GradientDiagnostics,
        PerformanceMonitor,
    )
    from checkpointing import CheckpointManager, compute_dataset_hash
    from logging_utils import LoggerFactory, MetricAggregator
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model with diagnostics
print("\n[TEST 2] Testing model diagnostics...")
try:
    from mamba_pytorch import MambaLM

    model = MambaLM(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        d_state=16,
        mode="tustin",
    )

    x = torch.randint(0, 256, (2, 128))  # Small batch

    # Test forward pass with diagnostics
    logits, diagnostics = model(x, return_diagnostics=True)

    assert 'A_bar' in diagnostics, "Missing A_bar in diagnostics"
    assert 'delta' in diagnostics, "Missing delta in diagnostics"

    print(f"✓ Model diagnostics work")
    print(f"  Diagnostics keys: {list(diagnostics.keys())}")

except Exception as e:
    print(f"✗ Model diagnostics failed: {e}")
    sys.exit(1)

# Test 3: Stability metrics
print("\n[TEST 3] Testing stability metrics...")
try:
    stability_metrics = StabilityAnalyzer.analyze_all(
        A_bar=diagnostics['A_bar'],
        A=diagnostics['A'],
        delta=diagnostics['delta'],
    )

    assert 'spectral_radius' in stability_metrics
    assert 'tustin_denominator_condition' in stability_metrics
    assert 'effective_receptive_field' in stability_metrics

    print("✓ Stability metrics computed")
    for key, value in stability_metrics.items():
        print(f"  {key}: {value}")

except Exception as e:
    print(f"✗ Stability metrics failed: {e}")
    sys.exit(1)

# Test 4: Discretization error (RK4)
print("\n[TEST 4] Testing discretization error analysis...")
try:
    # Create small test case
    # Extract single channel for simplified test
    A = diagnostics['A'][:, 0].cpu()  # (D,) - first state dimension
    B = diagnostics['B'][0, 0, :, 0].cpu()  # (D,) - first batch, first time, all channels, first state
    C = diagnostics['C'][0, 0, :, 0].cpu()  # (D,) - same
    delta = diagnostics['delta'][0, :, 0].cpu()  # (L,) - first batch, all time, first channel
    x_seq = diagnostics['x_input'][0, :, :].cpu()  # (L, D) - first batch

    # For RK4 test, we need A, B, C as (D, N=1) since we extracted first state dim
    A_simple = A.unsqueeze(-1)  # (D, 1)
    B_simple = B.unsqueeze(-1)  # (D, 1)
    C_simple = C.unsqueeze(-1)  # (D, 1)

    # Ground truth using RK4
    y_continuous = DiscretizationErrorAnalyzer.solve_continuous_ssm(
        A=A_simple,
        B=B_simple,
        C=C_simple,
        x_seq=x_seq,
        delta=delta,
        num_substeps=10,
    )

    print(f"✓ RK4 ground truth computed")
    print(f"  Output shape: {y_continuous.shape}")

except Exception as e:
    print(f"✗ Discretization error analysis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Gradient diagnostics
print("\n[TEST 5] Testing gradient diagnostics...")
try:
    # Create dummy gradients
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param) * 0.01

    grad_diagnostics = GradientDiagnostics.analyze_all(model)

    assert 'norms' in grad_diagnostics
    assert 'statistics' in grad_diagnostics
    assert 'issues' in grad_diagnostics

    print("✓ Gradient diagnostics computed")
    print(f"  Tracked parameters: {len(grad_diagnostics['norms'])}")
    print(f"  Mean gradient: {grad_diagnostics['statistics']['grad_mean']:.6f}")

except Exception as e:
    print(f"✗ Gradient diagnostics failed: {e}")
    sys.exit(1)

# Test 6: Performance monitoring
print("\n[TEST 6] Testing performance monitoring...")
try:
    device = torch.device('cpu')

    mem_stats = PerformanceMonitor.get_gpu_memory_usage(device)
    assert 'vram_peak_mb' in mem_stats

    throughput = PerformanceMonitor.compute_throughput(
        iter_num=100,
        start_time=0,
        batch_size=2,
        seq_len=128,
        gradient_accumulation_steps=4,
    )
    assert 'tokens_per_sec' in throughput

    print("✓ Performance monitoring works")
    print(f"  Throughput metrics: {list(throughput.keys())}")

except Exception as e:
    print(f"✗ Performance monitoring failed: {e}")
    sys.exit(1)

# Test 7: Checkpoint manager
print("\n[TEST 7] Testing checkpoint manager...")
try:
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=tmpdir,
            checkpoint_interval_hours=0.001,  # Very short for testing
            keep_last_n=2,
        )

        # Should save first checkpoint
        assert ckpt_mgr.should_save_checkpoint() == True

        # Save a checkpoint
        ckpt_path = ckpt_mgr.save_checkpoint(
            step=100,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=None,
            metrics={'loss': 0.5},
            config={'test': True},
            dataset_version={'train_hash': 'abc123'},
            wall_clock_time=3600,
        )

        assert os.path.exists(ckpt_path)

        # Load checkpoint
        loaded = ckpt_mgr.load_latest_checkpoint()
        assert loaded is not None
        assert loaded['step'] == 100

        print("✓ Checkpoint manager works")
        print(f"  Checkpoint saved and loaded successfully")

except Exception as e:
    print(f"✗ Checkpoint manager failed: {e}")
    sys.exit(1)

# Test 8: Logger factory
print("\n[TEST 8] Testing logger factory...")
try:
    # Test JSON logger (no W&B required)
    logger = LoggerFactory.create_logger(
        backend='json',
        log_dir='test_logs',
    )

    # Test metric aggregator
    metric_agg = MetricAggregator(logger)
    metric_agg.add_metric('test/loss', 0.5)
    metric_agg.add_metric('test/accuracy', 0.8)
    metric_agg.flush(step=100)

    print("✓ Logger and metric aggregator work")

    # Cleanup
    import shutil
    if os.path.exists('test_logs'):
        shutil.rmtree('test_logs')

except Exception as e:
    print(f"✗ Logger failed: {e}")
    sys.exit(1)

# Test 9: Dataset versioning
print("\n[TEST 9] Testing dataset versioning...")
try:
    # Test with a dummy file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        temp_path = f.name

    try:
        hash_val = compute_dataset_hash(temp_path)
        assert len(hash_val) == 32  # MD5 hash length
        print(f"✓ Dataset hashing works")
        print(f"  Hash: {hash_val[:16]}...")
    finally:
        os.remove(temp_path)

except Exception as e:
    print(f"✗ Dataset versioning failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nResearch training system is ready to use.")
print("\nNext steps:")
print("  1. Install W&B: pip install wandb")
print("  2. Login to W&B: wandb login")
print("  3. Run research training: python train_pathx_research.py")
print("="*80 + "\n")

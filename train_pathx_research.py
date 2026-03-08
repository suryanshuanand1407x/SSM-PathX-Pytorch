"""
Path-X Research-Grade Training Script
======================================
Gold-standard training for Tustin Mamba paper with comprehensive metrics.

Features:
- Time-based checkpointing (every 2 hours)
- W&B logging with full metric suite
- Numerical stability analysis (spectral radius, condition numbers, ERF)
- Discretization error vs RK4 ground truth
- Gradient diagnostics
- Performance benchmarking (VRAM, throughput)
- Dataset versioning
- Auto-resume from checkpoints
"""

import os
import sys
import time
import pickle
import logging
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional

# Local imports
from mamba_pytorch import MambaLM, count_parameters, optimize_for_mps, enable_fast_inference
from research_metrics import (
    StabilityAnalyzer,
    DiscretizationErrorAnalyzer,
    GradientDiagnostics,
    PerformanceMonitor,
)
from checkpointing import CheckpointManager, compute_dataset_version
from logging_utils import LoggerFactory, MetricAggregator

# =============================================================================
# HYPERPARAMETERS - Research Configuration
# =============================================================================

# Model Architecture
VOCAB_SIZE = 256              # Grayscale pixel values (0-255)
MAX_SEQ_LEN = 16384          # Full 128x128 grid (flattened)
D_MODEL = 64                # Model dimension
N_LAYERS = 6                 # Number of Mamba blocks
D_STATE = 16                 # SSM state dimension
D_CONV = 4                   # Convolution kernel size
EXPAND = 2                   # Expansion factor
MODE = "tustin"              # Discretization: "tustin", "vanilla", or "zoh"

# Training Configuration
BATCH_SIZE = 4               # Batch size (limited by 16k sequence length)
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 2 × 16 = 32
LEARNING_RATE = 3e-4         # Initial learning rate
WEIGHT_DECAY = 0.1           # AdamW weight decay
MAX_ITERS = 20000            # Maximum training iterations
WARMUP_ITERS = 500           # LR warmup iterations
LR_DECAY_ITERS = MAX_ITERS   # LR decay schedule
MIN_LR = 3e-5                # Minimum learning rate
GRAD_CLIP = 1.0              # Gradient clipping threshold

# Evaluation & Logging
EVAL_INTERVAL = 500          # Evaluate every N iterations
EVAL_ITERS = 100             # Number of batches for evaluation
LOG_INTERVAL = 10            # Log training stats every N iterations

# Research Metrics
ENABLE_STABILITY_ANALYSIS = True     # Spectral radius, condition numbers, ERF
ENABLE_DISCRETIZATION_ERROR = True   # RK4 ground truth comparison
ENABLE_GRADIENT_DIAGNOSTICS = True   # Gradient norm tracking
RK4_SUBSTEPS = 10                    # RK4 integration substeps
SYNTHETIC_VAL_SEQ_LEN = 128          # Shorter sequences for RK4 (faster)

# Checkpointing (Time-based)
CHECKPOINT_DIR = "checkpoints_research"
CHECKPOINT_INTERVAL_HOURS = 2.0      # Save every 2 hours
KEEP_LAST_N_CHECKPOINTS = 5          # Keep only 5 most recent

# Logging
WANDB_PROJECT = "tustin-mamba-pathx"
WANDB_ENTITY = None                  # Set to your W&B username or team
USE_WANDB = True                     # Enable W&B logging

# Precision & Performance
USE_BFLOAT16 = True          # Mixed precision training
NUM_WORKERS = 4              # DataLoader workers
PIN_MEMORY = True            # Pin memory for faster GPU transfer

# Paths
DATA_DIR = "archive"
RESULTS_DIR = "results/pathx_research"

# Random Seed
SEED = 12

# =============================================================================
# Setup Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_research.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# =============================================================================
# Device Detection
# =============================================================================

def get_device() -> torch.device:
    """Detect best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps')
            device = torch.device('mps')
            logging.info(f"✓ Using Apple Metal (MPS) acceleration")
            return device
        except Exception as e:
            logging.warning(f"MPS available but failed: {e}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device

    device = torch.device('cpu')
    logging.warning(f"⚠ Using CPU (training will be slow)")
    return device


def configure_memory_management(device: torch.device):
    """
    Configure PyTorch memory management to prevent system hangs.

    Sets up:
    - CUDA memory allocator with fragmentation limits
    - Garbage collection optimization
    - Memory allocation strategy to prevent OOM
    """
    if device.type == 'cuda':
        # Configure CUDA memory allocator
        # max_split_size_mb: Prevents large allocations from fragmenting memory
        # This helps avoid OOM errors by limiting how memory is split
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

        # Set memory fraction to leave some room for system
        # This prevents taking up 100% of VRAM which can hang the system
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True

        # Clear any cached memory from previous runs
        torch.cuda.empty_cache()

        logging.info("✓ CUDA memory management configured:")
        logging.info("  - Max split size: 512 MB")
        logging.info("  - Expandable segments: Enabled")
        logging.info("  - Memory fraction: 95%")
        logging.info("  - CuDNN benchmark: Enabled")

    elif device.type == 'mps':
        # MPS-specific memory management and performance optimizations
        optimize_for_mps()

        # Clear any cached memory
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        logging.info("✓ MPS memory management configured")
        logging.info("  - Parallel scan optimized for Metal backend")
        logging.info("  - Float32 precision for linear algebra (no CPU fallback)")
        logging.info("  - Memory fraction: 80%")

    # Python garbage collection optimization
    # Collect less frequently to reduce overhead, but collect when we do
    gc.set_threshold(700, 10, 10)
    logging.info("✓ Garbage collection optimized")


def cleanup_memory(device: torch.device, force_gc: bool = False):
    """
    Clean up memory caches to prevent accumulation.

    Args:
        device: Device to clean up memory for
        force_gc: Whether to force Python garbage collection
    """
    if device.type == 'cuda':
        # Empty CUDA cache
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # Empty MPS cache if available
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Force garbage collection if requested
    if force_gc:
        gc.collect()


# =============================================================================
# Dataset Class
# =============================================================================

class PathXDataset(Dataset):
    """Path-X Dataset from Long Range Arena."""

    def __init__(self, data_path: str):
        logging.info(f"Loading dataset: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        logging.info(f"  Loaded {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.data[idx]
        x = torch.from_numpy(example['input_ids_0']).long()
        y = torch.tensor(example['label'], dtype=torch.long)
        return x, y


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(iter_num: int) -> float:
    """Cosine learning rate schedule with warmup."""
    if iter_num < WARMUP_ITERS:
        return LEARNING_RATE * iter_num / WARMUP_ITERS
    if iter_num > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (iter_num - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# =============================================================================
# Research Metric Collection
# =============================================================================

def collect_stability_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Collect numerical stability metrics from model.

    Returns metrics:
    - spectral_radius: Max eigenvalue of A_bar
    - tustin_denominator_condition: Condition number of (I - Δ/2*A)
    - effective_receptive_field: ERF length
    """
    model.eval()

    # Get a single batch for analysis
    x, y = next(iter(val_loader))
    x = x.to(device)

    with torch.no_grad():
        # Forward pass with diagnostics
        _, diagnostics = model(x, return_diagnostics=True)

        # Analyze stability
        stability_metrics = StabilityAnalyzer.analyze_all(
            A_bar=diagnostics['A_bar'],
            A=diagnostics['A'],
            delta=diagnostics['delta'],
        )

    model.train()
    return stability_metrics


def collect_discretization_error(
    model: nn.Module,
    device: torch.device,
    seq_len: int = SYNTHETIC_VAL_SEQ_LEN,
) -> Dict[str, float]:
    """
    Compute discretization error vs RK4 ground truth.

    Uses a small synthetic batch to avoid expensive RK4 integration
    on full 16k sequences.
    """
    model.eval()

    # Create synthetic validation batch
    batch_size = 2
    x_synth = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(device)

    with torch.no_grad():
        # Forward pass with diagnostics
        logits, diagnostics = model(x_synth, return_diagnostics=True)

        # Get discrete SSM output (after embedding projection)
        # We'll use the first layer's output for analysis
        y_discrete = logits.mean(dim=-1)  # (B, L) - average over vocab

        # Extract SSM parameters for first sample
        A = diagnostics['A'][0].cpu()  # (N,)
        B = diagnostics['B'][0, 0, 0].cpu()  # (N,) - first timestep, first channel
        C = diagnostics['C'][0, 0, 0].cpu()  # (N,)
        delta = diagnostics['delta'][0, :, 0].cpu()  # (L,) - first channel
        x_input = diagnostics['x_input'][0, :, 0].cpu()  # (L,) - first channel

        # Compute RK4 ground truth (simplified single-channel analysis)
        # Note: Full analysis would require per-channel computation
        error_metrics = DiscretizationErrorAnalyzer.analyze_discretization_error(
            y_discrete=y_discrete[0, :].cpu(),  # First sample
            A=A,
            B=B,
            C=C,
            x_seq=x_input,
            delta=delta,
            num_substeps=RK4_SUBSTEPS,
        )

    model.train()
    return error_metrics


def collect_gradient_metrics(model: nn.Module) -> Dict[str, float]:
    """Collect gradient diagnostics."""
    grad_diagnostics = GradientDiagnostics.analyze_all(
        model=model,
        param_patterns=['A_log', 'dt_proj', 'delta_norm', 'D'],
    )

    # Flatten for logging
    metrics = {}
    for name, norm in grad_diagnostics['norms'].items():
        metrics[f'gradients/norm_{name}'] = norm

    for name, value in grad_diagnostics['statistics'].items():
        metrics[f'gradients/{name}'] = value

    for name, flag in grad_diagnostics['issues'].items():
        metrics[f'gradients/{name}'] = int(flag)

    return metrics


# =============================================================================
# Evaluation Function
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_iters: int = EVAL_ITERS,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_iters:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BFLOAT16):
            logits = model(x)
            logits_pooled = logits.mean(dim=1)
            binary_logits = logits_pooled[:, :2]
            loss = F.cross_entropy(binary_logits, y)

        predictions = binary_logits.argmax(dim=-1)
        correct = (predictions == y).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += y.size(0)

    avg_loss = total_loss / min(max_iters, len(dataloader))
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    model.train()
    return {'loss': avg_loss, 'accuracy': accuracy}


# =============================================================================
# Training Function
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
):
    """Research-grade training loop with comprehensive metrics."""

    # Setup checkpoint manager (time-based)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval_hours=CHECKPOINT_INTERVAL_HOURS,
        keep_last_n=KEEP_LAST_N_CHECKPOINTS,
    )

    # Setup logger (W&B or JSON fallback)
    logger = LoggerFactory.create_logger(
        backend='wandb' if USE_WANDB else 'json',
        project=WANDB_PROJECT,
        config=config,
        name=f"{MODE}_pathx_{time.strftime('%Y%m%d_%H%M%S')}",
        log_dir=RESULTS_DIR,
    )

    # Metric aggregator
    metric_agg = MetricAggregator(logger)

    # Dataset versioning
    dataset_version = compute_dataset_version({
        'train': os.path.join(DATA_DIR, 'lra-pathfinder128-curv_contour_length_14.train.pickle'),
        'val': os.path.join(DATA_DIR, 'lra-pathfinder128-curv_contour_length_14.dev.pickle'),
    })
    logger.log_config({'dataset_version': dataset_version})

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # Gradient scaler (CUDA only)
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=USE_BFLOAT16)
    else:
        scaler = None

    # Try to resume from checkpoint
    checkpoint_state = checkpoint_manager.load_latest_checkpoint()
    if checkpoint_state is not None:
        # Handle torch.compile key mismatch: compiled models have "_orig_mod." prefix
        state_dict = checkpoint_state['model_state_dict']
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        # Check if we need to add/remove the "_orig_mod." prefix
        if model_keys != checkpoint_keys:
            # Case 1: Model is compiled, checkpoint is not -> add prefix
            if any(k.startswith('_orig_mod.') for k in model_keys):
                logging.info("  Adapting checkpoint for compiled model (adding _orig_mod. prefix)...")
                state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
            # Case 2: Model is not compiled, checkpoint is compiled -> remove prefix
            elif any(k.startswith('_orig_mod.') for k in checkpoint_keys):
                logging.info("  Adapting checkpoint from compiled model (removing _orig_mod. prefix)...")
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        if scaler is not None and checkpoint_state['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint_state['scaler_state_dict'])
        start_iter = checkpoint_state['step'] + 1
        global_start_time = time.time() - checkpoint_state.get('wall_clock_time', 0)

        # Restore RNG states
        torch.set_rng_state(checkpoint_state['random_state']['torch'])
        if checkpoint_state['random_state']['torch_cuda'] is not None:
            torch.cuda.set_rng_state(checkpoint_state['random_state']['torch_cuda'])

        logging.info(f"✓ Resumed from step {start_iter}")
    else:
        start_iter = 0
        global_start_time = time.time()

    # Training state
    best_val_loss = float('inf')
    step_times = []

    # Reset peak memory stats
    PerformanceMonitor.reset_peak_memory(device)

    logging.info("\n" + "="*80)
    logging.info("RESEARCH TRAINING START")
    logging.info("="*80)
    logging.info(f"Model: {MODE.upper()} mode")
    logging.info(f"Parameters: {count_parameters(model):,}")
    logging.info(f"Device: {device}")
    logging.info(f"Precision: {'bfloat16' if USE_BFLOAT16 else 'float32'}")
    logging.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL_HOURS} hours")
    logging.info(f"Research metrics: Stability={ENABLE_STABILITY_ANALYSIS}, "
                 f"Discretization={ENABLE_DISCRETIZATION_ERROR}, "
                 f"Gradients={ENABLE_GRADIENT_DIAGNOSTICS}")
    logging.info("="*80 + "\n")

    # Training loop
    train_iter = iter(train_loader)
    model.train()
    optimizer.zero_grad()

    for iter_num in range(start_iter, MAX_ITERS):
        step_start = time.time()

        # === Get batch ===
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        # === Update learning rate ===
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # === Forward pass ===
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BFLOAT16):
            logits = model(x)
            logits_pooled = logits.mean(dim=1)
            binary_logits = logits_pooled[:, :2]
            loss = F.cross_entropy(binary_logits, y)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        # === Backward pass ===
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # === Gradient accumulation ===
        if (iter_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # Collect gradient metrics BEFORE clipping
            if ENABLE_GRADIENT_DIAGNOSTICS:
                grad_metrics = collect_gradient_metrics(model)
                for k, v in grad_metrics.items():
                    metric_agg.add_metric(k, v)

            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        step_times.append(time.time() - step_start)

        # === Logging (every LOG_INTERVAL) ===
        if iter_num % LOG_INTERVAL == 0:
            # Training metrics
            metric_agg.add_metric('train/loss', loss.item() * GRADIENT_ACCUMULATION_STEPS)
            metric_agg.add_metric('train/lr', lr)

            # Performance metrics
            throughput = PerformanceMonitor.compute_throughput(
                iter_num=iter_num,
                start_time=global_start_time,
                batch_size=BATCH_SIZE,
                seq_len=MAX_SEQ_LEN,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            )
            metric_agg.add_metrics({f'performance/{k}': v for k, v in throughput.items()})

            # Timing stats
            timing_stats = PerformanceMonitor.compute_timing_stats(step_times)
            metric_agg.add_metrics({f'timing/{k}': v for k, v in timing_stats.items()})

            # Wall clock time
            elapsed_hours = (time.time() - global_start_time) / 3600
            metric_agg.add_metric('timing/wall_clock_hours', elapsed_hours)

            # Flush metrics
            metric_agg.flush(step=iter_num)

            # Console log
            logging.info(
                f"iter {iter_num:5d} | loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | "
                f"lr {lr:.2e} | {throughput['tokens_per_sec']:.0f} tok/s | "
                f"{timing_stats['ms_per_step_mean']:.1f} ms/step"
            )

        # === Periodic Memory Cleanup (every 100 iterations) ===
        if iter_num % 100 == 0 and iter_num > 0:
            cleanup_memory(device, force_gc=False)

        # === Evaluation (every EVAL_INTERVAL) ===
        if iter_num % EVAL_INTERVAL == 0 and iter_num > 0:
            logging.info("\n" + "-"*80)
            logging.info(f"EVALUATION at iter {iter_num}")
            logging.info("-"*80)

            eval_start = time.time()

            # Standard validation metrics
            val_metrics = evaluate(model, val_loader, device)
            metric_agg.add_metric('val/loss', val_metrics['loss'])
            metric_agg.add_metric('val/accuracy', val_metrics['accuracy'])

            logging.info(f"Val loss: {val_metrics['loss']:.4f} | Val accuracy: {val_metrics['accuracy']:.4f}")

            # === Research Metrics ===
            if ENABLE_STABILITY_ANALYSIS:
                try:
                    stability_metrics = collect_stability_metrics(model, val_loader, device)
                    metric_agg.add_metrics({f'stability/{k}': v for k, v in stability_metrics.items()})
                    logging.info(f"Spectral radius: {stability_metrics['spectral_radius']:.6f}")
                    logging.info(f"Condition number: {stability_metrics['tustin_denominator_condition']:.6f}")
                    logging.info(f"ERF: {stability_metrics['effective_receptive_field']}")
                except Exception as e:
                    logging.warning(f"Stability analysis failed: {e}")

            if ENABLE_DISCRETIZATION_ERROR:
                try:
                    error_metrics = collect_discretization_error(model, device)
                    metric_agg.add_metrics({f'discretization/{k}': v for k, v in error_metrics.items()})
                    logging.info(f"L2 error (vs RK4): {error_metrics['l2_error']:.6f}")
                except Exception as e:
                    logging.warning(f"Discretization error analysis failed: {e}")

            # GPU memory
            mem_stats = PerformanceMonitor.get_gpu_memory_usage(device)
            metric_agg.add_metrics({f'memory/{k}': v for k, v in mem_stats.items()})
            if device.type == 'cuda':
                logging.info(f"VRAM peak: {mem_stats['vram_peak_mb']:.1f} MB")

            eval_time = time.time() - eval_start
            logging.info(f"Evaluation time: {eval_time:.1f}s")
            logging.info("-"*80 + "\n")

            # Flush evaluation metrics
            metric_agg.flush(step=iter_num)

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_manager.save_best_checkpoint(
                    step=iter_num,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    metrics=val_metrics,
                    config=config,
                    dataset_version=dataset_version,
                )

            # Clean up memory after evaluation
            cleanup_memory(device, force_gc=True)

        # === Time-based Checkpointing ===
        if checkpoint_manager.should_save_checkpoint():
            wall_clock_time = time.time() - global_start_time
            checkpoint_manager.save_checkpoint(
                step=iter_num,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                metrics={'train_loss': loss.item() * GRADIENT_ACCUMULATION_STEPS},
                config=config,
                dataset_version=dataset_version,
                wall_clock_time=wall_clock_time,
            )
            # Clean up memory after checkpoint save
            cleanup_memory(device, force_gc=False)

    # === Final Stats ===
    total_time = time.time() - global_start_time
    logging.info("\n" + "="*80)
    logging.info("TRAINING COMPLETE")
    logging.info("="*80)
    logging.info(f"Total wall-clock time: {total_time/3600:.2f} hours")
    logging.info(f"Best val loss: {best_val_loss:.4f}")
    logging.info("="*80 + "\n")

    # Final memory cleanup
    cleanup_memory(device, force_gc=True)
    logging.info("✓ Final memory cleanup complete")

    # Finish logging
    logger.finish()


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main training function."""

    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Get device
    device = get_device()

    # Configure memory management to prevent system hangs
    configure_memory_management(device)

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load datasets
    logging.info("\n" + "="*80)
    logging.info("LOADING DATASETS")
    logging.info("="*80)

    train_dataset = PathXDataset(
        os.path.join(DATA_DIR, "lra-pathfinder128-curv_contour_length_14.train.pickle")
    )
    val_dataset = PathXDataset(
        os.path.join(DATA_DIR, "lra-pathfinder128-curv_contour_length_14.dev.pickle")
    )

    logging.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    logging.info("="*80 + "\n")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == 'cuda',
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == 'cuda',
        persistent_workers=NUM_WORKERS > 0,
    )

    # Initialize model
    logging.info("="*80)
    logging.info("INITIALIZING MODEL")
    logging.info("="*80)

    model = MambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        mode=MODE,
    )

    model = model.to(device)

    # Enable torch.compile for faster inference (PyTorch 2.0+)
    try:
        model = enable_fast_inference(model, device)
    except Exception as e:
        logging.warning(f"Failed to enable torch.compile: {e}")

    logging.info(f"Parameters: {count_parameters(model):,}")
    logging.info("="*80 + "\n")

    # Build config dictionary
    config = {
        'model': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'n_layers': N_LAYERS,
            'd_state': D_STATE,
            'd_conv': D_CONV,
            'expand': EXPAND,
            'mode': MODE,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'max_iters': MAX_ITERS,
            'grad_clip': GRAD_CLIP,
        },
        'research': {
            'stability_analysis': ENABLE_STABILITY_ANALYSIS,
            'discretization_error': ENABLE_DISCRETIZATION_ERROR,
            'gradient_diagnostics': ENABLE_GRADIENT_DIAGNOSTICS,
            'rk4_substeps': RK4_SUBSTEPS,
        },
    }

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )


if __name__ == "__main__":
    main()

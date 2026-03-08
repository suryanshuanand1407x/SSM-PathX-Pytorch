# Hyperparameter Management Guide

## ✅ Problem Solved

The training system now **automatically detects** when you change hyperparameters and handles incompatible checkpoints gracefully.

---

## How It Works

### Automatic Validation

When you start training, the system:
1. ✅ Loads the latest checkpoint
2. ✅ Compares checkpoint config with current config
3. ✅ Detects mismatches in critical parameters
4. ✅ Shows clear warning and starts from scratch if incompatible

### Critical Parameters (Checked Automatically)

These parameters affect model architecture and are validated:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | 256 |
| `d_model` | Model dimension | 64 |
| `n_layers` | Number of layers | 6 |
| `d_state` | SSM state dimension | 16 |
| `d_conv` | Convolution kernel size | 4 |
| `expand` | Expansion factor | 2 |

**If ANY of these change**, the checkpoint is incompatible.

---

## Example Scenarios

### Scenario 1: Changed D_MODEL (Your Current Case)

**What you did:**
```python
D_MODEL = 64  # Old value (saved in checkpoint)
↓
D_MODEL = 16  # New value (current code)
```

**What happens:**
```
⚠️  CHECKPOINT HYPERPARAMETER MISMATCH DETECTED!
The checkpoint was saved with different hyperparameters:
  • d_model: 64 -> 16

Options:
  1. Delete old checkpoints: rm -rf checkpoints_research/*
  2. Revert hyperparameters to match checkpoint
  3. Use a different checkpoint directory

⚠️  Starting training from scratch (ignoring incompatible checkpoint)
```

**System behavior:**
- ✅ Training starts from step 0 (fresh)
- ✅ No crash or confusing errors
- ✅ New checkpoints saved with D_MODEL=16

### Scenario 2: Only Changed Training Hyperparameters

**What you did:**
```python
BATCH_SIZE = 2 -> 4          # OK to change
LEARNING_RATE = 3e-4 -> 1e-4  # OK to change
GRAD_CLIP = 1.0 -> 0.5        # OK to change
```

**What happens:**
- ✅ Checkpoint loads successfully
- ✅ Training resumes from where it left off
- ⚠️ Training hyperparameters are NOT validated (you're responsible for tracking)

---

## Recommended Workflow

### Option 1: Clean Slate (Recommended)

When changing model architecture, start fresh:

```bash
# Delete old checkpoints
rm -rf checkpoints_research/*

# Start training
python train_pathx_research.py
```

**Pros:**
- Clean experiments
- No confusion
- W&B tracks each run separately

**Cons:**
- Lose previous training progress

### Option 2: Use Separate Checkpoint Directories

Keep checkpoints for different configurations:

```python
# In train_pathx_research.py
CHECKPOINT_DIR = f"checkpoints_research_{D_MODEL}d_{N_LAYERS}l"
```

Example:
```
checkpoints_research_64d_6l/   # D_MODEL=64, N_LAYERS=6
checkpoints_research_16d_4l/   # D_MODEL=16, N_LAYERS=4
checkpoints_research_128d_8l/  # D_MODEL=128, N_LAYERS=8
```

**Pros:**
- Keep all experiments
- Easy to resume any configuration

**Cons:**
- More disk space

### Option 3: Revert Hyperparameters

If you want to resume training:

```python
# Revert to checkpoint values
D_MODEL = 64  # Was 16
N_LAYERS = 6  # Keep same
```

**Pros:**
- Continue training where you left off

**Cons:**
- Can't test new configuration

---

## Manual Checkpoint Cleanup

### Delete All Checkpoints

```bash
rm -rf checkpoints_research/*
```

### Delete Specific Checkpoint

```bash
# List checkpoints
ls -lh checkpoints_research/

# Delete specific checkpoint
rm checkpoints_research/checkpoint_step0_20260308_120552.pt
```

### Keep Only Best Checkpoint

```bash
# Delete regular checkpoints, keep best
rm checkpoints_research/checkpoint_*.pt
# Keep: checkpoints_research/best_checkpoint.pt
```

---

## Training Hyperparameters (Not Validated)

These can be changed freely without invalidating checkpoints:

### Safe to Change
- `BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `GRAD_CLIP`
- `WARMUP_ITERS`
- `LR_DECAY_ITERS`
- `MIN_LR`
- `MAX_ITERS`
- `EVAL_INTERVAL`
- `LOG_INTERVAL`

### Example
```python
# Change these anytime:
BATCH_SIZE = 4  # Was 2
LEARNING_RATE = 1e-4  # Was 3e-4
```

Checkpoint will load fine, and training continues with new values.

---

## Model Mode Parameter

The `mode` parameter (`"tustin"`, `"vanilla"`, `"zoh"`) is special:

**Changing mode WILL break checkpoint loading** because:
- Different modes have different parameters
- Example: `delta_norm` only exists in `"tustin"` mode

**Solution:** Treat mode changes like architecture changes - start fresh.

---

## W&B Run Management

When starting fresh due to hyperparameter changes:

### Option 1: Let W&B Create New Run (Automatic)
```python
# No changes needed - system creates new run ID automatically
```

**Result:** Each run gets unique ID, easy to compare.

### Option 2: Group Related Experiments
```python
# In train_pathx_research.py, modify logger setup:
logger = LoggerFactory.create_logger(
    backend='wandb',
    project=WANDB_PROJECT,
    config=config,
    name=f"{MODE}_d{D_MODEL}_l{N_LAYERS}_{time.strftime('%Y%m%d_%H%M%S')}",
    group=f"{MODE}_hyperparameter_search",  # Add this
    log_dir=RESULTS_DIR,
)
```

**Result:** Runs grouped in W&B for easy comparison.

---

## Troubleshooting

### Issue: Still getting shape mismatch errors

**Cause:** Validation didn't catch all parameters.

**Solution:**
```bash
# Nuclear option - delete everything
rm -rf checkpoints_research/*
rm -rf wandb/*  # Optional: clean W&B cache
```

### Issue: Want to keep old checkpoint but start fresh

**Solution:**
```bash
# Rename old checkpoint directory
mv checkpoints_research checkpoints_research_old_64d

# Start training (creates new checkpoints_research/)
python train_pathx_research.py
```

### Issue: Accidentally deleted checkpoint mid-training

**Solution:**
- If you have `best_checkpoint.pt`, you can still resume from best validation checkpoint
- Otherwise, training starts from scratch (not the end of the world!)

---

## Summary

✅ **Automatic Detection:** System detects incompatible checkpoints
✅ **Clear Warnings:** Tells you exactly what changed
✅ **Graceful Handling:** Starts fresh instead of crashing
✅ **No Manual Checking:** You don't need to remember to delete checkpoints

**Best Practice:** When changing architecture hyperparameters (d_model, n_layers, etc.), run:
```bash
rm -rf checkpoints_research/* && python train_pathx_research.py
```

---

## Current Status

Your training is now configured to:
1. ✅ Detect D_MODEL mismatch (64 -> 16)
2. ✅ Show warning message
3. ✅ Start training from scratch with D_MODEL=16
4. ✅ Save new checkpoints compatible with D_MODEL=16

**You can now run:**
```bash
python train_pathx_research.py
```

And it will work correctly! 🎉

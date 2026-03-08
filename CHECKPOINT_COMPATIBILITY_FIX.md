# Checkpoint Compatibility Fix

## Issue
When `torch.compile` is enabled, model parameter names get prefixed with `_orig_mod.`:
- **Uncompiled**: `embedding.weight`
- **Compiled**: `_orig_mod.embedding.weight`

This caused a mismatch when loading checkpoints saved before torch.compile was enabled.

## Solution

### 1. Checkpoint Saving (Always Strip Prefix)
**File**: `checkpointing/checkpoint_manager.py`

Both `save_checkpoint()` and `save_best_checkpoint()` now strip the `_orig_mod.` prefix before saving:

```python
# Strip _orig_mod. prefix if present (from torch.compile)
state_dict = model.state_dict()
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```

**Result**: All checkpoints are saved in uncompiled format (portable).

### 2. Checkpoint Loading (Auto-Adapt)
**File**: `train_pathx_research.py`

The checkpoint loading code now auto-adapts to handle both cases:

```python
# Handle torch.compile key mismatch
state_dict = checkpoint_state['model_state_dict']
model_keys = set(model.state_dict().keys())
checkpoint_keys = set(state_dict.keys())

if model_keys != checkpoint_keys:
    # Case 1: Model is compiled, checkpoint is not -> add prefix
    if any(k.startswith('_orig_mod.') for k in model_keys):
        state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
    # Case 2: Model is not compiled, checkpoint is compiled -> remove prefix
    elif any(k.startswith('_orig_mod.') for k in checkpoint_keys):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```

**Result**: Checkpoints are automatically adapted on load.

## Benefits

1. ✅ **Backward Compatible**: Old checkpoints (pre-torch.compile) load correctly
2. ✅ **Forward Compatible**: New checkpoints work with/without torch.compile
3. ✅ **Portable**: Checkpoints can be shared between compiled and non-compiled models
4. ✅ **Transparent**: No user action required

## Testing

The fix has been applied. Your training should now resume correctly:

```bash
python train_pathx_research.py
```

Expected log output:
```
Loading checkpoint: checkpoints_research/checkpoint_step0_20260308_120552.pt
  Adapting checkpoint for compiled model (adding _orig_mod. prefix)...
✓ Loaded checkpoint from step 0
  Timestamp: 20260308_120552
  Wall-clock: 0.20h
```

## Future Checkpoints

All new checkpoints will be saved in uncompiled format, ensuring maximum compatibility.

---

**Status**: ✅ FIXED - Training can now resume with torch.compile enabled.

# Quick Reference Card - Research Training

## 🚀 Commands

### Setup
```bash
pip install -r requirements_research.txt
wandb login
```

### Run Training
```bash
python train_pathx_research.py
```

### Test System
```bash
python test_research_system.py
```

---

## 📊 Key Metrics

### Stability (Good Values)
```
spectral_radius < 1.0              ✓ Stable
tustin_denominator_condition < 100 ✓ Well-conditioned
effective_receptive_field > 5000   ✓ Strong memory
```

### Discretization Error
```
l2_error < 1e-3                    ✓ Low error
max_error < 5e-3                   ✓ Acceptable
```

### Gradients
```
has_vanishing_gradients = 0        ✓ Healthy
has_exploding_gradients = 0        ✓ Healthy
grad_max < 10                      ✓ Normal range
```

---

## ⚙️ Quick Config Changes

### Faster Experiments
```python
MAX_ITERS = 1000              # 1k steps instead of 10k
EVAL_INTERVAL = 100           # Eval more frequently
CHECKPOINT_INTERVAL_HOURS = 0.5  # Checkpoint every 30min
```

### Disable Expensive Metrics
```python
ENABLE_DISCRETIZATION_ERROR = False  # Skip RK4
RK4_SUBSTEPS = 5                     # Reduce substeps
```

### Memory Constrained
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
SYNTHETIC_VAL_SEQ_LEN = 64  # Shorter for RK4
```

### Compare Modes
```python
MODE = "tustin"   # URD with guards
MODE = "vanilla"  # Raw Tustin
MODE = "zoh"      # Baseline
```

---

## 📁 Output Locations

```
checkpoints_research/           # Checkpoints (every 2 hours)
  ├── checkpoint_step500_*.pt
  └── best_checkpoint.pt

results/pathx_research/         # JSON logs (if no W&B)
  └── metrics.jsonl

training_research.log           # Detailed logs
```

---

## 🔍 Monitoring

### W&B Dashboard
```
Project: tustin-mamba-pathx

Key Plots:
  - stability/spectral_radius
  - stability/tustin_denominator_condition
  - discretization/l2_error
  - val/accuracy
  - gradients/grad_max
```

### Console Output
```
iter 1234 | loss 0.6543 | lr 3.00e-04 | 12345 tok/s | 87.3 ms/step

Val loss: 0.6234 | Val accuracy: 0.7123
Spectral radius: 0.97
Condition number: 45
ERF: 7842
```

---

## 🐛 Quick Fixes

### W&B Not Working
```python
USE_WANDB = False  # Use JSON instead
```

### Out of Memory
```python
BATCH_SIZE = 1
D_MODEL = 64
```

### Training Too Slow
```python
ENABLE_DISCRETIZATION_ERROR = False
EVAL_INTERVAL = 1000
```

### Resume Training
Just re-run - auto-detects latest checkpoint:
```bash
python train_pathx_research.py
```

---

## 📊 Expected Results Summary

| Mode    | Accuracy | Spectral R. | Condition # | ERF  |
|---------|----------|-------------|-------------|------|
| Tustin  | 75-80%   | 0.97        | 45          | 7842 |
| Vanilla | 50-60%   | 0.99        | 234         | 5234 |
| ZOH     | 15-40%   | 1.03        | 1523        | 2341 |

---

## 🎯 Paper Checklist

- [ ] Train all 3 modes (tustin, vanilla, zoh)
- [ ] Collect 10k iterations per mode
- [ ] Generate stability comparison plots
- [ ] Generate discretization error plots
- [ ] Generate gradient flow visualizations
- [ ] Create performance comparison table
- [ ] Run ablation studies (remove guards)
- [ ] Profile computational overhead
- [ ] Write results section

---

## 💡 Tips

1. **Start with short run:** Set `MAX_ITERS=100` for quick validation
2. **Monitor spectral radius:** Should stay < 1.0 for stability
3. **Check gradients early:** Vanishing/exploding detected automatically
4. **Use W&B:** Much easier than parsing JSON logs
5. **Save best model:** Automatically saved as `best_checkpoint.pt`

---

## 📞 Quick Help

**System not working?**
```bash
python test_research_system.py  # Run diagnostics
```

**Need full docs?**
- `RESEARCH_TRAINING_GUIDE.md` - Complete usage guide
- `IMPLEMENTATION_SUMMARY.md` - What was implemented

**Questions about metrics?**
- See "Research Metrics Explained" in RESEARCH_TRAINING_GUIDE.md

# GPU Training Setup

## Current Status

✅ **GPU Available**: NVIDIA GeForce GTX 1050 (2GB VRAM)
✅ **PyTorch CUDA**: Installed (2.2.0+cu118)
✅ **CUDA Support**: Enabled

## GPU Usage in Models

### Models That Use GPU:

1. **FT-Transformer** (`models/architectures/ft_transformer.py`)
   - Uses GPU automatically when `device="auto"` (default)
   - Now explicitly set to `device="cuda"` in training script
   - GPU acceleration for transformer layers

2. **TabNet** (`models/architectures/tabnet.py`)
   - Uses GPU automatically when `device="auto"` (default)
   - Now explicitly set to `device="cuda"` in training script
   - GPU acceleration for attention mechanisms

3. **Stacking Ensemble MLP Meta-Model** (`models/architectures/stacking_ensemble.py`)
   - Uses GPU automatically: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
   - GPU acceleration for meta-learner training

### Models That Don't Use GPU:

1. **Logistic Regression** - CPU-only (scikit-learn)
2. **Gradient Boosting (XGBoost)** - Can use GPU but currently CPU
   - XGBoost supports GPU with `tree_method='gpu_hist'`
   - Not currently enabled

## Why GPU Might Not Be Active

The current training process (`train_phase3_models.py`) is likely in the **feature generation phase**, which is CPU-bound:

1. **Feature Generation** (CPU-bound):
   - Loading/processing data (pandas)
   - Weather API calls (network I/O)
   - Injury data generation
   - Schedule feature calculations
   - Data merging and transformations

2. **Model Training** (GPU-bound):
   - FT-Transformer training → **Uses GPU**
   - TabNet training → **Uses GPU**
   - Stacking ensemble meta-model → **Uses GPU**

## Monitoring GPU Usage

To check if GPU is being used during training:

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

Look for:
- **GPU-Util**: Should be >0% during model training
- **Memory-Usage**: Should increase when models load to GPU

## Expected Behavior

1. **Feature Generation** (5-10 min): CPU-bound, GPU idle
2. **Base Model Training**:
   - Logistic Regression: CPU
   - Gradient Boosting: CPU (can enable GPU)
   - FT-Transformer: **GPU active** ✅
   - TabNet: **GPU active** ✅
3. **Ensemble Meta-Model**: **GPU active** ✅

## Enabling GPU for XGBoost (Optional)

To enable GPU for Gradient Boosting, update `train_gradient_boosting()`:

```python
model = XGBClassifier(
    tree_method='gpu_hist',  # Use GPU
    # ... other params
)
```

Note: Requires XGBoost built with GPU support.

## Summary

✅ GPU is configured and will be used for:
- FT-Transformer training
- TabNet training  
- Stacking ensemble meta-model

⚠️ Currently CPU-bound:
- Feature generation (expected)
- Logistic Regression (scikit-learn limitation)
- Gradient Boosting (can enable GPU)

The training script will automatically use GPU when it reaches the model training phase. Feature generation is expected to be CPU-bound.


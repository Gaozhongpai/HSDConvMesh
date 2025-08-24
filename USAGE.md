# Code Usage Guide

## Overview

This repository provides multiple ways to run the spectral dictionary mesh autoencoder:

## 1. Quick Start (Backwards Compatible)

For existing users, the original `pai3DMM.py` script still works but is now much cleaner:

```bash
python pai3DMM.py
```

Configure by editing the parameters at the top of `pai3DMM.py` (lines 34-39):

```python
root_dir = 'dataset/COMA-dataset'   # Dataset path
is_hierarchical = True              # True: HSDConv, False: SDConv
is_same_param = False               # Parameter configuration variant
is_old_filter = False               # False: updated filter (-x), True: original
mode = 'test'                       # 'train' or 'test'
```

## 2. Recommended (New Projects)

For new projects, use the modular approach with configuration files:

```bash
python main.py
```

This uses:
- `config.py` - Clean configuration management with dataclasses
- `setup_utils.py` - Utility functions for setup and data loading
- `main.py` - Clean main execution script

## 3. Configuration Options

### Model Variants
- **SDConv**: Original spectral dictionary convolution (IJCAI 2021)
- **HSDConv**: Hierarchical version with adaptive sampling (TPAMI extension)
- **-x suffix**: Updated spectral filter (recommended)

### Key Parameters
- `is_hierarchical=True`: Enables adaptive hierarchical mapping (HSDConv)
- `is_old_filter=False`: Uses improved spectral filter
- `base_size=32`: Dictionary size (affects model size vs accuracy trade-off)
- `filter_sizes_enc/dec`: Network architecture channel sizes

### Dataset Support
- COMA: Human faces (5,023 vertices)
- DFAUST: Human bodies (6,890 vertices)  
- MANO: Human hands

## 4. Model Names

The system automatically generates model names based on configuration:
- `SDConvFinal-x`: Base model with updated filter
- `HSDConvFinal-x`: Hierarchical model with updated filter
- `HSDConvFinal-x-param`: Parameter variant

## 5. Directory Structure

Results are organized as:
```
dataset/[DATASET]/results/[MODEL_NAME]/latent_[SIZE]/
├── checkpoints/        # Model checkpoints
├── summaries/          # Tensorboard logs
├── samples/           # Training samples
└── predictions/       # Test predictions
```

## 6. Training vs Testing

**Training:**
```python
mode = 'train'
```

**Testing:**
```python
mode = 'test'
```

## 7. Performance Tips

- Use `is_hierarchical=True` for better performance on diverse poses
- Use `is_old_filter=False` for improved spectral filtering
- Adjust `base_size` to balance model size vs accuracy
- Use COMA dataset for faces, DFAUST for bodies
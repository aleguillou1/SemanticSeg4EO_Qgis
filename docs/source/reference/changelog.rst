.. _changelog:

Changelog
=========

All notable changes to SemanticSeg4EO are documented here.

Version 1.1.0 — Robust Patch Extraction
-----------------------------------------

**Patch Extraction**

- **Batch mode**: extract patches from multiple image/label pairs at once
- **Per-image grids**: optionally use a different ``Grid_N.shp`` for each image pair
- **Custom file naming patterns**: define your own regex if files don't follow
  the ``Image_N`` / ``Label_N`` convention
- **CRS validation**: warns if image, label, and grid have mismatched coordinate
  reference systems
- **Configurable interpolation**: ``bilinear``, ``nearest``, ``bicubic``, ``lanczos``
  (labels always use nearest-neighbor)
- **Configurable compression**: ``deflate``, ``lzw``, ``none``
- **Detailed per-pair statistics**: patch counts, split breakdowns, and error
  reporting for each pair in batch mode
- **Metadata output**: per-pair JSON metadata files alongside extracted patches

**Internal**

- Improved UTF-8 handling for file paths with special characters (é, è, ê, etc.)
- Suppressed spurious GDAL/rasterio warnings in the output log


Version 1.0.0 — Initial Release
---------------------------------

**Core features**

- External Python environment architecture (Conda, venv, or system Python)
- Environment Setup Wizard with verification step
- Patch Extraction: single mode with georeferenced GeoTIFF output
- Model Training with configurable architectures, loss functions, and augmentation
- Prediction: sliding window inference with optional Gaussian blending

**Architectures (v1.0)**

- ``unet-dropout`` (built-in, no deps)
- Full SMP support: ``unet``, ``unet++``, ``manet``, ``linknet``, ``fpn``,
  ``pspnet``, ``pan``, ``deeplabv3``, ``deeplabv3+``
- Transformer models: ``segformer-b0..b5``, ``unetformer``,
  ``hrnet-w18/w32/w48``, ``swin-unet``

**Training features (v1.0)**

- 5 augmentation levels (none → extreme)
- 15+ loss functions with binary/multiclass auto-conversion
- AMP mixed precision
- Encoder freeze/unfreeze strategy
- LR warmup
- K-Fold cross-validation
- Per-class metrics

**Prediction features (v1.0)**

- Sliding window inference
- Gaussian blending
- Batch inference
- Confidence map output
- Auto-load result in QGIS
- Encoder override for old checkpoints

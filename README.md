# SemanticSeg4EO — QGIS Plugin

A QGIS plugin for semantic segmentation of Earth Observation imagery using deep learning.

> **Safe by design:** all processing runs in an external Python environment via subprocess. Nothing is installed into QGIS — your installation stays clean and stable.

---

## Features

- **Patch extraction** from large satellite images (single & batch mode, georeferenced GeoTIFF output)
- **Model training** — 20+ architectures: U-Net, DeepLabV3+, SegFormer, HRNet, SwinUNet, ConvNeXt…
- **Large-image prediction** with seamless reconstruction, Gaussian blending and confidence maps
- Binary and multi-class segmentation
- K-Fold cross-validation, class weighting, mixed-precision training

## Architecture

```
┌──────────────────────────────────────────────────┐
│  QGIS                                            │
│  ├── Plugin GUI (PyQt5 — no external deps)       │
│  └── Subprocess launcher                         │
└──────────────────┬───────────────────────────────┘
                   │  JSON params + subprocess
                   ▼
┌──────────────────────────────────────────────────┐
│  External Python (Conda / venv)                  │
│  PyTorch · rasterio · SMP · geopandas · …        │
└──────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install the plugin

Download the latest **`SemanticSeg4EO_QGIS.zip`** from the [Releases](../../releases) page.

In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP** → select the file → **Install**.

### 2. Create an external Python environment

```bash
# Conda (recommended)
conda create -n semanticseg4eo python=3.10 -y
conda activate semanticseg4eo

# Install PyTorch (pick ONE)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118    # CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121    # CUDA 12.1

# Install the rest
pip install -r requirements_external.txt
```

> See `requirements_external.txt` in this repo for the full dependency list.  
> **Always install PyTorch first** — the requirements file does not include it because the correct build depends on your hardware.

### 3. Point the plugin to your Python

Open SemanticSeg4EO in QGIS and click **Browse python…** in the status bar.  
Navigate to `python.exe` (Windows) or `bin/python` (Linux/macOS) inside your environment — done.

Alternatively, click **Configure Environment** for auto-detection of installed Conda/venv environments.

## Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Page | Description |
|------|-------------|
| [Environment Setup](docs/environment_setup.rst) | How to create, configure and verify the external environment |
| [Patch Extraction](docs/patch_extraction.rst) | Single & batch extraction, parameters, CRS validation |
| [Model Training](docs/model_training.rst) | Architectures, losses, augmentation, k-fold, advanced options |
| [Prediction](docs/prediction.rst) | Large-image inference, Gaussian blending, confidence maps |

## Repository Contents

```
.
├── SemanticSeg4EO_QGIS.zip        ← installable plugin (download from Releases)
├── requirements_external.txt       ← dependencies for the external environment
├── docs/                           ← full documentation (RST)
├── LICENSE
└── README.md
```

## Requirements

| Side | What you need |
|------|---------------|
| **QGIS** | QGIS ≥ 3.16 — nothing else |
| **External env** | Python ≥ 3.8, PyTorch ≥ 1.10, numpy < 2, rasterio, geopandas, opencv-python, tifffile, imagecodecs, segmentation-models-pytorch, scipy, scikit-learn, matplotlib, tqdm |
| **Optional** | `transformers` (SegFormer), `timm` (HRNet, SwinUNet, ConvNeXt) |

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Adrien Leguillou**  
Research Engineer — LETG, Université de Bretagne Occidentale  
📧 adrien.leguillou@univ-brest.fr

## Related

[SemanticSeg4EO standalone framework](https://github.com/aleguillou1/SemanticSeg4EO) — the same training and prediction scripts usable outside QGIS.

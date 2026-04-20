#  SemanticSeg4EO — QGIS Plugin

[![QGIS](https://img.shields.io/badge/QGIS-3.16+-green.svg)](https://qgis.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

> **A unified framework for semantic segmentation of Earth Observation imagery — directly inside QGIS.**

SemanticSeg4EO provides a complete **end-to-end pipeline**, from dataset preparation to large-scale inference, while keeping your QGIS environment clean and stable.

---

## Key Features

- 🔄 **End-to-End Workflow**  
  Patch extraction → Training → Large-scale inference

- 🧠 **State-of-the-Art Models**  
  20+ architectures supported:
  *U-Net, DeepLabV3+, SegFormer, HRNet, SwinUNet, ConvNeXt, ...*

- 🛰️ **Advanced Inference**
  - Gaussian blending for seamless mosaics  
  - Confidence map generation  

- 🎓 **Research-Grade Tools**
  - K-Fold cross-validation  
  - Class weighting  
  - Automatic Mixed Precision (AMP)

---

## 🧩 Architecture

The plugin cleanly separates GIS interaction from heavy deep learning workloads:

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


> 💡 **Stability by Design**  
> All heavy processing runs in an isolated Python environment → no dependency conflicts with QGIS.

---

## Getting Started

### 1. Install the Plugin

1. Download `SemanticSeg4EO.zip` from this repository  
2. Open QGIS  
3. Go to:  
   **Plugins → Manage and Install Plugins → Install from ZIP**  
4. Select the `.zip` file and click **Install**

> ℹ️ The full source code is available in `src/` for transparency and reproducibility.

---

### 2. Set Up the Python Environment

We recommend **Conda**:

```bash
# Create environment
conda create -n semanticseg4eo python=3.10 -y
conda activate semanticseg4eo

# Install PyTorch (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements_external.txt
```

or use Environnement.yaml (CPU) / Environnement_GPU (for GPU) : 

```bash
conda env create -f environment.yml (or environnement_GPU
conda activate semanticseg4eo
```

Important: Always install PyTorch first to match your specific hardware (CPU or CUDA version). Check pytorch.org for the correct command.

3. Connect the Plugin
Open SemanticSeg4EO in QGIS
Click "Browse python..." (bottom status bar)
Select:
python.exe (Windows)
bin/python (Linux/macOS)

✅ Optional: Use "Configure Environment" for automatic Conda detection.

Project Structure

```text
├── src/                          # Full Python source code
├── SemanticSeg4EO.zip            # QGIS plugin package
├── docs/                         # Documentation & guides
└── requirements_external.txt     # External environment dependencies
```

Citation

If you use this software in your research, please cite:

Le Guillou, A. (2026). SemanticSeg4EO: An Open-source unified framework and QGIS Plugin for Semantic Segmentation in Earth Observation. SoftwareX (Submitted).
Contact
Adrien Le Guillou - Research Engineer
LETG, Université de Bretagne Occidentale
📧 adrien.leguillou@univ-brest.fr



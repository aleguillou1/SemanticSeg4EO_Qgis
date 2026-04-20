# SemanticSeg4EO - QGIS Plugin

## Semantic Segmentation for Earth Observation Imagery

A QGIS plugin for semantic segmentation of satellite imagery using deep learning.

**⚠️ IMPORTANT: This plugin uses an EXTERNAL Python environment for processing. No PyTorch dependencies are installed in QGIS, which keeps your QGIS installation safe and stable.**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  QGIS                                                       │
│  ├── Plugin GUI (PyQt5 native - no external deps)           │
│  └── Subprocess launcher                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ JSON params + subprocess
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  External Python Environment (Conda or venv)                │
│  ├── PyTorch                                                │
│  ├── segmentation-models-pytorch                            │
│  ├── rasterio, geopandas, etc.                              │
│  └── Processing scripts                                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Step 1: Install the QGIS Plugin

1. Download `SemanticSeg4EO_QGIS.zip`
2. In QGIS: **Plugins** → **Manage and Install Plugins** → **Install from ZIP**
3. Select the ZIP file and click **Install**
4. Enable "SemanticSeg4EO" in the plugin list

### Step 2: Create the External Python Environment

The plugin will guide you through this with a wizard, but here are the manual steps:

#### Option A: Conda (Recommended)

```bash
# Create environment
conda create -n semanticseg4eo python=3.10 -y

# Activate
conda activate semanticseg4eo

# Install PyTorch (CPU version - smaller, works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR for GPU with CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for GPU with CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install rasterio tifffile geopandas opencv-python
pip install segmentation-models-pytorch
pip install numpy scipy scikit-learn matplotlib tqdm
```

#### Option B: Python venv

```bash
# Create environment
python3 -m venv ~/semanticseg4eo_env

# Activate (Linux/Mac)
source ~/semanticseg4eo_env/bin/activate

# Activate (Windows)
# ~/semanticseg4eo_env/Scripts/activate

# Install dependencies (same as above)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install rasterio tifffile geopandas opencv-python
pip install segmentation-models-pytorch
pip install numpy scipy scikit-learn matplotlib tqdm
```

### Step 3: Configure the Plugin

1. Open QGIS and click on **SemanticSeg4EO** in the toolbar or menu
2. Click **"⚙️ Configure Environment"**
3. Follow the wizard to point to your Conda or venv environment
4. Click **"Verify Environment"** to check everything works

## Usage

### Tab 1: Patch Extraction

Extract training patches from large satellite images. Two modes are available:

#### Single Mode

For one image + one label + one grid:

1. **Satellite Image**: Multi-band GeoTIFF (e.g., Sentinel-2)
2. **Labels/Mask**: Single-band GeoTIFF with class values
3. **Grid (Shapefile)**: Polygon grid defining patch locations
4. **Output Directory**: Where patches will be saved

#### Batch Mode

For multiple image/label pairs at once:

1. **Data Directory**: Folder containing your files with naming convention:
   ```
   data_folder/
   ├── Image_1.tif
   ├── Label_1.tif
   ├── Image_2.tif
   ├── Label_2.tif
   ├── Grid_1.shp    ← optional per-image grid
   ├── Grid_2.shp    ← optional per-image grid
   └── ...
   ```
2. **Shared Grid**: A single grid applied to all images (unless per-image grids are used)
3. **Per-image grids** (optional): Check this to use `Grid_N.shp` matching each `Image_N.tif`
4. **Custom patterns** (advanced): Define your own regex if files don't follow `Image_N` / `Label_N` naming

Batch mode automatically discovers and pairs files, adds a prefix (`img1_`, `img2_`, ...) to avoid filename collisions, and reports per-pair statistics.

#### Parameters (both modes)

- **Patch Size**: Typically 224 or 256 pixels
- **Image Channels**: Number of bands in your image
- **Interpolation**: `bilinear` (default for images), `nearest`, `bicubic`, or `lanczos`. Labels always use nearest-neighbor.
- **Compression**: `deflate` (default), `lzw`, or `none`
- **Validate CRS**: Warns if image, label, and grid have mismatched coordinate systems
- **Train/Val/Test Ratio**: Data split proportions (must sum to 1.0)

All output patches are **georeferenced GeoTIFFs** that can be loaded directly in QGIS.

### Tab 2: Model Training

Train deep learning segmentation models:

**Mode:**
- **Binary**: Two classes (e.g., water vs. land, building vs. not-building)
- **Multi-class**: Multiple land cover classes (set number of classes)

**Architecture:**
- `unet-dropout`: Simple U-Net (works without SMP)
- `unet`, `unet++`: Standard U-Net variants
- `deeplabv3`, `deeplabv3+`: DeepLab models
- `fpn`: Feature Pyramid Network
- `segformer-b0` to `segformer-b5`: SegFormer (requires `transformers`)
- `hrnet-w18/w32/w48`: High-Resolution Net (requires `timm`)
- `swin-unet`: Swin Transformer U-Net
- `unetformer`: U-Net + Transformer hybrid

**Parameters:**
- **Epochs**: 50-200 typically
- **Batch Size**: 4-8 (lower if GPU memory issues)
- **Learning Rate**: 0.0003 for pretrained, 0.001 from scratch
- **Device**: `cuda` for GPU, `cpu` for CPU

### Tab 3: Prediction

Apply trained models to new satellite images:

1. Select your trained model (`.pth` file)
2. Select input image to segment
3. Choose output location
4. Configure parameters:
   - **Patch Size**: 512-1024 (larger = faster but more memory)
   - **Overlap**: 128-256 (larger = smoother edges)
   - **Threshold**: For binary mode (0.5 default)
   - **Gaussian blending**: Eliminates grid artifacts at patch boundaries

## Data Preparation

### Masks Format

**Binary mode:**
- Values: 0 (background), 1 (foreground)
- Example: Water detection → 0=land, 1=water

**Multi-class mode:**
- Values: 0 to N-1 (where N is number of classes)
- Example: Land cover → 0=water, 1=forest, 2=urban, 3=agriculture, etc.

### Grid Shapefile

Create a grid of polygons covering your study area. Each polygon becomes one training patch.

You can create a grid in QGIS:
**Vector** → **Research Tools** → **Create Grid**

## Troubleshooting

### "Environment not configured"

The plugin can't find the external Python environment.
1. Click "Configure Environment"
2. Make sure you created the Conda env or venv
3. Verify the path is correct

### "torch not found"

PyTorch isn't installed in the external environment:
```bash
conda activate semanticseg4eo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Batch mode: "No matching pairs found"

Make sure your files follow the naming convention (`Image_1.tif`, `Label_1.tif`, ...) or define custom patterns in the advanced options.

### CRS mismatch warning

The image, label, and grid have different coordinate systems. While extraction will proceed, results may be incorrect. Reproject your data to a common CRS first.

### CUDA out of memory

- Reduce batch size (2 or 1)
- Use smaller patch size
- Switch to CPU for prediction

### Process hangs

- Check the log output for errors
- Try with a smaller test image first
- Verify input file formats are correct

## Requirements

### QGIS Side (no special requirements)
- QGIS 3.16+
- That's it! No pip install needed.

### External Environment
- Python 3.8+
- PyTorch 1.10+
- rasterio
- geopandas
- opencv-python
- segmentation-models-pytorch (optional but recommended)
- numpy, scipy, scikit-learn, matplotlib
- transformers, timm (optional, for modern architectures)

## Why External Environment?

QGIS uses a modified Python environment with specific versions of GDAL, numpy, and other libraries. Installing PyTorch and its dependencies directly in QGIS can:

- Break GDAL/rasterio due to version conflicts
- Corrupt the QGIS Python environment
- Cause crashes and instability

By using an external environment:
- QGIS remains stable
- You can update PyTorch independently
- Easy to manage GPU/CPU versions
- Isolates all deep learning dependencies

## License

MIT License - see LICENSE file

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/semanticseg4eo/qgis-plugin/issues)
- Check existing issues for solutions

.. _faq:

Frequently Asked Questions
===========================

.. contents:: On this page
   :local:
   :depth: 2


General Questions
------------------

**Why does the plugin use an external environment instead of installing PyTorch in QGIS?**

QGIS uses a customised Python environment with pinned versions of GDAL, PyQt5, and
numpy. Installing PyTorch and its dependencies directly into QGIS commonly causes
version conflicts that break QGIS functionality or cause crashes.
By using an external environment, your QGIS installation remains stable, and you can
freely manage PyTorch versions (CPU/GPU, CUDA versions) independently.

See :doc:`../getting_started/environment_setup` for the full explanation.


**Does the plugin work on Linux, macOS, and Windows?**

Yes. The plugin is designed to work on all three platforms. Environment isolation
is handled differently per platform (Windows requires filtering OSGeo4W paths from
``PATH``; Linux/macOS require filtering QGIS paths), but this is handled automatically.


**Can I use the plugin with QGIS 3.16? 3.22? 3.34?**

The plugin requires QGIS 3.16 or later. It has been developed against the PyQt5 API
that has been stable since 3.16. No processing features depend on a QGIS version
beyond 3.16.


**Do I need a GPU?**

No. All three modules (extraction, training, prediction) can run on CPU.
A GPU is strongly recommended for training and prediction as it is typically
10–50× faster. Any NVIDIA GPU with at least 4GB VRAM is sufficient for most
use cases.


**Which CUDA version should I install?**

Run ``nvidia-smi`` in a terminal. The CUDA version shown is the **maximum** version
supported by your driver. Choose the highest PyTorch CUDA wheel that does not exceed
your driver's CUDA version:

- Driver CUDA ≥ 12.1 → install PyTorch with ``--index-url .../cu121``
- Driver CUDA 11.6–12.0 → install PyTorch with ``--index-url .../cu118``
- Older driver or no GPU → install the CPU version


Patch Extraction Questions
---------------------------

**What format should my satellite image be in?**

Multi-band georeferenced GeoTIFF (``.tif``). Any number of bands is supported.
Ensure the file has a valid CRS (coordinate reference system) for best results.


**What format should my label mask be in?**

Single-band GeoTIFF with integer pixel values:

- Binary: 0 = background, 1 = foreground
- Multi-class: 0, 1, 2, ... (N-1) = class indices

The spatial extent and CRS should match the image. Use nearest-neighbour resampling
when creating or reprojecting label rasters to avoid introducing spurious class values.


**How do I create a grid shapefile in QGIS?**

Go to **Vector → Research Tools → Create Grid**.
Set the grid type to *Rectangle (polygon)*, set the extent to your study area,
and set the horizontal and vertical spacing to the desired cell size in map units.

For example, if your image is at 10m resolution and you want 256×256 pixel patches:
cell width = cell height = 256 × 10 = 2560m.


**My labels have a NoData value (e.g., 255). Will this cause problems?**

Patches where the label mask is entirely NoData are typically excluded automatically
during training (the DataLoader skips masks with no valid pixels). However, if your
NoData value (e.g., 255) is within the class range, it may be interpreted as a valid
class. Reclassify your labels to remove NoData values before extraction.


**What happens if a patch falls outside the image extent?**

Patches that fall partially or entirely outside the image extent are handled gracefully
— the out-of-bounds region is filled with zeros. These patches are still saved
and included in the dataset.


Training Questions
-------------------

**Should I use pretrained weights?**

Almost always yes, for SMP-based architectures. Pretrained ImageNet encoders provide
a much better starting point than random initialisation, even for non-RGB multispectral
data. The first convolution layer is adapted to match your number of input channels
by averaging or repeating the pretrained weights.

For ``unet-dropout`` (built-in), no pretrained weights are available.


**How many patches do I need for good results?**

This depends on the complexity of your task:

- Minimum viable: ~200–500 patches (use aggressive augmentation)
- Good starting point: 1,000–5,000 patches
- Comfortable: 5,000–20,000 patches
- Diminishing returns above ~50,000 patches

Patches should cover the diversity of conditions present in your test area.


**The model predicts everything as one class. What should I do?**

This is typically a class imbalance problem. Try:

1. Enable **Class weights** in the optimization group
2. Switch to a loss function designed for imbalance (``binary_focal_dice``,
   ``binary_tversky``)
3. Use ``aggressive`` or ``extreme`` augmentation (minority oversampling is enabled)
4. Check your labels are correctly formatted (correct class values)


**What is K-Fold cross-validation and when should I use it?**

K-Fold CV trains K models on different subsets of your data and reports
mean/std metrics across all folds. Use it when:

- Your dataset is small and a single train/val split may not be representative
- You need statistically robust performance estimates (e.g., for publication)
- You want to compare architectures fairly

K-Fold takes K× longer than standard training, so it is not recommended
for rapid experimentation.


**Where is the trained model saved?**

The best model (by validation IoU) is saved as ``best_model.pth`` in the output
directory. Training curves are saved as ``training_curves.png`` in the same folder.


Prediction Questions
---------------------

**Why is the output prediction blurry at patch boundaries?**

Make sure **Gaussian blending** is enabled (it is by default). This applies a
2D Gaussian weight to each patch before merging, smoothing transitions.


**Can I run prediction on an image with a different resolution than training data?**

Yes, but results may degrade. The model was trained on patches at a specific
pixel resolution. Feeding it imagery at a different GSD (ground sampling distance)
will effectively change the apparent object sizes. For best results, resample the
input image to the same resolution as your training data before prediction.


**Can I save the raw probability output (not just the final class map)?**

Yes. Check **Save confidence map** in the Prediction tab. A second GeoTIFF with
the suffix ``_confidence.tif`` is saved alongside the classification output,
containing floating-point probability values.


**The prediction result is not automatically loaded in QGIS. Why?**

Make sure **Add result to QGIS** is checked in the Model Overrides group of the
Prediction tab. This option is enabled by default.
Also check that the output path is a valid, writable location.


**Can I predict with a model trained outside the plugin?**

Yes, as long as the architecture and checkpoint format are compatible.
The checkpoint must be a standard PyTorch ``state_dict`` saved with ``torch.save()``.
You may need to use the **Encoder override** option to specify the correct encoder
if the checkpoint does not include metadata.

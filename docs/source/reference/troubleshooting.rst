.. _troubleshooting:

Troubleshooting
===============

This page covers the most common errors and how to resolve them.

.. contents:: On this page
   :local:
   :depth: 2


Environment Issues
-------------------

"Environment not configured"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** The environment status bar shows *"Not configured"* and processing
does not start.

**Solution:**

1. Click **Configure Environment** in the status bar.
2. Follow the Environment Setup Wizard.
3. Make sure you have created the Conda environment or venv *before* configuring the plugin.

See :doc:`../getting_started/environment_setup` for step-by-step instructions.


"Python executable not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** The wizard reports it cannot find the Python executable.

**Solutions:**

- **Conda**: Make sure the environment name is correct and matches what you created
  with ``conda create -n <name>``. The wizard searches standard Conda installation paths.
  If your Conda is installed in a non-standard location, use the *Browse* button to
  locate ``python.exe`` (Windows) or ``python`` (Linux/macOS) manually.
- **venv**: Check that the path in the wizard points to the root of the virtual
  environment folder (the folder that contains ``Scripts/`` on Windows or ``bin/`` on
  Linux/macOS).
- **System Python**: Check that the path points directly to the ``python`` or
  ``python3`` executable.


"torch not found" during verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Verification shows ``✗ torch: NOT FOUND``.

**Solution:** Activate the environment and install PyTorch:

.. code-block:: bash

   # Conda
   conda activate semanticseg4eo
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # venv
   source ~/semanticseg4eo_env/bin/activate  # Linux/macOS
   # or
   C:\semanticseg4eo_env\Scripts\activate     # Windows
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Then re-run **Verify Environment** in the wizard.


"segmentation_models_pytorch not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Verification shows SMP not found. Architectures like ``unet``,
``deeplabv3+``, etc. are unavailable.

**Solution:**

.. code-block:: bash

   conda activate semanticseg4eo
   pip install segmentation-models-pytorch


"transformers / timm not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** SegFormer, HRNet, or SwinUNet throw import errors.

**Solution:**

.. code-block:: bash

   conda activate semanticseg4eo
   pip install transformers timm


Patch Extraction Issues
------------------------

"No matching pairs found" (batch mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Batch extraction reports ``No matching pairs found`` and exits.

**Solutions:**

- Check that your files follow the expected naming convention:
  ``Image_1.tif``, ``Label_1.tif``, ``Image_2.tif``, ``Label_2.tif``, ...
- The match is **case-insensitive**, so ``image_1.tif`` also works.
- If your files use a different naming scheme, expand **Advanced Options** in the
  batch mode group and enter custom regex patterns with a capture group for the
  numeric index. Example: ``survey_(\d+)_rgb`` for files like ``survey_01_rgb.tif``.


CRS mismatch warning
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** The log shows a ``WARNING: CRS mismatch!`` message.

**Explanation:** The image, label, and/or grid files have different coordinate
reference systems. Extraction continues but results may be spatially misaligned.

**Solution:** Reproject all inputs to the same CRS before running extraction.

In QGIS, use **Raster → Projections → Warp (Reproject)** for rasters, or
**Vector → Data Management Tools → Reproject Layer** for the grid shapefile.

Alternatively, use GDAL:

.. code-block:: bash

   gdalwarp -t_srs EPSG:32631 input_label.tif label_reprojected.tif
   ogr2ogr -t_srs EPSG:32631 grid_reprojected.shp grid.shp


"Ratios must sum to 1.0"
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Extraction does not start; an error about ratios is shown.

**Solution:** Adjust the train, val, and test ratio spinboxes so they sum
to a value between 0.95 and 1.05. The running total is shown next to the fields.
Example: 0.70 + 0.20 + 0.10 = 1.00 ✓


Training Issues
----------------

"CUDA out of memory"
~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Training crashes with a ``RuntimeError: CUDA out of memory`` message.

**Solutions (try in order):**

1. Reduce **Batch Size** to 2 or 1
2. Reduce **Patch Size** during extraction (smaller patches)
3. Switch to a lighter architecture (e.g., ``unet`` with ``resnet18`` instead of
   ``resnet101``)
4. Enable **Mixed Precision (AMP)** to reduce VRAM usage
5. Switch **Device** to ``cpu`` (slower but no VRAM limit)


Model doesn't converge (loss doesn't decrease)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Possible causes and solutions:**

- **Learning rate too high**: Try reducing LR by 10× (e.g., from 0.001 to 0.0001)
- **Learning rate too low**: Try increasing LR slightly
- **Wrong mode**: Make sure *Mode* (binary/multiclass) matches your label values
- **Wrong number of classes**: In multiclass mode, check *Classes* matches the
  number of unique values in your labels
- **Bad data normalization**: The plugin normalizes images to [0,1]; check your
  images don't have extreme outlier values or completely black/white regions


Validation loss increases after a few epochs (overfitting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

- Increase augmentation level (e.g., from ``basic`` to ``advanced``)
- Increase dropout rate (e.g., from 0.3 to 0.5)
- Enable early stopping with adequate patience
- Reduce model complexity (simpler architecture or encoder)
- Collect more training data


Training hangs with no output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

- Check the external environment is correctly configured (run **Verify Environment**)
- Try a very small test dataset first (e.g., 10 patches)
- Check disk space — the output directory must have write access
- On Windows, check that no antivirus is blocking the subprocess
- Click **Detach Log** to see the full output in a separate window


Prediction Issues
------------------

"Model architecture mismatch"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Prediction fails with an error about tensor size or architecture.

**Cause:** The checkpoint was saved with a different architecture or number
of channels than what is currently configured.

**Solution:**

- Make sure the architecture selected during prediction matches the one used
  for training (this is stored in the checkpoint metadata and used automatically)
- If the checkpoint is old and lacks metadata, manually set the **Encoder**
  override to match the encoder used during training
- Verify **Input Channels** matches the image and training data


Grid artifacts in output (no Gaussian blending)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** The output segmentation shows a regular grid pattern.

**Solution:** Enable **Gaussian blending** in the Prediction tab.
This is enabled by default but may have been unchecked.


Output is all one class
~~~~~~~~~~~~~~~~~~~~~~~~

**Possible causes:**

- **Threshold too high/low** (binary mode): Adjust the threshold slider.
  If all pixels are classified as background, lower the threshold.
  If all are foreground, raise it.
- **Wrong mode**: Make sure the prediction mode (binary/multiclass) matches
  how the model was trained
- **Untrained model**: The model may not have converged — check training metrics


Slow prediction
~~~~~~~~~~~~~~~~

**Solutions:**

- Switch **Device** to ``cuda`` if you have a GPU
- Increase inference **Batch Size** (e.g., 4–8 on GPU)
- Increase **Patch Size** (larger patches = fewer forward passes)


Windows-Specific Issues
------------------------

Qt DLL conflicts
~~~~~~~~~~~~~~~~~

**Symptom:** Processing crashes immediately on Windows with DLL-related errors.

**Cause:** QGIS Qt libraries conflict with PyTorch's Qt libraries on the PATH.

**Solution:** The plugin automatically filters QGIS paths from the subprocess
environment. If you still encounter this issue:

- Make sure you are using Conda (not a system Python that shares QGIS paths)
- Reinstall the plugin to get the latest environment isolation fixes


Paths with special characters (é, è, ê, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Errors containing ``UnicodeDecodeError`` or ``can't decode byte 0xe9``.

**Explanation:** The plugin forces UTF-8 encoding (``PYTHONUTF8=1``) in the
subprocess environment. This should handle French and other non-ASCII characters
in file paths automatically.

If you still encounter this, try moving your data to a path without special characters
as a workaround.


Getting More Help
------------------

If your issue is not covered here:

1. Check the **Output & Progress** log for the full error message
2. Click **Detach Log** for a more readable view
3. Search or open an issue on the
   `GitHub Issues page <https://github.com/aleguillou1/SemanticSeg4EO/issues>`_

When reporting an issue, please include:

- Your operating system and QGIS version
- Your environment type (Conda / venv / system) and Python version
- The full error message from the output log
- The architecture and parameters you were using

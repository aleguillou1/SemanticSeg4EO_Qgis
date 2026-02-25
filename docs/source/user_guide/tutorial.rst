.. _tutorial:

Tutorial: Land Cover Classification from BD Ortho IRC
=======================================================

This step-by-step tutorial walks you through a complete workflow — from raw data
to a classified map — using SemanticSeg4EO. It is designed as a starting point to
help you get familiar with the tool.

All data used and produced in this tutorial (raw imagery, labels, grid, extracted
patches, trained model and predictions) are freely available on Zenodo:

.. TODO:: Insert Zenodo link below once published.

📦 **Dataset & results:** `https://doi.org/10.5281/zenodo.XXXXXXX <https://doi.org/10.5281/zenodo.XXXXXXX>`_

.. contents:: In this tutorial
   :local:
   :depth: 2


Input Data
-----------

This tutorial uses the following inputs:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Input
     - Description
   * - **Satellite image**
     - BD Ortho at 20 cm resolution, IRC (Infrared–Red–Green) composite — 3 bands
   * - **Labels**
     - Land cover mask derived from OCS GE, 5 classes (values 0 to 4)
   * - **Grid**
     - Polygon shapefile defining the patch locations over the study area

.. figure:: ../_images/tuto_input_data.png
   :alt: Input data overview
   :align: center
   :width: 90%

   *Overview of the input data loaded in QGIS: BD Ortho IRC image, OCS GE labels,
   and the extraction grid.*


Step 1 — Patch Extraction
---------------------------

Open SemanticSeg4EO and navigate to the **Patch Extraction** tab.

Select the input files:

- **Satellite Image:** the BD Ortho IRC GeoTIFF
- **Labels / Mask:** the OCS GE raster (5 classes, 0–4)
- **Grid:** the polygon shapefile

Set the extraction parameters:

.. list-table::
   :widths: 35 30
   :header-rows: 0

   * - Patch size
     - ``224``
   * - Image channels
     - ``3`` (IRC)
   * - Data type
     - ``int16``
   * - Train / Val / Test ratio
     - 0.75 / 0.15 / 0.10

.. figure:: ../_images/tuto_patch_extraction_params.png
   :alt: Patch extraction parameters
   :align: center
   :width: 80%

   *Patch extraction tab with the parameters used in this tutorial.*

Click **Run Extraction**. The plugin extracts **1 435 patches** into the output
directory, organised as follows:

.. code-block:: text

   output_dir/
   └── Patch/
       ├── train/
       │   ├── images/    (1 076 patches)
       │   └── labels/
       ├── validation/
       │   ├── images/    (215 patches)
       │   └── labels/
       └── test/
           ├── images/    (144 patches)
           └── labels/

Each patch is a georeferenced GeoTIFF that can be loaded directly in QGIS for
visual inspection.


Step 2 — Model Training
-------------------------

Navigate to the **Model Training** tab.

Select the dataset:

- **Dataset directory:** the ``Patch/`` folder created in Step 1

.. figure:: ../_images/tuto_dataset_folder.png
   :alt: Dataset folder structure
   :align: center
   :width: 70%

   *The Patch folder containing train, validation and test splits.*

Configure the training:

.. list-table::
   :widths: 35 30
   :header-rows: 0

   * - Mode
     - ``Multi-class``
   * - Number of classes
     - ``5``
   * - Architecture
     - ``segformer-b2``
   * - Epochs
     - as needed (see training curves)
   * - Device
     - ``cuda`` (GPU) or ``cpu``

.. figure:: ../_images/tuto_training_params.png
   :alt: Training parameters
   :align: center
   :width: 80%

   *Model training tab configured for SegFormer-B2, multi-class (5 classes).*

Click **Run Training**. The plugin launches the training in the external
environment. Progress and metrics are displayed in the log panel.

Once training is complete, the output directory contains:

.. code-block:: text

   trained_models/
   ├── model_best_iou.pth           ← best IoU checkpoint
   ├── model_best_loss.pth          ← best validation loss checkpoint
   ├── model_final_model.pth        ← final model weights
   ├── model_metrics.json           ← detailed metrics history
   └── model_training_plot.png      ← training curves

.. figure:: ../_images/tuto_training_results.png
   :alt: Training results
   :align: center
   :width: 90%

   *Training results: metrics summary and training curves.*

.. figure:: ../_images/tuto_training_plot.png
   :alt: Training curves
   :align: center
   :width: 70%

   *Loss and IoU curves over the training epochs.*


Step 3 — Prediction on an Independent Image
----------------------------------------------

To evaluate the generalisation ability of the model, we apply it to an
**independent image** that was not used during training. This image has the same
characteristics as the training data: BD Ortho at 20 cm resolution, IRC composite.

Navigate to the **Prediction** tab and configure:

- **Model:** the ``.pth`` file from Step 2 (e.g. ``model_best_iou.pth``)
- **Input image:** the independent BD Ortho IRC scene
- **Output:** path for the prediction GeoTIFF

.. figure:: ../_images/tuto_prediction_params.png
   :alt: Prediction parameters
   :align: center
   :width: 80%

   *Prediction tab: model, independent input image and output path.*

Click **Run Prediction**. The plugin performs patch-based inference with seamless
reconstruction.

.. figure:: ../_images/tuto_prediction_result.png
   :alt: Prediction result
   :align: center
   :width: 90%

   *Prediction result on the independent image, displayed in QGIS alongside the
   original BD Ortho IRC.*


Summary
---------

This tutorial covered the three main steps of a SemanticSeg4EO workflow:

1. **Patch extraction** — preparing a training dataset from large satellite imagery
2. **Model training** — training a SegFormer-B2 on 5 land cover classes
3. **Prediction** — applying the trained model to a new, unseen image

The numerical scores (IoU, F1, accuracy) provide a quantitative assessment of model
performance, while the visual prediction on an independent scene demonstrates the
model's ability to generalise beyond the training area.

.. note::
   All data (BD Ortho IRC images, OCS GE labels, grid shapefile, extracted patches,
   trained model, and predictions) are freely available for download on Zenodo so
   that you can reproduce this tutorial exactly:

   📦 `https://doi.org/10.5281/zenodo.XXXXXXX <https://doi.org/10.5281/zenodo.XXXXXXX>`_

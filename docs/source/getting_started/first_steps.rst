.. _first_steps:

First Steps
===========

Once the plugin is installed and your environment is configured, this page
walks you through the complete SemanticSeg4EO workflow from raw satellite
imagery to a segmentation map.

.. contents:: On this page
   :local:
   :depth: 2


Opening the Plugin
------------------

You can open the SemanticSeg4EO dialog in two ways:

- **Toolbar**: Click the SemanticSeg4EO icon in the QGIS toolbar
- **Menu**: Go to **Raster → &SemanticSeg4EO → SemanticSeg4EO**

The main dialog opens with three tabs:

- **Patch Extraction** — Slice large images into training patches
- **Model Training** — Train a segmentation model on your patches
- **Prediction** — Apply your trained model to new imagery

.. figure:: ../_images/main_dialog_overview.png
   :alt: Main dialog overview
   :align: center
   :width: 90%

   *The main SemanticSeg4EO dialog with its three workflow tabs.*

.. .. [INSERT SCREENSHOT: Full main dialog — all three tabs visible]


The Output & Progress Panel
-----------------------------

At the bottom of the dialog is the **Output & Progress** panel. It shows:

- A scrolling log of all messages from the external Python process
- A progress bar updated in real time
- A **Cancel** button to interrupt processing at any time
- A **Detach Log** button to open the log in a separate resizable window

The panel can be resized by dragging the horizontal splitter between the tabs
and the output area.


Typical Workflow
-----------------

A typical SemanticSeg4EO workflow has three stages:

.. code-block:: text

   1. PREPARE DATA
      Large satellite GeoTIFF (image) + Label GeoTIFF (mask) + Grid Shapefile
              ↓
   2. EXTRACT PATCHES  (Tab 1)
      Georeferenced patch GeoTIFFs organized in train/val/test folders
              ↓
   3. TRAIN MODEL  (Tab 2)
      .pth checkpoint file saved to output directory
              ↓
   4. PREDICT  (Tab 3)
      Segmentation GeoTIFF loaded automatically in QGIS

Preparing Your Data
-------------------

Before running the plugin, you need three inputs:

**1. Satellite Image (GeoTIFF)**

A multi-band georeferenced GeoTIFF. For example:

- Sentinel-2 (10–60m resolution, 4–12 bands)
- SPOT, Pléiades, or any other sensor
- Any number of bands (configured via *Input Channels*)

**2. Label / Mask (GeoTIFF)**

A single-band GeoTIFF with integer class values:

- **Binary mode**: pixel values are ``0`` (background) or ``1`` (class of interest)
- **Multi-class mode**: pixel values are ``0`` to ``N-1`` (one value per class)

The label raster must cover the same geographic extent as (or overlap with) your image.

**3. Grid Shapefile**

A polygon grid defining patch locations. Each polygon becomes one patch.

.. tip::
   To create a grid in QGIS, go to **Vector → Research Tools → Create Grid**.
   Set the grid extent to your study area and the cell size to your desired patch size
   in map units (e.g., 256 × 10m = 2560m cells for a 256-pixel patch at 10m resolution).

Your data folder can be organized however you like. For batch extraction, the recommended
convention is:

.. code-block:: text

   data_folder/
   ├── Image_1.tif          ← satellite image
   ├── Label_1.tif          ← label/mask
   ├── Image_2.tif
   ├── Label_2.tif
   ├── Grid_1.shp           ← optional per-image grid
   ├── Grid.shp             ← shared grid (used for all images)
   └── ...


Checking the Environment Before Processing
-------------------------------------------

Before clicking **Start Extraction**, **Start Training**, or running **Prediction**,
always check the environment status bar at the top of the dialog.

- 🟢 **Green / configured**: environment is ready
- 🔴 **Not configured**: click **Configure Environment** first

Processing will not start if no environment is configured.


Cancelling a Process
---------------------

Any running process can be cancelled at any time by clicking the **Cancel** button
in the Output & Progress panel. The external Python process will be terminated gracefully.

If a process is running when you try to close the dialog, QGIS will ask for
confirmation before cancelling and closing.


Next Steps
----------

- :doc:`../user_guide/patch_extraction` — Detailed patch extraction options
- :doc:`../user_guide/model_training` — All training parameters explained
- :doc:`../user_guide/prediction` — Running predictions on new images

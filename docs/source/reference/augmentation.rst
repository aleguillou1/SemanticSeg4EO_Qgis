.. _augmentation:

Data Augmentation
=================

SemanticSeg4EO implements a custom augmentation pipeline specifically designed
for multi-channel geospatial imagery. Five progressive levels are available,
each building on the previous one.

.. contents:: On this page
   :local:
   :depth: 2


Augmentation Levels
--------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Level
     - Description
   * - ``none``
     - No augmentation. Patches are fed to the model as-is.
       Use when your dataset is already large and diverse, or for debugging.
   * - ``basic``
     - Geometric only: horizontal flip, vertical flip, 90° rotation.
       Safe starting point for most datasets. **Recommended default.**
   * - ``advanced``
     - ``basic`` + scaling, brightness, contrast, gamma, per-channel noise.
       Adds radiometric variability to handle sensor differences and
       atmospheric conditions.
   * - ``aggressive``
     - ``advanced`` + random crop, elastic deformation, Gaussian blur,
       Gaussian noise, coarse dropout (Cutout), channel dropout,
       minority class oversampling.
       Stronger regularization for small datasets.
   * - ``extreme``
     - ``aggressive`` + free rotation (±45°), grid distortion,
       MixUp, CutMix, channel shuffle (if > 3 bands).
       Maximum regularization. Use only if the model overfits severely.


Detailed Transform Catalogue
------------------------------

Geometric Transforms
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Transform
     - Level
     - Description
   * - **Horizontal flip**
     - basic+
     - Mirror image and mask left-right (p=0.5)
   * - **Vertical flip**
     - basic+
     - Mirror image and mask top-bottom (p=0.5)
   * - **90° rotation**
     - basic+
     - Rotate by 90°, 180°, or 270° (p=0.5)
   * - **Scale**
     - advanced+
     - Random zoom in/out (0.85–1.15×, p=0.4)
   * - **Random crop**
     - aggressive+
     - Random crop and resize to patch size (scale 0.7–1.0, p=0.4)
   * - **Elastic deformation**
     - aggressive+
     - Local deformations simulating terrain distortion
       (alpha=120, sigma=12, p=0.25)
   * - **Free rotation**
     - extreme
     - Rotate by any angle ±45° (p=0.3)
   * - **Grid distortion**
     - extreme
     - Grid-based spatial distortion (limit=0.3, p=0.2)

Radiometric Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

These transforms apply only to the image, not the label mask.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Transform
     - Level
     - Description
   * - **Brightness**
     - advanced+
     - Multiply all pixel values by a factor in [0.85, 1.15] (p=0.4)
   * - **Contrast**
     - advanced+
     - Adjust contrast around the mean (±0.15, p=0.4)
   * - **Gamma**
     - advanced+
     - Apply gamma correction (0.85–1.15, p=0.3)
   * - **Per-channel noise**
     - advanced+
     - Add Gaussian noise to a single random channel (std=0.02, p=0.2)
   * - **Gaussian noise**
     - aggressive+
     - Add Gaussian noise to all channels (std=0.025, p=0.25)
   * - **Gaussian blur**
     - aggressive+
     - Smooth image with Gaussian kernel (sigma 0.5–2.0, p=0.25)

Dropout and Cutout
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Transform
     - Level
     - Description
   * - **Coarse dropout**
     - aggressive+
     - Zero out up to 5 rectangular regions (max 32×32 px, p=0.2).
       Simulates cloud shadows or missing data.
   * - **Channel dropout**
     - aggressive+
     - Zero out 1–2 random channels (p=0.15).
       Only when ``in_channels > 2``. Simulates missing spectral bands.

MixUp / CutMix
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Transform
     - Level
     - Description
   * - **MixUp**
     - extreme
     - Linearly blend two training samples and their labels (alpha=0.3, p=0.2)
   * - **CutMix**
     - extreme
     - Replace a random rectangular region with content from another sample
       (alpha=1.0, p=0.2)
   * - **Channel shuffle**
     - extreme
     - Randomly permute band order (p=0.1). Only when ``in_channels > 3``.

Minority Class Oversampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At ``aggressive`` and ``extreme`` levels, minority class oversampling is enabled:
patches containing the minority class (class index 1 by default) are identified
and augmented with extra transforms to improve model sensitivity to rare classes.


Choosing an Augmentation Level
--------------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Scenario
     - Recommended level
   * - Large diverse dataset (>10,000 patches)
     - ``none`` or ``basic``
   * - Medium dataset (1,000–10,000 patches)
     - ``basic`` or ``advanced``
   * - Small dataset (<1,000 patches)
     - ``advanced`` or ``aggressive``
   * - Very small dataset (<200 patches)
     - ``aggressive`` or ``extreme``
   * - Model overfits quickly
     - Increase by one level
   * - Model underfits or unstable training
     - Decrease by one level

.. tip::
   Start with ``basic`` augmentation and increase if you observe overfitting
   (widening gap between training and validation loss after early epochs).

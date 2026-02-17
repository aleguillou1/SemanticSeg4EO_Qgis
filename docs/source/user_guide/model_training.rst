.. _model_training:

Model Training
==============

The **Model Training** tab lets you configure and launch training of a deep learning
segmentation model on the patches extracted in the previous step.
Parameters are organized into **Basic** and **Advanced** sub-tabs.

.. figure:: ../_images/training_tab_overview.png
   :alt: Model Training tab
   :align: center
   :width: 90%

   *The Model Training tab, showing Basic Parameters.*

.. .. [INSERT SCREENSHOT: Model Training tab — Basic Parameters sub-tab]

.. contents:: On this page
   :local:
   :depth: 2


Dataset Configuration
----------------------

At the top of the tab, configure the dataset before choosing any model parameters.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - **Dataset Directory**
     - Path to the folder created by Patch Extraction.
       Must contain ``train/``, ``val/`` (and optionally ``test/``) subfolders.
   * - **Mode**
     - ``binary`` — two-class segmentation (background vs. one class).
       ``multiclass`` — multiple land cover classes.
   * - **Classes**
     - Number of output classes. Only active in ``multiclass`` mode.
       Set this to the number of unique integer values in your label masks.


Basic Parameters
-----------------

The **Basic Parameters** sub-tab is the main interface for most users.
Parameters are split into two columns: **Model** (left) and
**Optimization + Regularization** (right).

.. figure:: ../_images/training_basic_subtab.png
   :alt: Basic Parameters sub-tab
   :align: center
   :width: 90%

   *Basic Parameters: model selection on the left, optimization on the right.*

.. .. [INSERT SCREENSHOT: Basic Parameters sub-tab]


Model Group
~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Architecture**
     - ``unet-dropout``
     - The segmentation model architecture. See :doc:`../reference/architectures`
       for the full list and requirements.
   * - **Encoder**
     - ``resnet34``
     - Backbone encoder for SMP-based architectures.
       Not shown for self-contained architectures (SegFormer, HRNet, SwinUNet, unet-dropout).
       See :doc:`../reference/architectures`.
   * - **Input Channels**
     - 4
     - Number of bands in your image patches. Must match the actual patch data.
       For Sentinel-2 RGB: 3, for 4-band (RGB+NIR): 4.
   * - **ImageNet pretrained**
     - unchecked
     - Use pretrained ImageNet weights for the encoder (strongly recommended
       when using SMP architectures). Not available for ``unet-dropout``.


Training Group
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Epochs**
     - 100
     - Maximum number of training epochs. With early stopping enabled,
       training may stop before this limit.
       Typical range: 50–200.
   * - **Batch Size**
     - 4
     - Number of patches per training step.
       Reduce to 2 or 1 if you encounter GPU out-of-memory errors.
   * - **Learning Rate**
     - 0.0003
     - Initial learning rate for the optimizer (Adam).
       Recommended: 0.0001–0.0003 for pretrained encoders, 0.001 for training from scratch.
   * - **Device**
     - ``cuda``
     - ``cuda``: use the GPU (requires NVIDIA GPU and correct PyTorch CUDA version).
       ``cpu``: use the CPU (slower but always works).


Optimization Group
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **LR Scheduler**
     - ``reduce_on_plateau``
     - Strategy for reducing the learning rate during training.
       See :ref:`lr_schedulers` below.
   * - **Loss Function**
     - ``binary_dice_bce`` / ``dice_ce``
     - Loss function used during training (auto-switched based on mode).
       See :doc:`../reference/loss_functions`.
   * - **Augmentation**
     - ``basic``
     - Level of data augmentation applied during training.
       See :doc:`../reference/augmentation`.
   * - **Class weights**
     - unchecked
     - Compute and apply per-class weights to address class imbalance.
       Useful when one class is much rarer than others.


Regularization Group
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Freeze encoder**
     - unchecked
     - Freeze the encoder weights for the first N epochs, training only the decoder.
       Useful to stabilize training when using a pretrained encoder.
   * - **Freeze epochs**
     - 5
     - Number of epochs to keep the encoder frozen before unfreezing.
       Only active when "Freeze encoder" is checked.
   * - **Early stopping**
     - checked
     - Automatically stop training if the validation metric does not improve.
   * - **Patience**
     - 15
     - Number of epochs to wait for improvement before stopping.
       Increase for noisier datasets.


Output
------

At the bottom of the tab, configure where the trained model is saved:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - **Output Directory**
     - Folder where the trained model (``.pth`` checkpoint) and training plots
       will be saved. Defaults to ``./trained_models``.

.. tip::
   The best checkpoint (by validation metric) is saved automatically as
   ``best_model.pth`` in the output directory. Training curves (loss and metric
   vs. epoch) are also saved as a PNG image.


.. _lr_schedulers:

Learning Rate Schedulers
--------------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Scheduler
     - Description
   * - ``reduce_on_plateau``
     - Reduces LR by a factor when the validation metric stops improving.
       Good default for most datasets.
   * - ``cosine_annealing``
     - Smoothly decreases LR following a cosine curve. Effective for
       pretrained models.
   * - ``one_cycle``
     - Increases LR briefly then decreases it (super-convergence). Fast
       convergence on clean datasets.
   * - ``none``
     - Fixed learning rate for the entire training run.


Advanced Parameters
--------------------

The **Advanced Parameters** sub-tab exposes fine-grained control over model
regularization, loss function hyperparameters, and cross-validation.

.. figure:: ../_images/training_advanced_subtab.png
   :alt: Advanced Parameters sub-tab
   :align: center
   :width: 90%

   *Advanced Parameters: fine-tuning options.*

.. .. [INSERT SCREENSHOT: Advanced Parameters sub-tab]

.. note::
   These parameters have sensible defaults. Only adjust them if you understand
   their effect on training.


Model Fine-Tuning
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Dropout Rate**
     - 0.3
     - Probability of dropping a neuron during training (regularization).
       0.0 disables dropout. Range: 0.0–0.7.
   * - **Mixed Precision (AMP)**
     - unchecked
     - Use FP16 for faster training with less GPU memory.
       **GPU only** — has no effect on CPU.
   * - **Warmup Epochs**
     - 0
     - Gradually increase LR from 0 to the target value over N epochs
       before the main scheduler takes over. Useful with large pretrained models.


Loss Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Focal Loss parameters:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Focal Gamma**
     - 2.0
     - Controls how much the loss focuses on hard-to-classify examples.
       Higher values → more focus on difficult pixels. Range: 0.5–5.0.
   * - **Focal Alpha**
     - 0.25
     - Weight for the positive class in the focal loss.
       Increase for highly imbalanced datasets. Range: 0.1–0.9.

Tversky Loss parameters:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Tversky Alpha (FP)**
     - 0.3
     - Weight assigned to false positives.
       Lower = more false positives tolerated (higher recall).
   * - **Tversky Beta (FN)**
     - 0.7
     - Weight assigned to false negatives.
       Higher = stronger penalty for missed detections (higher recall).


K-Fold Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~

When enabled, K-Fold CV replaces the fixed train/val/test split:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Enable K-Fold**
     - unchecked
     - Pool all patches (train + val + test) and split into K folds.
       K models are trained and evaluated sequentially.
   * - **Folds (K)**
     - 5
     - Number of folds. Typical values: 5 or 10.

K-Fold outputs results as **mean ± standard deviation** and provides
95% confidence intervals over all folds.

.. note::
   K-Fold training takes K times longer than standard training. Start with K=5
   for initial experiments.


Reading the Training Log
-------------------------

During training, the output panel shows epoch-by-epoch progress:

.. code-block:: text

   Epoch  1/100 | Train Loss: 0.4321 | Val Loss: 0.3980 | Val IoU: 0.6123
   Epoch  2/100 | Train Loss: 0.3876 | Val Loss: 0.3541 | Val IoU: 0.6458
   ...
   Epoch 18/100 | Train Loss: 0.2103 | Val Loss: 0.2890 | Val IoU: 0.7812
   → Best model saved (IoU: 0.7812)
   ...
   Early stopping: no improvement for 15 epochs.
   Training complete. Best model: epoch 18, Val IoU: 0.7812

For multi-class mode, per-class IoU metrics are also reported at the end.


Recommended Configurations
----------------------------

**Fast experiment (CPU, small dataset)**

- Architecture: ``unet-dropout``
- Encoder: N/A
- Epochs: 30, Batch: 4, LR: 0.001
- Augmentation: ``basic``
- Loss: ``binary_dice_bce`` or ``dice_ce``
- Device: ``cpu``

**Production (GPU, pretrained)**

- Architecture: ``unet`` or ``deeplabv3+``
- Encoder: ``resnet34`` or ``efficientnet-b3``
- Pretrained: ✓
- Epochs: 100, Batch: 8, LR: 0.0003
- Augmentation: ``advanced``
- Loss: ``binary_dice_bce`` / ``dice_ce``
- Freeze encoder: ✓ (5 epochs), Early stopping: ✓ (15 patience)
- Device: ``cuda``
- AMP: ✓

**Modern architecture (Transformer)**

- Architecture: ``segformer-b2`` or ``unetformer``
- Encoder: N/A (self-contained)
- Epochs: 80, Batch: 4, LR: 0.0001
- Augmentation: ``advanced``
- Warmup Epochs: 5
- Device: ``cuda``, AMP: ✓

.. _parameters:

Complete Parameter Reference
=============================

This page provides a consolidated reference of all parameters across the three
plugin tabs.

.. contents:: On this page
   :local:
   :depth: 2


Patch Extraction Parameters
-----------------------------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``mode``
     - string
     - ``single``
     - Extraction mode: ``single`` or ``batch``
   * - ``image_path``
     - path
     - —
     - Path to the satellite GeoTIFF (single mode)
   * - ``label_path``
     - path
     - —
     - Path to the label GeoTIFF (single mode)
   * - ``grid_path``
     - path
     - —
     - Path to the grid shapefile
   * - ``data_dir``
     - path
     - —
     - Path to data folder containing image/label pairs (batch mode)
   * - ``output_dir``
     - path
     - —
     - Output folder for extracted patches
   * - ``patch_size``
     - int
     - 256
     - Output patch size in pixels (width = height)
   * - ``in_channels``
     - int
     - 4
     - Expected number of image bands (for validation)
   * - ``interpolation``
     - string
     - ``bilinear``
     - Resampling for image patches: ``bilinear``, ``nearest``, ``bicubic``, ``lanczos``
   * - ``compression``
     - string
     - ``deflate``
     - Output TIFF compression: ``deflate``, ``lzw``, ``none``
   * - ``validate_crs``
     - bool
     - ``true``
     - Warn on CRS mismatch between image, label, and grid
   * - ``id_column``
     - string
     - ``id``
     - Grid attribute column used for patch naming
   * - ``train_ratio``
     - float
     - 0.70
     - Fraction of patches for training (0.0–1.0)
   * - ``val_ratio``
     - float
     - 0.20
     - Fraction of patches for validation (0.0–1.0)
   * - ``test_ratio``
     - float
     - 0.10
     - Fraction of patches for testing (0.0–1.0)
   * - ``use_per_image_grid``
     - bool
     - ``false``
     - Use per-image ``Grid_N.shp`` files (batch mode)
   * - ``image_pattern``
     - regex
     - ``image_(\d+)``
     - Regex for matching image files (batch mode, advanced)
   * - ``label_pattern``
     - regex
     - ``label_(\d+)``
     - Regex for matching label files (batch mode, advanced)


Model Training Parameters
--------------------------

Basic Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dataset_root``
     - path
     - —
     - Root directory of the patch dataset (must contain ``train/``, ``val/``)
   * - ``mode``
     - string
     - ``binary``
     - ``binary`` or ``multiclass``
   * - ``num_classes``
     - int
     - 2
     - Number of classes (multiclass mode only; ignored for binary)
   * - ``model_name``
     - string
     - ``unet-dropout``
     - Segmentation architecture name (see :doc:`architectures`)
   * - ``encoder_name``
     - string
     - ``resnet34``
     - Encoder backbone (SMP architectures and unetformer only)
   * - ``pretrained``
     - bool
     - ``false``
     - Use ImageNet pretrained encoder weights
   * - ``in_channels``
     - int
     - 4
     - Number of input image channels
   * - ``epochs``
     - int
     - 100
     - Maximum training epochs
   * - ``batch_size``
     - int
     - 4
     - Training batch size
   * - ``learning_rate``
     - float
     - 0.0003
     - Initial learning rate (Adam optimizer)
   * - ``device``
     - string
     - ``cuda``
     - ``cuda`` or ``cpu``
   * - ``save_dir``
     - path
     - ``./trained_models``
     - Directory for saving checkpoints and plots
   * - ``scheduler``
     - string
     - ``reduce_on_plateau``
     - LR scheduler: ``reduce_on_plateau``, ``cosine_annealing``, ``one_cycle``, ``none``
   * - ``loss_function``
     - string
     - ``binary_dice_bce``
     - Loss function name (see :doc:`loss_functions`)
   * - ``augmentation_level``
     - string
     - ``basic``
     - Augmentation level: ``none``, ``basic``, ``advanced``, ``aggressive``, ``extreme``
   * - ``use_class_weights``
     - bool
     - ``false``
     - Compute and apply per-class weights
   * - ``freeze_encoder``
     - bool
     - ``false``
     - Freeze encoder for first N epochs
   * - ``freeze_epochs``
     - int
     - 5
     - Number of epochs to keep encoder frozen
   * - ``early_stopping``
     - bool
     - ``true``
     - Enable early stopping
   * - ``patience``
     - int
     - 15
     - Early stopping patience (epochs without improvement)

Advanced Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dropout_rate``
     - float
     - 0.3
     - Dropout probability (0.0 = disabled)
   * - ``use_amp``
     - bool
     - ``false``
     - Enable mixed precision training (FP16, GPU only)
   * - ``warmup_epochs``
     - int
     - 0
     - Number of LR warmup epochs
   * - ``focal_gamma``
     - float
     - 2.0
     - Focal loss gamma parameter (0.5–5.0)
   * - ``focal_alpha``
     - float
     - 0.25
     - Focal loss alpha parameter (0.1–0.9)
   * - ``tversky_alpha``
     - float
     - 0.3
     - Tversky loss false positive weight
   * - ``tversky_beta``
     - float
     - 0.7
     - Tversky loss false negative weight
   * - ``use_kfold``
     - bool
     - ``false``
     - Enable K-Fold cross-validation
   * - ``n_splits``
     - int
     - 5
     - Number of folds for K-Fold CV


Prediction Parameters
----------------------

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``model_path``
     - path
     - —
     - Path to the trained ``.pth`` checkpoint
   * - ``input_path``
     - path
     - —
     - Path to the input satellite GeoTIFF
   * - ``output_path``
     - path
     - —
     - Path for the output segmentation GeoTIFF
   * - ``patch_size``
     - int
     - 512
     - Inference patch size in pixels (64–2048)
   * - ``overlap``
     - int
     - 128
     - Overlap between adjacent patches in pixels
   * - ``threshold``
     - float
     - 0.50
     - Decision threshold for binary mode (0.0–1.0)
   * - ``device``
     - string
     - ``cuda``
     - ``cuda`` or ``cpu``
   * - ``batch_size``
     - int
     - 1
     - Inference batch size (number of patches per forward pass)
   * - ``gaussian_blending``
     - bool
     - ``true``
     - Apply Gaussian blending when merging patch predictions
   * - ``save_confidence``
     - bool
     - ``false``
     - Save raw probability map alongside the classification result
   * - ``dropout_rate``
     - float
     - 0.3
     - Dropout rate (must match training value)
   * - ``encoder_name``
     - string
     - *(auto)*
     - Override encoder name (leave empty for auto-detection from checkpoint)

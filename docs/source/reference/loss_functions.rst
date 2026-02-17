.. _loss_functions:

Loss Functions
==============

SemanticSeg4EO provides a comprehensive set of loss functions for both binary
and multi-class segmentation. The loss function list automatically switches
when you change the training mode.

.. contents:: On this page
   :local:
   :depth: 2


Binary Mode Loss Functions
---------------------------

These losses are available when **Mode = binary**.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Loss Function
     - Description
   * - ``binary_dice_bce``
     - **Recommended default.** Combination of Dice loss and Binary Cross-Entropy
       (50/50 weight). Handles both overlap quality and pixel-level accuracy.
   * - ``binary_focal_dice``
     - Combination of Focal loss and Dice loss. Best for strongly imbalanced
       datasets where the foreground class is rare.
   * - ``bce``
     - Standard Binary Cross-Entropy. Simple and fast. Works well for
       balanced classes.
   * - ``binary_dice``
     - Pure Dice loss. Focuses entirely on region overlap. Insensitive to
       class imbalance.
   * - ``binary_focal``
     - Focal loss. Down-weights easy examples and focuses on hard-to-classify
       pixels. Use with the *Focal Gamma* and *Focal Alpha* parameters.
   * - ``binary_tversky``
     - Tversky loss. Generalisation of Dice that allows control over the
       trade-off between false positives (alpha) and false negatives (beta).
   * - ``binary_focal_tversky``
     - Combination of Focal and Tversky losses.
       Best for very imbalanced data where recall is critical.


Multi-class Mode Loss Functions
---------------------------------

These losses are available when **Mode = multiclass**.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Loss Function
     - Description
   * - ``dice_ce``
     - **Recommended default.** Combination of multi-class Dice and
       Cross-Entropy. Robust on most datasets.
   * - ``focal_dice``
     - Combination of multi-class Focal loss and Dice loss.
       Good for imbalanced multi-class problems.
   * - ``ce``
     - Standard Cross-Entropy. Classic choice, works well for balanced classes.
   * - ``dice``
     - Multi-class Dice loss. Handles class imbalance well.
   * - ``focal``
     - Multi-class Focal loss. Focuses training on difficult examples.
   * - ``tversky``
     - Multi-class Tversky loss. Control FP/FN balance with alpha and beta.
   * - ``focal_tversky``
     - Multi-class Focal + Tversky combination.
   * - ``combo``
     - Multi-loss combination. A weighted sum of Dice, Focal, and Cross-Entropy.


Loss Function Parameters
--------------------------

The **Advanced Parameters** tab exposes tuning parameters for Focal and Tversky losses.

Focal Loss Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - **Focal Gamma** γ
     - 2.0
     - Modulating factor. Higher values → more focus on difficult pixels
       (those close to the decision boundary).
       γ = 0 reduces to standard cross-entropy.
   * - **Focal Alpha** α
     - 0.25
     - Weight for the positive (foreground) class.
       Increase for highly imbalanced datasets (rare foreground).

Tversky Loss Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - **Tversky Alpha** (FP weight)
     - 0.3
     - Controls false positive penalty.
       Lower = more false positives tolerated = higher recall.
   * - **Tversky Beta** (FN weight)
     - 0.7
     - Controls false negative penalty.
       Higher = stronger penalty for missed detections = higher recall.

.. note::
   For Tversky loss: Alpha + Beta should sum to 1.0. The default values
   (0.3 + 0.7 = 1.0) bias the model towards recall, which is appropriate
   for rare class detection (e.g., small buildings, water bodies).


Choosing a Loss Function
--------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Scenario
     - Recommended loss
   * - Balanced binary dataset
     - ``binary_dice_bce``
   * - Imbalanced binary (rare foreground)
     - ``binary_focal_dice`` or ``binary_tversky``
   * - Balanced multi-class
     - ``dice_ce``
   * - Imbalanced multi-class (rare classes)
     - ``focal_dice`` or ``tversky``
   * - Need high recall (few missed detections)
     - ``binary_tversky`` (beta=0.7) or ``binary_focal_tversky``
   * - Need high precision (few false positives)
     - ``binary_tversky`` (alpha=0.7, beta=0.3)


Auto-Conversion
----------------

When you switch the training mode between ``binary`` and ``multiclass``, the
loss function dropdown automatically selects a sensible equivalent:

- ``binary_dice_bce`` ↔ ``dice_ce``
- ``binary_focal`` ↔ ``focal``
- ``binary_tversky`` ↔ ``tversky``

You can always override this auto-selection manually.

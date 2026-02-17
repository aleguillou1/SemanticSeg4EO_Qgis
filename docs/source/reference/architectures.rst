.. _architectures:

Architectures & Encoders
=========================

SemanticSeg4EO supports 20+ segmentation architectures from three categories:
built-in, SMP-based, and modern Transformer models. This page documents each
architecture, its requirements, and recommended use cases.

.. contents:: On this page
   :local:
   :depth: 2


Architecture Overview
----------------------

.. list-table::
   :widths: 20 15 20 45
   :header-rows: 1

   * - Architecture
     - Category
     - Requires
     - Notes
   * - ``unet-dropout``
     - Built-in
     - *(none)*
     - Simple U-Net with configurable dropout. Works without any optional
       dependencies. Good for quick experiments.
   * - ``unet``
     - SMP
     - ``smp``
     - Classic U-Net with skip connections. Reliable baseline.
   * - ``unet++``
     - SMP
     - ``smp``
     - Nested U-Net with dense skip connections. Often outperforms plain U-Net.
   * - ``manet``
     - SMP
     - ``smp``
     - Multi-scale Attention Network. Good for multi-scale features.
   * - ``linknet``
     - SMP
     - ``smp``
     - Lightweight encoder-decoder. Faster than U-Net.
   * - ``fpn``
     - SMP
     - ``smp``
     - Feature Pyramid Network. Effective for multi-scale objects.
   * - ``pspnet``
     - SMP
     - ``smp``
     - Pyramid Scene Parsing Network. Good for scene-level context.
   * - ``pan``
     - SMP
     - ``smp``
     - Path Aggregation Network. Strong multi-scale aggregation.
   * - ``deeplabv3``
     - SMP
     - ``smp``
     - DeepLab v3 with ASPP. Strong contextual reasoning.
   * - ``deeplabv3+``
     - SMP
     - ``smp``
     - DeepLab v3+ with improved decoder. Often best accuracy for dense prediction.
   * - ``segformer-b0``
     - Modern
     - ``transformers``
     - Smallest SegFormer. Fast, good accuracy-speed trade-off.
   * - ``segformer-b1``
     - Modern
     - ``transformers``
     - SegFormer B1.
   * - ``segformer-b2``
     - Modern
     - ``transformers``
     - SegFormer B2. Recommended Transformer baseline.
   * - ``segformer-b3``
     - Modern
     - ``transformers``
     - SegFormer B3.
   * - ``segformer-b4``
     - Modern
     - ``transformers``
     - SegFormer B4.
   * - ``segformer-b5``
     - Modern
     - ``transformers``
     - Largest SegFormer. Highest accuracy, most memory.
   * - ``unetformer``
     - Modern
     - ``smp``
     - U-Net decoder with Transformer encoder. Good hybrid approach.
   * - ``hrnet-w18``
     - Modern
     - ``timm``
     - HRNet-W18. High-resolution feature maps, 18-width.
   * - ``hrnet-w32``
     - Modern
     - ``timm``
     - HRNet-W32. Balanced accuracy/speed.
   * - ``hrnet-w48``
     - Modern
     - ``timm``
     - HRNet-W48. Highest accuracy in the HRNet family.
   * - ``swin-unet``
     - Modern
     - ``timm``
     - Swin Transformer U-Net. State-of-the-art for medical/EO segmentation.

.. note::
   ``smp`` = ``segmentation-models-pytorch``, ``transformers`` = HuggingFace Transformers,
   ``timm`` = PyTorch Image Models. Install these in your external environment
   (see :doc:`../getting_started/environment_setup`).


Built-in Architecture
----------------------

unet-dropout
~~~~~~~~~~~~~

A simple U-Net implementation built directly into the plugin, requiring no
optional dependencies. It includes:

- Configurable dropout at each decoder level (set via *Dropout Rate*)
- 4 encoder levels with max-pooling
- 4 decoder levels with skip connections and bilinear upsampling
- Works with any number of input channels

**When to use**: quick experiments, CPU-only setups, or when you cannot install
``segmentation-models-pytorch``.


SMP Architectures
------------------

All SMP architectures use the ``segmentation-models-pytorch`` library and support:

- Dozens of encoder backbones (see :ref:`encoders` below)
- ImageNet pretrained weights
- Configurable input channels (encoder first layer is adapted automatically)

Install SMP in your environment:

.. code-block:: bash

   pip install segmentation-models-pytorch


Modern / Transformer Architectures
------------------------------------

These architectures do not use a separate encoder — they are self-contained.
The **Encoder** dropdown is automatically hidden when one of these is selected.

SegFormer (b0–b5)
~~~~~~~~~~~~~~~~~~

SegFormer is a transformer-based segmentation model from NVIDIA that uses:

- A hierarchical transformer encoder (Mix Vision Transformer)
- A lightweight MLP decoder
- Very competitive performance on standard benchmarks

All six variants (b0–b5) are available, corresponding to increasing model sizes.

Requires: ``pip install transformers``

HRNet (w18, w32, w48)
~~~~~~~~~~~~~~~~~~~~~

HRNet (High-Resolution Network) maintains high-resolution feature representations
throughout the network, making it excellent for segmentation tasks requiring
fine spatial detail.

- w18/w32/w48 refer to the width (number of channels) at the highest resolution stream
- Higher width = more parameters and better accuracy

Requires: ``pip install timm``

SwinUNet
~~~~~~~~~

SwinUNet is a pure Transformer architecture using the Swin Transformer as both
encoder and decoder backbone, with skip connections following a U-Net structure.

Requires: ``pip install timm``

UNetFormer
~~~~~~~~~~~

UNetFormer combines a Transformer encoder with the U-Net decoder structure.
Uses a standard SMP encoder (see :ref:`encoders`) as the backbone.

Requires: ``pip install segmentation-models-pytorch``


.. _encoders:

Encoders
---------

For SMP-based architectures and UNetFormer, you select an **encoder backbone**
from the Encoder dropdown. The encoder is the feature extraction network;
the decoder learns to combine those features into a segmentation mask.

Encoder families available:

**ResNet family** (most common baseline):

.. code-block:: text

   resnet18, resnet34, resnet50, resnet101, resnet152

**ResNeXt family**:

.. code-block:: text

   resnext50_32x4d, resnext101_32x4d, resnext101_32x8d

**SE-Net family** (squeeze-and-excitation):

.. code-block:: text

   se_resnet50, se_resnet101, se_resnet152
   se_resnext50_32x4d, se_resnext101_32x4d
   senet154

**EfficientNet family**:

.. code-block:: text

   efficientnet-b0 → efficientnet-b7
   timm-efficientnet-b0 → timm-efficientnet-l2

**ResNeSt family** (via timm):

.. code-block:: text

   timm-resnest14d, timm-resnest26d, timm-resnest50d
   timm-resnest101e, timm-resnest200e, timm-resnest269e

**DenseNet family**:

.. code-block:: text

   densenet121, densenet169, densenet201, densenet161

**Inception family**:

.. code-block:: text

   inceptionresnetv2, inceptionv4

**MobileNet**:

.. code-block:: text

   mobilenet_v2

**DPN family**:

.. code-block:: text

   dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131

**VGG family**:

.. code-block:: text

   vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

**Mix Vision Transformer** (SegFormer backbone via SMP):

.. code-block:: text

   mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

**MobileOne**:

.. code-block:: text

   mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4

**ConvNeXt** (U-Net only, requires timm):

.. code-block:: text

   convnext_tiny, convnext_small, convnext_base
   convnext_large, convnext_xlarge


Encoder Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Use case
     - Recommended encoder
   * - Quick experiments / CPU
     - ``resnet18``, ``mobilenet_v2``, ``efficientnet-b0``
   * - Balanced (recommended default)
     - ``resnet34``, ``resnet50``, ``efficientnet-b3``
   * - Best accuracy (GPU required)
     - ``resnet101``, ``efficientnet-b5``, ``se_resnext50_32x4d``
   * - Very high accuracy, needs timm
     - ``convnext_base``, ``timm-efficientnet-b5``


Which Architecture Should I Choose?
-------------------------------------

As a starting point:

1. **First experiment** → ``unet-dropout`` (no dependencies, fast)
2. **Standard baseline** → ``unet`` or ``deeplabv3+`` with ``resnet34``
3. **Best accuracy** → ``segformer-b2`` or ``hrnet-w32`` (requires transformers/timm)
4. **Limited GPU memory** → ``fpn`` or ``linknet`` with ``efficientnet-b0``
5. **Fine spatial detail** → ``hrnet-w32`` or ``hrnet-w48``

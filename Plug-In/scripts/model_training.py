#!/usr/bin/env python3
"""
Model Training Script - SemanticSeg4EO Plugin (Standalone)
============================================================

This script is executed by the EXTERNAL Python environment (not QGIS).
Supports binary and multi-class segmentation with modern architectures.

Architectures:
  SMP: unet, unet++, deeplabv3+, deeplabv3, manet, fpn, pan, pspnet, linknet
  Modern: segformer-b0..b5, unetformer, hrnet-w18/w32/w48, swin-unet
  Custom: unet-dropout (built-in), convnext-unet (timm)

Features:
  - 5 augmentation levels (none, basic, advanced, aggressive, extreme)
  - Comprehensive loss functions (binary & multiclass, auto-conversion)
  - Mixed precision training (AMP)
  - Encoder freeze/unfreeze strategy
  - Warmup scheduler, cosine/plateau/onecycle
  - K-Fold cross-validation (optional)
  - Per-class metrics logging

Usage:
    python model_training.py params.json progress.json
"""

import os
import sys
import time
import logging
import random

# Fix OpenMP conflict on Windows (numpy/torch/mkl/matplotlib)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# ============ SUPPRESS RASTERIO/GDAL WARNINGS ============
os.environ['CPL_LOG'] = 'OFF'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_PAM_ENABLED'] = 'NO'
os.environ['PROJ_LIB'] = ''  # Suppress PROJ warnings

import json
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Suppress ALL rasterio/GDAL logging
logging.getLogger('rasterio').setLevel(logging.CRITICAL)
logging.getLogger('rasterio._env').setLevel(logging.CRITICAL)
logging.getLogger('rasterio.env').setLevel(logging.CRITICAL)
logging.getLogger('GDAL').setLevel(logging.CRITICAL)
logging.getLogger('fiona').setLevel(logging.CRITICAL)
logging.getLogger('fiona._env').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
warnings.filterwarnings('ignore', message='.*GDAL.*')
warnings.filterwarnings('ignore', message='.*CRS.*')
warnings.filterwarnings('ignore', message='.*proj.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# For reading TIFFs
try:
    # Redirect stderr temporarily to suppress C-level GDAL Unicode errors
    import io as _io
    _old_stderr = sys.stderr
    sys.stderr = _io.StringIO()
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rasterio.env.defenv()
        except Exception:
            pass
    sys.stderr = _old_stderr
    try:
        warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    except Exception:
        pass
    HAS_RASTERIO = True
except ImportError:
    sys.stderr = _old_stderr if '_old_stderr' in dir() else sys.stderr
    HAS_RASTERIO = False

try:
    import tifffile as tiff
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def read_tiff(filepath):
    """Read TIFF file using rasterio (preferred) or tifffile"""
    if HAS_RASTERIO:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(filepath) as src:
                data = src.read()
                if data.shape[0] == 1:
                    return data[0]
                return np.transpose(data, (1, 2, 0))
    elif HAS_TIFFFILE:
        return tiff.imread(filepath)
    else:
        raise ImportError("Neither rasterio nor tifffile is installed")


# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================

def check_dependencies():
    """Check available dependencies"""
    deps = {}
    try:
        import segmentation_models_pytorch as smp
        deps['smp'] = True
        deps['smp_version'] = smp.__version__
    except ImportError:
        deps['smp'] = False
    try:
        from transformers import SegformerForSemanticSegmentation
        deps['transformers'] = True
    except ImportError:
        deps['transformers'] = False
    try:
        import timm
        deps['timm'] = True
    except ImportError:
        deps['timm'] = False
    try:
        from sklearn.model_selection import KFold
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False
    return deps


DEPS = check_dependencies()


def update_progress(progress_file, progress, message, result=None):
    """Update progress file for QGIS to read"""
    data = {'progress': progress, 'message': message}
    if result:
        data['result'] = result
    with open(progress_file, 'w') as f:
        json.dump(data, f)


# ============================================================================
# AUGMENTATION LEVELS
# ============================================================================

class AugmentationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class AugmentationConfig:
    """Configuration for geospatial data augmentation"""
    enabled: bool = True
    prob: float = 0.8

    # Geometric
    flip_horizontal: bool = False
    flip_horizontal_prob: float = 0.5
    flip_vertical: bool = False
    flip_vertical_prob: float = 0.5
    rotation_90: bool = False
    rotation_90_prob: float = 0.5
    rotation_any: bool = False
    rotation_any_prob: float = 0.3
    rotation_any_limit: float = 30.0
    scale: bool = False
    scale_prob: float = 0.4
    scale_min: float = 0.8
    scale_max: float = 1.2
    random_crop: bool = False
    random_crop_prob: float = 0.5
    random_crop_scale_min: float = 0.7
    random_crop_scale_max: float = 1.0
    elastic: bool = False
    elastic_prob: float = 0.3
    elastic_alpha: float = 120.0
    elastic_sigma: float = 12.0
    grid_distort: bool = False
    grid_distort_prob: float = 0.2
    grid_distort_limit: float = 0.3

    # Pixel
    brightness: bool = False
    brightness_prob: float = 0.4
    brightness_limit: float = 0.15
    contrast: bool = False
    contrast_prob: float = 0.4
    contrast_limit: float = 0.15
    gamma: bool = False
    gamma_prob: float = 0.3
    gamma_min: float = 0.8
    gamma_max: float = 1.2
    gaussian_noise: bool = False
    gaussian_noise_prob: float = 0.3
    gaussian_noise_std: float = 0.03
    channel_noise: bool = False
    channel_noise_prob: float = 0.2
    channel_noise_std: float = 0.02
    channel_shuffle: bool = False
    channel_shuffle_prob: float = 0.1
    channel_dropout: bool = False
    channel_dropout_prob: float = 0.2
    channel_dropout_max_channels: int = 2

    # Blur
    gaussian_blur: bool = False
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma_min: float = 0.5
    gaussian_blur_sigma_max: float = 2.0
    motion_blur: bool = False
    motion_blur_prob: float = 0.2
    motion_blur_kernel: int = 7

    # Dropout/Cutout
    coarse_dropout: bool = False
    coarse_dropout_prob: float = 0.3
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height: int = 32
    coarse_dropout_max_width: int = 32
    coarse_dropout_fill_value: float = 0.0

    # MixUp / CutMix
    mixup: bool = False
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.4
    cutmix: bool = False
    cutmix_prob: float = 0.3
    cutmix_alpha: float = 1.0

    # Minority class focus
    minority_oversample: bool = False
    minority_class_idx: int = 1
    minority_augment_extra: bool = False


def get_augmentation_config(level: AugmentationLevel, in_channels: int = 4) -> AugmentationConfig:
    """Return augmentation config for given level, optimized for remote sensing."""
    config = AugmentationConfig()

    if level == AugmentationLevel.NONE:
        config.enabled = False
        return config

    # BASIC: geometric only
    if level in (AugmentationLevel.BASIC, AugmentationLevel.ADVANCED,
                 AugmentationLevel.AGGRESSIVE, AugmentationLevel.EXTREME):
        config.enabled = True
        config.prob = 0.8
        config.flip_horizontal = True
        config.flip_vertical = True
        config.rotation_90 = True

    # ADVANCED: + radiometric
    if level in (AugmentationLevel.ADVANCED, AugmentationLevel.AGGRESSIVE, AugmentationLevel.EXTREME):
        config.scale = True
        config.scale_prob = 0.4
        config.scale_min = 0.85
        config.scale_max = 1.15
        config.brightness = True
        config.brightness_prob = 0.4
        config.brightness_limit = 0.15
        config.contrast = True
        config.contrast_prob = 0.4
        config.contrast_limit = 0.15
        config.gamma = True
        config.gamma_prob = 0.3
        config.gamma_min = 0.85
        config.gamma_max = 1.15
        config.channel_noise = True
        config.channel_noise_prob = 0.2
        config.channel_noise_std = 0.02

    # AGGRESSIVE: + deformations
    if level in (AugmentationLevel.AGGRESSIVE, AugmentationLevel.EXTREME):
        config.random_crop = True
        config.random_crop_prob = 0.4
        config.elastic = True
        config.elastic_prob = 0.25
        config.gaussian_blur = True
        config.gaussian_blur_prob = 0.25
        config.gaussian_noise = True
        config.gaussian_noise_prob = 0.25
        config.gaussian_noise_std = 0.025
        config.coarse_dropout = True
        config.coarse_dropout_prob = 0.2
        config.coarse_dropout_max_holes = 5
        if in_channels > 2:
            config.channel_dropout = True
            config.channel_dropout_prob = 0.15
            config.channel_dropout_max_channels = min(2, in_channels - 1)
        config.minority_oversample = True
        config.minority_augment_extra = True

    # EXTREME: everything
    if level == AugmentationLevel.EXTREME:
        config.prob = 0.9
        config.rotation_any = True
        config.rotation_any_prob = 0.3
        config.rotation_any_limit = 45.0
        config.grid_distort = True
        config.grid_distort_prob = 0.2
        config.motion_blur = True
        config.motion_blur_prob = 0.15
        config.mixup = True
        config.mixup_prob = 0.2
        config.mixup_alpha = 0.3
        config.cutmix = True
        config.cutmix_prob = 0.2
        if in_channels > 3:
            config.channel_shuffle = True
            config.channel_shuffle_prob = 0.1

    return config


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

class AdvancedMultiChannelAugmentation:
    """Advanced augmentation for multi-channel geospatial images"""

    def __init__(self, config: AugmentationConfig, patch_size: int = 256, mode: str = 'multiclass'):
        self.config = config
        self.patch_size = patch_size
        self.mode = mode

    def __call__(self, img: torch.Tensor, mask: torch.Tensor,
                 has_minority_class: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.enabled or random.random() > self.config.prob:
            return img, mask

        img_np = img.numpy()
        mask_np = mask.squeeze(0).numpy() if (self.mode == 'binary' and mask.dim() == 3) else mask.numpy()

        # Geometric transforms
        if self.config.flip_horizontal and random.random() < self.config.flip_horizontal_prob:
            img_np = np.flip(img_np, axis=2).copy()
            mask_np = np.flip(mask_np, axis=-1).copy()
        if self.config.flip_vertical and random.random() < self.config.flip_vertical_prob:
            img_np = np.flip(img_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=-2 if mask_np.ndim > 1 else 0).copy()
        if self.config.rotation_90 and random.random() < self.config.rotation_90_prob:
            k = random.choice([1, 2, 3])
            img_np = np.rot90(img_np, k, axes=(1, 2)).copy()
            mask_np = np.rot90(mask_np, k, axes=(0, 1) if mask_np.ndim == 2 else (1, 2)).copy()

        # Pixel transforms
        if self.config.brightness and random.random() < self.config.brightness_prob:
            factor = 1.0 + random.uniform(-self.config.brightness_limit, self.config.brightness_limit)
            img_np = np.clip(img_np * factor, 0, 1)
        if self.config.contrast and random.random() < self.config.contrast_prob:
            factor = 1.0 + random.uniform(-self.config.contrast_limit, self.config.contrast_limit)
            mean = img_np.mean()
            img_np = np.clip((img_np - mean) * factor + mean, 0, 1)
        if self.config.gamma and random.random() < self.config.gamma_prob:
            g = random.uniform(self.config.gamma_min, self.config.gamma_max)
            img_np = np.clip(np.power(img_np + 1e-7, g), 0, 1)
        if self.config.gaussian_noise and random.random() < self.config.gaussian_noise_prob:
            noise = np.random.normal(0, self.config.gaussian_noise_std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 1)
        if self.config.channel_noise and random.random() < self.config.channel_noise_prob:
            c = random.randint(0, img_np.shape[0] - 1)
            noise = np.random.normal(0, self.config.channel_noise_std, img_np[c].shape).astype(np.float32)
            img_np[c] = np.clip(img_np[c] + noise, 0, 1)

        # Blur
        if self.config.gaussian_blur and random.random() < self.config.gaussian_blur_prob:
            sigma = random.uniform(self.config.gaussian_blur_sigma_min, self.config.gaussian_blur_sigma_max)
            from scipy.ndimage import gaussian_filter
            for c in range(img_np.shape[0]):
                img_np[c] = gaussian_filter(img_np[c], sigma=sigma)

        # Channel dropout
        if self.config.channel_dropout and random.random() < self.config.channel_dropout_prob:
            n_drop = random.randint(1, self.config.channel_dropout_max_channels)
            channels_to_drop = random.sample(range(img_np.shape[0]), min(n_drop, img_np.shape[0] - 1))
            for c in channels_to_drop:
                img_np[c] = 0.0

        # Coarse dropout
        if self.config.coarse_dropout and random.random() < self.config.coarse_dropout_prob:
            h, w = img_np.shape[1], img_np.shape[2]
            n_holes = random.randint(1, self.config.coarse_dropout_max_holes)
            for _ in range(n_holes):
                hole_h = random.randint(8, self.config.coarse_dropout_max_height)
                hole_w = random.randint(8, self.config.coarse_dropout_max_width)
                y = random.randint(0, max(0, h - hole_h))
                x = random.randint(0, max(0, w - hole_w))
                img_np[:, y:y + hole_h, x:x + hole_w] = self.config.coarse_dropout_fill_value

        # Convert back to tensors
        img = torch.from_numpy(img_np.copy()).float()
        if self.mode == 'binary':
            mask = torch.from_numpy(mask_np.copy()).unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask_np.copy()).long()

        return img, mask


# ============================================================================
# LOSS FUNCTIONS - BINARY
# ============================================================================

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class BinaryDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = BinaryDiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets.float())


class BinaryFocalDiceLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = BinaryDiceLoss()
        self.focal = BinaryFocalLoss(focal_alpha, focal_gamma)

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class BinaryFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()


# ============================================================================
# LOSS FUNCTIONS - MULTICLASS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6, weight=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth)
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.ce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        tp = (pred_soft * target_oh).sum(dim=(2, 3))
        fp = (pred_soft * (1 - target_oh)).sum(dim=(2, 3))
        fn = ((1 - pred_soft) * target_oh).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        if target.dtype == torch.float32:
            target = target.squeeze(1).long()
        pred_soft = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        tp = (pred_soft * target_oh).sum(dim=(2, 3))
        fp = (pred_soft * (1 - target_oh)).sum(dim=(2, 3))
        fn = ((1 - pred_soft) * target_oh).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()


class ComboLoss(nn.Module):
    """Combination of multiple losses"""
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, pred, target):
        total = 0
        for loss, w in zip(self.losses, self.weights):
            total = total + w * loss(pred, target)
        return total


# ============================================================================
# LOSS FACTORY - AUTO MODE VALIDATION
# ============================================================================

class LossFactory:
    """Factory with strict binary/multiclass validation and auto-conversion"""

    BINARY_LOSSES = {
        'bce', 'binary_dice', 'binary_focal', 'binary_focal_dice',
        'binary_tversky', 'binary_focal_tversky', 'binary_dice_bce',
        'dice_bce'  # legacy alias
    }
    MULTICLASS_LOSSES = {
        'ce', 'dice', 'focal', 'tversky', 'focal_tversky',
        'dice_ce', 'focal_dice', 'combo'
    }
    BINARY_TO_MULTICLASS = {
        'bce': 'ce', 'binary_dice': 'dice', 'binary_focal': 'focal',
        'binary_focal_dice': 'focal_dice', 'binary_tversky': 'tversky',
        'binary_focal_tversky': 'focal_tversky', 'binary_dice_bce': 'dice_ce',
        'dice_bce': 'dice_ce'
    }
    MULTICLASS_TO_BINARY = {v: k for k, v in BINARY_TO_MULTICLASS.items() if k != 'dice_bce'}

    @staticmethod
    def create(loss_type, mode, num_classes=2, class_weights=None,
               focal_alpha=0.25, focal_gamma=2.0,
               tversky_alpha=0.3, tversky_beta=0.7,
               dice_weight=0.5, ce_weight=0.5):
        """Create loss function with auto mode conversion"""
        loss_type = loss_type.lower()

        # Auto-convert between binary <-> multiclass
        if mode == 'binary':
            if loss_type not in LossFactory.BINARY_LOSSES:
                if loss_type in LossFactory.MULTICLASS_LOSSES:
                    new = LossFactory.MULTICLASS_TO_BINARY.get(loss_type, 'binary_focal_dice')
                    print(f"  ⚠ Auto-conversion: '{loss_type}' → '{new}' (mode binary)")
                    loss_type = new
                else:
                    print(f"  ⚠ Unknown loss '{loss_type}', using 'binary_focal_dice'")
                    loss_type = 'binary_focal_dice'
        else:
            if loss_type not in LossFactory.MULTICLASS_LOSSES:
                if loss_type in LossFactory.BINARY_LOSSES:
                    new = LossFactory.BINARY_TO_MULTICLASS.get(loss_type, 'focal_dice')
                    print(f"  ⚠ Auto-conversion: '{loss_type}' → '{new}' (mode multiclass)")
                    loss_type = new
                else:
                    print(f"  ⚠ Unknown loss '{loss_type}', using 'focal_dice'")
                    loss_type = 'focal_dice'

        # Binary losses
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'binary_dice':
            return BinaryDiceLoss()
        elif loss_type == 'binary_focal':
            return BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'binary_focal_dice':
            return BinaryFocalDiceLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                                       dice_weight=dice_weight, focal_weight=ce_weight)
        elif loss_type == 'binary_tversky':
            return BinaryTverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif loss_type == 'binary_focal_tversky':
            return BinaryFocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif loss_type in ('binary_dice_bce', 'dice_bce'):
            return BinaryDiceBCELoss(dice_weight=dice_weight, bce_weight=ce_weight)
        # Multiclass losses
        elif loss_type == 'ce':
            return nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'focal':
            return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=class_weights)
        elif loss_type == 'tversky':
            return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif loss_type == 'focal_tversky':
            return FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif loss_type == 'dice_ce':
            return DiceCELoss(dice_weight=dice_weight, ce_weight=ce_weight, weight=class_weights)
        elif loss_type == 'focal_dice':
            return ComboLoss(
                [FocalLoss(gamma=focal_gamma, weight=class_weights), DiceLoss()],
                [ce_weight, dice_weight]
            )
        elif loss_type == 'combo':
            return ComboLoss(
                [nn.CrossEntropyLoss(weight=class_weights), DiceLoss(),
                 FocalLoss(gamma=focal_gamma, weight=class_weights)],
                [0.4, 0.3, 0.3]
            )
        else:
            raise ValueError(f"Unknown loss: {loss_type}")

    @staticmethod
    def get_available(mode=None):
        if mode == 'binary':
            return sorted(LossFactory.BINARY_LOSSES)
        elif mode == 'multiclass':
            return sorted(LossFactory.MULTICLASS_LOSSES)
        return sorted(LossFactory.BINARY_LOSSES | LossFactory.MULTICLASS_LOSSES)


# ============================================================================
# MODELS - BUILT-IN
# ============================================================================

class SimpleUNet(nn.Module):
    """Simple U-Net implementation (no dependencies)"""
    def __init__(self, in_channels, classes, dropout=0.5):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

        self.enc1 = block(in_channels, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)
        self.bridge = block(512, 1024)
        self.dropout = nn.Dropout2d(dropout)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)
        self.out = nn.Conv2d(64, classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.dropout(self.bridge(self.pool(e4)))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ============================================================================
# MODELS - MODERN ARCHITECTURES
# ============================================================================

class ConvNeXtUNet(nn.Module):
    """UNet with ConvNeXt encoder via timm"""
    CONVNEXT_MODELS = {
        'convnext_tiny': 'convnext_tiny.fb_in22k_ft_in1k',
        'convnext_small': 'convnext_small.fb_in22k_ft_in1k',
        'convnext_base': 'convnext_base.fb_in22k_ft_in1k',
        'convnext_large': 'convnext_large.fb_in22k_ft_in1k',
        'convnext_xlarge': 'convnext_xlarge.fb_in22k_ft_in1k',
    }

    def __init__(self, encoder_name, num_classes, in_channels=3, pretrained=True, dropout_rate=0.3):
        super().__init__()
        if not DEPS.get('timm'):
            raise ImportError("ConvNeXt requires timm: pip install timm")
        import timm

        timm_name = self.CONVNEXT_MODELS.get(encoder_name, encoder_name)
        self.encoder = timm.create_model(timm_name, pretrained=pretrained,
                                          features_only=True, in_chans=in_channels, drop_rate=dropout_rate)
        enc_ch = self.encoder.feature_info.channels()
        dec_ch = [256, 128, 64, 32]
        self.center = self._conv_block(enc_ch[-1], dec_ch[0], dropout_rate)
        self.decoder_blocks = nn.ModuleList()
        for i, dc in enumerate(dec_ch):
            in_ch = dc + (enc_ch[-(i + 2)] if i < len(enc_ch) - 1 else 0)
            out_ch = dec_ch[i + 1] if i + 1 < len(dec_ch) else dc
            self.decoder_blocks.append(self._conv_block(in_ch, out_ch, dropout_rate))
        self.final = nn.Conv2d(dec_ch[-1], num_classes, 1)

    def _conv_block(self, in_ch, out_ch, dr):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dr),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        features = self.encoder(x)
        x_dec = self.center(features[-1])
        for i, blk in enumerate(self.decoder_blocks):
            x_dec = F.interpolate(x_dec, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(features) - 1:
                skip = features[-(i + 2)]
                if skip.shape[-2:] != x_dec.shape[-2:]:
                    skip = F.interpolate(skip, size=x_dec.shape[-2:], mode='bilinear', align_corners=False)
                x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = blk(x_dec)
        x_dec = F.interpolate(x_dec, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final(x_dec)


class SegFormerWrapper(nn.Module):
    """SegFormer B0-B5 wrapper"""
    VARIANTS = {
        'segformer-b0': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'segformer-b1': 'nvidia/segformer-b1-finetuned-ade-512-512',
        'segformer-b2': 'nvidia/segformer-b2-finetuned-ade-512-512',
        'segformer-b3': 'nvidia/segformer-b3-finetuned-ade-512-512',
        'segformer-b4': 'nvidia/segformer-b4-finetuned-ade-512-512',
        'segformer-b5': 'nvidia/segformer-b5-finetuned-ade-512-512',
    }
    VARIANT_CONFIGS = {
        'segformer-b0': dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
                             decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8]),
        'segformer-b1': dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8]),
        'segformer-b2': dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8]),
        'segformer-b3': dict(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8]),
        'segformer-b4': dict(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8]),
        'segformer-b5': dict(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512],
                             decoder_hidden_size=768, num_attention_heads=[1, 2, 5, 8]),
    }

    def __init__(self, variant, num_classes, in_channels, pretrained=True, dropout_rate=0.3):
        super().__init__()
        if not DEPS.get('transformers'):
            raise ImportError("pip install transformers")
        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        self.num_classes = num_classes
        self.in_channels = in_channels

        if pretrained and variant in self.VARIANTS:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.VARIANTS[variant], num_labels=num_classes, ignore_mismatched_sizes=True,
                hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
        else:
            config = None
            if variant in self.VARIANTS:
                try:
                    config = SegformerConfig.from_pretrained(
                        self.VARIANTS[variant], num_labels=num_classes, num_channels=in_channels,
                        hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
                except Exception:
                    pass
            if config is None and variant in self.VARIANT_CONFIGS:
                config = SegformerConfig(
                    num_labels=num_classes, num_channels=in_channels,
                    hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate,
                    **self.VARIANT_CONFIGS[variant])
            if config is None:
                config = SegformerConfig(num_labels=num_classes, num_channels=in_channels,
                                         hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
            self.model = SegformerForSemanticSegmentation(config)

        if in_channels != 3 and pretrained:
            self._adapt_input_channels(in_channels)

    def _adapt_input_channels(self, in_channels):
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        with torch.no_grad():
            if in_channels > 3:
                new_conv.weight[:, :3] = old_conv.weight
                for i in range(3, in_channels):
                    new_conv.weight[:, i] = old_conv.weight[:, i % 3]
            else:
                new_conv.weight = nn.Parameter(old_conv.weight[:, :in_channels])
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return F.interpolate(outputs.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


class UNetFormer(nn.Module):
    """UNetFormer with configurable dropout"""
    def __init__(self, num_classes, in_channels, encoder_name='resnet18', pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        if DEPS.get('timm'):
            import timm
            self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                              features_only=True, in_chans=in_channels)
            self.encoder_channels = self.encoder.feature_info.channels()
        elif DEPS.get('smp'):
            import segmentation_models_pytorch as smp
            aux = smp.Unet(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes,
                           encoder_weights='imagenet' if pretrained else None)
            self.encoder = aux.encoder
            self.encoder_channels = list(aux.encoder.out_channels[1:])
        else:
            raise ImportError("pip install timm or segmentation_models_pytorch")
        dec_ch = [256, 128, 64, 64]
        layers = nn.ModuleList()
        in_ch = self.encoder_channels[-1]
        for i, out_ch in enumerate(dec_ch):
            skip_ch = self.encoder_channels[-(i + 2)] if i < len(self.encoder_channels) - 1 else 0
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU()))
            in_ch = out_ch
        self.decoder = layers
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        if isinstance(features, dict):
            features = list(features.values())
        out = features[-1]
        for i, layer in enumerate(self.decoder):
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(features) - 1:
                skip = features[-(i + 2)]
                if skip.shape[-2:] != out.shape[-2:]:
                    skip = F.interpolate(skip, size=out.shape[-2:], mode='bilinear')
                out = torch.cat([out, skip], dim=1)
            out = layer(out)
        out = self.final_conv(out)
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


class HRNetSegmentation(nn.Module):
    """HRNet with configurable dropout"""
    def __init__(self, variant, num_classes, in_channels, pretrained=True, dropout_rate=0.3):
        super().__init__()
        if not DEPS.get('timm'):
            raise ImportError("pip install timm")
        import timm
        self.backbone = timm.create_model(variant, pretrained=pretrained,
                                           features_only=True, in_chans=in_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            features = self.backbone(dummy)
            total_channels = sum(f.shape[1] for f in features)
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate), nn.Conv2d(256, num_classes, 1))

    def forward(self, x):
        features = self.backbone(x)
        target_size = features[0].shape[-2:]
        fused = []
        for f in features:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            fused.append(f)
        out = self.head(torch.cat(fused, dim=1))
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)


class SwinUNet(nn.Module):
    """Swin-UNet with configurable dropout"""
    def __init__(self, num_classes, in_channels, pretrained=True, dropout_rate=0.3):
        super().__init__()
        if not DEPS.get('timm'):
            raise ImportError("pip install timm")
        import timm
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained,
                                          features_only=True, in_chans=in_channels)
        ec = self.encoder.feature_info.channels()

        def _cb(ic, oc, dr):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.GELU(),
                nn.Dropout2d(p=dr),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.GELU())

        self.up4 = nn.ConvTranspose2d(ec[-1], 256, 2, stride=2)
        self.conv4 = _cb(256 + ec[-2], 256, dropout_rate)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = _cb(128 + ec[-3], 128, dropout_rate)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = _cb(64 + ec[-4], 64, dropout_rate)
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=4), nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        f = self.encoder(x)
        d4 = torch.cat([self.up4(f[-1]), F.interpolate(f[-2], size=self.up4(f[-1]).shape[-2:], mode='bilinear')], 1)
        d4 = self.conv4(d4)
        d3 = torch.cat([self.up3(d4), F.interpolate(f[-3], size=self.up3(d4).shape[-2:], mode='bilinear')], 1)
        d3 = self.conv3(d3)
        d2 = torch.cat([self.up2(d3), F.interpolate(f[-4], size=self.up2(d3).shape[-2:], mode='bilinear')], 1)
        d2 = self.conv2(d2)
        return F.interpolate(self.final(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)


# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Factory for creating all supported segmentation models"""

    MODERN_MODELS = {
        'segformer-b0': ('segformer', 'b0'), 'segformer-b1': ('segformer', 'b1'),
        'segformer-b2': ('segformer', 'b2'), 'segformer-b3': ('segformer', 'b3'),
        'segformer-b4': ('segformer', 'b4'), 'segformer-b5': ('segformer', 'b5'),
        'unetformer': ('unetformer', None),
        'hrnet-w18': ('hrnet', 'hrnet_w18'), 'hrnet-w32': ('hrnet', 'hrnet_w32'),
        'hrnet-w48': ('hrnet', 'hrnet_w48'),
        'swin-unet': ('swin-unet', None),
    }
    SMP_MODELS = ['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'manet', 'fpn', 'pan', 'pspnet', 'linknet']
    CONVNEXT_ENCODERS = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge']

    @classmethod
    def list_models(cls):
        models = ['unet-dropout'] + list(cls.MODERN_MODELS.keys())
        if DEPS.get('smp'):
            models.extend(cls.SMP_MODELS)
        return models

    @classmethod
    def list_encoders(cls):
        encoders = ['resnet34', 'resnet50', 'resnet101',
                     'efficientnet-b0', 'efficientnet-b3', 'efficientnet-b4',
                     'se_resnext50_32x4d']
        if DEPS.get('timm'):
            encoders.extend(cls.CONVNEXT_ENCODERS)
        return encoders

    @classmethod
    def create(cls, model_name, encoder_name='resnet34', in_channels=4, num_classes=2,
               mode='multiclass', pretrained=False, dropout_rate=0.3):
        """Build a segmentation model"""
        name = model_name.lower()
        actual_classes = 1 if mode == 'binary' else num_classes

        # Built-in fallback
        if name == 'unet-dropout':
            return SimpleUNet(in_channels, actual_classes, dropout_rate)

        # ConvNeXt encoder → custom UNet
        if encoder_name.lower() in cls.CONVNEXT_ENCODERS:
            if name in ('unet', 'u-net', 'u_net'):
                print(f"  ✨ ConvNeXt-UNet (encoder: {encoder_name})")
                return ConvNeXtUNet(encoder_name, actual_classes, in_channels, pretrained, dropout_rate)
            else:
                print(f"  ⚠ ConvNeXt only supported with UNet, falling back to UNet")
                return ConvNeXtUNet(encoder_name, actual_classes, in_channels, pretrained, dropout_rate)

        # Modern architectures
        if name in cls.MODERN_MODELS:
            model_type, variant = cls.MODERN_MODELS[name]
            if model_type == 'segformer':
                return SegFormerWrapper(name, actual_classes, in_channels, pretrained, dropout_rate)
            elif model_type == 'unetformer':
                return UNetFormer(actual_classes, in_channels, encoder_name, pretrained, dropout_rate)
            elif model_type == 'hrnet':
                return HRNetSegmentation(variant, actual_classes, in_channels, pretrained, dropout_rate)
            elif model_type == 'swin-unet':
                return SwinUNet(actual_classes, in_channels, pretrained, dropout_rate)

        # SMP models
        if DEPS.get('smp'):
            import segmentation_models_pytorch as smp
            smp_map = {
                'unet': smp.Unet, 'unet++': smp.UnetPlusPlus, 'deeplabv3+': smp.DeepLabV3Plus,
                'deeplabv3': smp.DeepLabV3, 'manet': smp.MAnet, 'fpn': smp.FPN,
                'pan': smp.PAN, 'pspnet': smp.PSPNet, 'linknet': smp.Linknet,
            }
            # Normalize name variants
            name_variants = {
                'unet': ['unet', 'u-net', 'u_net'],
                'unet++': ['unet++', 'unetplusplus', 'unetpp'],
                'deeplabv3+': ['deeplabv3+', 'deeplabv3plus'],
                'deeplabv3': ['deeplabv3', 'deeplab-v3'],
                'manet': ['manet', 'ma-net'], 'fpn': ['fpn'], 'pan': ['pan'],
                'pspnet': ['pspnet', 'psp-net'], 'linknet': ['linknet', 'link-net'],
            }
            for canonical, variants in name_variants.items():
                if name in variants:
                    try:
                        model = smp_map[canonical](
                            encoder_name=encoder_name, in_channels=in_channels,
                            classes=actual_classes,
                            encoder_weights='imagenet' if pretrained else None,
                            activation=None)
                        print(f"  ✓ Created {canonical} with {encoder_name} encoder")
                        return model
                    except Exception as e:
                        print(f"  ⚠ Failed to create {canonical} with {encoder_name}: {e}")
                        print(f"    Falling back to unet-dropout")
                        return SimpleUNet(in_channels, actual_classes, dropout_rate)

        # Final fallback
        print(f"  ⚠ Unknown model '{model_name}', falling back to unet-dropout")
        return SimpleUNet(in_channels, actual_classes, dropout_rate)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(output, target, mode, num_classes, threshold=0.5):
    """Compute comprehensive segmentation metrics"""
    with torch.no_grad():
        if mode == 'binary':
            pred = (torch.sigmoid(output) > threshold).float()
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            if pred.dim() == 4 and pred.size(1) == 1:
                pred = pred.squeeze(1)

            tp = ((pred == 1) & (target == 1)).sum().float()
            tn = ((pred == 0) & (target == 0)).sum().float()
            fp = ((pred == 1) & (target == 0)).sum().float()
            fn = ((pred == 0) & (target == 1)).sum().float()
            eps = 1e-6

            iou = (tp / (tp + fp + fn + eps)).item()
            precision = (tp / (tp + fp + eps)).item()
            recall = (tp / (tp + fn + eps)).item()
            f1 = 2 * precision * recall / (precision + recall + eps)
            accuracy = ((tp + tn) / (tp + tn + fp + fn + eps)).item()

            return {'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}
        else:
            pred = output.argmax(dim=1)
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1)
            if target.dtype == torch.float32:
                target = target.long()

            ious, f1s, precisions, recalls = [], [], [], []
            per_class = {}
            for c in range(num_classes):
                pred_c, target_c = (pred == c), (target == c)
                tp = (pred_c & target_c).sum().float()
                fp = (pred_c & ~target_c).sum().float()
                fn = (~pred_c & target_c).sum().float()
                eps = 1e-6
                iou_c = (tp / (tp + fp + fn + eps)).item()
                prec_c = (tp / (tp + fp + eps)).item()
                rec_c = (tp / (tp + fn + eps)).item()
                f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
                per_class[f'iou_class_{c}'] = iou_c
                per_class[f'f1_class_{c}'] = f1_c
                if (tp + fp + fn) > 0:
                    ious.append(iou_c)
                    f1s.append(f1_c)
                    precisions.append(prec_c)
                    recalls.append(rec_c)

            correct = (pred == target).sum().float()
            accuracy = (correct / target.numel()).item()

            return {
                'mean_iou': np.mean(ious) if ious else 0.0,
                'f1': np.mean(f1s) if f1s else 0.0,
                'precision': np.mean(precisions) if precisions else 0.0,
                'recall': np.mean(recalls) if recalls else 0.0,
                'accuracy': accuracy,
                'per_class': per_class
            }


# ============================================================================
# DATASET
# ============================================================================

class SegmentationDataset(Dataset):
    """Dataset for segmentation patches"""

    def __init__(self, folder, mode='binary', num_classes=2, transform=None, in_channels=None):
        self.image_dir = Path(folder) / 'images'
        self.mask_dir = Path(folder) / 'labels'
        self.mode = mode
        self.num_classes = num_classes
        self.transform = transform
        self.in_channels = in_channels

        exts = {'.tif', '.tiff', '.TIF', '.TIFF'}
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix in exts])
        self.masks = sorted([p for p in self.mask_dir.iterdir() if p.suffix in exts])

        # Build pairs by matching stems
        self.pairs = []
        mask_dict = {m.stem: m for m in self.masks}
        for img_path in self.images:
            if img_path.stem in mask_dict:
                self.pairs.append((img_path, mask_dict[img_path.stem]))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No matching image-mask pairs in {folder}")
        print(f"  Dataset: {len(self.pairs)} pairs in {folder}")

        # Minority class analysis (for weighted sampling)
        self.has_minority_class = []
        for _, lbl_path in self.pairs:
            label = read_tiff(str(lbl_path))
            self.has_minority_class.append(bool((label == 1).any()))

    def get_sample_weights(self):
        n_minority = sum(self.has_minority_class)
        n_majority = len(self.pairs) - n_minority
        if n_minority == 0:
            return torch.ones(len(self.pairs))
        w_minority = n_majority / n_minority if n_minority > 0 else 1.0
        return torch.tensor([w_minority if h else 1.0 for h in self.has_minority_class])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = read_tiff(str(img_path)).astype(np.float32)
        mask = read_tiff(str(mask_path))

        # Ensure spatial dimensions match
        if img.ndim == 2:
            img_h, img_w = img.shape
        else:
            img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if mask_h != img_h or mask_w != img_w:
            mask = cv2.resize(mask.astype(np.float32), (img_w, img_h),
                              interpolation=cv2.INTER_NEAREST).astype(mask.dtype)

        # Normalize (99th percentile)
        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / (p99 + 1e-6), 0, 1)

        # Handle dimensions
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Handle mask
        if self.mode == 'binary':
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask[np.newaxis, ...]).float()
        else:
            mask = np.clip(mask.astype(np.int64), 0, self.num_classes - 1)
            mask = torch.from_numpy(mask).long()

        # Augmentation
        if self.transform:
            has_min = self.has_minority_class[idx] if idx < len(self.has_minority_class) else False
            try:
                img, mask = self.transform(img, mask, has_min)
            except TypeError:
                img, mask = self.transform(img, mask)

        return img, mask


# ============================================================================
# K-FOLD DATASET AND CROSS-VALIDATOR
# ============================================================================

class _DatasetFromPairs(Dataset):
    """Dataset created from explicit list of (image_path, mask_path) pairs.
    Used by K-Fold cross-validation to create train/val splits from merged data."""

    def __init__(self, pairs, mode='binary', num_classes=2, transform=None, in_channels=None):
        self.pairs = pairs
        self.mode = mode
        self.num_classes = num_classes
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = read_tiff(str(img_path)).astype(np.float32)
        msk = read_tiff(str(mask_path))

        if img.ndim == 2:
            img_h, img_w = img.shape
        else:
            img_h, img_w = img.shape[:2]
        msk_h, msk_w = msk.shape[:2]
        if msk_h != img_h or msk_w != img_w:
            msk = cv2.resize(msk.astype(np.float32), (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST).astype(msk.dtype)

        if np.max(img) > 0:
            p99 = np.percentile(img, 99)
            img = np.clip(img / (p99 + 1e-6), 0, 1)

        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.mode == 'binary':
            msk = (msk > 0).astype(np.float32)
            msk = torch.from_numpy(msk[np.newaxis, ...]).float()
        else:
            msk = np.clip(msk.astype(np.int64), 0, self.num_classes - 1)
            msk = torch.from_numpy(msk).long()

        if self.transform:
            try:
                img, msk = self.transform(img, msk, False)
            except TypeError:
                img, msk = self.transform(img, msk)

        return img, msk


class KFoldCrossValidator:
    """K-Fold cross-validation for segmentation datasets.
    Merges all subdirectories (train/validation/test) and creates K folds."""

    def __init__(self, dataset_root, subdirectories=None, n_splits=5,
                 random_state=42, mode='binary', num_classes=2):
        self.dataset_root = Path(dataset_root)
        self.subdirectories = subdirectories or ['train', 'validation', 'test']
        self.n_splits = n_splits
        self.random_state = random_state
        self.mode = mode
        self.num_classes = num_classes
        self.folds = []

    def prepare_folds(self):
        """Collect all pairs from all subdirs, then create K-Fold splits."""
        all_pairs = []
        valid_extensions = {'.tif', '.tiff', '.TIF', '.TIFF'}

        # Try Patch/subdir/images or subdir/images
        for subdir in self.subdirectories:
            for base in [self.dataset_root / 'Patch', self.dataset_root]:
                image_dir = base / subdir / 'images'
                mask_dir = base / subdir / 'labels'
                if image_dir.exists() and mask_dir.exists():
                    images = sorted([p for p in image_dir.iterdir()
                                     if p.is_file() and p.suffix in valid_extensions])
                    masks = {m.stem: m for m in mask_dir.iterdir()
                             if m.is_file() and m.suffix in valid_extensions}
                    for img_path in images:
                        if img_path.stem in masks:
                            all_pairs.append((img_path, masks[img_path.stem]))
                    break  # found this subdir, don't check other base

        if not all_pairs:
            raise RuntimeError("No image-mask pairs found for K-Fold.")

        print(f"\nK-Fold: {len(all_pairs)} total pairs collected")

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_pairs)):
            train_pairs = [all_pairs[i] for i in train_idx]
            val_pairs = [all_pairs[i] for i in val_idx]
            self.folds.append({
                'fold': fold_idx,
                'train_pairs': train_pairs,
                'val_pairs': val_pairs,
            })
            print(f"  Fold {fold_idx + 1}: {len(train_pairs)} train, {len(val_pairs)} val")

    def get_fold_datasets(self, fold_idx, transform=None):
        fold = self.folds[fold_idx]
        train_ds = _DatasetFromPairs(fold['train_pairs'], transform=transform,
                                     mode=self.mode, num_classes=self.num_classes)
        val_ds = _DatasetFromPairs(fold['val_pairs'], transform=None,
                                   mode=self.mode, num_classes=self.num_classes)
        return train_ds, val_ds


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def freeze_encoder(model):
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['encoder', 'backbone', 'segformer.encoder']):
            param.requires_grad = False


def unfreeze_encoder(model):
    for name, param in model.named_parameters():
        if any(x in name.lower() for x in ['encoder', 'backbone', 'segformer.encoder']):
            param.requires_grad = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_class_weights(dataloader, device, mode, num_classes):
    """Calculate class weights from training data"""
    print("  Calculating class weights...")
    if mode == 'binary':
        total, positive = 0, 0
        for _, masks in dataloader:
            positive += (masks > 0).sum().item()
            total += masks.numel()
        negative = total - positive
        weight = negative / (positive + 1e-6)
        print(f"    Positive: {positive:,} ({100 * positive / total:.2f}%), Weight: {weight:.2f}")
        return torch.tensor([weight], device=device)
    else:
        class_counts = torch.zeros(num_classes)
        total = 0
        for _, masks in dataloader:
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            masks = masks.long()
            for c in range(num_classes):
                class_counts[c] += (masks == c).sum().item()
            total += masks.numel()
        weights = total / (num_classes * class_counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        for c in range(num_classes):
            print(f"    Class {c}: {class_counts[c]:,.0f} px ({100 * class_counts[c] / total:.2f}%), weight: {weights[c]:.2f}")
        return weights.to(device)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(params, progress_file):
    """Main training function - called via JSON params from QGIS plugin"""

    sys.stdout.reconfigure(line_buffering=True)

    # === BASIC PARAMETERS ===
    dataset_root = params['dataset_root']
    mode = params.get('mode', 'binary')
    num_classes = params.get('num_classes', 2)
    model_name = params.get('model_name', 'unet')
    encoder_name = params.get('encoder_name', 'resnet34')
    pretrained = params.get('pretrained', False)
    in_channels = params.get('in_channels', 10)
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 4)
    lr = params.get('learning_rate', 0.001)
    device_str = params.get('device', 'cuda')
    save_dir = params.get('save_dir', './trained_models')

    # === ADVANCED PARAMETERS (new) ===
    scheduler_type = params.get('scheduler', 'reduce_on_plateau')
    freeze_encoder_flag = params.get('freeze_encoder', False)
    freeze_epochs = params.get('freeze_epochs', 5)
    early_stopping = params.get('early_stopping', True)
    patience = params.get('patience', 15)
    loss_function = params.get('loss_function', 'dice_ce')
    use_class_weights = params.get('use_class_weights', False)
    dropout_rate = params.get('dropout_rate', 0.3)
    augmentation_level = params.get('augmentation_level', 'none')  # none/basic/advanced/aggressive/extreme
    use_amp = params.get('use_amp', False)
    warmup_epochs = params.get('warmup_epochs', 0)
    focal_gamma = params.get('focal_gamma', 2.0)
    focal_alpha = params.get('focal_alpha', 0.25)
    tversky_alpha = params.get('tversky_alpha', 0.3)
    tversky_beta = params.get('tversky_beta', 0.7)

    # K-Fold parameters
    use_kfold = params.get('use_kfold', False)
    n_splits = params.get('n_splits', 5)

    # Legacy augmentation support
    if params.get('data_augmentation', False) and augmentation_level == 'none':
        augmentation_level = 'basic'

    update_progress(progress_file, 0, "Initializing...")

    # Print configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION - SemanticSeg4EO")
    print("=" * 60)
    print(f"Model: {model_name} + {encoder_name}")
    print(f"Mode: {mode} ({num_classes} classes)")
    print(f"Input channels: {in_channels}")
    print(f"Pretrained: {pretrained}")
    print(f"Dropout: {dropout_rate}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Loss: {loss_function}")
    print(f"Augmentation: {augmentation_level}")
    print(f"Mixed Precision (AMP): {use_amp}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Class weights: {use_class_weights}")
    print(f"Freeze encoder: {freeze_encoder_flag} ({freeze_epochs} epochs)")
    print(f"Early stopping: {early_stopping} (patience={patience})")
    if use_kfold:
        print(f"K-Fold: {n_splits} folds")
    print("=" * 60)
    sys.stdout.flush()

    # Device
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
        try:
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass
    else:
        device = torch.device('cpu')
        if device_str == 'cuda':
            print("⚠ CUDA not available, using CPU")

    os.makedirs(save_dir, exist_ok=True)
    update_progress(progress_file, 5, "Loading datasets...")

    # Dataset paths
    train_path = os.path.join(dataset_root, 'Patch', 'train')
    val_path = os.path.join(dataset_root, 'Patch', 'validation')
    if not os.path.exists(train_path):
        train_path = os.path.join(dataset_root, 'train')
        val_path = os.path.join(dataset_root, 'validation')

    # Augmentation
    try:
        aug_level = AugmentationLevel(augmentation_level)
    except ValueError:
        aug_level = AugmentationLevel.NONE
    aug_config = get_augmentation_config(aug_level, in_channels)
    transform = AdvancedMultiChannelAugmentation(aug_config, mode=mode) if aug_config.enabled else None

    # ================================================================
    # K-FOLD CROSS-VALIDATION BRANCH
    # ================================================================
    if use_kfold:
        if not DEPS.get('sklearn'):
            print("ERROR: K-Fold requires scikit-learn: pip install scikit-learn")
            update_progress(progress_file, -1, "scikit-learn not installed")
            return "Error: scikit-learn required for K-Fold"

        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION ({n_splits} folds)")
        print(f"{'='*60}")

        cv = KFoldCrossValidator(
            dataset_root=dataset_root,
            n_splits=n_splits,
            random_state=42,
            mode=mode,
            num_classes=num_classes,
        )
        cv.prepare_folds()

        actual_classes = 1 if mode == 'binary' else num_classes
        all_fold_results = []

        for fold_idx in range(n_splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'='*60}")
            sys.stdout.flush()

            fold_train_ds, fold_val_ds = cv.get_fold_datasets(fold_idx, transform=transform)
            fold_bs = min(batch_size, len(fold_train_ds))
            fold_train_loader = DataLoader(fold_train_ds, batch_size=fold_bs, shuffle=True, num_workers=0)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=fold_bs, num_workers=0)

            print(f"  Train: {len(fold_train_ds)}, Val: {len(fold_val_ds)}")

            # Create fresh model for each fold
            model = ModelFactory.create(model_name, encoder_name, in_channels,
                                        actual_classes, mode, pretrained, dropout_rate)
            model = model.to(device)

            # Loss
            class_weight = None
            if use_class_weights:
                class_weight = calculate_class_weights(fold_train_loader, device, mode, num_classes)
            criterion = LossFactory.create(loss_function, mode, num_classes, class_weight,
                                           focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                                           tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)

            # Optimizer + Scheduler
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, weight_decay=1e-4)
            if scheduler_type == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            elif scheduler_type in ('cosine_annealing', 'cosine'):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            elif scheduler_type == 'one_cycle':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10,
                                                                epochs=epochs, steps_per_epoch=len(fold_train_loader))
            else:
                scheduler = None

            scaler = None
            if use_amp and device.type == 'cuda':
                try:
                    from torch.cuda.amp import GradScaler
                    scaler = GradScaler()
                except ImportError:
                    pass

            best_iou = 0.0
            best_loss = float('inf')
            best_f1 = 0.0
            epochs_no_improve = 0
            
            # Historique d'entraînement pour ce fold (comme dans le baseline)
            fold_history = {
                'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [],
                'val_precision': [], 'val_recall': [], 'val_accuracy': [], 'lr': []
            }

            for epoch in range(epochs):
                current_lr = optimizer.param_groups[0]['lr']
                if epoch < warmup_epochs:
                    warmup_lr = lr * (epoch + 1) / warmup_epochs
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

                model.train()
                train_loss = 0.0
                for imgs, masks in fold_train_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    optimizer.zero_grad()
                    if scaler is not None:
                        from torch.cuda.amp import autocast
                        with autocast():
                            outputs = model(imgs)
                            loss = criterion(outputs, masks)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(imgs)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                    if scheduler_type == 'one_cycle' and scheduler:
                        scheduler.step()
                    train_loss += loss.item()
                train_loss /= len(fold_train_loader)

                model.eval()
                val_metrics = []
                with torch.no_grad():
                    for imgs, masks in fold_val_loader:
                        imgs, masks = imgs.to(device), masks.to(device)
                        outputs = model(imgs)
                        m = compute_metrics(outputs, masks, mode, num_classes)
                        val_metrics.append(m)

                val_iou = np.mean([m.get('iou', m.get('mean_iou', 0)) for m in val_metrics])
                val_f1 = np.mean([m['f1'] for m in val_metrics])
                val_precision = np.mean([m.get('precision', 0) for m in val_metrics])
                val_recall = np.mean([m.get('recall', 0) for m in val_metrics])
                val_accuracy = np.mean([m.get('accuracy', 0) for m in val_metrics])
                val_loss = train_loss  # Approximation (pas de validation loss séparé dans le code actuel)
                
                # Sauvegarder l'historique
                current_lr = optimizer.param_groups[0]['lr']
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['val_iou'].append(val_iou)
                fold_history['val_f1'].append(val_f1)
                fold_history['val_precision'].append(val_precision)
                fold_history['val_recall'].append(val_recall)
                fold_history['val_accuracy'].append(val_accuracy)
                fold_history['lr'].append(current_lr)

                if scheduler and scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_iou)
                elif scheduler and scheduler_type in ('cosine_annealing', 'cosine'):
                    scheduler.step()

                # Sauvegarder les meilleurs modèles (comme dans le baseline)
                model_metadata = {
                    'model_name': model_name, 'mode': mode,
                    'in_channels': in_channels, 'num_classes': actual_classes,
                    'encoder_name': encoder_name, 'dropout_rate': dropout_rate,
                    'fold': fold_idx + 1,
                }
                
                # Best IoU
                if val_iou > best_iou:
                    best_iou = val_iou
                    epochs_no_improve = 0
                    torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata},
                               os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_best_iou.pth'))
                else:
                    epochs_no_improve += 1
                
                # Best Loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata},
                               os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_best_loss.pth'))
                
                # Best F1
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata},
                               os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_best_f1.pth'))

                # Progress - afficher plus régulièrement
                fold_progress = 5 + int(90 * ((fold_idx * epochs + epoch + 1) / (n_splits * epochs)))
                update_progress(progress_file, min(fold_progress, 95),
                                f"Fold {fold_idx+1}/{n_splits} — Epoch {epoch+1}/{epochs}: IoU={val_iou:.4f}, F1={val_f1:.4f}")

                # Afficher toutes les 3 epochs au lieu de 10 pour un meilleur suivi
                if epoch % 3 == 0 or epoch == epochs - 1 or epoch < 5:
                    print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, IoU={val_iou:.4f}, F1={val_f1:.4f}")
                    sys.stdout.flush()

                if early_stopping and epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            # ============================================================
            # SAUVEGARDES COMPLÈTES POUR CE FOLD (comme dans le baseline)
            # ============================================================
            
            # 1. Sauvegarder l'historique JSON
            # Extraire le niveau d'augmentation de manière sûre
            try:
                if hasattr(aug_config, 'level'):
                    aug_level_str = aug_config.level
                elif hasattr(aug_config, 'enabled'):
                    aug_level_str = 'enabled' if aug_config.enabled else 'none'
                else:
                    aug_level_str = str(aug_level) if 'aug_level' in locals() else 'unknown'
            except:
                aug_level_str = 'unknown'
            
            # Convertir fold_history en types Python natifs (éviter numpy)
            serializable_history = {}
            for key, values in fold_history.items():
                serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            
            fold_json = {
                'fold': fold_idx + 1,
                'model_name': model_name,
                'architecture': model_name,
                'encoder': encoder_name,
                'best_val_iou': float(best_iou),
                'best_val_loss': float(best_loss),
                'best_val_f1': float(best_f1),
                'training_epochs': len(fold_history['train_loss']),
                'training_history': serializable_history,
                'val_metrics': {
                    'mean_iou': float(best_iou),
                    'mean_f1': float(best_f1),
                    'precision': float(val_precision),
                    'recall': float(val_recall),
                    'accuracy': float(val_accuracy)
                },
                'config': {
                    'batch_size': int(batch_size),
                    'learning_rate': float(lr),
                    'epochs': int(epochs),
                    'loss_function': str(loss_function),
                    'use_class_weights': bool(use_class_weights),
                    'augmentation_level': aug_level_str,
                    'freeze_encoder': bool(freeze_encoder),
                    'warmup_epochs': int(warmup_epochs),
                    'use_amp': bool(use_amp),
                    'dropout_rate': float(dropout_rate)
                }
            }
            
            fold_json_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_metrics.json')
            with open(fold_json_path, 'w') as f:
                json.dump(fold_json, f, indent=2)
            print(f"  ✓ JSON sauvegardé: {fold_json_path}")
            
            # 2. Sauvegarder CSV log
            try:
                import csv
                csv_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_training_log.csv')
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1', 
                                   'val_precision', 'val_recall', 'val_accuracy', 'lr'])
                    for i in range(len(fold_history['train_loss'])):
                        writer.writerow([
                            i + 1,
                            fold_history['train_loss'][i],
                            fold_history['val_loss'][i],
                            fold_history['val_iou'][i],
                            fold_history['val_f1'][i],
                            fold_history['val_precision'][i],
                            fold_history['val_recall'][i],
                            fold_history['val_accuracy'][i],
                            fold_history['lr'][i]
                        ])
                print(f"  ✓ CSV log sauvegardé: {csv_path}")
            except Exception as e:
                print(f"  ⚠ Impossible de sauvegarder le CSV: {e}")
            
            # 3. Créer le plot d'entraînement
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                epochs_range = range(1, len(fold_history['train_loss']) + 1)
                
                # Loss
                axes[0, 0].plot(epochs_range, fold_history['train_loss'], 'b-', label='Train', linewidth=2)
                axes[0, 0].plot(epochs_range, fold_history['val_loss'], 'r-', label='Validation', linewidth=2)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title(f'Fold {fold_idx + 1} - Training & Validation Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # IoU
                axes[0, 1].plot(epochs_range, fold_history['val_iou'], 'g-', linewidth=2)
                axes[0, 1].axhline(y=best_iou, color='r', linestyle='--', label=f'Best: {best_iou:.4f}')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('IoU')
                axes[0, 1].set_title(f'Fold {fold_idx + 1} - Validation IoU')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # F1
                axes[1, 0].plot(epochs_range, fold_history['val_f1'], 'm-', linewidth=2)
                axes[1, 0].axhline(y=best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.4f}')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].set_title(f'Fold {fold_idx + 1} - Validation F1')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Learning Rate
                axes[1, 1].plot(epochs_range, fold_history['lr'], 'c-', linewidth=2)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title(f'Fold {fold_idx + 1} - Learning Rate Schedule')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}_training_plot.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Plot sauvegardé: {plot_path}")
            except Exception as e:
                print(f"  ⚠ Impossible de créer le plot: {e}")
            
            # Collect fold results (convertir en types Python natifs)
            fold_result = {
                'fold': int(fold_idx + 1), 
                'best_iou': float(best_iou), 
                'best_f1': float(best_f1),
                'best_loss': float(best_loss),
                'precision': float(val_precision), 
                'recall': float(val_recall),
                'accuracy': float(val_accuracy),
                'epochs_completed': int(len(fold_history['train_loss'])),
                'history': fold_history  # Sera filtré dans le JSON global
            }
            all_fold_results.append(fold_result)
            print(f"\n  ✓ Fold {fold_idx + 1} terminé - Best IoU: {best_iou:.4f} | Best F1: {best_f1:.4f} | Best Loss: {best_loss:.4f}")

            # Free GPU memory
            del model, optimizer, criterion
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # ---- K-Fold Summary ----
        ious = [r['best_iou'] for r in all_fold_results]
        f1s = [r['best_f1'] for r in all_fold_results]
        losses = [r['best_loss'] for r in all_fold_results]
        precisions = [r['precision'] for r in all_fold_results]
        recalls = [r['recall'] for r in all_fold_results]
        accuracies = [r['accuracy'] for r in all_fold_results]
        
        mean_iou, std_iou = np.mean(ious), np.std(ious)
        mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
        mean_loss, std_loss = np.mean(losses), np.std(losses)
        mean_prec, std_prec = np.mean(precisions), np.std(precisions)
        mean_rec, std_rec = np.mean(recalls), np.std(recalls)
        mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)

        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION RESULTS ({n_splits} folds)")
        print(f"{'='*60}")
        for r in all_fold_results:
            print(f"  Fold {r['fold']}: IoU={r['best_iou']:.4f}, F1={r['best_f1']:.4f}, Loss={r['best_loss']:.4f}")
        print(f"\n  Mean IoU      : {mean_iou:.4f} +/- {std_iou:.4f}")
        print(f"  Mean F1       : {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  Mean Loss     : {mean_loss:.4f} +/- {std_loss:.4f}")
        print(f"  Mean Precision: {mean_prec:.4f} +/- {std_prec:.4f}")
        print(f"  Mean Recall   : {mean_rec:.4f} +/- {std_rec:.4f}")
        print(f"  Mean Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}")

        # 95% confidence interval
        ci_iou = None
        ci_f1 = None
        if n_splits > 1:
            try:
                import scipy.stats
                ci_iou = scipy.stats.t.interval(0.95, n_splits - 1, loc=mean_iou,
                                                scale=std_iou / np.sqrt(n_splits))
                ci_f1 = scipy.stats.t.interval(0.95, n_splits - 1, loc=mean_f1,
                                               scale=std_f1 / np.sqrt(n_splits))
                print(f"  IoU 95% CI: [{ci_iou[0]:.4f}, {ci_iou[1]:.4f}]")
                print(f"  F1  95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
            except Exception:
                pass

        print(f"{'='*60}")
        sys.stdout.flush()

        # Save K-Fold results complets (comme dans le baseline)
        # Extraire le niveau d'augmentation de manière sûre
        try:
            if hasattr(aug_config, 'level'):
                aug_level_str = aug_config.level
            elif hasattr(aug_config, 'enabled'):
                aug_level_str = 'enabled' if aug_config.enabled else 'none'
            else:
                aug_level_str = str(aug_level) if 'aug_level' in locals() else 'unknown'
        except:
            aug_level_str = 'unknown'
        
        kfold_results = {
            'n_splits': int(n_splits),
            'model_name': str(model_name),
            'architecture': str(model_name),
            'encoder': str(encoder_name),
            'cv_stats': {
                'mean_iou': float(mean_iou), 'std_iou': float(std_iou),
                'mean_f1': float(mean_f1), 'std_f1': float(std_f1),
                'mean_loss': float(mean_loss), 'std_loss': float(std_loss),
                'mean_precision': float(mean_prec), 'std_precision': float(std_prec),
                'mean_recall': float(mean_rec), 'std_recall': float(std_rec),
                'mean_accuracy': float(mean_acc), 'std_accuracy': float(std_acc),
                'all_ious': [float(x) for x in ious],
                'all_f1s': [float(x) for x in f1s],
                'all_losses': [float(x) for x in losses],
                'all_precisions': [float(x) for x in precisions],
                'all_recalls': [float(x) for x in recalls],
                'all_accuracies': [float(x) for x in accuracies],
                'ci_iou_95': {'lower': float(ci_iou[0]), 'upper': float(ci_iou[1])} if ci_iou else None,
                'ci_f1_95': {'lower': float(ci_f1[0]), 'upper': float(ci_f1[1])} if ci_f1 else None,
            },
            'fold_results': [{k: v for k, v in fr.items() if k != 'history'} for fr in all_fold_results],
            'config': {
                'batch_size': int(batch_size),
                'learning_rate': float(lr),
                'epochs': int(epochs),
                'loss_function': str(loss_function),
                'use_class_weights': bool(use_class_weights),
                'augmentation_level': aug_level_str,
                'freeze_encoder': bool(freeze_encoder),
                'warmup_epochs': int(warmup_epochs),
                'use_amp': bool(use_amp),
                'dropout_rate': float(dropout_rate),
                'early_stopping': bool(early_stopping),
                'patience': int(patience)
            },
            'save_dir': str(save_dir)
        }
        
        json_path = os.path.join(save_dir, f'{model_name}_kfold_complete_results.json')
        with open(json_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        print(f"\n✓ Résultats globaux K-Fold sauvegardés: {json_path}")
        print(f"✓ Fichiers par fold: {n_splits} x (JSON + CSV + Plot + 3 modèles .pth)")

        summary = f"K-Fold ({n_splits}): Mean IoU={mean_iou:.4f}±{std_iou:.4f}, Mean F1={mean_f1:.4f}±{std_f1:.4f}"
        update_progress(progress_file, 100, "Complete!", summary)
        return summary

    # ================================================================
    # STANDARD TRAINING (no K-Fold)
    # ================================================================

    # Datasets
    train_ds = SegmentationDataset(train_path, mode, num_classes, transform, in_channels)
    val_ds = SegmentationDataset(val_path, mode, num_classes, None, in_channels)

    # Weighted sampling for minority oversampling
    sampler = None
    shuffle = True
    if aug_config.enabled and aug_config.minority_oversample:
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print("  ✓ Using WeightedRandomSampler for minority oversampling")

    batch_size = min(batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")
    update_progress(progress_file, 10, f"Loaded {len(train_ds)} train, {len(val_ds)} val samples")

    # Model
    actual_classes = 1 if mode == 'binary' else num_classes
    model = ModelFactory.create(model_name, encoder_name, in_channels, actual_classes, mode, pretrained, dropout_rate)
    model = model.to(device)

    # Freeze encoder
    encoder_frozen = False
    if freeze_encoder_flag and model_name != 'unet-dropout':
        freeze_encoder(model)
        encoder_frozen = True
        print(f"  ✓ Encoder frozen for first {freeze_epochs} epochs")
        print(f"  Trainable: {count_parameters(model):,} / {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"  Parameters: {count_parameters(model):,} (all trainable)")
    sys.stdout.flush()

    # Class weights
    class_weight = None
    if use_class_weights:
        class_weight = calculate_class_weights(train_loader, device, mode, num_classes)

    # Loss function (using LossFactory)
    criterion = LossFactory.create(
        loss_function, mode, num_classes, class_weight,
        focal_alpha=focal_alpha, focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)
    print(f"  ✓ Loss: {loss_function}")

    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Scheduler
    if scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    elif scheduler_type in ('cosine_annealing', 'cosine'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=len(train_loader))
    else:
        scheduler = None
    print(f"  ✓ Scheduler: {scheduler_type}")

    # Mixed precision
    scaler = None
    if use_amp and device.type == 'cuda':
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            print("  ✓ Mixed Precision (AMP) enabled")
        except ImportError:
            print("  ⚠ AMP not available")
    sys.stdout.flush()

    print(f"\nStarting training for {epochs} epochs...")
    update_progress(progress_file, 15, f"Training {model_name}...")

    # Training loop
    best_loss = float('inf')
    best_iou = 0.0
    epochs_without_improvement = 0
    stopped_early = False
    history = {
        'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [], 'val_accuracy': [], 'lr': []
    }

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]['lr']

        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            current_lr = warmup_lr

        # Unfreeze encoder
        if encoder_frozen and epoch == freeze_epochs:
            unfreeze_encoder(model)
            encoder_frozen = False
            print(f"\n{'=' * 50}")
            print(f"✓ Epoch {epoch + 1}: ENCODER UNFROZEN!")
            print(f"  Trainable: {count_parameters(model):,}")
            print(f"{'=' * 50}")
            sys.stdout.flush()

        # Train
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            if scheduler_type == 'one_cycle' and scheduler is not None:
                scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_metrics = []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                metrics = compute_metrics(outputs, masks, mode, num_classes)
                all_metrics.append(metrics)

        val_loss /= len(val_loader)

        # Aggregate metrics
        if mode == 'binary':
            val_iou = np.mean([m['iou'] for m in all_metrics])
        else:
            val_iou = np.mean([m['mean_iou'] for m in all_metrics])

        val_f1 = np.mean([m['f1'] for m in all_metrics])
        val_precision = np.mean([m['precision'] for m in all_metrics])
        val_recall = np.mean([m['recall'] for m in all_metrics])
        val_accuracy = np.mean([m['accuracy'] for m in all_metrics])

        # Scheduler step
        if scheduler is not None and scheduler_type == 'reduce_on_plateau':
            scheduler.step(val_iou)
        elif scheduler is not None and scheduler_type in ('cosine_annealing', 'cosine'):
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)

        # Save best models - include all metadata needed for prediction
        model_metadata = {
            'model_name': model_name,
            'mode': mode,
            'in_channels': in_channels,
            'num_classes': actual_classes,
            'encoder_name': encoder_name,
            'dropout_rate': dropout_rate,
        }

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata},
                       os.path.join(save_dir, f'{model_name}_best_loss.pth'))

        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata},
                       os.path.join(save_dir, f'{model_name}_best_iou.pth'))
        else:
            epochs_without_improvement += 1

        # Progress
        progress = 15 + int(80 * (epoch + 1) / epochs)
        iou_name = 'IoU' if mode == 'binary' else 'mIoU'
        msg = f"Epoch {epoch + 1}/{epochs}: Loss={train_loss:.4f}, Val={val_loss:.4f}, {iou_name}={val_iou:.4f}, F1={val_f1:.4f} ({epoch_time:.1f}s)"
        print(msg)
        print(f"  Prec={val_precision:.4f}, Rec={val_recall:.4f}, Acc={val_accuracy:.4f}, LR={current_lr:.6f}")
        sys.stdout.flush()
        update_progress(progress_file, progress, f"Epoch {epoch + 1}: {iou_name}={val_iou:.4f}, F1={val_f1:.4f}")

        # Early stopping
        if early_stopping and epochs_without_improvement >= patience:
            print(f"\n⚠ Early stopping! No improvement for {patience} epochs. Best {iou_name}: {best_iou:.4f}")
            stopped_early = True
            break

    # Save final model
    final_path = os.path.join(save_dir, f'{model_name}_{mode}_final.pth')
    model_metadata['best_val_loss'] = best_loss
    model_metadata['best_val_iou'] = best_iou
    torch.save({'model_state_dict': model.state_dict(), 'metadata': model_metadata}, final_path)

    # Save history
    with open(os.path.join(save_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    iou_name = 'IoU' if mode == 'binary' else 'mIoU'
    actual_epochs = len(history['train_loss'])
    print(f"Epochs completed: {actual_epochs}")
    print(f"Best {iou_name}: {best_iou:.4f}")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Final F1: {history['val_f1'][-1]:.4f}")
    print(f"Final Precision: {history['val_precision'][-1]:.4f}")
    print(f"Final Recall: {history['val_recall'][-1]:.4f}")
    print(f"Final Accuracy: {history['val_accuracy'][-1]:.4f}")
    print("=" * 60)
    sys.stdout.flush()

    # Generate training plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_range = range(1, actual_epochs + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', lw=2)
        axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val', lw=2)
        axes[0, 0].set_title('Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs_range, history['val_iou'], 'g-', label=iou_name, lw=2)
        axes[0, 1].plot(epochs_range, history['val_f1'], 'b-', label='F1', lw=2)
        axes[0, 1].axhline(y=best_iou, color='green', ls='--', alpha=0.5)
        axes[0, 1].set_title(f'{iou_name} & F1', fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        axes[1, 0].plot(epochs_range, history['val_precision'], 'c-', label='Precision', lw=2)
        axes[1, 0].plot(epochs_range, history['val_recall'], 'm-', label='Recall', lw=2)
        axes[1, 0].plot(epochs_range, history['val_accuracy'], 'y-', label='Accuracy', lw=2)
        axes[1, 0].set_title('Precision, Recall & Accuracy', fontweight='bold')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        axes[1, 1].plot(epochs_range, history['lr'], 'r-', lw=2)
        axes[1, 1].set_title('Learning Rate', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        status = "(Early Stopped)" if stopped_early else "(Completed)"
        fig.suptitle(f'{model_name.upper()} + {encoder_name} - {actual_epochs} epochs {status}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"✓ Training plot saved: {plot_path}")
    except Exception as e:
        print(f"⚠ Could not generate plot: {e}")

    iou_name = 'IoU' if mode == 'binary' else 'mIoU'
    if stopped_early:
        summary = f"Training stopped at epoch {actual_epochs}. Best {iou_name}: {best_iou:.4f}"
    else:
        summary = f"Training complete! Best {iou_name}: {best_iou:.4f}, Best Loss: {best_loss:.4f}"

    print(f"\n{summary}")
    print(f"Models saved to: {save_dir}")
    sys.stdout.flush()
    update_progress(progress_file, 100, "Complete!", summary)
    return summary


def main():
    if len(sys.argv) != 3:
        print("Usage: python model_training.py params.json progress.json")
        sys.exit(1)

    params_file = sys.argv[1]
    progress_file = sys.argv[2]

    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        result = train_model(params, progress_file)
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        update_progress(progress_file, -1, error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()

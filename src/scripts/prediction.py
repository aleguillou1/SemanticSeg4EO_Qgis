#!/usr/bin/env python3
"""
Prediction Script - SemanticSeg4EO Plugin (Standalone)
========================================================

This script is executed by the EXTERNAL Python environment (not QGIS).
Handles large image prediction with seamless reconstruction.

Imports model architectures from model_training.py to ensure perfect parity.
Supports ALL architectures trained via the plugin:
  SMP: unet, unet++, deeplabv3+, deeplabv3, manet, fpn, pan, pspnet, linknet
  Modern: segformer-b0..b5, unetformer, hrnet-w18/w32/w48, swin-unet
  Custom: unet-dropout, convnext-unet

Usage:
    python prediction.py params.json progress.json
"""

import os
import sys
import math
import json
import warnings
import logging

# ============ SUPPRESS RASTERIO/GDAL WARNINGS ============
os.environ['CPL_LOG'] = '/dev/null' if sys.platform != 'win32' else 'NUL'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_PAM_ENABLED'] = 'NO'
os.environ['PROJ_LIB'] = ''

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rasterio with full error suppression
import rasterio
from rasterio.windows import Window
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        rasterio.env.defenv()
    except Exception:
        pass
try:
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
except Exception:
    pass

# ============================================================================
# IMPORT MODEL ARCHITECTURES FROM model_training.py
# ============================================================================
# This ensures EXACT parity between training and prediction architectures.
# If model_training.py is in the same directory, we import from it directly.

_IMPORTED_FROM_TRAINING = False

try:
    # Add script directory to path
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    from model_training import (
        ModelFactory,
        SimpleUNet,
        DEPS as TRAINING_DEPS,
        check_dependencies,
    )
    _IMPORTED_FROM_TRAINING = True
    DEPS = TRAINING_DEPS
    print("✓ Model architectures imported from model_training.py")
except ImportError as e:
    print(f"⚠ Could not import from model_training.py: {e}")
    print("  Using built-in model definitions (limited architectures)")
    _IMPORTED_FROM_TRAINING = False

    # Fallback: check dependencies locally
    def check_dependencies():
        deps = {}
        try:
            import segmentation_models_pytorch as smp
            deps['smp'] = True
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
        return deps

    DEPS = check_dependencies()

    # Fallback SimpleUNet (exact copy from model_training.py)
    class SimpleUNet(nn.Module):
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

    # Fallback ModelFactory (SMP + built-in only)
    class ModelFactory:
        @classmethod
        def list_models(cls):
            models = ['unet-dropout']
            if DEPS.get('smp'):
                models.extend(['unet', 'unet++', 'deeplabv3+', 'deeplabv3', 'manet', 'fpn', 'pan', 'pspnet', 'linknet'])
            return models

        @classmethod
        def create(cls, model_name, encoder_name='resnet34', in_channels=4, num_classes=2,
                   mode='multiclass', pretrained=False, dropout_rate=0.3):
            name = model_name.lower()
            actual = 1 if mode == 'binary' else num_classes

            if name == 'unet-dropout' or not DEPS.get('smp'):
                return SimpleUNet(in_channels, actual, dropout_rate)

            if DEPS.get('smp'):
                import segmentation_models_pytorch as smp
                smp_map = {
                    'unet': smp.Unet, 'unet++': smp.UnetPlusPlus, 'deeplabv3+': smp.DeepLabV3Plus,
                    'deeplabv3': smp.DeepLabV3, 'manet': smp.MAnet, 'fpn': smp.FPN,
                    'pan': smp.PAN, 'pspnet': smp.PSPNet, 'linknet': smp.Linknet,
                }
                for key, model_cls in smp_map.items():
                    if name == key or name.replace('-', '').replace('_', '') == key.replace('+', 'plus').replace('-', ''):
                        try:
                            return model_cls(encoder_name=encoder_name, in_channels=in_channels,
                                             classes=actual, encoder_weights=None, activation=None)
                        except Exception as e:
                            print(f"  ⚠ Failed to create {key}: {e}")
                            return SimpleUNet(in_channels, actual, dropout_rate)

            return SimpleUNet(in_channels, actual, dropout_rate)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_progress(progress_file, progress, message, result=None):
    """Update progress file"""
    data = {'progress': progress, 'message': message}
    if result:
        data['result'] = result
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def build_model(name, in_channels, classes, encoder='resnet34', mode='binary', dropout_rate=0.3):
    """
    Build segmentation model for inference.
    Uses ModelFactory from model_training.py for full architecture support.
    """
    return ModelFactory.create(
        model_name=name,
        encoder_name=encoder,
        in_channels=in_channels,
        num_classes=classes if mode != 'binary' else 1,
        mode=mode,
        pretrained=False,  # Never load pretrained for inference
        dropout_rate=dropout_rate
    )


# ============================================================================
# MODEL LOADING - MULTI-FORMAT SUPPORT
# ============================================================================

def load_model(model_path, device, encoder_override=None, dropout_rate=0.3):
    """
    Load model from checkpoint - supports multiple formats:
    - Plugin QGIS format: metadata nested in 'metadata' key
    - Modern v3 format: config in 'config' key
    - External training format: metadata at root level
    - K-Fold format: pure state_dict
    - Auto-detection from state_dict weights
    """
    print(f"Loading model: {model_path}")
    sys.stdout.flush()

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Initialize metadata
    in_channels = None
    num_classes = None
    model_name = None
    encoder = encoder_override or None  # Use override if provided
    mode = None
    detected_dropout = dropout_rate

    # === LOOK FOR EXTERNAL METRICS/CONFIG FILES ===
    model_dir = os.path.dirname(model_path)
    model_basename = os.path.basename(model_path).replace('.pth', '')

    possible_config_files = [
        os.path.join(model_dir, f"{model_basename}_metrics.json"),
        os.path.join(model_dir, f"{model_basename.replace('_best_miou', '')}_metrics.json"),
        os.path.join(model_dir, f"{model_basename.replace('_best_loss', '')}_metrics.json"),
        os.path.join(model_dir, f"{model_basename.replace('_best_iou', '')}_metrics.json"),
        os.path.join(model_dir, "metrics.json"),
        os.path.join(model_dir, "config.json"),
    ]
    if os.path.exists(model_dir):
        for f_name in os.listdir(model_dir):
            if f_name.endswith(('_metrics.json', '_config.json')):
                possible_config_files.append(os.path.join(model_dir, f_name))

    for cfg_file in possible_config_files:
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, 'r') as f:
                    ext_config = json.load(f)
                print(f"  Found config: {os.path.basename(cfg_file)}")
                in_channels = in_channels or ext_config.get('in_channels')
                num_classes = num_classes or ext_config.get('nb_classes', ext_config.get('num_classes'))
                model_name = model_name or ext_config.get('model_name')
                if encoder is None:
                    encoder = ext_config.get('encoder_name')
                detected_dropout = ext_config.get('dropout_rate', detected_dropout)
                mode = mode or ext_config.get('mode')
                break
            except Exception as e:
                print(f"  Warning: Could not read {cfg_file}: {e}")

    # === EXTRACT FROM CHECKPOINT ===
    state_dict = None

    if isinstance(checkpoint, dict):
        # Get state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            first_key = list(checkpoint.keys())[0] if checkpoint else ''
            if '.' in first_key or 'weight' in first_key or 'bias' in first_key:
                state_dict = checkpoint
            else:
                state_dict = checkpoint

        # Format 1: Nested metadata (plugin QGIS format)
        if 'metadata' in checkpoint and isinstance(checkpoint['metadata'], dict):
            meta = checkpoint['metadata']
            in_channels = in_channels or meta.get('in_channels')
            num_classes = num_classes or meta.get('num_classes', meta.get('nb_classes'))
            model_name = model_name or meta.get('model_name')
            if encoder is None:
                encoder = meta.get('encoder_name')
            detected_dropout = meta.get('dropout_rate', detected_dropout)
            mode = mode or meta.get('mode')

        # Format 2: Config key (modern v3 format)
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            cfg = checkpoint['config']
            in_channels = in_channels or cfg.get('in_channels')
            num_classes = num_classes or cfg.get('num_classes')
            model_name = model_name or cfg.get('model_name')
            if encoder is None:
                encoder = cfg.get('encoder_name')
            detected_dropout = cfg.get('dropout_rate', detected_dropout)
            mode = mode or cfg.get('mode')
            # Auto-correct binary mode num_classes
            if mode == 'binary':
                num_classes = 1

        # Format 3: Root-level metadata
        if in_channels is None:
            in_channels = checkpoint.get('in_channels')
        if num_classes is None:
            num_classes = checkpoint.get('num_classes', checkpoint.get('nb_classes'))
        if model_name is None:
            model_name = checkpoint.get('model_name')
        if encoder is None:
            encoder = checkpoint.get('encoder_name')
    else:
        state_dict = checkpoint

    # === AUTO-DETECT FROM STATE_DICT WEIGHTS ===
    if state_dict is not None:
        # Remove 'module.' prefix if DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Detect in_channels
        if in_channels is None:
            for key, tensor in state_dict.items():
                if tensor.dim() == 4:
                    if any(p in key.lower() for p in ['encoder.conv1', 'enc1', 'input', 'stem', 'conv1.weight',
                                                       'patch_embeddings']):
                        in_channels = tensor.shape[1]
                        print(f"  Auto-detected in_channels from {key}: {in_channels}")
                        break
            if in_channels is None:
                for key, tensor in state_dict.items():
                    if tensor.dim() == 4 and 1 <= tensor.shape[1] <= 20:
                        in_channels = tensor.shape[1]
                        print(f"  Inferred in_channels from {key}: {in_channels}")
                        break

        # Detect num_classes
        if num_classes is None:
            for key, tensor in state_dict.items():
                if tensor.dim() == 4:
                    if any(p in key.lower() for p in ['segmentation_head', 'final_conv', 'out.', 'head.',
                                                       'classifier', 'final.']):
                        num_classes = tensor.shape[0]
                        print(f"  Auto-detected num_classes from {key}: {num_classes}")
                        break

        # Detect model architecture
        if model_name is None:
            keys_str = ' '.join(state_dict.keys()).lower()
            if 'segformer' in keys_str or 'patch_embeddings' in keys_str:
                # Try to detect SegFormer variant
                for variant in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']:
                    model_name = f'segformer-b2'  # default, hard to detect variant
                    break
            elif 'backbone' in keys_str and 'hrnet' in keys_str:
                model_name = 'hrnet-w18'
            elif 'swin' in keys_str:
                model_name = 'swin-unet'
            elif 'decoder.blocks' in keys_str or 'dense' in keys_str:
                model_name = 'unet++'
            elif 'aspp' in keys_str:
                model_name = 'deeplabv3+'
            elif 'fpn' in keys_str or 'pyramid' in keys_str:
                model_name = 'fpn'
            elif 'decoder' in keys_str and 'encoder' in keys_str:
                model_name = 'unet'
            elif 'enc1' in keys_str and 'dec1' in keys_str:
                model_name = 'unet-dropout'
            else:
                model_name = 'unet'
            print(f"  Auto-detected architecture: {model_name}")

    # Determine mode
    if mode is None:
        mode = 'binary' if (num_classes is not None and num_classes == 1) else 'multiclass'

    # Set defaults
    in_channels = in_channels or 4
    num_classes = num_classes or 1
    encoder = encoder or 'resnet34'

    print(f"\n  === Model Configuration ===")
    print(f"  Architecture: {model_name}")
    print(f"  Encoder: {encoder}")
    print(f"  Mode: {mode}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output classes: {num_classes}")
    print(f"  Dropout rate: {detected_dropout}")
    if _IMPORTED_FROM_TRAINING:
        print(f"  Source: model_training.py (full architecture support)")
    else:
        print(f"  Source: built-in fallback (SMP + SimpleUNet only)")
    print(f"  ============================\n")
    sys.stdout.flush()

    # Build model
    model = build_model(model_name, in_channels, num_classes, encoder, mode, detected_dropout)

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  ✓ Weights loaded (strict)")
    except RuntimeError:
        print("  ⚠ Strict loading failed, trying non-strict...")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Missing keys ({len(missing)}): {missing[:3]}...")
            if unexpected:
                print(f"    Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
            print("  ✓ Weights loaded (non-strict)")
        except Exception as e2:
            print(f"  ✗ Failed to load weights: {e2}")
            raise

    model = model.to(device)
    model.eval()
    sys.stdout.flush()
    return model, in_channels, num_classes, mode


# ============================================================================
# GAUSSIAN WEIGHTING WINDOW
# ============================================================================

def create_gaussian_window(height, width, sigma_scale=0.25):
    """Create 2D gaussian window for seamless blending"""
    sigma_y = height * sigma_scale
    sigma_x = width * sigma_scale
    y = np.arange(height) - (height - 1) / 2.0
    x = np.arange(width) - (width - 1) / 2.0
    gy = np.exp(-(y ** 2) / (2 * sigma_y ** 2))
    gx = np.exp(-(x ** 2) / (2 * sigma_x ** 2))
    window = np.outer(gy, gx).astype(np.float32)
    window /= window.max()
    window = np.clip(window, 0.01, 1.0)
    return window


def create_weight_mask(size, overlap):
    """Create simple blending weight mask (legacy fallback)"""
    weight = np.ones((size, size), dtype=np.float32)
    fade = min(overlap // 2, size // 4)
    if fade > 0:
        for i in range(fade):
            w = (i + 1) / (fade + 1)
            weight[i, :] *= w
            weight[-(i + 1), :] *= w
            weight[:, i] *= w
            weight[:, -(i + 1)] *= w
    return weight


# ============================================================================
# PREDICTION
# ============================================================================

def predict_image(params, progress_file):
    """Run prediction on large image"""

    sys.stdout.reconfigure(line_buffering=True)

    model_path = params['model_path']
    input_path = params['input_path']
    output_path = params['output_path']
    patch_size = params.get('patch_size', 512)
    overlap = params.get('overlap', 128)
    threshold = params.get('threshold', 0.5)
    device_str = params.get('device', 'cuda')
    save_confidence = params.get('save_confidence', False)
    encoder_override = params.get('encoder_name', None)  # NEW: explicit encoder selection
    dropout_rate = params.get('dropout_rate', 0.3)  # NEW: dropout rate
    batch_size_inference = params.get('batch_size', 1)  # NEW: batch inference
    use_gaussian = params.get('gaussian_blending', True)  # NEW: gaussian vs simple blending

    update_progress(progress_file, 0, "Loading model...")

    # Device
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if device_str == 'cuda':
            print("⚠ CUDA not available, using CPU")

    print(f"Using device: {device}")
    sys.stdout.flush()

    model, in_channels, num_classes, mode = load_model(model_path, device, encoder_override, dropout_rate)

    update_progress(progress_file, 10, "Opening input image...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(input_path) as src:
            height, width = src.height, src.width
            profile = src.profile.copy()

            print(f"Image: {width}x{height}, {src.count} bands")
            update_progress(progress_file, 15, f"Image: {width}x{height}")

            # Grid
            stride = patch_size - overlap
            n_rows = max(1, (height - overlap) // stride + (1 if (height - overlap) % stride > 0 else 0))
            n_cols = max(1, (width - overlap) // stride + (1 if (width - overlap) % stride > 0 else 0))
            total = n_rows * n_cols
            print(f"Processing {total} patches ({n_rows}x{n_cols}), batch_size={batch_size_inference}")

            # Blending window
            if use_gaussian:
                weight = create_gaussian_window(patch_size, patch_size)
            else:
                weight = create_weight_mask(patch_size, overlap)

            # Accumulators
            if mode == 'binary':
                pred_sum = np.zeros((height, width), dtype=np.float64)
            else:
                pred_sum = np.zeros((num_classes, height, width), dtype=np.float64)
            count_map = np.zeros((height, width), dtype=np.float64)

            # Process patches (with optional batching)
            batch_patches = []
            batch_positions = []
            patch_idx = 0

            for row in range(n_rows):
                for col in range(n_cols):
                    y = min(row * stride, max(0, height - patch_size))
                    x = min(col * stride, max(0, width - patch_size))

                    window = Window(x, y, min(patch_size, width - x), min(patch_size, height - y))
                    patch = src.read(window=window)

                    # Pad if patch is smaller than patch_size
                    if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                        padded = np.zeros((patch.shape[0], patch_size, patch_size), dtype=patch.dtype)
                        padded[:, :patch.shape[1], :patch.shape[2]] = patch
                        patch = padded

                    # Handle channels
                    if patch.shape[0] > in_channels:
                        patch = patch[:in_channels]
                    elif patch.shape[0] < in_channels:
                        repeats = math.ceil(in_channels / patch.shape[0])
                        patch = np.repeat(patch, repeats, axis=0)[:in_channels]

                    # Normalize (percentile 99, same as training)
                    patch = patch.astype(np.float32)
                    if np.max(patch) > 0:
                        p99 = np.percentile(patch, 99)
                        if p99 > 0:
                            patch = np.clip(patch / p99, 0, 1)

                    batch_patches.append(patch)
                    batch_positions.append((x, y))

                    # Process batch
                    if len(batch_patches) >= batch_size_inference or (row == n_rows - 1 and col == n_cols - 1):
                        with torch.no_grad():
                            batch_tensor = torch.from_numpy(np.stack(batch_patches)).float().to(device)
                            output = model(batch_tensor)

                            # Handle different output formats
                            if isinstance(output, dict):
                                output = output.get('out', output.get('logits', list(output.values())[0]))
                            elif isinstance(output, (tuple, list)):
                                output = output[0]

                            if mode == 'binary':
                                probs = torch.sigmoid(output).cpu().numpy()
                                if probs.ndim == 4:
                                    probs = probs[:, 0]  # (B, H, W)
                            else:
                                probs = F.softmax(output, dim=1).cpu().numpy()  # (B, C, H, W)

                        # Accumulate each patch
                        for b_idx in range(len(batch_patches)):
                            bx, by = batch_positions[b_idx]
                            eff_h = min(patch_size, height - by)
                            eff_w = min(patch_size, width - bx)
                            w = weight[:eff_h, :eff_w]

                            if mode == 'binary':
                                pred_sum[by:by + eff_h, bx:bx + eff_w] += probs[b_idx, :eff_h, :eff_w] * w
                            else:
                                for c in range(num_classes):
                                    pred_sum[c, by:by + eff_h, bx:bx + eff_w] += probs[b_idx, c, :eff_h, :eff_w] * w
                            count_map[by:by + eff_h, bx:bx + eff_w] += w

                        patch_idx += len(batch_patches)
                        batch_patches.clear()
                        batch_positions.clear()

                        if patch_idx % 20 == 0 or patch_idx == total:
                            progress = 15 + int(75 * patch_idx / total)
                            update_progress(progress_file, progress, f"Processed {patch_idx}/{total} patches")

    update_progress(progress_file, 90, "Finalizing...")

    # Average
    count_map = np.maximum(count_map, 1e-8)

    if mode == 'binary':
        pred_avg = pred_sum / count_map
        final_mask = (pred_avg > threshold).astype(np.uint8)
        confidence = pred_avg.astype(np.float32)
    else:
        pred_avg = pred_sum / count_map
        final_mask = np.argmax(pred_avg, axis=0).astype(np.uint8)
        confidence = np.max(pred_avg, axis=0).astype(np.float32)

    update_progress(progress_file, 95, "Saving output...")

    # Save prediction
    out_profile = profile.copy()
    out_profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw'})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(final_mask, 1)

    print(f"Saved prediction: {output_path}")

    # Save confidence
    if save_confidence:
        conf_path = output_path.replace('.tif', '_confidence.tif')
        conf_profile = out_profile.copy()
        conf_profile['dtype'] = 'float32'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(conf_path, 'w', **conf_profile) as dst:
                dst.write(confidence, 1)
        print(f"Saved confidence: {conf_path}")

    # Statistics
    unique, counts = np.unique(final_mask, return_counts=True)
    total_px = counts.sum()

    stats = []
    for u, c in zip(unique, counts):
        pct = (c / total_px) * 100
        if mode == 'binary':
            label = 'Foreground' if u == 1 else 'Background'
        else:
            label = f'Class {u}'
        stats.append(f"{label}: {pct:.1f}%")
        print(f"  {label}: {c:,} pixels ({pct:.1f}%)")

    summary = f"Prediction saved to {output_path}\n" + ", ".join(stats)
    update_progress(progress_file, 100, "Complete!", summary)

    return summary


def main():
    if len(sys.argv) != 3:
        print("Usage: python prediction.py params.json progress.json")
        sys.exit(1)

    params_file = sys.argv[1]
    progress_file = sys.argv[2]

    try:
        with open(params_file, 'r') as f:
            params = json.load(f)

        result = predict_image(params, progress_file)

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        update_progress(progress_file, -1, error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()

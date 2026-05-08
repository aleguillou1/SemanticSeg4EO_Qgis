#!/usr/bin/env python3
"""
Patch Extraction Script - Robust Version for QGIS Plugin
==========================================================
Supports:
  - Single mode: 1 image + 1 label + 1 grid
  - Batch mode: multiple images/labels with shared or per-image grids
  - Georeferenced GeoTIFF output (rasterio)
  - Configurable interpolation, compression, ID column
  - CRS validation between image and label
  - Robust error handling with detailed reporting
  - Progress reporting via JSON file (for QGIS plugin)

Batch mode naming convention:
    Images: Image_1.tif, Image_2.tif, ...  (or image_1.tif, custom patterns)
    Labels: Label_1.tif, Label_2.tif, ...  (or label_1.tif, custom patterns)
    Grids:  Grid_1.shp, Grid_2.shp, ...   (optional per-image grids)

     Copyright (c) 2026 Le Guillou Adrien
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
"""

"""

import os
import re
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

# ============ SUPPRESS RASTERIO/GDAL WARNINGS ============
os.environ['CPL_LOG'] = 'OFF'
os.environ['CPL_LOG_ERRORS'] = 'OFF'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['GDAL_PAM_ENABLED'] = 'NO'
logging.getLogger('rasterio').setLevel(logging.CRITICAL)
logging.getLogger('rasterio._env').setLevel(logging.CRITICAL)
logging.getLogger('rasterio.env').setLevel(logging.CRITICAL)
logging.getLogger('fiona').setLevel(logging.CRITICAL)
logging.getLogger('fiona._env').setLevel(logging.CRITICAL)

import numpy as np

# Redirect stderr temporarily to suppress C-level GDAL Unicode errors (é in paths)
import io as _io
_old_stderr = sys.stderr
sys.stderr = _io.StringIO()
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
sys.stderr = _old_stderr

import geopandas as gpd
import cv2

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        rasterio.env.defenv()
    except Exception:
        pass


# ============================================================
# Data classes
# ============================================================

@dataclass
class ImageLabelPair:
    """Represents a matched image-label pair with optional grid."""
    pair_id: str
    image_path: str
    label_path: str
    grid_path: Optional[str] = None

    def __repr__(self):
        return (f"Pair({self.pair_id}: "
                f"{Path(self.image_path).name} <-> {Path(self.label_path).name}"
                f"{' + ' + Path(self.grid_path).name if self.grid_path else ''})")


@dataclass
class ExtractionStats:
    """Accumulates extraction statistics."""
    total_patches: int = 0
    processed: int = 0
    failed: int = 0
    resized: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    pairs_processed: int = 0
    per_pair: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def merge(self, other: 'ExtractionStats', pair_id: str = ""):
        self.total_patches += other.total_patches
        self.processed += other.processed
        self.failed += other.failed
        self.resized += other.resized
        self.train_count += other.train_count
        self.val_count += other.val_count
        self.test_count += other.test_count
        self.pairs_processed += 1
        if pair_id:
            self.per_pair[pair_id] = {
                'total': other.total_patches,
                'processed': other.processed,
                'failed': other.failed,
            }
        self.errors.extend(other.errors)


# ============================================================
# Progress helper
# ============================================================

def update_progress(progress_file: str, progress: int, message: str, result=None):
    """Update progress file for QGIS to read."""
    data = {'progress': progress, 'message': message}
    if result:
        data['result'] = result
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass


# ============================================================
# Georeferenced TIFF writing
# ============================================================

INTERPOLATION_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
}


def save_georeferenced_tiff(filepath, data, transform, crs,
                            is_label=False, compression='deflate',
                            output_dtype=None):
    """
    Save numpy array as GeoTIFF with georeferencing.

    Args:
        filepath: Output path
        data: (H,W) for labels or (H,W,C) for images
        transform: rasterio Affine transform
        crs: CRS object
        is_label: single-band uint8
        compression: 'deflate', 'lzw', or None
        output_dtype: 'float32', 'int16', 'uint16', 'uint8' (images only, labels always uint8)
    """
    if is_label or data.ndim == 2:
        height, width = data.shape
        count = 1
        dtype = 'uint8'
        data_out = data.astype(np.uint8).reshape(1, height, width)
    else:
        height, width, count = data.shape
        # Determine output dtype
        if output_dtype == 'int16':
            dtype = 'int16'
            data_out = np.transpose(data.astype(np.int16), (2, 0, 1))
        elif output_dtype == 'uint16':
            dtype = 'uint16'
            data_out = np.transpose(data.astype(np.uint16), (2, 0, 1))
        elif output_dtype == 'uint8':
            dtype = 'uint8'
            data_out = np.transpose(data.astype(np.uint8), (2, 0, 1))
        else:  # float32 (default)
            dtype = 'float32'
            data_out = np.transpose(data.astype(np.float32), (2, 0, 1))

    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'width': width,
        'height': height,
        'count': count,
        'crs': crs,
        'transform': transform,
    }
    if compression:
        profile['compress'] = compression

    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data_out)


# ============================================================
# ID column auto-detection
# ============================================================

_ID_CANDIDATES = ['AUTO', 'ID', 'FID', 'id', 'fid', 'OBJECTID', 'ObjectID',
                  'patch_id', 'PATCH_ID', 'gid', 'GID']


def find_id_column(gdf: gpd.GeoDataFrame, hint: str = 'AUTO') -> str:
    """Find or create a suitable ID column in the GeoDataFrame."""
    if hint != 'AUTO' and hint in gdf.columns:
        return hint

    for col in _ID_CANDIDATES:
        if col in gdf.columns:
            return col

    # Fallback: create a sequential ID
    gdf['_auto_id'] = range(len(gdf))
    return '_auto_id'


# ============================================================
# CRS validation
# ============================================================

def validate_crs(image_path: str, label_path: str, grid_path: str):
    """
    Check CRS compatibility between image, label, and grid.
    Warns but does not fail — user may know what they're doing.
    """
    issues = []
    try:
        with rasterio.open(image_path) as src:
            img_crs = src.crs
        with rasterio.open(label_path) as src:
            lbl_crs = src.crs
        grid_crs = gpd.read_file(grid_path, rows=0).crs

        if img_crs and lbl_crs and img_crs != lbl_crs:
            issues.append(f"CRS mismatch: image={img_crs} vs label={lbl_crs}")
        if img_crs and grid_crs and img_crs != grid_crs:
            issues.append(f"CRS mismatch: image={img_crs} vs grid={grid_crs}")
    except Exception as e:
        issues.append(f"CRS check error: {e}")

    return issues


# ============================================================
# Batch pair discovery
# ============================================================

def find_matching_pairs(
    data_dir: str,
    image_pattern: str = r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$',
    label_pattern: str = r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$',
    grid_pattern: str = r'^[Gg]rid[_-]?(\d+)\.(shp|gpkg)$',
    recursive: bool = False,
) -> Tuple[List[ImageLabelPair], Dict]:
    """
    Find matching image-label pairs in a directory.

    Returns:
        (list of matched pairs, stats dict)
    """
    images, labels, grids = {}, {}, {}

    img_re = re.compile(image_pattern)
    lbl_re = re.compile(label_pattern)
    grd_re = re.compile(grid_pattern)

    walker = os.walk(data_dir) if recursive else [(data_dir, [], os.listdir(data_dir))]

    for root, _dirs, files in walker:
        for fname in files:
            fpath = os.path.join(root, fname)

            m = img_re.match(fname)
            if m:
                images[m.group(1)] = fpath
                continue
            m = lbl_re.match(fname)
            if m:
                labels[m.group(1)] = fpath
                continue
            m = grd_re.match(fname)
            if m:
                grids[m.group(1)] = fpath

    matched_ids = sorted(
        set(images) & set(labels),
        key=lambda x: int(x) if x.isdigit() else x
    )

    pairs = [
        ImageLabelPair(
            pair_id=pid,
            image_path=images[pid],
            label_path=labels[pid],
            grid_path=grids.get(pid),
        )
        for pid in matched_ids
    ]

    stats = {
        'total_images': len(images),
        'total_labels': len(labels),
        'total_grids': len(grids),
        'matched_pairs': len(pairs),
        'unmatched_images': sorted(set(images) - set(labels)),
        'unmatched_labels': sorted(set(labels) - set(images)),
    }
    return pairs, stats


# ============================================================
# Core single-pair extraction (used by both modes)
# ============================================================

def _extract_single_pair(
    image_path: str,
    label_path: str,
    grid_path: str,
    output_dir: str,
    patch_size: int = 224,
    image_channels: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    id_column: str = 'AUTO',
    random_seed: Optional[int] = 42,
    interpolation: str = 'bilinear',
    compression: str = 'deflate',
    prefix: str = '',
    existing_splits: Optional[Dict] = None,
    progress_callback=None,
    output_dtype: str = 'float32',
) -> ExtractionStats:
    """
    Extract patches from a single image+label+grid triplet.

    Args:
        progress_callback: callable(current_index, total) for progress updates
    Returns:
        ExtractionStats for this pair
    """
    stats = ExtractionStats()

    # Interpolation
    interp_img = INTERPOLATION_MAP.get(interpolation, cv2.INTER_LINEAR)

    # Create output dirs
    for split in ['train', 'validation', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

    # Load grid
    grid = gpd.read_file(grid_path)
    stats.total_patches = len(grid)

    # ID column
    id_col = find_id_column(grid, id_column)

    # Shuffle & split indices
    indices = list(grid.index)
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(indices)

    if existing_splits is None:
        n_train = int(train_ratio * len(indices))
        n_val = int(val_ratio * len(indices))
        train_idx = set(indices[:n_train])
        val_idx = set(indices[n_train:n_train + n_val])
        test_idx = set(indices[n_train + n_val:])
    else:
        train_idx = existing_splits['train']
        val_idx = existing_splits['validation']
        test_idx = existing_splits['test']

    stats.train_count = len(train_idx)
    stats.val_count = len(val_idx)
    stats.test_count = len(test_idx)

    print(f"  Grid: {len(grid)} patches  |  "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
          flush=True)

    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lbl:
        crs = src_img.crs

        for i, (idx, row) in enumerate(grid.iterrows()):
            geom = row['geometry']
            raw_id = row[id_col]

            # Build filename
            if isinstance(raw_id, float):
                patch_id = f"patch_{int(raw_id)}"
            else:
                patch_id = f"patch_{str(raw_id).replace('.', '_').replace(' ', '_')}"
            if prefix:
                patch_id = f"{prefix}_{patch_id}"

            # Determine split
            if idx in train_idx:
                split = 'train'
            elif idx in val_idx:
                split = 'validation'
            else:
                split = 'test'

            try:
                # Mask / crop
                img_data, _img_tf = mask(src_img, [geom], crop=True)
                lbl_data, _lbl_tf = mask(src_lbl, [geom], crop=True)

                _, h, w = img_data.shape
                needs_resize = (h != patch_size or w != patch_size)

                # --- Resize image ---
                resized_img = np.zeros(
                    (image_channels, patch_size, patch_size), dtype=np.float32
                )
                for c in range(min(image_channels, img_data.shape[0])):
                    resized_img[c] = cv2.resize(
                        img_data[c].astype(np.float32),
                        (patch_size, patch_size),
                        interpolation=interp_img,
                    )

                # --- Resize label (nearest to preserve classes) ---
                lbl_2d = lbl_data[0] if lbl_data.ndim == 3 else lbl_data
                resized_lbl = cv2.resize(
                    lbl_2d.astype(np.float32),
                    (patch_size, patch_size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)

                if needs_resize:
                    stats.resized += 1

                # HWC for saving
                img_hwc = np.transpose(resized_img, (1, 2, 0))

                # Georeferenced transform from geometry bounds
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                patch_transform = from_bounds(
                    bounds[0], bounds[1], bounds[2], bounds[3],
                    patch_size, patch_size,
                )

                # Save
                img_out = os.path.join(output_dir, split, 'images', f'{patch_id}.tif')
                lbl_out = os.path.join(output_dir, split, 'labels', f'{patch_id}.tif')

                save_georeferenced_tiff(
                    img_out, img_hwc, patch_transform, crs,
                    is_label=False, compression=compression,
                    output_dtype=output_dtype,
                )
                save_georeferenced_tiff(
                    lbl_out, resized_lbl, patch_transform, crs,
                    is_label=True, compression=compression,
                )

                stats.processed += 1

            except Exception as e:
                stats.failed += 1
                msg = f"Failed {patch_id}: {e}"
                if len(stats.errors) < 20:
                    stats.errors.append(msg)
                if stats.failed <= 5:
                    print(f"  WARNING {msg}", flush=True)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, stats.total_patches)

            # Console progress
            if (i + 1) % 200 == 0 or i == stats.total_patches - 1:
                print(f"  Progress: {i+1}/{stats.total_patches} | "
                      f"OK={stats.processed} Failed={stats.failed}",
                      flush=True)

    return stats


# ============================================================
# Public entry-point: supports single & batch
# ============================================================

def extract_patches(params: Dict, progress_file: str):
    """
    Main entry point called by QGIS plugin via subprocess.

    Params dict keys:
      --- Required (single mode) ---
        image_path, label_path, grid_path, output_dir

      --- Required (batch mode) ---
        mode: 'batch'
        data_dir: directory with Image_N / Label_N files
        grid_path: shared grid  (or per-image grids if use_per_image_grid=True)
        output_dir

      --- Optional ---
        patch_size (224), image_channels (10),
        train_ratio (0.7), val_ratio (0.2), test_ratio (0.1),
        id_column ('AUTO'), random_seed (42),
        interpolation ('bilinear'), compression ('deflate'),
        image_pattern, label_pattern, grid_pattern,
        use_per_image_grid (False), recursive (False),
        validate_crs (True)
    """
    sys.stdout.reconfigure(line_buffering=True)

    mode = params.get('mode', 'single')
    output_dir = params['output_dir']
    patch_size = params.get('patch_size', 224)
    image_channels = params.get('image_channels', 10)
    train_ratio = params.get('train_ratio', 0.7)
    val_ratio = params.get('val_ratio', 0.2)
    test_ratio = params.get('test_ratio', 0.1)
    id_column = params.get('id_column', 'AUTO')
    random_seed = params.get('random_seed', 42)
    interpolation = params.get('interpolation', 'bilinear')
    compression = params.get('compression', 'deflate')
    do_crs_check = params.get('validate_crs', True)
    output_dtype = params.get('output_dtype', 'float32')

    # Normalise ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    print("=" * 60)
    print(f"PATCH EXTRACTION — mode: {mode.upper()}")
    print("=" * 60)
    print(f"  Patch size   : {patch_size}x{patch_size}")
    print(f"  Channels     : {image_channels}")
    print(f"  Split        : train={train_ratio:.0%} val={val_ratio:.0%} test={test_ratio:.0%}")
    print(f"  Interpolation: {interpolation}")
    print(f"  Compression  : {compression}")
    print(f"  Image dtype  : {output_dtype}")
    print(f"  Output       : {output_dir}")
    print("=" * 60, flush=True)

    update_progress(progress_file, 2, "Initialising...")

    # ----------------------------------------------------------
    # Build list of (image, label, grid) triplets
    # ----------------------------------------------------------
    triplets: List[Tuple[str, str, str, str]] = []  # (pair_id, img, lbl, grid)

    if mode == 'batch':
        data_dir = params['data_dir']
        shared_grid = params.get('grid_path', '')
        use_per_image = params.get('use_per_image_grid', False)
        recursive = params.get('recursive', False)
        img_pattern = params.get('image_pattern', r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$')
        lbl_pattern = params.get('label_pattern', r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$')
        grd_pattern = params.get('grid_pattern', r'^[Gg]rid[_-]?(\d+)\.(shp|gpkg)$')

        pairs, scan = find_matching_pairs(
            data_dir, img_pattern, lbl_pattern, grd_pattern, recursive,
        )

        print(f"\nScan results: {scan['total_images']} images, "
              f"{scan['total_labels']} labels, {scan['matched_pairs']} matched pairs")
        if scan['unmatched_images']:
            print(f"  Unmatched images (IDs): {scan['unmatched_images']}")
        if scan['unmatched_labels']:
            print(f"  Unmatched labels (IDs): {scan['unmatched_labels']}")
        sys.stdout.flush()

        if not pairs:
            msg = ("No matching image-label pairs found! "
                   "Expected naming: Image_1.tif / Label_1.tif etc.")
            update_progress(progress_file, -1, msg)
            raise FileNotFoundError(msg)

        for pair in pairs:
            grid = pair.grid_path if (use_per_image and pair.grid_path) else shared_grid
            if not grid:
                print(f"  Skipping pair {pair.pair_id}: no grid available", flush=True)
                continue
            triplets.append((f"img{pair.pair_id}", pair.image_path, pair.label_path, grid))

    else:
        # Single mode
        image_path = params['image_path']
        label_path = params['label_path']
        grid_path = params['grid_path']

        for p, n in [(image_path, 'Image'), (label_path, 'Labels'), (grid_path, 'Grid')]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{n} not found: {p}")

        triplets.append(('', image_path, label_path, grid_path))

    if not triplets:
        msg = "No valid image-label-grid triplets to process."
        update_progress(progress_file, -1, msg)
        raise ValueError(msg)

    print(f"\nWill process {len(triplets)} triplet(s).", flush=True)
    update_progress(progress_file, 5, f"Processing {len(triplets)} triplet(s)...")

    # ----------------------------------------------------------
    # CRS validation
    # ----------------------------------------------------------
    if do_crs_check:
        for pair_id, img, lbl, grd in triplets:
            issues = validate_crs(img, lbl, grd)
            for issue in issues:
                name = pair_id or 'single'
                print(f"  CRS WARNING [{name}]: {issue}", flush=True)

    # ----------------------------------------------------------
    # Process each triplet
    # ----------------------------------------------------------
    global_stats = ExtractionStats()

    # Count total patches across all triplets for global progress
    total_patches_all = 0
    for _, _, _, grd in triplets:
        try:
            total_patches_all += len(gpd.read_file(grd))
        except Exception:
            pass

    patches_done_global = 0

    for t_idx, (pair_id, img_path, lbl_path, grd_path) in enumerate(triplets):
        print(f"\n{'—'*50}")
        print(f"Triplet {t_idx+1}/{len(triplets)}: {pair_id or 'single'}")
        print(f"  Image: {Path(img_path).name}")
        print(f"  Label: {Path(lbl_path).name}")
        print(f"  Grid : {Path(grd_path).name}", flush=True)

        # Per-pair seed for reproducibility while varying between pairs
        seed = (random_seed + t_idx) if random_seed is not None else None

        def _progress_cb(current, total):
            nonlocal patches_done_global
            patches_done_global_now = patches_done_global + current
            pct = 5 + int(90 * patches_done_global_now / max(total_patches_all, 1))
            pct = min(pct, 95)
            update_progress(
                progress_file, pct,
                f"Pair {t_idx+1}/{len(triplets)} — patch {current}/{total}"
            )

        try:
            pair_stats = _extract_single_pair(
                image_path=img_path,
                label_path=lbl_path,
                grid_path=grd_path,
                output_dir=output_dir,
                patch_size=patch_size,
                image_channels=image_channels,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                id_column=id_column,
                random_seed=seed,
                interpolation=interpolation,
                compression=compression,
                prefix=pair_id,
                progress_callback=_progress_cb,
                output_dtype=output_dtype,
            )
            global_stats.merge(pair_stats, pair_id or 'single')
            patches_done_global += pair_stats.total_patches

            print(f"  Done: {pair_stats.processed} OK, {pair_stats.failed} failed", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            global_stats.errors.append(f"Pair {pair_id}: {e}")
            print(f"  ERROR processing pair: {e}", flush=True)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"  Pairs processed : {global_stats.pairs_processed}")
    print(f"  Total patches   : {global_stats.total_patches}")
    print(f"  Saved OK        : {global_stats.processed}")
    print(f"  Failed          : {global_stats.failed}")
    print(f"  Resized         : {global_stats.resized}")
    print(f"  Train / Val / Test : "
          f"{global_stats.train_count} / {global_stats.val_count} / {global_stats.test_count}")
    print(f"  Output          : {output_dir}")
    if global_stats.errors:
        print(f"\n  First errors:")
        for err in global_stats.errors[:10]:
            print(f"    - {err}")
    print("=" * 60, flush=True)

    # ----------------------------------------------------------
    # Save metadata
    # ----------------------------------------------------------
    metadata = {
        'mode': mode,
        'parameters': {
            'patch_size': patch_size,
            'image_channels': image_channels,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'interpolation': interpolation,
            'compression': compression,
            'id_column': id_column,
            'random_seed': random_seed,
        },
        'statistics': {
            'pairs_processed': global_stats.pairs_processed,
            'total_patches': global_stats.total_patches,
            'processed': global_stats.processed,
            'failed': global_stats.failed,
            'resized': global_stats.resized,
            'train': global_stats.train_count,
            'validation': global_stats.val_count,
            'test': global_stats.test_count,
        },
        'per_pair': global_stats.per_pair,
        'georeferenced': True,
        'crs': None,  # filled below
    }

    # Grab CRS from first triplet
    try:
        with rasterio.open(triplets[0][1]) as src:
            metadata['crs'] = str(src.crs) if src.crs else None
    except Exception:
        pass

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    summary = (f"Complete: {global_stats.processed} georeferenced patches saved"
               f" ({global_stats.failed} failed)"
               f" from {global_stats.pairs_processed} pair(s)")
    update_progress(progress_file, 100, "Complete!", summary)
    return summary


# ============================================================
# CLI entry point
# ============================================================

def main():
    if len(sys.argv) != 3:
        print("Usage: python patch_extraction.py params.json progress.json")
        sys.exit(1)

    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            params = json.load(f)
        extract_patches(params, sys.argv[2])
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        try:
            update_progress(sys.argv[2], -1, str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()

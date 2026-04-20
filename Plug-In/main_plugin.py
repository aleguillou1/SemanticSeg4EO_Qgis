"""
SemanticSeg4EO - Main Plugin Module (QGIS-Safe)
================================================

This module ONLY uses:
- PyQt5 (native to QGIS)
- QGIS API
- Standard library (os, json, subprocess, etc.)

NO external dependencies (torch, rasterio, etc.) are imported here.
All processing is done via subprocess in an external Python environment.

v3.0 - Added:
  - 20+ architectures (SegFormer, HRNet, SwinUNet, ConvNeXt, UNetFormer)
  - 5 augmentation levels (none→extreme)
  - 15+ loss functions with auto-conversion
  - Advanced parameters tab (AMP, warmup, dropout, focal/tversky params)
  - Prediction: encoder selection, batch inference, gaussian blending
"""

import os
import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QGroupBox, QFormLayout, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QSlider,
    QRadioButton, QButtonGroup, QScrollArea, QFrame, QSplitter
)
from qgis.PyQt.QtGui import QIcon, QFont, QTextCursor
from qgis.core import QgsProject, QgsRasterLayer

# Plugin directory
PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = PLUGIN_DIR / 'config.json'


class PluginConfig:
    """Manage plugin configuration (stored in JSON)"""
    
    DEFAULT_CONFIG = {
        'python_path': '',
        'env_type': 'none',  # 'none', 'conda', 'venv', 'system'
        'conda_env_name': 'semanticseg4eo',
        'venv_path': '',
        'last_check': '',
        'dependencies_ok': False
    }
    
    @classmethod
    def load(cls):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                return {**cls.DEFAULT_CONFIG, **config}
            except:
                pass
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save(cls, config):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def get_python_executable(cls):
        """Get the configured Python executable path.
        
        Simple approach: always use the stored python_path directly.
        No more guessing - the user picks the exact executable.
        """
        config = cls.load()
        python_path = config.get('python_path', '')
        if python_path and os.path.isfile(python_path):
            return python_path
        return None


class ProcessRunner(QThread):
    """Run external Python scripts in a separate thread
    
    FIXED VERSION v4: 
    - Does NOT set PYTHONHOME (causes CUDA/DLL conflicts)
    - REMOVES QGIS paths from PATH (Qt DLL conflicts)
    - Sets UTF-8 encoding for French characters in paths (é, è, ê...)
    - Sets GDAL_DATA/PROJ_LIB to conda environment (not QGIS)
    """
    
    output_received = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, python_path, script_path, params_file, progress_file):
        super().__init__()
        self.python_path = python_path
        self.script_path = script_path
        self.params_file = params_file
        self.progress_file = progress_file
        self.process = None
        self.is_cancelled = False
    
    def _get_clean_env(self):
        """
        Create a CLEAN environment for running Python with CUDA.
        
        CRITICAL: QGIS injects Qt DLLs and other libraries that conflict
        with PyTorch/CUDA. We need to remove them.
        """
        env = os.environ.copy()
        
        python_dir = os.path.dirname(self.python_path)
        
        # ============ REMOVE PROBLEMATIC VARIABLES ============
        # These cause DLL conflicts with PyTorch/CUDA
        vars_to_remove = [
            'PYTHONHOME',        # Breaks Python module resolution
            'PYTHONPATH',        # Can load wrong modules
            'QT_PLUGIN_PATH',    # QGIS Qt conflicts with PyTorch Qt
            'QT_QPA_PLATFORM_PLUGIN_PATH',  # Qt platform plugins
            'QGIS_PREFIX_PATH',  # QGIS paths
            'QGIS_DEBUG',
            'QGIS_PLUGINPATH',
            'GDAL_DATA',         # Remove QGIS's GDAL, we'll set conda's below
            'PROJ_LIB',          # Remove QGIS's PROJ, we'll set conda's below
        ]
        
        for var in vars_to_remove:
            env.pop(var, None)
        
        # ============ SET ENCODING FOR FRENCH CHARACTERS ============
        # This fixes "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9"
        # which happens with French characters like é, è, ê in file paths
        env['PYTHONUTF8'] = '1'  # Force Python to use UTF-8
        env['PYTHONIOENCODING'] = 'utf-8'  # Force I/O encoding
        
        # ============ SUPPRESS GDAL/RASTERIO UNICODE ERRORS ============
        # GDAL tries to log paths with French chars (é=0xe9) and crashes
        env['CPL_LOG'] = 'OFF'
        env['CPL_LOG_ERRORS'] = 'OFF'
        env['GDAL_PAM_ENABLED'] = 'NO'
        env['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = 'NO'
        
        # ============ CLEAN THE PATH ============
        # Remove any path containing 'QGIS', 'qgis', or OSGeo4W
        if sys.platform == 'win32':
            current_path = env.get('PATH', '')
            path_parts = current_path.split(';')
            
            # Filter out QGIS-related paths
            clean_paths = []
            blocked_keywords = ['qgis', 'osgeo4w', 'grass']
            
            for p in path_parts:
                p_lower = p.lower()
                # Keep the path only if it doesn't contain QGIS-related keywords
                if not any(kw in p_lower for kw in blocked_keywords):
                    clean_paths.append(p)
            
            # Add conda environment paths at the BEGINNING (highest priority)
            conda_paths = [
                python_dir,
                os.path.join(python_dir, 'Library', 'bin'),
                os.path.join(python_dir, 'Scripts'),
                os.path.join(python_dir, 'Library', 'usr', 'bin'),
                os.path.join(python_dir, 'Library', 'mingw-w64', 'bin'),
                os.path.join(python_dir, 'DLLs'),
            ]
            
            # Final PATH: conda paths first, then cleaned system paths
            env['PATH'] = ';'.join(conda_paths + clean_paths)
            
            # Set CONDA_PREFIX
            env['CONDA_PREFIX'] = python_dir
            
            # ============ SET GDAL/PROJ FOR CONDA (not QGIS) ============
            # This fixes "Cannot find gdalvrt.xsd (GDAL_DATA is not defined)"
            conda_gdal_data = os.path.join(python_dir, 'Library', 'share', 'gdal')
            conda_proj_lib = os.path.join(python_dir, 'Library', 'share', 'proj')
            
            if os.path.exists(conda_gdal_data):
                env['GDAL_DATA'] = conda_gdal_data
            if os.path.exists(conda_proj_lib):
                env['PROJ_LIB'] = conda_proj_lib
            
        else:
            # Linux/Mac - simpler cleanup
            current_path = env.get('PATH', '')
            path_parts = current_path.split(':')
            clean_paths = [p for p in path_parts if 'qgis' not in p.lower()]
            env['PATH'] = python_dir + ':' + ':'.join(clean_paths)
        
        return env
    
    def run(self):
        try:
            # Get CLEAN environment (no QGIS contamination)
            env = self._get_clean_env()
            env['PYTHONUNBUFFERED'] = '1'
            
            cmd = [self.python_path, '-u', str(self.script_path), self.params_file, self.progress_file]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace undecodable characters instead of crashing
            )
            
            # Monitor progress file and stdout
            last_progress = 0
            while True:
                if self.is_cancelled:
                    self.process.terminate()
                    self.finished.emit(False, "Cancelled by user")
                    return
                
                line = self.process.stdout.readline()
                if line:
                    self.output_received.emit(line.strip())
                
                if os.path.exists(self.progress_file):
                    try:
                        with open(self.progress_file, 'r', encoding='utf-8') as f:
                            progress_data = json.load(f)
                            progress = progress_data.get('progress', 0)
                            message = progress_data.get('message', '')
                            if progress != last_progress:
                                self.progress_updated.emit(progress, message)
                                last_progress = progress
                    except:
                        pass
                
                if self.process.poll() is not None:
                    break
                
                time.sleep(0.1)
            
            # Read remaining output
            remaining = self.process.stdout.read()
            if remaining:
                noise_markers = ['UnicodeDecodeError', 'Exception ignored in',
                                 'rasterio._env', 'pyogrio/core.py',
                                 'local._env.start', '_register_drivers',
                                 'invalid continuation byte']
                for line in remaining.split('\n'):
                    stripped = line.strip()
                    if stripped and not any(n in stripped for n in noise_markers):
                        self.output_received.emit(stripped)
            
            # Check result
            if self.process.returncode == 0:
                result_msg = "Completed successfully"
                if os.path.exists(self.progress_file):
                    try:
                        with open(self.progress_file, 'r', encoding='utf-8') as f:
                            progress_data = json.load(f)
                            result_msg = progress_data.get('result', result_msg)
                    except:
                        pass
                self.finished.emit(True, result_msg)
            else:
                self.finished.emit(False, f"Process exited with code {self.process.returncode}")
                
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def cancel(self):
        self.is_cancelled = True
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except:
                pass


class EnvironmentConfigDialog(QDialog):
    """Simple, user-friendly dialog to configure the external Python environment.
    
    Three ways to set up (all on one screen):
    1. Browse directly for python.exe / python3
    2. Pick from auto-detected environments (conda, venv)
    3. Type / paste the path manually
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SemanticSeg4EO – Python Environment")
        self.setMinimumSize(680, 620)
        self._selected_path = ""
        self._setup_ui()
        # Auto-scan on open
        QTimer.singleShot(200, self._scan_environments)
    
    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        root = QVBoxLayout()
        
        # ── Current status ──────────────────────────────────────────────
        current_config = PluginConfig.load()
        current_path = current_config.get('python_path', '')
        
        if current_path and os.path.isfile(current_path):
            status_text = f"<span style='color:green;'>✔ Current: <b>{current_path}</b></span>"
        else:
            status_text = "<span style='color:#c0392b;'>✖ No Python environment configured yet</span>"
        
        status_lbl = QLabel(status_text)
        status_lbl.setWordWrap(True)
        root.addWidget(status_lbl)
        
        root.addSpacing(4)
        
        # ── Option 1 : Browse ───────────────────────────────────────────
        browse_group = QGroupBox("① Browse for python executable")
        bg_layout = QVBoxLayout()
        
        browse_hint = QLabel(
            "Select the <b>python.exe</b> (Windows) or <b>python</b> / <b>python3</b> (Linux/Mac) "
            "from your Conda or venv environment."
        )
        browse_hint.setWordWrap(True)
        bg_layout.addWidget(browse_hint)
        
        # Typical locations hint
        if sys.platform == 'win32':
            loc_hint = (
                "<i>Typical locations:<br>"
                " • Conda:  C:\\Users\\&lt;you&gt;\\miniconda3\\envs\\&lt;env&gt;\\python.exe<br>"
                " • venv:   C:\\&lt;folder&gt;\\Scripts\\python.exe</i>"
            )
        else:
            loc_hint = (
                "<i>Typical locations:<br>"
                " • Conda:  ~/miniconda3/envs/&lt;env&gt;/bin/python<br>"
                " • venv:   ~/&lt;env&gt;/bin/python</i>"
            )
        loc_lbl = QLabel(loc_hint)
        loc_lbl.setStyleSheet("color: #666; font-size: 11px;")
        loc_lbl.setWordWrap(True)
        bg_layout.addWidget(loc_lbl)
        
        self.btn_browse = QPushButton("  Browse for python…")
        self.btn_browse.setMinimumHeight(36)
        self.btn_browse.setStyleSheet(
            "QPushButton { background-color: #2980b9; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #3498db; }"
        )
        self.btn_browse.clicked.connect(self._browse_python)
        bg_layout.addWidget(self.btn_browse)
        
        browse_group.setLayout(bg_layout)
        root.addWidget(browse_group)
        
        # ── Option 2 : Auto-detected environments ──────────────────────
        detect_group = QGroupBox("② Or pick a detected environment")
        dg_layout = QVBoxLayout()
        
        scan_row = QHBoxLayout()
        self.btn_scan = QPushButton("Scan again")
        self.btn_scan.clicked.connect(self._scan_environments)
        scan_row.addWidget(self.btn_scan)
        self.scan_status = QLabel("")
        scan_row.addWidget(self.scan_status)
        scan_row.addStretch()
        dg_layout.addLayout(scan_row)
        
        self.env_list = QComboBox()
        self.env_list.setMinimumHeight(28)
        self.env_list.currentIndexChanged.connect(self._on_env_selected)
        dg_layout.addWidget(self.env_list)
        
        detect_group.setLayout(dg_layout)
        root.addWidget(detect_group)
        
        # ── Option 3 : Manual path ─────────────────────────────────────
        manual_group = QGroupBox("③ Or type / paste the path")
        mg_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Full path to python executable…")
        if current_path:
            self.path_edit.setText(current_path)
            self._selected_path = current_path
        self.path_edit.textChanged.connect(self._on_path_edited)
        mg_layout.addWidget(self.path_edit)
        manual_group.setLayout(mg_layout)
        root.addWidget(manual_group)
        
        # ── Verify section ──────────────────────────────────────────────
        verify_group = QGroupBox("Verification")
        vg_layout = QVBoxLayout()
        
        self.btn_verify = QPushButton("Verify selected Python")
        self.btn_verify.setMinimumHeight(32)
        self.btn_verify.clicked.connect(self._verify)
        vg_layout.addWidget(self.btn_verify)
        
        self.verify_output = QTextEdit()
        self.verify_output.setReadOnly(True)
        self.verify_output.setFont(QFont("Courier", 9))
        self.verify_output.setMaximumHeight(160)
        vg_layout.addWidget(self.verify_output)
        
        self.verify_status = QLabel("")
        vg_layout.addWidget(self.verify_status)
        
        verify_group.setLayout(vg_layout)
        root.addWidget(verify_group)
        
        # ── Buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        
        self.btn_install_help = QPushButton("How to install?")
        self.btn_install_help.clicked.connect(self._show_install_help)
        btn_row.addWidget(self.btn_install_help)
        
        btn_row.addStretch()
        
        self.btn_save = QPushButton("Save && Close")
        self.btn_save.setMinimumHeight(32)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #bdc3c7; }"
        )
        self.btn_save.clicked.connect(self._save_and_close)
        btn_row.addWidget(self.btn_save)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        
        root.addLayout(btn_row)
        self.setLayout(root)
    
    # ------------------------------------------------------------------ Browse
    def _browse_python(self):
        if sys.platform == 'win32':
            filter_str = "Python executable (python.exe);;All files (*)"
            start_dir = str(Path.home())
        else:
            filter_str = "All files (*)"
            start_dir = str(Path.home())
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select python executable", start_dir, filter_str
        )
        if filename:
            self.path_edit.setText(filename)
            self._selected_path = filename
    
    # ------------------------------------------------------------------ Scan
    def _scan_environments(self):
        """Auto-detect conda and venv environments on the system."""
        self.env_list.clear()
        self.env_list.addItem("— select —", "")
        self.scan_status.setText("Scanning…")
        
        found = []
        
        # ---- Conda environments ----
        conda_bases = []
        
        # From CONDA_PREFIX (active conda)
        cp = os.environ.get('CONDA_PREFIX', '')
        if cp:
            envs_dir = Path(cp).parent
            if envs_dir.is_dir():
                conda_bases.append(envs_dir)
            # Also the base itself might have an envs subdir
            base_envs = Path(cp).parent.parent / 'envs'
            if base_envs.is_dir():
                conda_bases.append(base_envs)
        
        # Common conda locations
        home = Path.home()
        for base_name in ['miniconda3', 'Miniconda3', 'anaconda3', 'Anaconda3',
                          'miniforge3', 'Miniforge3', 'mambaforge', 'Mambaforge']:
            candidate = home / base_name / 'envs'
            if candidate.is_dir():
                conda_bases.append(candidate)
        
        if sys.platform == 'win32':
            for base_name in ['miniconda3', 'Miniconda3', 'anaconda3', 'Anaconda3',
                              'miniforge3', 'mambaforge']:
                candidate = home / 'AppData' / 'Local' / base_name / 'envs'
                if candidate.is_dir():
                    conda_bases.append(candidate)
            # ProgramData
            pd = Path(os.environ.get('ProgramData', 'C:/ProgramData'))
            for base_name in ['miniconda3', 'Anaconda3']:
                candidate = pd / base_name / 'envs'
                if candidate.is_dir():
                    conda_bases.append(candidate)
        else:
            for p in [Path('/opt/conda/envs'), Path('/opt/miniconda3/envs')]:
                if p.is_dir():
                    conda_bases.append(p)
        
        # De-duplicate
        seen_dirs = set()
        for envs_dir in conda_bases:
            envs_dir = envs_dir.resolve()
            if envs_dir in seen_dirs:
                continue
            seen_dirs.add(envs_dir)
            
            try:
                for entry in sorted(envs_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    if sys.platform == 'win32':
                        py = entry / 'python.exe'
                    else:
                        py = entry / 'bin' / 'python'
                    if py.is_file():
                        label = f"[conda] {entry.name}  —  {py}"
                        found.append((label, str(py)))
            except PermissionError:
                pass
        
        # Also check conda base environment itself
        for base_name in ['miniconda3', 'Miniconda3', 'anaconda3', 'Anaconda3',
                          'miniforge3', 'mambaforge']:
            base_dir = home / base_name
            if sys.platform == 'win32':
                py = base_dir / 'python.exe'
            else:
                py = base_dir / 'bin' / 'python'
            if py.is_file():
                label = f"[conda base] {base_name}  —  {py}"
                if (label, str(py)) not in found:
                    found.append((label, str(py)))
        
        # ---- venv: look in common locations ----
        venv_candidates = [
            home / 'semanticseg4eo_env',
            home / '.virtualenvs',
            home / 'envs',
            home / 'venvs',
        ]
        if sys.platform == 'win32':
            venv_candidates.append(Path('C:/') / 'semanticseg4eo_env')
        
        for venv_dir in venv_candidates:
            if not venv_dir.is_dir():
                continue
            # Check if it's a venv itself
            if sys.platform == 'win32':
                py = venv_dir / 'Scripts' / 'python.exe'
            else:
                py = venv_dir / 'bin' / 'python'
            if py.is_file():
                label = f"[venv] {venv_dir.name}  —  {py}"
                found.append((label, str(py)))
            else:
                # Maybe it's a folder of venvs
                try:
                    for entry in sorted(venv_dir.iterdir()):
                        if not entry.is_dir():
                            continue
                        if sys.platform == 'win32':
                            py = entry / 'Scripts' / 'python.exe'
                        else:
                            py = entry / 'bin' / 'python'
                        if py.is_file():
                            label = f"[venv] {entry.name}  —  {py}"
                            found.append((label, str(py)))
                except PermissionError:
                    pass
        
        # Populate combo
        for label, path in found:
            self.env_list.addItem(label, path)
        
        n = len(found)
        self.scan_status.setText(f"{n} environment{'s' if n != 1 else ''} found" if n else "No environments detected")
    
    def _on_env_selected(self, index):
        path = self.env_list.currentData()
        if path:
            self.path_edit.setText(path)
            self._selected_path = path
    
    def _on_path_edited(self, text):
        self._selected_path = text.strip()
    
    # ------------------------------------------------------------------ Verify
    def _verify(self):
        python_path = self._selected_path
        if not python_path:
            self.verify_output.setPlainText("Please select or enter a Python path first.")
            return
        
        if not os.path.isfile(python_path):
            self.verify_output.setPlainText(f"File not found:\n{python_path}")
            self.verify_status.setText("❌ File does not exist")
            self.verify_status.setStyleSheet("color: red; font-weight: bold;")
            return
        
        self.verify_output.clear()
        self.verify_status.setText("Checking…")
        self.verify_output.append(f"Python: {python_path}\n")
        
        env = self._get_clean_env(python_path)
        
        # Python version
        try:
            r = subprocess.run(
                [python_path, '--version'],
                capture_output=True, text=True, timeout=10, env=env
            )
            self.verify_output.append(r.stdout.strip() or r.stderr.strip())
        except Exception as e:
            self.verify_output.append(f"Cannot run Python: {e}")
            self.verify_status.setText("❌ Cannot execute")
            self.verify_status.setStyleSheet("color: red; font-weight: bold;")
            return
        
        # Required packages
        required = ['torch', 'numpy', 'rasterio', 'tifffile']
        optional = ['segmentation_models_pytorch', 'geopandas', 'cv2',
                     'transformers', 'timm']
        
        all_ok = True
        self.verify_output.append("\nRequired packages:")
        for pkg in required:
            try:
                code = f"import {pkg}; print(getattr({pkg}, '__version__', 'OK'))"
                r = subprocess.run(
                    [python_path, '-c', code],
                    capture_output=True, text=True, timeout=30, env=env
                )
                if r.returncode == 0:
                    self.verify_output.append(f"  ✔ {pkg}: {r.stdout.strip()}")
                else:
                    self.verify_output.append(f"  ✖ {pkg}: NOT FOUND")
                    all_ok = False
            except Exception as e:
                self.verify_output.append(f"  ✖ {pkg}: error ({e})")
                all_ok = False
        
        self.verify_output.append("\nOptional packages:")
        for pkg in optional:
            try:
                r = subprocess.run(
                    [python_path, '-c', f'import {pkg}'],
                    capture_output=True, text=True, timeout=30, env=env
                )
                status = "✔ installed" if r.returncode == 0 else "— not installed"
                self.verify_output.append(f"  {status}: {pkg}")
            except:
                self.verify_output.append(f"  — {pkg}: unknown")
        
        # CUDA
        try:
            r = subprocess.run(
                [python_path, '-c',
                 'import torch; print("CUDA:", "available" if torch.cuda.is_available() else "not available")'],
                capture_output=True, text=True, timeout=30, env=env
            )
            if r.returncode == 0:
                self.verify_output.append(f"\n{r.stdout.strip()}")
        except:
            pass
        
        if all_ok:
            self.verify_status.setText("✔ Environment is ready!")
            self.verify_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.verify_status.setText("⚠ Some required packages are missing – see above")
            self.verify_status.setStyleSheet("color: #e67e22; font-weight: bold;")
    
    # ------------------------------------------------------------------ Helpers
    def _get_clean_env(self, python_path):
        """Build a clean env dict for subprocess (remove QGIS paths)."""
        env = os.environ.copy()
        python_dir = os.path.dirname(python_path)
        
        for var in ['PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH',
                     'QT_QPA_PLATFORM_PLUGIN_PATH', 'QGIS_PREFIX_PATH']:
            env.pop(var, None)
        
        if sys.platform == 'win32':
            parts = env.get('PATH', '').split(';')
            clean = [p for p in parts
                     if not any(k in p.lower() for k in ['qgis', 'osgeo4w', 'grass'])]
            conda_paths = [
                python_dir,
                os.path.join(python_dir, 'Library', 'bin'),
                os.path.join(python_dir, 'Scripts'),
            ]
            env['PATH'] = ';'.join(conda_paths + clean)
            env['CONDA_PREFIX'] = python_dir
        else:
            parts = env.get('PATH', '').split(':')
            clean = [p for p in parts if 'qgis' not in p.lower()]
            env['PATH'] = python_dir + ':' + ':'.join(clean)
        
        return env
    
    def _detect_env_type(self, python_path):
        """Guess env_type from the path for config storage."""
        p = python_path.lower().replace('\\', '/')
        if '/envs/' in p and ('conda' in p or 'miniconda' in p
                              or 'anaconda' in p or 'miniforge' in p
                              or 'mambaforge' in p):
            return 'conda'
        if '/scripts/python' in p or '/bin/python' in p:
            # Check for pyvenv.cfg (venv marker)
            parent = Path(python_path).resolve().parent.parent
            if (parent / 'pyvenv.cfg').is_file():
                return 'venv'
        return 'system'
    
    # ------------------------------------------------------------------ Save
    def _save_and_close(self):
        python_path = self._selected_path
        if not python_path or not os.path.isfile(python_path):
            QMessageBox.warning(
                self, "No valid Python selected",
                "Please select a valid python executable before saving.\n\n"
                "Use the Browse button or pick from the detected list."
            )
            return
        
        env_type = self._detect_env_type(python_path)
        
        config = PluginConfig.load()
        config['python_path'] = python_path
        config['env_type'] = env_type
        config['dependencies_ok'] = True
        
        # Store venv/conda metadata for display purposes
        if env_type == 'conda':
            # Extract env name from path
            parts = Path(python_path).parts
            try:
                idx = [p.lower() for p in parts].index('envs')
                config['conda_env_name'] = parts[idx + 1]
            except (ValueError, IndexError):
                config['conda_env_name'] = ''
        elif env_type == 'venv':
            config['venv_path'] = str(Path(python_path).resolve().parent.parent)
        
        PluginConfig.save(config)
        self.accept()
    
    # ------------------------------------------------------------------ Help
    def _show_install_help(self):
        help_text = """
<h3>How to create a Python environment</h3>

<h4>Option A – Conda (recommended)</h4>
<pre>
conda create -n semanticseg4eo python=3.10 -y
conda activate semanticseg4eo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install rasterio tifffile geopandas opencv-python
pip install segmentation-models-pytorch
pip install numpy scipy scikit-learn matplotlib tqdm
</pre>

<h4>Option B – venv</h4>
<pre>
python -m venv ~/semanticseg4eo_env
source ~/semanticseg4eo_env/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install rasterio tifffile geopandas opencv-python
pip install segmentation-models-pytorch
pip install numpy scipy scikit-learn matplotlib tqdm
</pre>

<p><b>For GPU (NVIDIA):</b> replace the torch install line with:<br>
<code>pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118</code></p>

<p>After creating the environment, come back here and <b>Browse</b> to its 
<code>python.exe</code> (Windows) or <code>python</code> (Linux/Mac).</p>
        """
        QMessageBox.information(self, "Installation Help", help_text)


class SemanticSeg4EODialog(QDialog):
    """Main dialog with tabs for extraction, training, and prediction"""
    
    # =====================================================================
    # ARCHITECTURE / ENCODER / LOSS DEFINITIONS
    # These must match what model_training.py and prediction.py support
    # =====================================================================
    
    # Models organized by category for the combo box
    ALL_ARCHITECTURES = [
        # Built-in (no SMP/timm needed)
        'unet-dropout',
        # SMP architectures
        'unet', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan',
        'deeplabv3', 'deeplabv3+',
        # Modern (require transformers/timm)
        'segformer-b0', 'segformer-b1', 'segformer-b2', 'segformer-b3',
        'segformer-b4', 'segformer-b5',
        'unetformer',
        'hrnet-w18', 'hrnet-w32', 'hrnet-w48',
        'swin-unet',
    ]
    
    # Standard SMP encoders (complete list)
    STANDARD_ENCODERS = [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        # ResNeXt family
        'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
        # SE-Net family
        'se_resnet50', 'se_resnet101', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',
        'senet154',
        # EfficientNet family
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
        # timm-EfficientNet (higher quality pretrained)
        'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
        'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5',
        'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8',
        'timm-efficientnet-l2',
        # ResNeSt (timm)
        'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d',
        'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e',
        # DenseNet family
        'densenet121', 'densenet169', 'densenet201', 'densenet161',
        # Inception family
        'inceptionresnetv2', 'inceptionv4',
        # MobileNet
        'mobilenet_v2',
        # DPN family
        'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
        # VGG family
        'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
        # Mix Vision Transformer (SegFormer backbone via SMP)
        'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
        # MobileOne
        'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4',
    ]
    
    # ConvNeXt encoders (only with UNet architecture)
    CONVNEXT_ENCODERS = [
        'convnext_tiny', 'convnext_small', 'convnext_base',
        'convnext_large', 'convnext_xlarge',
    ]
    
    # Models that don't use an encoder (self-contained)
    MODELS_WITHOUT_ENCODER = [
        'unet-dropout',
        'segformer-b0', 'segformer-b1', 'segformer-b2', 'segformer-b3',
        'segformer-b4', 'segformer-b5',
        'hrnet-w18', 'hrnet-w32', 'hrnet-w48',
        'swin-unet',
    ]
    
    # Loss functions by mode
    LOSSES_BINARY = [
        'binary_dice_bce', 'binary_focal_dice', 'bce', 'binary_dice',
        'binary_focal', 'binary_tversky', 'binary_focal_tversky',
    ]
    LOSSES_MULTICLASS = [
        'dice_ce', 'focal_dice', 'ce', 'dice', 'focal',
        'tversky', 'focal_tversky', 'combo',
    ]
    
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.runner = None
        self.temp_dir = None
        
        self.setWindowTitle("SemanticSeg4EO - Semantic Segmentation v3")
        self.setMinimumSize(1000, 750)
        self.resize(1050, 800)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Environment status bar
        self.env_status_bar = QGroupBox("Environment Status")
        env_layout = QHBoxLayout()
        
        self.env_status_icon = QLabel("●")
        self.env_status_icon.setFixedWidth(16)
        env_layout.addWidget(self.env_status_icon)
        
        self.env_status_label = QLabel("Not configured")
        self.env_status_label.setWordWrap(True)
        env_layout.addWidget(self.env_status_label, 1)
        
        self.btn_quick_browse = QPushButton("Browse python…")
        self.btn_quick_browse.setToolTip("Quickly select a python executable without opening the full dialog")
        self.btn_quick_browse.clicked.connect(self._quick_browse_python)
        env_layout.addWidget(self.btn_quick_browse)
        
        self.btn_configure = QPushButton("Configure Environment")
        self.btn_configure.clicked.connect(self.open_environment_wizard)
        env_layout.addWidget(self.btn_configure)
        
        self.env_status_bar.setLayout(env_layout)
        layout.addWidget(self.env_status_bar)
        
        # === SPLITTER: tabs on top, output on bottom (resizable) ===
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_extraction_tab(), "Patch Extraction")
        self.tabs.addTab(self._create_training_tab(), "Model Training")
        self.tabs.addTab(self._create_prediction_tab(), "Prediction")
        self.main_splitter.addWidget(self.tabs)
        
        # Output section (in splitter — user can resize by dragging)
        output_widget = QWidget()
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        
        output_header = QHBoxLayout()
        output_header.addWidget(QLabel("<b>Output & Progress</b>"))
        output_header.addStretch()
        self.btn_detach_log = QPushButton("Detach Log")
        self.btn_detach_log.setToolTip("Open log in a separate resizable window")
        self.btn_detach_log.clicked.connect(self._detach_log_window)
        output_header.addWidget(self.btn_detach_log)
        output_layout.addLayout(output_header)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        output_layout.addWidget(self.log_text)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        output_layout.addWidget(self.progress_bar)
        
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_operation)
        btn_layout.addWidget(self.btn_cancel)
        
        btn_layout.addStretch()
        
        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)
        btn_layout.addWidget(self.btn_help)
        
        output_layout.addLayout(btn_layout)
        output_widget.setLayout(output_layout)
        self.main_splitter.addWidget(output_widget)
        
        # Set initial splitter proportions (tabs bigger, output smaller)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([450, 300])
        
        layout.addWidget(self.main_splitter)
        
        self.setLayout(layout)
        
        # Detached log window reference
        self._detached_log = None
        
        # Check environment on startup
        self.update_environment_status()
    
    # =====================================================================
    # EXTRACTION TAB (unchanged)
    # =====================================================================
    
    def _create_extraction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # --- Mode selector ---
        mode_group = QGroupBox("Extraction Mode")
        mode_layout = QHBoxLayout()

        self.extract_mode_group = QButtonGroup(self)
        self.extract_mode_single = QRadioButton("Single (1 image + 1 label + 1 grid)")
        self.extract_mode_single.setChecked(True)
        self.extract_mode_batch = QRadioButton("Batch (multiple images / labels)")
        self.extract_mode_group.addButton(self.extract_mode_single, 0)
        self.extract_mode_group.addButton(self.extract_mode_batch, 1)
        mode_layout.addWidget(self.extract_mode_single)
        mode_layout.addWidget(self.extract_mode_batch)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self.extract_mode_group.idToggled.connect(self._on_extraction_mode_changed)

        # --- Scrollable area for inputs (changes depending on mode) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # === SINGLE MODE inputs ===
        self.extract_single_group = QGroupBox("Input Files (Single)")
        single_layout = QFormLayout()

        self.extract_image = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.extract_image, "GeoTIFF (*.tif *.tiff)"))
        h = QHBoxLayout()
        h.addWidget(self.extract_image)
        h.addWidget(btn)
        single_layout.addRow("Satellite Image:", h)

        self.extract_label = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.extract_label, "GeoTIFF (*.tif *.tiff)"))
        h = QHBoxLayout()
        h.addWidget(self.extract_label)
        h.addWidget(btn)
        single_layout.addRow("Labels / Mask:", h)

        self.extract_grid = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.extract_grid, "Shapefile (*.shp);;GeoPackage (*.gpkg)"))
        h = QHBoxLayout()
        h.addWidget(self.extract_grid)
        h.addWidget(btn)
        single_layout.addRow("Grid (Shapefile):", h)

        self.extract_single_group.setLayout(single_layout)
        scroll_layout.addWidget(self.extract_single_group)

        # === BATCH MODE inputs ===
        self.extract_batch_group = QGroupBox("Input Files (Batch)")
        batch_layout = QFormLayout()

        self.extract_data_dir = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.extract_data_dir))
        h = QHBoxLayout()
        h.addWidget(self.extract_data_dir)
        h.addWidget(btn)
        batch_layout.addRow("Data Directory:", h)

        batch_help = QLabel(
            "<i>Expected: Image_1.tif, Label_1.tif, Image_2.tif, Label_2.tif, ...</i>"
        )
        batch_help.setStyleSheet("color: gray; font-size: 10px;")
        batch_layout.addRow("", batch_help)

        self.extract_batch_grid = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(
            self.extract_batch_grid, "Shapefile (*.shp);;GeoPackage (*.gpkg)"))
        h = QHBoxLayout()
        h.addWidget(self.extract_batch_grid)
        h.addWidget(btn)
        batch_layout.addRow("Shared Grid:", h)

        self.extract_per_image_grid = QCheckBox("Use per-image grids (Grid_1.shp, Grid_2.shp, ...)")
        self.extract_per_image_grid.setToolTip(
            "If checked, each Image_N.tif will use Grid_N.shp from the data directory.\n"
            "Shared grid is used as fallback if a per-image grid is missing."
        )
        batch_layout.addRow("", self.extract_per_image_grid)

        self.extract_recursive = QCheckBox("Search subdirectories recursively")
        batch_layout.addRow("", self.extract_recursive)

        # Custom patterns (collapsed by default)
        self.extract_custom_patterns = QCheckBox("Custom file patterns (advanced)")
        batch_layout.addRow("", self.extract_custom_patterns)

        self.extract_img_pattern = QLineEdit(r'^[Ii]mage[_-]?(\d+)\.tif{1,2}$')
        self.extract_img_pattern.setVisible(False)
        batch_layout.addRow("Image pattern:", self.extract_img_pattern)

        self.extract_lbl_pattern = QLineEdit(r'^[Ll]abel[_-]?(\d+)\.tif{1,2}$')
        self.extract_lbl_pattern.setVisible(False)
        batch_layout.addRow("Label pattern:", self.extract_lbl_pattern)

        self.extract_custom_patterns.toggled.connect(self.extract_img_pattern.setVisible)
        self.extract_custom_patterns.toggled.connect(self.extract_lbl_pattern.setVisible)

        self.extract_batch_group.setLayout(batch_layout)
        self.extract_batch_group.setVisible(False)  # Hidden by default
        scroll_layout.addWidget(self.extract_batch_group)

        # === OUTPUT directory (shared) ===
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()

        self.extract_output = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.extract_output))
        h = QHBoxLayout()
        h.addWidget(self.extract_output)
        h.addWidget(btn)
        output_layout.addRow("Output Directory:", h)

        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)

        # === Parameters ===
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        self.extract_patch_size = QComboBox()
        self.extract_patch_size.addItems(['64', '128', '224', '256', '512'])
        self.extract_patch_size.setCurrentText('224')
        params_layout.addRow("Patch Size:", self.extract_patch_size)

        self.extract_channels = QSpinBox()
        self.extract_channels.setRange(1, 50)
        self.extract_channels.setValue(10)
        params_layout.addRow("Image Channels:", self.extract_channels)

        self.extract_output_dtype = QComboBox()
        self.extract_output_dtype.addItems(['float32', 'int16', 'uint16', 'uint8'])
        self.extract_output_dtype.setToolTip(
            "float32: for data with indices, MNS, MNT, normalized values\n"
            "int16: for raw satellite imagery (Sentinel-2 L2A, etc.)\n"
            "uint16: for raw drone/aerial RGB imagery\n"
            "uint8: for standard 8-bit RGB images (0-255)"
        )
        params_layout.addRow("Output Data Type:", self.extract_output_dtype)

        self.extract_interpolation = QComboBox()
        self.extract_interpolation.addItems(['bilinear', 'nearest', 'bicubic', 'lanczos'])
        self.extract_interpolation.setToolTip(
            "bilinear: good default for images\n"
            "nearest: preserves exact values (labels always use nearest)\n"
            "bicubic/lanczos: higher quality, slower"
        )
        params_layout.addRow("Interpolation:", self.extract_interpolation)

        self.extract_compression = QComboBox()
        self.extract_compression.addItems(['deflate', 'lzw', 'none'])
        params_layout.addRow("Compression:", self.extract_compression)

        self.extract_dtype = QComboBox()
        self.extract_dtype.addItems(['float32', 'int16', 'uint8', 'uint16'])
        self.extract_dtype.setToolTip(
            "float32: for data with indices, DSM, DTM, normalized values\n"
            "int16: for raw reflectance or signed integer data\n"
            "uint16: for unsigned integer satellite data (e.g. Sentinel-2 L1C)\n"
            "uint8: for RGB imagery (0-255)"
        )
        params_layout.addRow("Image Data Type:", self.extract_dtype)

        self.extract_validate_crs = QCheckBox("Validate CRS compatibility")
        self.extract_validate_crs.setChecked(True)
        params_layout.addRow("", self.extract_validate_crs)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        params_layout.addRow(line)

        # Split ratios
        self.extract_train_ratio = QDoubleSpinBox()
        self.extract_train_ratio.setRange(0.1, 0.9)
        self.extract_train_ratio.setSingleStep(0.05)
        self.extract_train_ratio.setValue(0.70)
        self.extract_train_ratio.valueChanged.connect(self._update_test_ratio_display)
        params_layout.addRow("Train Ratio:", self.extract_train_ratio)

        self.extract_val_ratio = QDoubleSpinBox()
        self.extract_val_ratio.setRange(0.05, 0.4)
        self.extract_val_ratio.setSingleStep(0.05)
        self.extract_val_ratio.setValue(0.20)
        self.extract_val_ratio.valueChanged.connect(self._update_test_ratio_display)
        params_layout.addRow("Validation Ratio:", self.extract_val_ratio)

        self.extract_test_ratio = QDoubleSpinBox()
        self.extract_test_ratio.setRange(0.0, 0.3)
        self.extract_test_ratio.setSingleStep(0.05)
        self.extract_test_ratio.setValue(0.10)
        self.extract_test_ratio.valueChanged.connect(self._update_test_ratio_display)
        params_layout.addRow("Test Ratio:", self.extract_test_ratio)

        self.extract_ratio_info = QLabel("Total: 1.00")
        self.extract_ratio_info.setStyleSheet("color: green;")
        params_layout.addRow("", self.extract_ratio_info)

        params_group.setLayout(params_layout)
        scroll_layout.addWidget(params_group)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Run button
        self.extract_run = QPushButton("Start Extraction")
        self.extract_run.setMinimumHeight(40)
        self.extract_run.clicked.connect(self.run_extraction)
        layout.addWidget(self.extract_run)

        tab.setLayout(layout)
        return tab
    
    def _on_extraction_mode_changed(self, btn_id, checked):
        """Toggle visibility between single and batch mode inputs."""
        if not checked:
            return
        is_batch = (btn_id == 1)
        self.extract_single_group.setVisible(not is_batch)
        self.extract_batch_group.setVisible(is_batch)

    # =====================================================================
    # TRAINING TAB (v3 - with sub-tabs: Basic + Advanced)
    # =====================================================================
    
    def _create_training_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout()
        
        # === TOP: Dataset selection (full width) ===
        data_group = QGroupBox("Dataset")
        data_layout = QFormLayout()
        
        self.train_dataset = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.train_dataset))
        h = QHBoxLayout()
        h.addWidget(self.train_dataset)
        h.addWidget(btn)
        data_layout.addRow("Dataset Directory:", h)
        
        # Mode and classes on same row
        mode_layout = QHBoxLayout()
        self.train_mode = QComboBox()
        self.train_mode.addItems(['binary', 'multiclass'])
        self.train_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.train_mode)
        mode_layout.addSpacing(20)
        self.train_classes = QSpinBox()
        self.train_classes.setRange(2, 50)
        self.train_classes.setValue(2)  # Binary = 2 classes
        self.train_classes.setEnabled(False)  # Disabled for binary mode
        mode_layout.addWidget(QLabel("Classes:"))
        mode_layout.addWidget(self.train_classes)
        mode_layout.addStretch()
        data_layout.addRow("", mode_layout)
        
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)
        
        # === SUB-TABS for training parameters ===
        self.train_subtabs = QTabWidget()
        self.train_subtabs.addTab(self._create_training_basic_subtab(), "Basic Parameters")
        self.train_subtabs.addTab(self._create_training_advanced_subtab(), "Advanced Parameters")
        main_layout.addWidget(self.train_subtabs)
        
        # === BOTTOM: Output + Run button ===
        bottom_layout = QHBoxLayout()
        
        bottom_layout.addWidget(QLabel("Output:"))
        self.train_save = QLineEdit("./trained_models")
        bottom_layout.addWidget(self.train_save)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_folder(self.train_save))
        bottom_layout.addWidget(btn)
        
        self.train_run = QPushButton("Start Training")
        self.train_run.setMinimumHeight(35)
        self.train_run.setMinimumWidth(150)
        self.train_run.clicked.connect(self.run_training)
        bottom_layout.addWidget(self.train_run)
        
        main_layout.addLayout(bottom_layout)
        
        tab.setLayout(main_layout)
        return tab
    
    def _create_training_basic_subtab(self):
        """Basic parameters sub-tab - for most users"""
        widget = QWidget()
        columns_layout = QHBoxLayout()
        
        # --- LEFT COLUMN: Model ---
        left_column = QVBoxLayout()
        
        model_group = QGroupBox("Model")
        model_layout = QFormLayout()
        
        self.train_arch = QComboBox()
        self.train_arch.addItems(self.ALL_ARCHITECTURES)
        self.train_arch.setToolTip(
            "=== Built-in ===\n"
            "  unet-dropout: Simple U-Net (no deps)\n\n"
            "=== SMP Architectures (need smp) ===\n"
            "  unet, unet++, deeplabv3+, manet, fpn, etc.\n\n"
            "=== Modern (need transformers/timm) ===\n"
            "  segformer-b0..b5: SegFormer (Transformer)\n"
            "  unetformer: U-Net + Transformer\n"
            "  hrnet-w18/w32/w48: High-Res Net\n"
            "  swin-unet: Swin Transformer U-Net"
        )
        self.train_arch.currentTextChanged.connect(self._on_arch_changed)
        model_layout.addRow("Architecture:", self.train_arch)
        
        self.train_encoder = QComboBox()
        self.train_encoder.addItems(self.STANDARD_ENCODERS + self.CONVNEXT_ENCODERS)
        self.train_encoder.setCurrentText('resnet34')
        self.train_encoder.setToolTip(
            "Encoder backbone (for SMP & UNetFormer)\n\n"
            "=== Lightweight ===\n"
            "  resnet18, mobilenet_v2, efficientnet-b0/b1\n\n"
            "=== Balanced (recommended) ===\n"
            "  resnet34, resnet50, efficientnet-b3\n\n"
            "=== Heavy (best accuracy) ===\n"
            "  resnet101, efficientnet-b5, se_resnext50\n\n"
            "=== ConvNeXt (UNet only, needs timm) ===\n"
            "  convnext_tiny/small/base/large/xlarge"
        )
        model_layout.addRow("Encoder:", self.train_encoder)
        
        # Encoder info label (dynamic)
        self.train_encoder_info = QLabel("")
        self.train_encoder_info.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addRow("", self.train_encoder_info)
        
        self.train_channels = QSpinBox()
        self.train_channels.setRange(1, 50)
        self.train_channels.setValue(4)
        model_layout.addRow("Input Channels:", self.train_channels)
        
        self.train_pretrained = QCheckBox("ImageNet pretrained")
        self.train_pretrained.setToolTip("Use pretrained encoder weights (recommended)")
        model_layout.addRow("", self.train_pretrained)
        
        model_group.setLayout(model_layout)
        left_column.addWidget(model_group)
        
        # Training params group
        train_group = QGroupBox("Training")
        train_layout = QFormLayout()
        
        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 1000)
        self.train_epochs.setValue(100)
        train_layout.addRow("Epochs:", self.train_epochs)
        
        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 64)
        self.train_batch.setValue(4)
        train_layout.addRow("Batch Size:", self.train_batch)
        
        self.train_lr = QLineEdit("0.0003")
        self.train_lr.setToolTip("0.0001-0.0003 for pretrained, 0.001 for scratch")
        train_layout.addRow("Learning Rate:", self.train_lr)
        
        self.train_device = QComboBox()
        self.train_device.addItems(['cuda', 'cpu'])
        train_layout.addRow("Device:", self.train_device)
        
        train_group.setLayout(train_layout)
        left_column.addWidget(train_group)
        left_column.addStretch()
        
        columns_layout.addLayout(left_column)
        
        # --- RIGHT COLUMN: Optimization + Regularization ---
        right_column = QVBoxLayout()
        
        optim_group = QGroupBox("Optimization")
        optim_layout = QFormLayout()
        
        self.train_scheduler = QComboBox()
        self.train_scheduler.addItems(['reduce_on_plateau', 'cosine_annealing', 'one_cycle', 'none'])
        self.train_scheduler.setToolTip(
            "reduce_on_plateau: Reduce LR when metric stagnates\n"
            "cosine_annealing: Gradual LR decrease\n"
            "one_cycle: Fast convergence\n"
            "none: Fixed LR"
        )
        optim_layout.addRow("LR Scheduler:", self.train_scheduler)
        
        self.train_loss_fn = QComboBox()
        # Will be populated by _on_mode_changed
        self.train_loss_fn.setToolTip(
            "Loss function (auto-converted if mode changes)\n\n"
            "dice_ce / binary_dice_bce: Balanced (recommended)\n"
            "focal / binary_focal: For class imbalance\n"
            "tversky: Control precision/recall balance\n"
            "combo: Multi-loss combination"
        )
        optim_layout.addRow("Loss Function:", self.train_loss_fn)
        
        self.train_augmentation = QComboBox()
        self.train_augmentation.addItems(['none', 'basic', 'advanced', 'aggressive', 'extreme'])
        self.train_augmentation.setCurrentText('basic')
        self.train_augmentation.setToolTip(
            "none: No augmentation\n"
            "basic: Flips + Rotation90\n"
            "advanced: + Scale, Brightness, Contrast, Gamma\n"
            "aggressive: + Elastic, Blur, Noise, Dropout, Oversampling\n"
            "extreme: + Free rotation, GridDistort, MixUp, CutMix"
        )
        optim_layout.addRow("Augmentation:", self.train_augmentation)
        
        self.train_weights = QCheckBox("Class weights (imbalanced data)")
        optim_layout.addRow("", self.train_weights)
        
        optim_group.setLayout(optim_layout)
        right_column.addWidget(optim_group)
        
        # Regularization group
        reg_group = QGroupBox("Regularization")
        reg_layout = QFormLayout()
        
        # Freeze encoder row
        freeze_layout = QHBoxLayout()
        self.train_freeze_encoder = QCheckBox("Freeze encoder")
        self.train_freeze_encoder.setToolTip("Freeze encoder weights for first N epochs")
        freeze_layout.addWidget(self.train_freeze_encoder)
        self.train_freeze_epochs = QSpinBox()
        self.train_freeze_epochs.setRange(1, 50)
        self.train_freeze_epochs.setValue(5)
        self.train_freeze_epochs.setFixedWidth(60)
        freeze_layout.addWidget(self.train_freeze_epochs)
        freeze_layout.addWidget(QLabel("epochs"))
        freeze_layout.addStretch()
        reg_layout.addRow("", freeze_layout)
        
        # Early stopping row
        early_layout = QHBoxLayout()
        self.train_early_stopping = QCheckBox("Early stopping")
        self.train_early_stopping.setChecked(True)
        self.train_early_stopping.setToolTip("Stop training if metric doesn't improve")
        early_layout.addWidget(self.train_early_stopping)
        self.train_patience = QSpinBox()
        self.train_patience.setRange(5, 100)
        self.train_patience.setValue(15)
        self.train_patience.setFixedWidth(60)
        early_layout.addWidget(self.train_patience)
        early_layout.addWidget(QLabel("patience"))
        early_layout.addStretch()
        reg_layout.addRow("", early_layout)
        
        reg_group.setLayout(reg_layout)
        right_column.addWidget(reg_group)
        right_column.addStretch()
        
        columns_layout.addLayout(right_column)
        widget.setLayout(columns_layout)
        
        # Initialize loss function list based on current mode
        self._on_mode_changed(self.train_mode.currentText())
        # Initialize encoder visibility
        self._on_arch_changed(self.train_arch.currentText())
        
        return widget
    
    def _create_training_advanced_subtab(self):
        """Advanced parameters sub-tab - for experienced users"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "<i>These parameters have sensible defaults. Only change them if you know what you're doing.</i>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; padding: 5px;")
        layout.addWidget(info_label)
        
        columns_layout = QHBoxLayout()
        
        # --- LEFT: Model fine-tuning ---
        left_col = QVBoxLayout()
        
        model_adv_group = QGroupBox("Model Fine-tuning")
        model_adv_layout = QFormLayout()
        
        self.train_dropout = QDoubleSpinBox()
        self.train_dropout.setRange(0.0, 0.7)
        self.train_dropout.setSingleStep(0.05)
        self.train_dropout.setValue(0.3)
        self.train_dropout.setToolTip("Dropout rate for regularization (0.0 = no dropout)")
        model_adv_layout.addRow("Dropout Rate:", self.train_dropout)
        
        self.train_amp = QCheckBox("Mixed Precision (AMP)")
        self.train_amp.setToolTip("Use FP16 for faster training with lower VRAM (GPU only)")
        model_adv_layout.addRow("", self.train_amp)
        
        self.train_warmup = QSpinBox()
        self.train_warmup.setRange(0, 50)
        self.train_warmup.setValue(0)
        self.train_warmup.setToolTip("Number of warmup epochs (gradual LR increase)")
        model_adv_layout.addRow("Warmup Epochs:", self.train_warmup)
        
        model_adv_group.setLayout(model_adv_layout)
        left_col.addWidget(model_adv_group)
        
        left_col.addStretch()
        columns_layout.addLayout(left_col)
        
        # --- RIGHT: Loss function parameters ---
        right_col = QVBoxLayout()
        
        loss_params_group = QGroupBox("Loss Function Parameters")
        loss_params_layout = QFormLayout()
        
        self.train_focal_gamma = QDoubleSpinBox()
        self.train_focal_gamma.setRange(0.5, 5.0)
        self.train_focal_gamma.setSingleStep(0.25)
        self.train_focal_gamma.setValue(2.0)
        self.train_focal_gamma.setToolTip(
            "Focal gamma: higher = more focus on hard examples\n"
            "Default: 2.0. Range: 0.5-5.0"
        )
        loss_params_layout.addRow("Focal Gamma:", self.train_focal_gamma)
        
        self.train_focal_alpha = QDoubleSpinBox()
        self.train_focal_alpha.setRange(0.1, 0.9)
        self.train_focal_alpha.setSingleStep(0.05)
        self.train_focal_alpha.setValue(0.25)
        self.train_focal_alpha.setToolTip(
            "Focal alpha: weight for positive class\n"
            "Default: 0.25"
        )
        loss_params_layout.addRow("Focal Alpha:", self.train_focal_alpha)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #cccccc;")
        loss_params_layout.addRow(sep)
        
        self.train_tversky_alpha = QDoubleSpinBox()
        self.train_tversky_alpha.setRange(0.1, 0.9)
        self.train_tversky_alpha.setSingleStep(0.05)
        self.train_tversky_alpha.setValue(0.3)
        self.train_tversky_alpha.setToolTip(
            "Tversky alpha: false positive weight\n"
            "Lower alpha = more false positives allowed (higher recall)\n"
            "Default: 0.3"
        )
        loss_params_layout.addRow("Tversky Alpha (FP):", self.train_tversky_alpha)
        
        self.train_tversky_beta = QDoubleSpinBox()
        self.train_tversky_beta.setRange(0.1, 0.9)
        self.train_tversky_beta.setSingleStep(0.05)
        self.train_tversky_beta.setValue(0.7)
        self.train_tversky_beta.setToolTip(
            "Tversky beta: false negative weight\n"
            "Higher beta = penalize missed detections more (higher recall)\n"
            "Default: 0.7"
        )
        loss_params_layout.addRow("Tversky Beta (FN):", self.train_tversky_beta)
        
        loss_params_group.setLayout(loss_params_layout)
        right_col.addWidget(loss_params_group)
        
        right_col.addStretch()
        columns_layout.addLayout(right_col)
        
        layout.addLayout(columns_layout)
        
        # --- K-Fold Cross-Validation section (full width) ---
        kfold_group = QGroupBox("K-Fold Cross-Validation")
        kfold_layout = QFormLayout()
        
        kfold_enable_layout = QHBoxLayout()
        self.train_use_kfold = QCheckBox("Enable K-Fold Cross-Validation")
        self.train_use_kfold.setToolTip(
            "Use K-Fold CV instead of fixed train/val/test split.\n"
            "All patches (train+val+test) are pooled and split into K folds.\n"
            "Provides robust metrics with confidence intervals."
        )
        kfold_enable_layout.addWidget(self.train_use_kfold)
        
        self.train_n_splits = QSpinBox()
        self.train_n_splits.setRange(2, 20)
        self.train_n_splits.setValue(5)
        self.train_n_splits.setFixedWidth(60)
        self.train_n_splits.setEnabled(False)
        kfold_enable_layout.addWidget(QLabel("Folds:"))
        kfold_enable_layout.addWidget(self.train_n_splits)
        kfold_enable_layout.addStretch()
        kfold_layout.addRow("", kfold_enable_layout)
        
        kfold_info = QLabel(
            "<i>K-Fold trains K models, each validated on a different fold. "
            "Results include mean/std and 95% confidence intervals.</i>"
        )
        kfold_info.setWordWrap(True)
        kfold_info.setStyleSheet("color: gray; font-size: 10px;")
        kfold_layout.addRow("", kfold_info)
        
        self.train_use_kfold.toggled.connect(self.train_n_splits.setEnabled)
        
        kfold_group.setLayout(kfold_layout)
        layout.addWidget(kfold_group)
        
        widget.setLayout(layout)
        return widget
    
    # =====================================================================
    # PREDICTION TAB (v3 - with encoder, batch, gaussian blending)
    # =====================================================================
    
    def _create_prediction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Input
        input_group = QGroupBox("Model and Input")
        input_layout = QFormLayout()
        
        self.pred_model = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.pred_model, "PyTorch Model (*.pth)"))
        h = QHBoxLayout()
        h.addWidget(self.pred_model)
        h.addWidget(btn)
        input_layout.addRow("Model (.pth):", h)
        
        self.pred_input = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.pred_input, "GeoTIFF (*.tif *.tiff)"))
        h = QHBoxLayout()
        h.addWidget(self.pred_input)
        h.addWidget(btn)
        input_layout.addRow("Input Image:", h)
        
        self.pred_output = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_save(self.pred_output, "GeoTIFF (*.tif)"))
        h = QHBoxLayout()
        h.addWidget(self.pred_output)
        h.addWidget(btn)
        input_layout.addRow("Output File:", h)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Two columns for parameters
        params_columns = QHBoxLayout()
        
        # Left: Basic prediction params
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.pred_patch = QSpinBox()
        self.pred_patch.setRange(64, 2048)
        self.pred_patch.setSingleStep(32)
        self.pred_patch.setValue(512)
        self.pred_patch.setToolTip("Patch size for inference (larger = faster but more VRAM)")
        params_layout.addRow("Patch Size:", self.pred_patch)
        
        self.pred_overlap = QSpinBox()
        self.pred_overlap.setRange(16, 512)
        self.pred_overlap.setSingleStep(16)
        self.pred_overlap.setValue(128)
        self.pred_overlap.setToolTip("Overlap between patches for smoother results")
        params_layout.addRow("Overlap:", self.pred_overlap)
        
        self.pred_threshold = QSlider(Qt.Horizontal)
        self.pred_threshold.setRange(0, 100)
        self.pred_threshold.setValue(50)
        self.pred_threshold_label = QLabel("0.50")
        self.pred_threshold.valueChanged.connect(
            lambda v: self.pred_threshold_label.setText(f"{v/100:.2f}"))
        h = QHBoxLayout()
        h.addWidget(self.pred_threshold)
        h.addWidget(self.pred_threshold_label)
        params_layout.addRow("Threshold:", h)
        
        self.pred_device = QComboBox()
        self.pred_device.addItems(['cuda', 'cpu'])
        params_layout.addRow("Device:", self.pred_device)
        
        params_group.setLayout(params_layout)
        params_columns.addWidget(params_group)
        
        # Right: Model overrides + options
        model_group = QGroupBox("Model Overrides (optional)")
        model_layout = QFormLayout()
        
        self.pred_encoder = QComboBox()
        self.pred_encoder.addItems(['(auto-detect)'] + self.STANDARD_ENCODERS + self.CONVNEXT_ENCODERS)
        self.pred_encoder.setToolTip(
            "Override encoder backbone for prediction.\n"
            "Leave as (auto-detect) to read from checkpoint metadata.\n"
            "Only needed for old checkpoints without metadata."
        )
        model_layout.addRow("Encoder:", self.pred_encoder)
        
        self.pred_dropout = QDoubleSpinBox()
        self.pred_dropout.setRange(0.0, 0.7)
        self.pred_dropout.setSingleStep(0.05)
        self.pred_dropout.setValue(0.3)
        self.pred_dropout.setToolTip("Dropout rate (must match training)")
        model_layout.addRow("Dropout Rate:", self.pred_dropout)
        
        self.pred_batch = QSpinBox()
        self.pred_batch.setRange(1, 32)
        self.pred_batch.setValue(1)
        self.pred_batch.setToolTip("Inference batch size (higher = faster on GPU)")
        model_layout.addRow("Batch Size:", self.pred_batch)
        
        self.pred_gaussian = QCheckBox("Gaussian blending")
        self.pred_gaussian.setChecked(True)
        self.pred_gaussian.setToolTip("Eliminates grid artifacts at patch boundaries")
        model_layout.addRow("", self.pred_gaussian)
        
        self.pred_confidence = QCheckBox("Save confidence map")
        model_layout.addRow("", self.pred_confidence)
        
        self.pred_add_qgis = QCheckBox("Add result to QGIS")
        self.pred_add_qgis.setChecked(True)
        model_layout.addRow("", self.pred_add_qgis)
        
        model_group.setLayout(model_layout)
        params_columns.addWidget(model_group)
        
        layout.addLayout(params_columns)
        
        # Run button
        self.pred_run = QPushButton("Start Prediction")
        self.pred_run.setMinimumHeight(40)
        self.pred_run.clicked.connect(self.run_prediction)
        layout.addWidget(self.pred_run)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    # =====================================================================
    # UI CALLBACKS
    # =====================================================================
    
    def _on_mode_changed(self, mode):
        """Update UI when mode changes"""
        if mode == 'binary':
            self.train_classes.setValue(2)
            self.train_classes.setEnabled(False)
            losses = self.LOSSES_BINARY
        else:
            self.train_classes.setEnabled(True)
            if self.train_classes.value() == 2:
                self.train_classes.setValue(6)
            losses = self.LOSSES_MULTICLASS
        
        # Update loss function combo
        current_loss = self.train_loss_fn.currentText()
        self.train_loss_fn.clear()
        self.train_loss_fn.addItems(losses)
        # Try to keep similar loss
        if current_loss in losses:
            self.train_loss_fn.setCurrentText(current_loss)
        else:
            self.train_loss_fn.setCurrentIndex(0)
    
    def _on_arch_changed(self, arch):
        """Update encoder combo based on selected architecture"""
        if arch in self.MODELS_WITHOUT_ENCODER:
            self.train_encoder.setEnabled(False)
            self.train_encoder_info.setText("This architecture has a built-in encoder")
        else:
            self.train_encoder.setEnabled(True)
            
            # Save current selection
            current = self.train_encoder.currentText()
            self.train_encoder.blockSignals(True)
            self.train_encoder.clear()
            
            if arch == 'unet':
                # UNet supports all encoders INCLUDING ConvNeXt
                self.train_encoder.addItems(self.STANDARD_ENCODERS + self.CONVNEXT_ENCODERS)
                self.train_encoder_info.setText("UNet supports all encoders including ConvNeXt (needs timm)")
            else:
                # Other SMP models: standard encoders only (no ConvNeXt)
                self.train_encoder.addItems(self.STANDARD_ENCODERS)
                self.train_encoder_info.setText("")
                # If current was a ConvNeXt, reset to resnet34
                if current in self.CONVNEXT_ENCODERS:
                    current = 'resnet34'
            
            # Restore selection if still valid
            idx = self.train_encoder.findText(current)
            if idx >= 0:
                self.train_encoder.setCurrentIndex(idx)
            else:
                self.train_encoder.setCurrentText('resnet34')
            self.train_encoder.blockSignals(False)
    
    def _browse_file(self, line_edit, filter_str):
        filename, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if filename:
            line_edit.setText(filename)
    
    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)
    
    def _browse_save(self, line_edit, filter_str):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", filter_str)
        if filename:
            line_edit.setText(filename)
    
    def _update_test_ratio_display(self):
        """Update the ratio display label"""
        train = self.extract_train_ratio.value()
        val = self.extract_val_ratio.value()
        test = self.extract_test_ratio.value()
        total = train + val + test
        
        if abs(total - 1.0) < 0.01:
            self.extract_ratio_info.setText(f"Total: {total:.2f}")
            self.extract_ratio_info.setStyleSheet("color: green;")
        else:
            self.extract_ratio_info.setText(f"Total: {total:.2f} (should be 1.0)")
            self.extract_ratio_info.setStyleSheet("color: orange;")
    
    def _detach_log_window(self):
        """Open log output in a separate resizable window."""
        if self._detached_log and self._detached_log.isVisible():
            self._detached_log.raise_()
            return
        self._detached_log = QDialog(self)
        self._detached_log.setWindowTitle("SemanticSeg4EO — Log Output")
        self._detached_log.setMinimumSize(700, 400)
        self._detached_log.resize(900, 500)
        lay = QVBoxLayout()
        self._detached_log_text = QTextEdit()
        self._detached_log_text.setReadOnly(True)
        self._detached_log_text.setFont(QFont("Courier", 9))
        self._detached_log_text.setPlainText(self.log_text.toPlainText())
        lay.addWidget(self._detached_log_text)
        self._detached_log.setLayout(lay)
        self._detached_log.show()
    
    # =====================================================================
    # LOGGING & STATUS
    # =====================================================================
    
    def log(self, message):
        # Filter out GDAL/rasterio Unicode spam
        if any(s in message for s in [
            'UnicodeDecodeError', 'rasterio._env', 'rasterio.env',
            'pyogrio.core', 'log_error', '_register_drivers',
            "can't decode byte 0xe", "invalid continuation byte",
            'Exception ignored in'
        ]):
            return
        self.log_text.append(message)
        self.log_text.moveCursor(QTextCursor.End)
        # Forward to detached log window if open
        if self._detached_log and self._detached_log.isVisible():
            self._detached_log_text.append(message)
            self._detached_log_text.moveCursor(QTextCursor.End)
    
    def update_environment_status(self):
        """Update the environment status display"""
        config = PluginConfig.load()
        python_path = PluginConfig.get_python_executable()
        
        if python_path and os.path.exists(python_path):
            env_type = config.get('env_type', 'system').upper()
            # Shorten path for display if too long
            display_path = python_path
            if len(display_path) > 80:
                display_path = '…' + display_path[-75:]
            self.env_status_label.setText(f"<b>{env_type}</b>: {display_path}")
            self.env_status_label.setStyleSheet("color: green;")
            self.env_status_icon.setText("●")
            self.env_status_icon.setStyleSheet("color: #27ae60; font-size: 14px;")
            self._set_buttons_enabled(True)
        else:
            self.env_status_label.setText(
                "No Python environment configured – click <b>Browse python…</b> or <b>Configure Environment</b>"
            )
            self.env_status_label.setStyleSheet("color: #c0392b;")
            self.env_status_icon.setText("●")
            self.env_status_icon.setStyleSheet("color: #c0392b; font-size: 14px;")
            self._set_buttons_enabled(False)
    
    def _set_buttons_enabled(self, enabled):
        self.extract_run.setEnabled(enabled)
        self.train_run.setEnabled(enabled)
        self.pred_run.setEnabled(enabled)
    
    def open_environment_wizard(self):
        dialog = EnvironmentConfigDialog(self)
        if dialog.exec_():
            self.update_environment_status()
            self.log("Environment configured successfully")
    
    def _quick_browse_python(self):
        """One-click shortcut: browse for python executable and save immediately."""
        if sys.platform == 'win32':
            filter_str = "Python executable (python.exe);;All files (*)"
        else:
            filter_str = "All files (*)"
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select python executable", str(Path.home()), filter_str
        )
        if not filename:
            return
        
        if not os.path.isfile(filename):
            QMessageBox.warning(self, "Error", f"File not found:\n{filename}")
            return
        
        # Detect env type from path
        p = filename.lower().replace('\\', '/')
        if '/envs/' in p and any(k in p for k in ['conda', 'miniconda', 'anaconda',
                                                    'miniforge', 'mambaforge']):
            env_type = 'conda'
        elif (Path(filename).resolve().parent.parent / 'pyvenv.cfg').is_file():
            env_type = 'venv'
        else:
            env_type = 'system'
        
        config = PluginConfig.load()
        config['python_path'] = filename
        config['env_type'] = env_type
        config['dependencies_ok'] = True
        PluginConfig.save(config)
        
        self.update_environment_status()
        self.log(f"Python set to: {filename}")
    
    def _check_environment(self):
        """Check if environment is configured"""
        python_path = PluginConfig.get_python_executable()
        if not python_path or not os.path.exists(python_path):
            QMessageBox.warning(
                self, "Environment Not Configured",
                "Please configure the Python environment first.\n\n"
                "Click 'Configure Environment' to set up."
            )
            return None
        return python_path
    
    def _start_process(self, script_name, params):
        """Start an external process"""
        python_path = self._check_environment()
        if not python_path:
            return
        
        # Create temp files for communication
        self.temp_dir = tempfile.mkdtemp(prefix='semanticseg4eo_')
        params_file = os.path.join(self.temp_dir, 'params.json')
        progress_file = os.path.join(self.temp_dir, 'progress.json')
        
        # Write parameters
        with open(params_file, 'w') as f:
            json.dump(params, f)
        
        # Get script path
        script_path = PLUGIN_DIR / 'scripts' / f'{script_name}.py'
        
        self.log(f"\n{'='*50}")
        self.log(f"Starting {script_name}...")
        self.log(f"Python: {python_path}")
        self.log(f"{'='*50}\n")
        
        # Disable buttons
        self._set_buttons_enabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start process
        self.runner = ProcessRunner(python_path, script_path, params_file, progress_file)
        self.runner.output_received.connect(self.log)
        self.runner.progress_updated.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.start()
    
    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        if message:
            self.log(f"[{value}%] {message}")
    
    def _on_finished(self, success, message):
        # Re-enable buttons
        self._set_buttons_enabled(True)
        self.btn_cancel.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            self.log(f"\n{message}")
            
            # Add prediction to QGIS if requested
            if (self.tabs.currentIndex() == 2 and  # Prediction tab
                self.pred_add_qgis.isChecked() and
                os.path.exists(self.pred_output.text())):
                self._add_layer_to_qgis(self.pred_output.text())
        else:
            self.progress_bar.setValue(0)
            self.log(f"\nError: {message}")
            QMessageBox.critical(self, "Error", message)
        
        # Cleanup thread properly
        if self.runner:
            self.runner.wait(1000)  # Wait up to 1 second for thread to finish
            self.runner.deleteLater()  # Schedule for deletion
            self.runner = None
        
        # Cleanup temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
            self.temp_dir = None
        
        self.log("\n--- Ready for next operation ---\n")
    
    def _add_layer_to_qgis(self, filepath):
        """Add a raster layer to QGIS"""
        name = os.path.splitext(os.path.basename(filepath))[0]
        layer = QgsRasterLayer(filepath, name)
        if layer.isValid():
            QgsProject.instance().addMapLayer(layer)
            self.log(f"Added layer to QGIS: {name}")
    
    def cancel_operation(self):
        if self.runner:
            self.runner.cancel()
            self.log("Cancelling...")
    
    # =====================================================================
    # RUN OPERATIONS
    # =====================================================================
    
    def run_extraction(self):
        is_batch = self.extract_mode_batch.isChecked()

        params = {
            'mode': 'batch' if is_batch else 'single',
            'output_dir': self.extract_output.text(),
            'patch_size': int(self.extract_patch_size.currentText()),
            'image_channels': self.extract_channels.value(),
            'train_ratio': self.extract_train_ratio.value(),
            'val_ratio': self.extract_val_ratio.value(),
            'test_ratio': self.extract_test_ratio.value(),
            'interpolation': self.extract_interpolation.currentText(),
            'compression': self.extract_compression.currentText()
                if self.extract_compression.currentText() != 'none' else None,
            'validate_crs': self.extract_validate_crs.isChecked(),
            'output_dtype': self.extract_dtype.currentText(),
            'output_dtype': self.extract_output_dtype.currentText(),
        }

        if is_batch:
            params['data_dir'] = self.extract_data_dir.text()
            params['grid_path'] = self.extract_batch_grid.text()
            params['use_per_image_grid'] = self.extract_per_image_grid.isChecked()
            params['recursive'] = self.extract_recursive.isChecked()

            if self.extract_custom_patterns.isChecked():
                params['image_pattern'] = self.extract_img_pattern.text()
                params['label_pattern'] = self.extract_lbl_pattern.text()

            # Validate batch fields
            if not params['data_dir']:
                QMessageBox.warning(self, "Error", "Please select a data directory")
                return
            if not params['grid_path'] and not params['use_per_image_grid']:
                QMessageBox.warning(self, "Error",
                    "Please select a shared grid, or enable per-image grids")
                return
        else:
            params['image_path'] = self.extract_image.text()
            params['label_path'] = self.extract_label.text()
            params['grid_path'] = self.extract_grid.text()

            # Validate single fields
            for key in ['image_path', 'label_path', 'grid_path']:
                if not params[key]:
                    QMessageBox.warning(self, "Error", "Please fill in all input fields")
                    return

        if not params['output_dir']:
            QMessageBox.warning(self, "Error", "Please select an output directory")
            return

        # Validate ratios
        total = params['train_ratio'] + params['val_ratio'] + params['test_ratio']
        if abs(total - 1.0) > 0.05:
            QMessageBox.warning(self, "Error",
                f"Ratios must sum to 1.0 (currently {total:.2f})")
            return

        self._start_process('patch_extraction', params)

    
    def run_training(self):
        try:
            lr = float(self.train_lr.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid learning rate")
            return
        
        params = {
            # === Basic parameters ===
            'dataset_root': self.train_dataset.text(),
            'mode': self.train_mode.currentText(),
            'num_classes': self.train_classes.value(),
            'model_name': self.train_arch.currentText(),
            'encoder_name': self.train_encoder.currentText(),
            'pretrained': self.train_pretrained.isChecked(),
            'in_channels': self.train_channels.value(),
            'epochs': self.train_epochs.value(),
            'batch_size': self.train_batch.value(),
            'learning_rate': lr,
            'device': self.train_device.currentText(),
            'save_dir': self.train_save.text(),
            # === Optimization ===
            'scheduler': self.train_scheduler.currentText(),
            'loss_function': self.train_loss_fn.currentText(),
            'augmentation_level': self.train_augmentation.currentText(),
            'use_class_weights': self.train_weights.isChecked(),
            # === Regularization ===
            'freeze_encoder': self.train_freeze_encoder.isChecked(),
            'freeze_epochs': self.train_freeze_epochs.value(),
            'early_stopping': self.train_early_stopping.isChecked(),
            'patience': self.train_patience.value(),
            # === Advanced parameters ===
            'dropout_rate': self.train_dropout.value(),
            'use_amp': self.train_amp.isChecked(),
            'warmup_epochs': self.train_warmup.value(),
            'focal_gamma': self.train_focal_gamma.value(),
            'focal_alpha': self.train_focal_alpha.value(),
            'tversky_alpha': self.train_tversky_alpha.value(),
            'tversky_beta': self.train_tversky_beta.value(),
            # === K-Fold Cross-Validation ===
            'use_kfold': self.train_use_kfold.isChecked(),
            'n_splits': self.train_n_splits.value(),
        }
        
        if not params['dataset_root']:
            QMessageBox.warning(self, "Error", "Please select a dataset directory")
            return
        
        self._start_process('model_training', params)
    
    def run_prediction(self):
        # Build encoder override
        encoder_override = self.pred_encoder.currentText()
        if encoder_override == '(auto-detect)':
            encoder_override = None
        
        params = {
            'model_path': self.pred_model.text(),
            'input_path': self.pred_input.text(),
            'output_path': self.pred_output.text(),
            'patch_size': self.pred_patch.value(),
            'overlap': self.pred_overlap.value(),
            'threshold': self.pred_threshold.value() / 100.0,
            'device': self.pred_device.currentText(),
            'save_confidence': self.pred_confidence.isChecked(),
            # === New v3 parameters ===
            'batch_size': self.pred_batch.value(),
            'gaussian_blending': self.pred_gaussian.isChecked(),
            'dropout_rate': self.pred_dropout.value(),
        }
        
        # Only add encoder_name if explicitly set
        if encoder_override:
            params['encoder_name'] = encoder_override
        
        for key in ['model_path', 'input_path', 'output_path']:
            if not params[key]:
                QMessageBox.warning(self, "Error", "Please fill in all fields")
                return
        
        self._start_process('prediction', params)
    
    def show_help(self):
        help_text = """
<h3>SemanticSeg4EO v3 - Help</h3>

<h4>Environment Setup</h4>
<p>This plugin uses an <b>external Python environment</b> (Conda or venv) 
for processing. Click "Configure Environment" to set it up.</p>

<h4>Patch Extraction</h4>
<p>Extract training patches from large satellite images using a grid shapefile.</p>

<h4>Model Training</h4>
<p>Train deep learning segmentation models. <b>20+ architectures</b> available:</p>
<ul>
<li><b>SMP</b>: U-Net, U-Net++, DeepLabV3+, MAnet, FPN, etc.</li>
<li><b>Modern</b>: SegFormer, HRNet, SwinUNet, UNetFormer</li>
<li><b>ConvNeXt</b>: UNet with ConvNeXt encoder (needs timm)</li>
</ul>

<h4>Advanced Parameters</h4>
<ul>
<li><b>Augmentation levels</b>: none, basic, advanced, aggressive, extreme</li>
<li><b>Loss functions</b>: 15+ options (Dice, Focal, Tversky, combos...)</li>
<li><b>AMP</b>: Mixed precision for faster training</li>
<li><b>Dropout</b>: Configurable regularization</li>
</ul>

<h4>Prediction</h4>
<p>Apply trained models to new satellite images.</p>
<ul>
<li><b>Gaussian blending</b>: Eliminates grid artifacts</li>
<li><b>Encoder override</b>: Specify encoder for old checkpoints</li>
<li><b>Batch inference</b>: Process multiple patches in parallel</li>
</ul>

<h4>Tips</h4>
<ul>
<li>Use GPU (cuda) for faster training/prediction</li>
<li>Start with 'basic' augmentation, increase if needed</li>
<li>Use class weights for imbalanced datasets</li>
<li>Install transformers + timm for modern architectures</li>
</ul>
        """
        QMessageBox.information(self, "Help", help_text)
    
    def closeEvent(self, event):
        if self.runner and self.runner.isRunning():
            reply = QMessageBox.question(
                self, "Confirm", "A process is running. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.runner.cancel()
                self.runner.wait()
            else:
                event.ignore()
                return
        event.accept()


class SemanticSeg4EOPlugin:
    """Main plugin class"""
    
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None
    
    def initGui(self):
        icon_path = str(PLUGIN_DIR / 'resources' / 'icon.png')
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        
        self.action = QAction(icon, "SemanticSeg4EO", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&SemanticSeg4EO", self.action)
    
    def unload(self):
        self.iface.removePluginRasterMenu("&SemanticSeg4EO", self.action)
        self.iface.removeToolBarIcon(self.action)
        if self.dialog:
            self.dialog.close()
    
    def run(self):
        if not self.dialog:
            self.dialog = SemanticSeg4EODialog(self.iface, self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()

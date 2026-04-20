"""
SemanticSeg4EO - QGIS Plugin for Semantic Segmentation
=======================================================

This plugin uses an EXTERNAL Python environment for heavy processing.
No PyTorch or deep learning libraries are loaded into QGIS.

Architecture:
- QGIS side: Only PyQt5 (native) for GUI
- External: Conda/venv with PyTorch for processing
- Communication: JSON files + subprocess
"""

def classFactory(iface):
    from .main_plugin import SemanticSeg4EOPlugin
    return SemanticSeg4EOPlugin(iface)

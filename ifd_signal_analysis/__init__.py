#!/usr/bin/env python3
"""
IFD Signal Analysis Utility - A comprehensive PyQt6 application for visualizing
and analyzing oscilloscope waveform data from Siglent Binary Format V4.0 files.

This package provides:
- Interactive matplotlib plots with zoom, pan, and navigation
- Support for single files or entire directories
- Channel management with visibility toggles
- Background loading and plotting for responsive UI
- Extensible architecture for future enhancements

Features:
- Open single files or entire directories
- Overlay multiple waveforms or display separately
- Interactive matplotlib plots with zoom and pan
- Channel visibility management
- Export plots to various formats
- Responsive UI with background operations

For internal development use only.

Author: Assistant
Version: 1.0.0
Dependencies: PyQt6, matplotlib, numpy, scipy
"""

from .utils.constants import APP_NAME, APP_VERSION

__version__ = APP_VERSION
__title__ = APP_NAME

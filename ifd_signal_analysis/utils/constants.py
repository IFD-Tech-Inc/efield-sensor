#!/usr/bin/env python3
"""
Shared constants and configurations for IFD Signal Analysis Utility.

This module contains application-wide constants, default settings, and shared
type definitions to prevent duplication across modules.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Application metadata
APP_NAME = 'IFD Signal Analysis Utility'
APP_VERSION = '1.0.0'
APP_ORGANIZATION = 'IFDSignalAnalysisUtility'
APP_DOMAIN = 'ifd.internal'

# Matplotlib backend configuration
MATPLOTLIB_BACKEND = 'QtAgg'

# Default UI settings
DEFAULT_WINDOW_SIZE = (1400, 900)
DEFAULT_WINDOW_POSITION = (100, 100)
DEFAULT_OVERLAY_MODE = True
DEFAULT_PLOT_FIGURE_SIZE = (12, 8)
DEFAULT_PLOT_DPI = 100

# Plot styling constants
PLOT_LINE_WIDTH = 1.0
PLOT_LINE_ALPHA = 0.8
PLOT_GRID_ALPHA = 0.3
PLOT_LEGEND_LOCATION = 'upper right'

# Channel list widget settings
CHANNEL_ITEM_MARGINS = (5, 2, 5, 2)
CHANNEL_FONT_FAMILY = "Consolas"
CHANNEL_FONT_SIZE = 9

# File information display settings
INFO_FONT_FAMILY = 'Consolas'
INFO_FONT_SIZE = 8
INFO_WIDGET_MAX_HEIGHT = 200

# Thread communication constants
LOAD_PROGRESS_STAGES = {
    'INIT': 10,
    'SCAN_DIR': 25,
    'READ_FILE': 25,
    'PARSE': 50,
    'PROCESS': 75,
    'FINALIZE': 90,
    'COMPLETE': 100
}

PLOT_PROGRESS_STAGES = {
    'START': 5,
    'PROCESS_MIN': 10,
    'PROCESS_MAX': 95,
    'FINALIZE': 95,
    'COMPLETE': 100
}

# File dialog settings
SIGLENT_FILE_FILTER = 'Siglent Binary Files (*.bin);;All Files (*)'
PLOT_SAVE_FILTER = 'PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)'
DEFAULT_PLOT_SAVE_NAME = 'waveform_plot.png'

# Status bar message durations (milliseconds)
STATUS_MESSAGE_SHORT = 2000
STATUS_MESSAGE_MEDIUM = 3000
STATUS_MESSAGE_LONG = 5000

# Progress dialog settings
PROGRESS_DIALOG_UPDATE_INTERVAL = 100  # milliseconds
SPLASH_SCREEN_DISPLAY_TIME = 1500  # milliseconds
SPLASH_SCREEN_STEPS = 15

# Default empty info text
DEFAULT_INFO_TEXT = """No data loaded.

Load a Siglent binary file (.bin) to see waveform information."""

# Color management for plotting
from matplotlib.colors import TABLEAU_COLORS
AVAILABLE_PLOT_COLORS = list(TABLEAU_COLORS.keys())

# Type aliases for better code readability
ChannelDataDict = Dict[str, Any]  # Channel name -> ChannelData mapping
ParserDict = Dict[str, Any]  # File key -> Parser mapping
PlotDataDict = Dict[str, Dict[str, Any]]  # Channel name -> plot data mapping
LoadResultDict = Dict[str, Any]  # Result from LoadWaveformThread
PlotResultDict = Dict[str, Any]  # Result from PlottingThread

# Error messages
ERROR_SIGLENT_PARSER_UNAVAILABLE = (
    "Siglent parser module is not available.\n"
    "Please ensure siglent_parser.py is in the same directory."
)

ERROR_PROGRESS_DIALOG_UNAVAILABLE = (
    "Warning: Progress dialog not available"
)

ERROR_SPLASH_SCREEN_UNAVAILABLE = (
    "Warning: Splash screen not available"
)

# Success messages
SUCCESS_LOAD_TEMPLATE = "Successfully loaded {count} channel(s) from {source}"
SUCCESS_PLOT_TEMPLATE = "Successfully plotted {count} channel(s)"
SUCCESS_SAVE_TEMPLATE = "Plot saved to {filename}"
SUCCESS_CLEAR_MESSAGE = "All data cleared"
SUCCESS_ZOOM_FIT_MESSAGE = "Zoomed to fit all data"

# Warning messages
WARNING_NO_CHANNELS = "No valid channel data found in the specified location."
WARNING_NO_ENABLED_CHANNELS = "No enabled channels with valid data found"
WARNING_NO_DATA_TO_SAVE = "No data to save. Load waveform data first."

# Loading operation descriptions
LOADING_FILE_DESC = "Loading file"
LOADING_DIRECTORY_DESC = "Loading directory"
RENDERING_PLOTS_DESC = "Rendering waveform plots"

# Default plot title and axis labels
DEFAULT_PLOT_TITLE = 'IFD Signal Analysis - Load data to begin'
PLOT_XLABEL = 'Time (s)'
PLOT_YLABEL = 'Voltage (V)'
PLOT_TITLE_TEMPLATE = 'IFD Signal Analysis - {count} channel(s) displayed'

# Application state messages
STATUS_READY = "Ready"
STATUS_LOADING = "Loading waveform data..."
STATUS_PLOTTING = "Preparing plots..."
STATUS_CANCELLED = "cancelled by user"

# Splitter panel settings
LEFT_PANEL_MIN_WIDTH = 300
LEFT_PANEL_MAX_WIDTH = 500
DEFAULT_SPLITTER_SIZES = [300, 1100]  # 30% left, 70% right

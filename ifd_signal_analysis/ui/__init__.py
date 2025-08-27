#!/usr/bin/env python3
"""
User interface modules for IFD Signal Analysis Utility.

This package contains all UI components including the main window,
plot canvas, and channel management widgets.
"""

from .plot_canvas import PlotCanvas
from .channel_list import ChannelListWidget
from .main_window import IFDSignalAnalysisMainWindow

__all__ = ['IFDSignalAnalysisMainWindow', 'PlotCanvas', 'ChannelListWidget']

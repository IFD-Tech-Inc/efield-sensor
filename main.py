#!/usr/bin/env python3
"""
Siglent Waveform Viewer - PyQt6 Application

A comprehensive GUI application for visualizing and analyzing oscilloscope waveform data
from Siglent Binary Format V4.0 files. Features include:

- Opening single files or entire directories
- Overlay multiple waveforms or display separately  
- Interactive matplotlib plots with zoom, pan, and navigation
- Channel management with visibility toggles
- Extensible architecture for future math operations

Author: Assistant
Version: 1.0.0
Dependencies: PyQt6, matplotlib, numpy, scipy
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for matplotlib

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QStatusBar, QMenuBar,
    QToolBar, QListWidget, QListWidgetItem, QCheckBox, QLabel,
    QPushButton, QComboBox, QSlider, QGroupBox, QTextEdit,
    QTabWidget, QScrollArea, QFrame, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont, QColor, QPalette

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.patches as patches

# Import our Siglent parser
try:
    from siglent_parser import SiglentBinaryParser, ChannelData
    SIGLENT_PARSER_AVAILABLE = True
except ImportError as e:
    SIGLENT_PARSER_AVAILABLE = False
    print(f"Warning: Siglent parser not available: {e}")


class LoadWaveformThread(QThread):
    """
    Background thread for loading waveform data to prevent GUI freezing.
    Emits signals with progress updates and results.
    """
    progress = pyqtSignal(str)  # Status message
    finished = pyqtSignal(dict)  # Channel data dictionary
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, file_path: str, is_directory: bool = False):
        super().__init__()
        self.file_path = file_path
        self.is_directory = is_directory
        
    def run(self):
        """Execute the waveform loading in the background."""
        try:
            if not SIGLENT_PARSER_AVAILABLE:
                self.error.emit("Siglent parser module not available")
                return
                
            parser = SiglentBinaryParser()
            
            if self.is_directory:
                self.progress.emit(f"Scanning directory: {self.file_path}")
                channels = parser.parse_directory(self.file_path)
                self.progress.emit(f"Loaded {len(channels)} channels from directory")
            else:
                self.progress.emit(f"Loading file: {Path(self.file_path).name}")
                channels = parser.parse_file(self.file_path)
                self.progress.emit(f"Loaded {len(channels)} channels from file")
            
            # Add metadata to channels
            result = {
                'channels': channels,
                'parser': parser,
                'source_path': self.file_path,
                'is_directory': self.is_directory
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = f"Failed to load waveform data: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class PlotCanvas(FigureCanvas):
    """
    Matplotlib canvas widget integrated with PyQt6.
    Handles all waveform plotting and visualization.
    """
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('white')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot styling
        plt.style.use('default')
        self.colors = list(TABLEAU_COLORS.keys())
        self.color_index = 0
        
        # Create initial empty plot
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Waveform Viewer - Load data to begin')
        
        # Store plot data for management
        self.plot_data = {}  # Store channel name -> plot line mapping
        self.channel_visibility = {}  # Store channel visibility state
        
        self.draw()
        
    def clear_all_plots(self):
        """Clear all plotted data and reset the canvas."""
        self.ax.clear()
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Waveform Viewer - Load data to begin')
        self.plot_data.clear()
        self.channel_visibility.clear()
        self.color_index = 0
        self.draw()
        
    def get_next_color(self):
        """Get the next color in the cycle."""
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
        
    def plot_channel(self, channel_name: str, channel_data: ChannelData, 
                    parser_header, overlay_mode: bool = True, visible: bool = True):
        """
        Plot a single channel's waveform data.
        
        Args:
            channel_name: Name of the channel (e.g., 'C1', 'C2')
            channel_data: ChannelData object containing waveform data
            parser_header: SiglentBinaryHeader for time axis calculation
            overlay_mode: If True, overlay on existing plot; if False, create subplot
            visible: Whether the channel should be visible initially
        """
        if len(channel_data.voltage_data) == 0:
            print(f"Warning: No data available for channel {channel_name}")
            return
            
        # Calculate time array
        time_array = channel_data.get_time_array(
            parser_header.time_div,
            parser_header.time_delay, 
            parser_header.sample_rate,
            parser_header.hori_div_num
        )
        
        if len(time_array) == 0:
            print(f"Warning: Could not calculate time array for channel {channel_name}")
            return
            
        # Get color for this channel
        color = self.get_next_color()
        
        # Plot the waveform
        if overlay_mode:
            line, = self.ax.plot(time_array, channel_data.voltage_data, 
                               label=channel_name, color=color, 
                               linewidth=1.0, alpha=0.8)
            line.set_visible(visible)
        else:
            # TODO: Implement subplot mode for separate channel display
            # For now, fall back to overlay mode
            line, = self.ax.plot(time_array, channel_data.voltage_data, 
                               label=channel_name, color=color,
                               linewidth=1.0, alpha=0.8)
            line.set_visible(visible)
            
        # Store the plot line for visibility management
        self.plot_data[channel_name] = {
            'line': line,
            'data': channel_data,
            'time': time_array,
            'color': color
        }
        self.channel_visibility[channel_name] = visible
        
        # Update plot appearance
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (V)')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Auto-scale the view
        self.ax.relim()
        self.ax.autoscale()
        
        # Update title with channel info
        num_visible = sum(self.channel_visibility.values())
        self.ax.set_title(f'Waveform Viewer - {num_visible} channel(s) displayed')
        
        self.draw()
        
    def set_channel_visibility(self, channel_name: str, visible: bool):
        """Toggle visibility of a specific channel."""
        if channel_name in self.plot_data:
            self.plot_data[channel_name]['line'].set_visible(visible)
            self.channel_visibility[channel_name] = visible
            
            # Update title
            num_visible = sum(self.channel_visibility.values())
            self.ax.set_title(f'Waveform Viewer - {num_visible} channel(s) displayed')
            
            # Update legend to only show visible channels
            handles = []
            labels = []
            for ch_name, plot_info in self.plot_data.items():
                if self.channel_visibility.get(ch_name, False):
                    handles.append(plot_info['line'])
                    labels.append(ch_name)
            
            if handles:
                self.ax.legend(handles, labels)
            else:
                self.ax.legend().set_visible(False)
                
            self.draw()
            
    def remove_channel(self, channel_name: str):
        """Remove a channel from the plot entirely."""
        if channel_name in self.plot_data:
            # Remove the line from the plot
            self.plot_data[channel_name]['line'].remove()
            
            # Remove from our tracking dictionaries
            del self.plot_data[channel_name]
            if channel_name in self.channel_visibility:
                del self.channel_visibility[channel_name]
            
            # Update legend
            if self.plot_data:
                handles = []
                labels = []
                for ch_name, plot_info in self.plot_data.items():
                    if self.channel_visibility.get(ch_name, False):
                        handles.append(plot_info['line'])
                        labels.append(ch_name)
                
                if handles:
                    self.ax.legend(handles, labels)
                else:
                    self.ax.legend().set_visible(False)
            else:
                # No channels left, reset to empty state
                self.clear_all_plots()
                return
                
            # Update title
            num_visible = sum(self.channel_visibility.values())
            self.ax.set_title(f'Waveform Viewer - {num_visible} channel(s) displayed')
            
            self.draw()


class ChannelListWidget(QListWidget):
    """
    Custom QListWidget for managing channel visibility and properties.
    Each item contains a checkbox for visibility control.
    """
    
    channel_visibility_changed = pyqtSignal(str, bool)  # channel_name, visible
    channel_removed = pyqtSignal(str)  # channel_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def add_channel(self, channel_name: str, channel_data: ChannelData, visible: bool = True):
        """Add a new channel to the list with checkbox for visibility control."""
        
        # Create the list item
        item = QListWidgetItem(self)
        
        # Create a custom widget for the item
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Checkbox for visibility
        checkbox = QCheckBox()
        checkbox.setChecked(visible)
        checkbox.stateChanged.connect(
            lambda state, ch=channel_name: self.channel_visibility_changed.emit(
                ch, state == Qt.CheckState.Checked.value
            )
        )
        
        # Channel information label
        info_text = f"{channel_name}"
        if hasattr(channel_data, 'volt_div_val'):
            volt_div = channel_data.volt_div_val.get_scaled_value()
            unit = channel_data.volt_div_val.get_unit_string()
            info_text += f" ({volt_div:.3f}{unit}/div)"
            
        if len(channel_data.voltage_data) > 0:
            info_text += f" [{len(channel_data.voltage_data)} samples]"
            
        label = QLabel(info_text)
        label.setFont(QFont("Consolas", 9))
        
        # Add widgets to layout
        layout.addWidget(checkbox)
        layout.addWidget(label)
        layout.addStretch()
        
        # Set the custom widget as the item widget
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        
        # Store channel name in item data for later reference
        item.setData(Qt.ItemDataRole.UserRole, channel_name)
        
    def clear_all_channels(self):
        """Remove all channels from the list."""
        self.clear()
        
    def show_context_menu(self, position):
        """Show right-click context menu for channel operations."""
        item = self.itemAt(position)
        if item is not None:
            channel_name = item.data(Qt.ItemDataRole.UserRole)
            
            from PyQt6.QtWidgets import QMenu
            menu = QMenu(self)
            
            remove_action = menu.addAction(f"Remove {channel_name}")
            remove_action.triggered.connect(lambda: self.remove_channel_item(channel_name))
            
            menu.exec(self.mapToGlobal(position))
            
    def remove_channel_item(self, channel_name: str):
        """Remove a specific channel from the list."""
        for i in range(self.count()):
            item = self.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == channel_name:
                self.takeItem(i)
                self.channel_removed.emit(channel_name)
                break


class WaveformViewerMainWindow(QMainWindow):
    """
    Main application window for the Siglent Waveform Viewer.
    
    Provides a comprehensive interface for loading, visualizing, and managing
    oscilloscope waveform data with support for multiple channels and files.
    """
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self.loaded_data = {}  # Store all loaded channel data
        self.parsers = {}  # Store parser instances for each loaded file
        self.settings = QSettings('SiglentWaveformViewer', 'Settings')
        
        # UI setup
        self.setWindowTitle('Siglent Waveform Viewer v1.0')
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize the UI components
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
        self.create_main_layout()
        self.create_keyboard_shortcuts()
        
        # Load application settings
        self.load_settings()
        
        # Show welcome message
        self.status_bar.showMessage('Ready - Load waveform data to begin analysis', 5000)
        
    def create_menu_bar(self):
        """Create the application menu bar with File, View, and Help menus."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('&File')
        
        open_file_action = QAction('&Open File...', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.setStatusTip('Open a single Siglent binary file')
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)
        
        open_dir_action = QAction('Open &Directory...', self)
        open_dir_action.setShortcut('Ctrl+D')
        open_dir_action.setStatusTip('Open all Siglent binary files in a directory')
        open_dir_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_dir_action)
        
        file_menu.addSeparator()
        
        clear_action = QAction('&Clear All', self)
        clear_action.setShortcut('Ctrl+N')
        clear_action.setStatusTip('Clear all loaded waveform data')
        clear_action.triggered.connect(self.clear_all_data)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menubar.addMenu('&View')
        
        self.overlay_action = QAction('&Overlay Mode', self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.setStatusTip('Display all channels on the same plot')
        view_menu.addAction(self.overlay_action)
        
        self.separate_action = QAction('&Separate Plots', self)
        self.separate_action.setCheckable(True)
        self.separate_action.setStatusTip('Display each channel in its own subplot')
        view_menu.addAction(self.separate_action)
        
        # Make view actions mutually exclusive
        self.overlay_action.triggered.connect(lambda: self.separate_action.setChecked(False))
        self.separate_action.triggered.connect(lambda: self.overlay_action.setChecked(False))
        
        view_menu.addSeparator()
        
        zoom_fit_action = QAction('Zoom to &Fit', self)
        zoom_fit_action.setShortcut('Ctrl+F')
        zoom_fit_action.setStatusTip('Auto-scale plot to fit all data')
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        view_menu.addAction(zoom_fit_action)
        
        # Help Menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About...', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    def create_tool_bar(self):
        """Create the main toolbar with commonly used actions."""
        toolbar = QToolBar()
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        
        # File operations
        open_file_btn = QPushButton('📁 Open File')
        open_file_btn.clicked.connect(self.open_file)
        toolbar.addWidget(open_file_btn)
        
        open_dir_btn = QPushButton('📂 Open Directory')  
        open_dir_btn.clicked.connect(self.open_directory)
        toolbar.addWidget(open_dir_btn)
        
        toolbar.addSeparator()
        
        clear_btn = QPushButton('🗑️ Clear All')
        clear_btn.clicked.connect(self.clear_all_data)
        toolbar.addWidget(clear_btn)
        
        self.addToolBar(toolbar)
        
    def create_status_bar(self):
        """Create the status bar for displaying application status."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.channel_count_label = QLabel('No data loaded')
        self.status_bar.addPermanentWidget(self.channel_count_label)
        
    def create_main_layout(self):
        """Create the main application layout with splitter panels."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Channel management
        left_panel = self.create_channel_panel()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(500)
        
        # Right panel - Plot area
        right_panel = self.create_plot_panel()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter ratio (30% left, 70% right)
        splitter.setSizes([300, 1100])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
    def create_channel_panel(self):
        """Create the left panel for channel management and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Channel list section
        channel_group = QGroupBox('Loaded Channels')
        channel_layout = QVBoxLayout(channel_group)
        
        self.channel_list = ChannelListWidget()
        self.channel_list.channel_visibility_changed.connect(self.on_channel_visibility_changed)
        self.channel_list.channel_removed.connect(self.on_channel_removed)
        channel_layout.addWidget(self.channel_list)
        
        layout.addWidget(channel_group)
        
        # Display controls section
        controls_group = QGroupBox('Display Controls')
        controls_layout = QVBoxLayout(controls_group)
        
        # View mode selection
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel('View Mode:'))
        
        self.overlay_radio = QRadioButton('Overlay')
        self.overlay_radio.setChecked(True)
        self.separate_radio = QRadioButton('Separate')
        
        view_layout.addWidget(self.overlay_radio)
        view_layout.addWidget(self.separate_radio)
        view_layout.addStretch()
        
        controls_layout.addLayout(view_layout)
        
        # Quick action buttons
        button_layout = QVBoxLayout()
        
        zoom_fit_btn = QPushButton('🔍 Zoom to Fit')
        zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        button_layout.addWidget(zoom_fit_btn)
        
        save_plot_btn = QPushButton('💾 Save Plot...')
        save_plot_btn.clicked.connect(self.save_plot)
        button_layout.addWidget(save_plot_btn)
        
        controls_layout.addLayout(button_layout)
        
        layout.addWidget(controls_group)
        
        # File info section
        info_group = QGroupBox('File Information')
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        self.info_text.setFont(QFont('Consolas', 8))
        self.info_text.setPlainText('No data loaded.\n\nLoad a Siglent binary file (.bin) to see waveform information.')
        
        info_layout.addWidget(self.info_text)
        layout.addWidget(info_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_plot_panel(self):
        """Create the right panel containing the matplotlib plot canvas."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the plot canvas
        self.plot_canvas = PlotCanvas()
        
        # Create navigation toolbar
        self.nav_toolbar = NavigationToolbar(self.plot_canvas, panel)
        
        # Add widgets to layout
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.plot_canvas)
        
        return panel
        
    def create_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for common operations."""
        # These are already handled in the menu actions, but we can add more here if needed
        pass
        
    def load_settings(self):
        """Load application settings from persistent storage."""
        # Restore window geometry
        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry'))
        
        # Restore other settings as needed
        overlay_mode = self.settings.value('overlay_mode', True, type=bool)
        self.overlay_radio.setChecked(overlay_mode)
        self.separate_radio.setChecked(not overlay_mode)
        
    def save_settings(self):
        """Save application settings to persistent storage."""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('overlay_mode', self.overlay_radio.isChecked())
        
    def open_file(self):
        """Open and load a single Siglent binary file."""
        if not SIGLENT_PARSER_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Siglent parser module is not available.\nPlease ensure siglent_parser.py is in the same directory.')
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Siglent Binary File',
            '',
            'Siglent Binary Files (*.bin);;All Files (*)'
        )
        
        if file_path:
            self.load_waveform_data(file_path, is_directory=False)
            
    def open_directory(self):
        """Open and load all Siglent binary files in a directory."""
        if not SIGLENT_PARSER_AVAILABLE:
            QMessageBox.critical(self, 'Error', 'Siglent parser module is not available.\nPlease ensure siglent_parser.py is in the same directory.')
            return
            
        directory = QFileDialog.getExistingDirectory(
            self,
            'Select Directory Containing Siglent Binary Files'
        )
        
        if directory:
            self.load_waveform_data(directory, is_directory=True)
            
    def load_waveform_data(self, path: str, is_directory: bool):
        """Load waveform data in a background thread."""
        self.status_bar.showMessage('Loading waveform data...', 0)
        
        # Create and start the loading thread
        self.load_thread = LoadWaveformThread(path, is_directory)
        self.load_thread.progress.connect(self.on_load_progress)
        self.load_thread.finished.connect(self.on_load_finished)
        self.load_thread.error.connect(self.on_load_error)
        self.load_thread.start()
        
    def on_load_progress(self, message: str):
        """Handle progress updates from the loading thread."""
        self.status_bar.showMessage(message, 0)
        
    def on_load_finished(self, result: dict):
        """Handle successful completion of waveform data loading."""
        try:
            channels = result['channels']
            parser = result['parser']
            source_path = result['source_path']
            
            if not channels:
                QMessageBox.warning(self, 'Warning', 'No valid channel data found in the specified location.')
                self.status_bar.showMessage('Ready', 5000)
                return
            
            # Store the loaded data
            file_key = Path(source_path).name if not result['is_directory'] else f"dir_{Path(source_path).name}"
            self.loaded_data[file_key] = channels
            self.parsers[file_key] = parser
            
            # Add channels to the UI
            overlay_mode = self.overlay_radio.isChecked()
            
            for channel_name, channel_data in channels.items():
                if channel_data.enabled and len(channel_data.voltage_data) > 0:
                    # Add to channel list
                    self.channel_list.add_channel(channel_name, channel_data, visible=True)
                    
                    # Plot the channel
                    self.plot_canvas.plot_channel(
                        channel_name, 
                        channel_data, 
                        parser.header, 
                        overlay_mode=overlay_mode,
                        visible=True
                    )
            
            # Update file info display
            self.update_file_info(parser, source_path, channels)
            
            # Update status
            enabled_channels = [name for name, data in channels.items() if data.enabled]
            self.channel_count_label.setText(f'{len(enabled_channels)} channel(s) loaded')
            self.status_bar.showMessage(f'Successfully loaded {len(enabled_channels)} channel(s) from {Path(source_path).name}', 5000)
            
        except Exception as e:
            self.on_load_error(f"Error processing loaded data: {str(e)}\n{traceback.format_exc()}")
            
    def on_load_error(self, error_message: str):
        """Handle errors during waveform data loading."""
        QMessageBox.critical(self, 'Load Error', f'Failed to load waveform data:\n\n{error_message}')
        self.status_bar.showMessage('Ready', 5000)
        
    def update_file_info(self, parser, source_path: str, channels: Dict[str, ChannelData]):
        """Update the file information display."""
        if not parser.header:
            return
            
        info_lines = []
        info_lines.append(f"File: {Path(source_path).name}")
        info_lines.append(f"Format Version: {parser.header.version}")
        info_lines.append(f"Sample Rate: {parser.header.sample_rate.get_scaled_value():.0f} Sa/s")
        info_lines.append(f"Wave Length: {parser.header.wave_length} samples")
        info_lines.append(f"Data Width: {'8-bit' if parser.header.data_width == 0 else '16-bit'}")
        info_lines.append(f"Time/Div: {parser.header.time_div.get_scaled_value():.6f} {parser.header.time_div.get_unit_string()}")
        
        info_lines.append("\nChannels:")
        for name, channel_data in channels.items():
            if channel_data.enabled:
                volt_div = channel_data.volt_div_val.get_scaled_value()
                unit = channel_data.volt_div_val.get_unit_string()
                info_lines.append(f"  {name}: {volt_div:.3f} {unit}/div, {len(channel_data.voltage_data)} samples")
        
        self.info_text.setPlainText('\n'.join(info_lines))
        
    def on_channel_visibility_changed(self, channel_name: str, visible: bool):
        """Handle channel visibility toggle from the channel list."""
        self.plot_canvas.set_channel_visibility(channel_name, visible)
        
    def on_channel_removed(self, channel_name: str):
        """Handle channel removal from the channel list."""
        self.plot_canvas.remove_channel(channel_name)
        
        # Update status
        remaining_channels = len(self.plot_canvas.plot_data)
        self.channel_count_label.setText(f'{remaining_channels} channel(s) loaded')
        
    def clear_all_data(self):
        """Clear all loaded waveform data."""
        reply = QMessageBox.question(
            self, 
            'Clear All Data',
            'Are you sure you want to clear all loaded waveform data?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear data structures
            self.loaded_data.clear()
            self.parsers.clear()
            
            # Clear UI components
            self.channel_list.clear_all_channels()
            self.plot_canvas.clear_all_plots()
            self.info_text.setPlainText('No data loaded.\n\nLoad a Siglent binary file (.bin) to see waveform information.')
            
            # Update status
            self.channel_count_label.setText('No data loaded')
            self.status_bar.showMessage('All data cleared', 3000)
            
    def zoom_to_fit(self):
        """Auto-scale the plot to fit all visible data."""
        if self.plot_canvas.plot_data:
            self.plot_canvas.ax.relim()
            self.plot_canvas.ax.autoscale()
            self.plot_canvas.draw()
            self.status_bar.showMessage('Zoomed to fit all data', 2000)
        
    def save_plot(self):
        """Save the current plot as an image file."""
        if not self.plot_canvas.plot_data:
            QMessageBox.information(self, 'Info', 'No data to save. Load waveform data first.')
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Plot',
            'waveform_plot.png',
            'PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)'
        )
        
        if file_path:
            try:
                self.plot_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_bar.showMessage(f'Plot saved to {Path(file_path).name}', 3000)
            except Exception as e:
                QMessageBox.critical(self, 'Save Error', f'Failed to save plot:\n{str(e)}')
        
    def show_about_dialog(self):
        """Display the About dialog."""
        about_text = """
        <h2>Siglent Waveform Viewer</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Author:</b> Assistant</p>
        
        <p>A comprehensive PyQt6 application for visualizing and analyzing 
        oscilloscope waveform data from Siglent Binary Format V4.0 files.</p>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Load single files or entire directories</li>
        <li>Interactive matplotlib plots with zoom and pan</li>
        <li>Overlay multiple waveforms or display separately</li>
        <li>Channel visibility management</li>
        <li>Export plots to various formats</li>
        <li>Extensible architecture for future enhancements</li>
        </ul>
        
        <p><b>Dependencies:</b> PyQt6, matplotlib, numpy, scipy</p>
        """
        
        QMessageBox.about(self, 'About Siglent Waveform Viewer', about_text)
        
    def closeEvent(self, event):
        """Handle application closing."""
        self.save_settings()
        event.accept()


def main():
    """Main application entry point."""
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName('Siglent Waveform Viewer')
    app.setApplicationVersion('1.0.0')
    app.setOrganizationName('SiglentWaveformViewer')
    app.setOrganizationDomain('github.com')
    
    # Set application style and theme
    app.setStyle('Fusion')  # Modern cross-platform style
    
    try:
        # Create and show main window
        window = WaveformViewerMainWindow()
        window.show()
        
        # Run the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        # Handle any uncaught exceptions
        QMessageBox.critical(
            None, 
            'Application Error', 
            f'An unexpected error occurred:\n\n{str(e)}\n\n{traceback.format_exc()}'
        )
        sys.exit(1)


if __name__ == '__main__':
    main()

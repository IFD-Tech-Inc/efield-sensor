#!/usr/bin/env python3
"""
Main window for IFD Signal Analysis Utility.

This module contains the main application window that coordinates all UI components
and handles user interactions for loading, visualizing, and managing waveform data.
"""

import traceback
from pathlib import Path
from typing import Dict, Optional, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QStatusBar, QToolBar, QPushButton,
    QGroupBox, QLabel, QRadioButton, QTextEdit
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QFont
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .plot_canvas import PlotCanvas
from .channel_list import ChannelListWidget
from ..threads import PlottingThread, LoadWaveformThread
from ..utils.constants import (
    APP_NAME, APP_VERSION, APP_ORGANIZATION,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_POSITION,
    LEFT_PANEL_MIN_WIDTH, LEFT_PANEL_MAX_WIDTH, DEFAULT_SPLITTER_SIZES,
    SIGLENT_FILE_FILTER, PLOT_SAVE_FILTER, DEFAULT_PLOT_SAVE_NAME,
    STATUS_MESSAGE_SHORT, STATUS_MESSAGE_MEDIUM, STATUS_MESSAGE_LONG,
    INFO_FONT_FAMILY, INFO_FONT_SIZE, INFO_WIDGET_MAX_HEIGHT,
    DEFAULT_INFO_TEXT, ERROR_SIGLENT_PARSER_UNAVAILABLE,
    WARNING_NO_CHANNELS, WARNING_NO_DATA_TO_SAVE,
    SUCCESS_PLOT_TEMPLATE, SUCCESS_SAVE_TEMPLATE, SUCCESS_CLEAR_MESSAGE,
    SUCCESS_ZOOM_FIT_MESSAGE, STATUS_READY, STATUS_LOADING, STATUS_PLOTTING,
    STATUS_CANCELLED, LOADING_FILE_DESC, LOADING_DIRECTORY_DESC,
    RENDERING_PLOTS_DESC,
    ChannelDataDict, LoadResultDict
)

# Import external dependencies with availability checks
try:
    from siglent_parser import ChannelData
    SIGLENT_PARSER_AVAILABLE = True
except ImportError:
    SIGLENT_PARSER_AVAILABLE = False
    ChannelData = None

try:
    from progressdialog import LoadingProgressDialog
    PROGRESS_DIALOG_AVAILABLE = True
except ImportError:
    PROGRESS_DIALOG_AVAILABLE = False


class IFDSignalAnalysisMainWindow(QMainWindow):
    """
    Main application window for the IFD Signal Analysis Utility.
    
    This window provides a comprehensive interface for loading, visualizing, and
    managing oscilloscope waveform data with support for multiple channels and files.
    It coordinates between UI components and manages background operations for
    responsive user experience.
    
    Attributes:
        loaded_data: Dictionary storing all loaded channel data
        parsers: Dictionary storing parser instances for each loaded file
        settings: QSettings object for persistent application settings
        progress_dialog: Progress dialog for background operations
        plot_canvas: Main plotting widget
        channel_list: Channel management widget
    """
    
    def __init__(self) -> None:
        """Initialize the main application window."""
        super().__init__()
        
        # Application state
        self.loaded_data: ChannelDataDict = {}  # Store all loaded channel data
        self.parsers: Dict[str, Any] = {}  # Store parser instances for each loaded file
        self.settings = QSettings(APP_ORGANIZATION, 'Settings')
        self.progress_dialog: Optional['LoadingProgressDialog'] = None
        
        # UI setup
        self._setup_window()
        self._create_ui_components()
        self._load_settings()
        
        # Show welcome message
        self.status_bar.showMessage(
            f'{APP_NAME} - Ready to load waveform data', 
            STATUS_MESSAGE_LONG
        )
        
    def _setup_window(self) -> None:
        """Configure the main window properties."""
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.setGeometry(*DEFAULT_WINDOW_POSITION, *DEFAULT_WINDOW_SIZE)
        
    def _create_ui_components(self) -> None:
        """Create all UI components in the correct order."""
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()
        self._create_main_layout()
        self._create_keyboard_shortcuts()
        
    # ==================== Menu and Toolbar Creation ====================
    
    def _create_menu_bar(self) -> None:
        """Create the application menu bar with File, View, and Help menus."""
        menubar = self.menuBar()
        
        # File Menu
        self._create_file_menu(menubar)
        
        # View Menu  
        self._create_view_menu(menubar)
        
        # Help Menu
        self._create_help_menu(menubar)
        
    def _create_file_menu(self, menubar: 'QMenuBar') -> None:
        """Create the File menu."""
        file_menu = menubar.addMenu('&File')
        
        # Open File action
        open_file_action = QAction('&Open File...', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.setStatusTip('Open a single Siglent binary file')
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)
        
        # Open Directory action
        open_dir_action = QAction('Open &Directory...', self)
        open_dir_action.setShortcut('Ctrl+D')
        open_dir_action.setStatusTip('Open all Siglent binary files in a directory')
        open_dir_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_dir_action)
        
        file_menu.addSeparator()
        
        # Clear All action
        clear_action = QAction('&Clear All', self)
        clear_action.setShortcut('Ctrl+N')
        clear_action.setStatusTip('Clear all loaded waveform data')
        clear_action.triggered.connect(self.clear_all_data)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def _create_view_menu(self, menubar: 'QMenuBar') -> None:
        """Create the View menu."""
        view_menu = menubar.addMenu('&View')
        
        # Overlay Mode action
        self.overlay_action = QAction('&Overlay Mode', self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.setStatusTip('Display all channels on the same plot')
        view_menu.addAction(self.overlay_action)
        
        # Separate Plots action
        self.separate_action = QAction('&Separate Plots', self)
        self.separate_action.setCheckable(True)
        self.separate_action.setStatusTip('Display each channel in its own subplot')
        view_menu.addAction(self.separate_action)
        
        # Make view actions mutually exclusive
        self.overlay_action.triggered.connect(
            lambda: self.separate_action.setChecked(False)
        )
        self.separate_action.triggered.connect(
            lambda: self.overlay_action.setChecked(False)
        )
        
        view_menu.addSeparator()
        
        # Zoom to Fit action
        zoom_fit_action = QAction('Zoom to &Fit', self)
        zoom_fit_action.setShortcut('Ctrl+F')
        zoom_fit_action.setStatusTip('Auto-scale plot to fit all data')
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        view_menu.addAction(zoom_fit_action)
        
        view_menu.addSeparator()
        
        # Reset Channel Scale action
        reset_selected_action = QAction('Reset Selected Channel Scale', self)
        reset_selected_action.setShortcut('R')
        reset_selected_action.setStatusTip('Reset Y-axis scaling for the selected channel')
        reset_selected_action.triggered.connect(self.reset_selected_channel_scale)
        view_menu.addAction(reset_selected_action)
        
        # Reset All Axes Scale action
        reset_all_action = QAction('Reset All Axes Scales', self)
        reset_all_action.setShortcut('Shift+R')
        reset_all_action.setStatusTip('Reset all Y-axes to auto-scaled engineering values')
        reset_all_action.triggered.connect(self.reset_all_axes_scales)
        view_menu.addAction(reset_all_action)
        
        # Clear Selection action
        clear_selection_action = QAction('Clear Selection', self)
        clear_selection_action.setShortcut('Escape')
        clear_selection_action.setStatusTip('Clear the current channel selection')
        clear_selection_action.triggered.connect(self._clear_channel_selection)
        view_menu.addAction(clear_selection_action)
        
    def _create_help_menu(self, menubar: 'QMenuBar') -> None:
        """Create the Help menu."""
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About...', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    def _create_tool_bar(self) -> None:
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
        
    def _create_status_bar(self) -> None:
        """Create the status bar for displaying application status."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.channel_count_label = QLabel('No data loaded')
        self.status_bar.addPermanentWidget(self.channel_count_label)
        
    # ==================== Layout Creation ====================
    
    def _create_main_layout(self) -> None:
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
        left_panel = self._create_channel_panel()
        left_panel.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)
        left_panel.setMaximumWidth(LEFT_PANEL_MAX_WIDTH)
        
        # Right panel - Plot area
        right_panel = self._create_plot_panel()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial splitter ratio
        splitter.setSizes(DEFAULT_SPLITTER_SIZES)
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
    def _create_channel_panel(self) -> QWidget:
        """Create the left panel for channel management and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Channel list section
        channel_group = QGroupBox('Channel Management')
        channel_layout = QVBoxLayout(channel_group)
        
        self.channel_list = ChannelListWidget()
        self.channel_list.channel_visibility_changed.connect(self._on_channel_visibility_changed)
        self.channel_list.channel_removed.connect(self._on_channel_removed)
        self.channel_list.channel_selected.connect(self._on_channel_selected_from_list)
        channel_layout.addWidget(self.channel_list)
        
        layout.addWidget(channel_group)
        
        # Quick action buttons (minimal set)
        layout.addWidget(self._create_quick_actions())
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def _create_quick_actions(self) -> QWidget:
        """Create minimal quick action controls."""
        actions_group = QGroupBox('Quick Actions')
        actions_layout = QVBoxLayout(actions_group)
        
        # Essential buttons only
        zoom_fit_btn = QPushButton('🔍 Zoom to Fit')
        zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        actions_layout.addWidget(zoom_fit_btn)
        
        save_plot_btn = QPushButton('💾 Save Plot...')
        save_plot_btn.clicked.connect(self.save_plot)
        actions_layout.addWidget(save_plot_btn)
        
        # Reset button for selected channel
        reset_btn = QPushButton('🔄 Reset Selected Scale')
        reset_btn.clicked.connect(self.reset_selected_channel_scale)
        actions_layout.addWidget(reset_btn)
        
        return actions_group
    
        
    def _create_plot_panel(self) -> QWidget:
        """Create the right panel containing the matplotlib plot canvas."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the plot canvas
        self.plot_canvas = PlotCanvas()
        
        # Connect plot canvas selection signal
        self.plot_canvas.channel_selected.connect(self._on_channel_selected_from_plot)
        
        # Create navigation toolbar
        self.nav_toolbar = NavigationToolbar(self.plot_canvas, panel)
        
        # Add widgets to layout
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.plot_canvas)
        
        return panel
        
    def _create_keyboard_shortcuts(self) -> None:
        """Set up keyboard shortcuts for common operations."""
        # These are already handled in the menu actions
        pass
        
    # ==================== Settings Management ====================
    
    def _load_settings(self) -> None:
        """Load application settings from persistent storage."""
        # Restore window geometry
        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry'))
        
        # Note: View mode settings removed with display controls
        
    def _save_settings(self) -> None:
        """Save application settings to persistent storage."""
        self.settings.setValue('geometry', self.saveGeometry())
        # Note: View mode settings removed with display controls
        
    # ==================== File Operations ====================
    
    def open_file(self) -> None:
        """Open and load a single Siglent binary file."""
        if not SIGLENT_PARSER_AVAILABLE:
            QMessageBox.critical(self, 'Error', ERROR_SIGLENT_PARSER_UNAVAILABLE)
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Siglent Binary File',
            '',
            SIGLENT_FILE_FILTER
        )
        
        if file_path:
            self._load_waveform_data(file_path, is_directory=False)
            
    def open_directory(self) -> None:
        """Open and load all Siglent binary files in a directory."""
        if not SIGLENT_PARSER_AVAILABLE:
            QMessageBox.critical(self, 'Error', ERROR_SIGLENT_PARSER_UNAVAILABLE)
            return
            
        directory = QFileDialog.getExistingDirectory(
            self,
            'Select Directory Containing Siglent Binary Files'
        )
        
        if directory:
            self._load_waveform_data(directory, is_directory=True)
    
    # ==================== Data Loading ====================
    
    def _load_waveform_data(self, path: str, is_directory: bool) -> None:
        """Load waveform data in a background thread with progress dialog."""
        self.status_bar.showMessage(STATUS_LOADING, 0)
        
        # Create progress dialog if available
        if PROGRESS_DIALOG_AVAILABLE:
            self.progress_dialog = LoadingProgressDialog(self)
            operation_desc = LOADING_DIRECTORY_DESC if is_directory else LOADING_FILE_DESC
            self.progress_dialog.start_loading(operation_desc)
        
        # Create and start the loading thread
        self.load_thread = LoadWaveformThread(path, is_directory)
        self.load_thread.progress.connect(self._on_load_progress)
        self.load_thread.progress_percentage.connect(self._on_load_progress_percentage)
        self.load_thread.finished.connect(self._on_load_finished)
        self.load_thread.error.connect(self._on_load_error)
        
        # Connect progress dialog cancellation if available
        if self.progress_dialog:
            self.progress_dialog.cancelled.connect(self._on_load_cancelled)
            
        self.load_thread.start()
        
    def _on_load_progress(self, message: str) -> None:
        """Handle progress updates from the loading thread."""
        self.status_bar.showMessage(message, 0)
        
    def _on_load_progress_percentage(self, percentage: int, message: str) -> None:
        """Handle progress percentage updates from the loading thread."""
        if self.progress_dialog and not self.progress_dialog.is_cancelled():
            if not self.progress_dialog.update_progress(percentage, message):
                # User cancelled, terminate the thread
                if hasattr(self, 'load_thread') and self.load_thread.isRunning():
                    self.load_thread.terminate()
                    self.load_thread.wait()
                    
    def _on_load_cancelled(self) -> None:
        """Handle cancellation of the loading operation."""
        if hasattr(self, 'load_thread') and self.load_thread.isRunning():
            self.load_thread.terminate()
            self.load_thread.wait()
        self.status_bar.showMessage(f'Loading {STATUS_CANCELLED}', STATUS_MESSAGE_MEDIUM)
        
    def _on_load_finished(self, result: LoadResultDict) -> None:
        """Handle successful completion of waveform data loading."""
        try:
            # Close loading progress dialog if it exists
            if self.progress_dialog:
                self.progress_dialog.finish_loading("Loading completed successfully")
                self.progress_dialog = None
                
            channels = result['channels']
            parser = result['parser']
            source_path = result['source_path']
            
            if not channels:
                QMessageBox.warning(self, 'Warning', WARNING_NO_CHANNELS)
                self.status_bar.showMessage(STATUS_READY, STATUS_MESSAGE_LONG)
                return
            
            # Store the loaded data
            file_key = (Path(source_path).name if not result['is_directory'] 
                       else f"dir_{Path(source_path).name}")
            self.loaded_data[file_key] = channels
            self.parsers[file_key] = parser
            
            # Add channels to the channel list (UI only)
            for channel_name, channel_data in channels.items():
                if channel_data.enabled and len(channel_data.voltage_data) > 0:
                    self.channel_list.add_channel(channel_name, channel_data, visible=True)
            
            
            # Start background plotting instead of direct plotting
            self._start_plotting(channels, parser)
            
        except Exception as e:
            self._on_load_error(
                f"Error processing loaded data: {str(e)}\n{traceback.format_exc()}"
            )
            
    def _on_load_error(self, error_message: str) -> None:
        """Handle errors during waveform data loading."""
        # Close progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        QMessageBox.critical(
            self, 'Load Error', 
            f'Failed to load waveform data:\n\n{error_message}'
        )
        self.status_bar.showMessage(STATUS_READY, STATUS_MESSAGE_LONG)
    
    # ==================== Plotting Operations ====================
    
    def _start_plotting(self, channels: ChannelDataDict, parser: Any) -> None:
        """Start background plotting of channels using PlottingThread."""
        self.status_bar.showMessage(STATUS_PLOTTING, 0)
        
        # Create progress dialog for plotting if available
        if PROGRESS_DIALOG_AVAILABLE:
            self.progress_dialog = LoadingProgressDialog(self)
            self.progress_dialog.setWindowTitle("Rendering Plots")
            self.progress_dialog.start_loading(RENDERING_PLOTS_DESC)
        
        # Default to overlay mode (since display controls removed)
        overlay_mode = True
        
        # Create and start the plotting thread
        self.plot_thread = PlottingThread(channels, parser.header, overlay_mode)
        self.plot_thread.progress.connect(self._on_plot_progress)
        self.plot_thread.progress_percentage.connect(self._on_plot_progress_percentage)
        self.plot_thread.channel_plotted.connect(self._on_channel_plotted)
        self.plot_thread.finished.connect(self._on_plotting_finished)
        self.plot_thread.error.connect(self._on_plotting_error)
        
        # Connect progress dialog cancellation if available
        if self.progress_dialog:
            self.progress_dialog.cancelled.connect(self._on_plotting_cancelled)
            
        self.plot_thread.start()
    
    def _on_plot_progress(self, message: str) -> None:
        """Handle progress updates from the plotting thread."""
        self.status_bar.showMessage(message, 0)
        
    def _on_plot_progress_percentage(self, percentage: int, message: str) -> None:
        """Handle progress percentage updates from the plotting thread."""
        if self.progress_dialog and not self.progress_dialog.is_cancelled():
            if not self.progress_dialog.update_progress(percentage, message):
                # User cancelled, terminate the thread
                if hasattr(self, 'plot_thread') and self.plot_thread.isRunning():
                    self.plot_thread.cancel()
                    self.plot_thread.wait()
    
    def _on_channel_plotted(self, channel_name: str, plot_data: Dict[str, Any]) -> None:
        """Handle individual channel plot completion (progressive rendering)."""
        # Skip progressive rendering - we'll use batch auto-scaling instead
        # This reduces rendering operations and enables optimal axis assignment
        pass
    
    def _on_plotting_finished(self, result: Dict[str, Any]) -> None:
        """Handle successful completion of plotting."""
        try:
            # Close plotting progress dialog if it exists
            if self.progress_dialog:
                self.progress_dialog.finish_loading("Rendering completed successfully")
                self.progress_dialog = None
                
            total_channels = result['total_channels']
            plot_data = result.get('plot_data', {})
            overlay_mode = result.get('overlay_mode', True)
            
            # Apply batch plot data with auto-scaling if we have any plot data
            if plot_data:
                self.plot_canvas.apply_batch_plot_data(plot_data, overlay_mode)
                
            # Update status
            self.channel_count_label.setText(f'{total_channels} channel(s) loaded')
            self.status_bar.showMessage(
                SUCCESS_PLOT_TEMPLATE.format(count=total_channels), 
                STATUS_MESSAGE_LONG
            )
            
        except Exception as e:
            self._on_plotting_error(
                f"Error finalizing plots: {str(e)}\n{traceback.format_exc()}"
            )
    
    def _on_plotting_cancelled(self) -> None:
        """Handle cancellation of the plotting operation."""
        if hasattr(self, 'plot_thread') and self.plot_thread.isRunning():
            self.plot_thread.cancel()
            self.plot_thread.wait()
        self.status_bar.showMessage(f'Plotting {STATUS_CANCELLED}', STATUS_MESSAGE_MEDIUM)
        
    def _on_plotting_error(self, error_message: str) -> None:
        """Handle errors during plotting."""
        # Close progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        QMessageBox.critical(
            self, 'Plotting Error', 
            f'Failed to render plots:\n\n{error_message}'
        )
        self.status_bar.showMessage(STATUS_READY, STATUS_MESSAGE_LONG)
        
    # ==================== Channel Management ====================
    
    def _on_channel_visibility_changed(self, channel_name: str, visible: bool) -> None:
        """Handle channel visibility toggle from the channel list."""
        self.plot_canvas.set_channel_visibility(channel_name, visible)
        
    def _on_channel_removed(self, channel_name: str) -> None:
        """Handle channel removal from the channel list."""
        self.plot_canvas.remove_channel(channel_name)
        
        # Update status
        remaining_channels = self.plot_canvas.get_channel_count()
        self.channel_count_label.setText(f'{remaining_channels} channel(s) loaded')
        
    # ==================== User Actions ====================
    
    def clear_all_data(self) -> None:
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
            
            # Update status
            self.channel_count_label.setText('No data loaded')
            self.status_bar.showMessage(SUCCESS_CLEAR_MESSAGE, STATUS_MESSAGE_MEDIUM)
            
    def zoom_to_fit(self) -> None:
        """Auto-scale the plot to fit all visible data across all y-axes."""
        if self.plot_canvas.has_data():
            self.plot_canvas.zoom_to_fit_all_axes()
            self.status_bar.showMessage(SUCCESS_ZOOM_FIT_MESSAGE, STATUS_MESSAGE_SHORT)
        
    def save_plot(self) -> None:
        """Save the current plot as an image file."""
        if not self.plot_canvas.has_data():
            QMessageBox.information(self, 'Info', WARNING_NO_DATA_TO_SAVE)
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Plot',
            DEFAULT_PLOT_SAVE_NAME,
            PLOT_SAVE_FILTER
        )
        
        if file_path:
            try:
                self.plot_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_bar.showMessage(
                    SUCCESS_SAVE_TEMPLATE.format(filename=Path(file_path).name), 
                    STATUS_MESSAGE_MEDIUM
                )
            except Exception as e:
                QMessageBox.critical(
                    self, 'Save Error', 
                    f'Failed to save plot:\n{str(e)}'
                )
    
    def reset_selected_channel_scale(self) -> None:
        """Reset the Y-axis scaling for the currently selected channel."""
        selected_channel = self.plot_canvas.get_selected_channel()
        if selected_channel:
            self.plot_canvas.reset_channel_scale(selected_channel)
            self._update_selection_status(selected_channel)  # Update status with new scale
            self.status_bar.showMessage(
                f'Reset scale for {selected_channel}', 
                STATUS_MESSAGE_SHORT
            )
        else:
            QMessageBox.information(
                self, 'No Selection', 
                'Please select a channel first to reset its scale.'
            )
            
    def reset_all_axes_scales(self) -> None:
        """Reset all Y-axes scaling to auto-scaled engineering values."""
        if self.plot_canvas.has_data():
            self.plot_canvas.reset_all_axes_scales()
            self.status_bar.showMessage(
                'Reset all axis scales to auto-scaled values', 
                STATUS_MESSAGE_SHORT
            )
        else:
            QMessageBox.information(
                self, 'No Data', 
                'No data available to reset scales.'
            )
        
    def show_about_dialog(self) -> None:
        """Display the About dialog."""
        about_text = f"""
        <h2>{APP_NAME}</h2>
        <p><b>Version:</b> {APP_VERSION}</p>
        <p><b>Author:</b> Assistant</p>
        <p><b>For internal development use only.</b></p>
        
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
        
        QMessageBox.about(self, f'About {APP_NAME}', about_text)
        
    # ==================== Channel Selection Coordination ====================
    
    def _on_channel_selected_from_list(self, channel_name: str) -> None:
        """
        Handle channel selection from the channel list widget.
        
        Args:
            channel_name: Name of the selected channel
        """
        # Update plot canvas selection (without triggering its signal)
        if self.plot_canvas.get_selected_channel() != channel_name:
            self.plot_canvas.select_channel(channel_name)
        
        # Update status bar to show selection and scale info
        self._update_selection_status(channel_name)
    
    def _on_channel_selected_from_plot(self, channel_name: str) -> None:
        """
        Handle channel selection from the plot canvas.
        
        Args:
            channel_name: Name of the selected channel
        """
        # Update channel list selection (without triggering its signal)
        if self.channel_list.get_selected_channel() != channel_name:
            self.channel_list.select_channel(channel_name)
        
        # Update status bar to show selection and scale info
        self._update_selection_status(channel_name)
    
    def _clear_channel_selection(self) -> None:
        """
        Clear channel selection from both list and plot.
        """
        self.channel_list.clear_selection()
        self.plot_canvas.clear_selection()
        
        # Update status bar
        self._update_selection_status(None)
    
    def _update_selection_status(self, channel_name: Optional[str]) -> None:
        """
        Update the status bar with current selection and scale information.
        
        Args:
            channel_name: Name of the selected channel, or None if no selection
        """
        if channel_name:
            scale_factor = self.plot_canvas.get_channel_scale_factor(channel_name)
            scale_text = f" (Scale: {scale_factor:.2f}x)" if scale_factor != 1.0 else ""
            message = f"Selected: {channel_name}{scale_text} | Use mouse wheel to zoom selected channel"
            self.status_bar.showMessage(message, 0)  # Permanent until changed
        else:
            # Show channel count when no selection
            channel_count = self.plot_canvas.get_channel_count()
            message = f"{channel_count} channel(s) loaded | Click a channel or waveform to select"
            self.status_bar.showMessage(message, 0)
        
    # ==================== Helper Methods ====================
    
    def _update_file_info(self, parser: Any, source_path: str, 
                         channels: ChannelDataDict) -> None:
        """Update the file information display."""
        if not parser.header:
            return
            
        info_lines = []
        info_lines.append(f"File: {Path(source_path).name}")
        info_lines.append(f"Format Version: {parser.header.version}")
        info_lines.append(
            f"Sample Rate: {parser.header.sample_rate.get_scaled_value():.0f} Sa/s"
        )
        info_lines.append(f"Wave Length: {parser.header.wave_length} samples")
        info_lines.append(
            f"Data Width: {'8-bit' if parser.header.data_width == 0 else '16-bit'}"
        )
        info_lines.append(
            f"Time/Div: {parser.header.time_div.get_scaled_value():.6f} "
            f"{parser.header.time_div.get_unit_string()}"
        )
        
        info_lines.append("\nChannels:")
        for name, channel_data in channels.items():
            if channel_data.enabled:
                volt_div = channel_data.volt_div_val.get_scaled_value()
                unit = channel_data.volt_div_val.get_unit_string()
                sample_count = len(channel_data.voltage_data)
                info_lines.append(
                    f"  {name}: {volt_div:.3f} {unit}/div, {sample_count} samples"
                )
        
        self.info_text.setPlainText('\n'.join(info_lines))
        
    # ==================== Event Handling ====================
        
    def closeEvent(self, event: 'QCloseEvent') -> None:
        """Handle application closing."""
        self._save_settings()
        event.accept()

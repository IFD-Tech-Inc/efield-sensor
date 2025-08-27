#!/usr/bin/env python3
"""
Plot canvas widget for IFD Signal Analysis Utility.

This module contains the PlotCanvas class that provides matplotlib integration
with PyQt6 for waveform visualization and interactive plotting.
"""

import matplotlib
from typing import Dict, Optional, Any, List
import numpy as np

# Set matplotlib backend before other matplotlib imports
from ..utils.constants import MATPLOTLIB_BACKEND
matplotlib.use(MATPLOTLIB_BACKEND)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, QMutex, pyqtSignal

from ..utils.constants import (
    DEFAULT_PLOT_FIGURE_SIZE,
    DEFAULT_PLOT_DPI,
    DEFAULT_PLOT_TITLE,
    PLOT_XLABEL,
    PLOT_YLABEL,
    PLOT_TITLE_TEMPLATE,
    PLOT_LINE_WIDTH,
    PLOT_LINE_ALPHA,
    PLOT_GRID_ALPHA,
    PLOT_LEGEND_LOCATION,
    AVAILABLE_PLOT_COLORS
)

# Import will be handled at runtime to avoid circular dependencies
try:
    from siglent_parser import ChannelData
    SIGLENT_PARSER_AVAILABLE = True
except ImportError:
    SIGLENT_PARSER_AVAILABLE = False
    ChannelData = None


class PlotCanvas(FigureCanvas):
    """
    Matplotlib canvas widget integrated with PyQt6.
    
    This widget handles all waveform plotting and visualization with thread-safe
    operations. It supports both overlay and separate plotting modes, channel
    visibility management, and batch rendering for performance optimization.
    
    Attributes:
        fig: Matplotlib figure object
        ax: Main axes object for plotting
        plot_data: Dictionary storing plot line references and metadata
        channel_visibility: Dictionary tracking channel visibility states
        selected_channel: Currently selected channel name
        channel_scale_factors: Y-axis scaling factors for each channel
    """
    
    # Signal for channel selection
    channel_selected = pyqtSignal(str)  # channel_name
    
    def __init__(self, parent: Optional['QWidget'] = None) -> None:
        """
        Initialize the plot canvas.
        
        Args:
            parent: Parent widget, if any
        """
        # Create matplotlib figure
        self.fig = Figure(figsize=DEFAULT_PLOT_FIGURE_SIZE, dpi=DEFAULT_PLOT_DPI)
        self.fig.patch.set_facecolor('white')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot styling
        plt.style.use('default')
        self.colors = AVAILABLE_PLOT_COLORS.copy()
        self.color_index = 0
        
        # Create initial empty plot
        self.ax = self.fig.add_subplot(111)
        self._setup_empty_plot()
        
        # Store plot data for management
        self.plot_data: Dict[str, Dict[str, Any]] = {}  # Channel name -> plot line mapping
        self.channel_visibility: Dict[str, bool] = {}  # Channel visibility state
        
        # Channel selection and scaling
        self.selected_channel: Optional[str] = None
        self.channel_scale_factors: Dict[str, float] = {}  # Channel name -> scale factor
        self.original_data: Dict[str, np.ndarray] = {}  # Store original voltage data
        
        # Pending updates queue for batch rendering
        self.pending_updates: List[str] = []
        self.batch_update_pending = False
        
        # Thread safety for plot operations
        self.plot_mutex = QMutex()
        
        # Connect matplotlib events for interactivity
        self.pick_connection = self.mpl_connect('pick_event', self._on_pick)
        self.scroll_connection = self.mpl_connect('scroll_event', self._on_scroll)
        
        # Flag to prevent scroll-triggered picks
        self.scroll_in_progress = False
        
        self.draw()
    
    def _setup_empty_plot(self) -> None:
        """Configure the initial empty plot appearance."""
        self.ax.set_xlabel(PLOT_XLABEL)
        self.ax.set_ylabel(PLOT_YLABEL)
        self.ax.grid(True, alpha=PLOT_GRID_ALPHA)
        self.ax.set_title(DEFAULT_PLOT_TITLE)
        
    def clear_all_plots(self) -> None:
        """Clear all plotted data and reset the canvas."""
        self.ax.clear()
        self._setup_empty_plot()
        self.plot_data.clear()
        self.channel_visibility.clear()
        self.selected_channel = None
        self.channel_scale_factors.clear()
        self.original_data.clear()
        self.color_index = 0
        self.draw()
        
    def get_next_color(self) -> str:
        """
        Get the next color in the cycle.
        
        Returns:
            Color name from the matplotlib color cycle
        """
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
        
    def plot_channel(self, channel_name: str, channel_data: 'ChannelData', 
                    parser_header: Any, overlay_mode: bool = True, 
                    visible: bool = True) -> None:
        """
        Plot a single channel's waveform data.
        
        This method is kept for backward compatibility but should generally
        be replaced by apply_plot_data for better performance.
        
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
        try:
            time_array = channel_data.get_time_array(
                parser_header.time_div,
                parser_header.time_delay, 
                parser_header.sample_rate,
                parser_header.hori_div_num
            )
        except Exception as e:
            print(f"Warning: Could not calculate time array for channel {channel_name}: {e}")
            return
        
        if len(time_array) == 0:
            print(f"Warning: Empty time array for channel {channel_name}")
            return
            
        # Get color for this channel
        color = self.get_next_color()
        
        # Plot the waveform
        line = self._create_plot_line(
            time_array, channel_data.voltage_data, channel_name, color, visible
        )
        
        # Store the plot line for visibility management
        self._store_plot_data(channel_name, line, channel_data, time_array, color, visible, channel_data.voltage_data)
        
        # Update plot appearance
        self.update_plot_appearance()
        
    def _create_plot_line(self, time_array: 'np.ndarray', voltage_data: 'np.ndarray',
                         channel_name: str, color: str, visible: bool) -> Any:
        """
        Create a plot line on the axes.
        
        Args:
            time_array: Time values for x-axis
            voltage_data: Voltage values for y-axis
            channel_name: Name for the legend
            color: Color for the line
            visible: Initial visibility state
            
        Returns:
            Matplotlib line object
        """
        line, = self.ax.plot(
            time_array, voltage_data, 
            label=channel_name, color=color, 
            linewidth=PLOT_LINE_WIDTH, alpha=PLOT_LINE_ALPHA,
            picker=5  # Enable line picking with 5 pixel tolerance
        )
        line.set_visible(visible)
        # Store channel name in line object for identification
        line.set_gid(channel_name)  # Set group ID for identification
        return line
    
    def _store_plot_data(self, channel_name: str, line: Any, channel_data: 'ChannelData',
                        time_array: 'np.ndarray', color: str, visible: bool, voltage_data: 'np.ndarray') -> None:
        """
        Store plot data for management.
        
        Args:
            channel_name: Name of the channel
            line: Matplotlib line object
            channel_data: Original channel data
            time_array: Time array data
            color: Line color
            visible: Visibility state
            voltage_data: Voltage data array for scaling operations
        """
        self.plot_data[channel_name] = {
            'line': line,
            'data': channel_data,
            'time': time_array,
            'color': color
        }
        self.channel_visibility[channel_name] = visible
        
        # Store original data and initialize scale factor
        self.original_data[channel_name] = voltage_data.copy()
        self.channel_scale_factors[channel_name] = 1.0
        
    def set_channel_visibility(self, channel_name: str, visible: bool) -> None:
        """
        Toggle visibility of a specific channel.
        
        Args:
            channel_name: Name of the channel to modify
            visible: Desired visibility state
        """
        if channel_name in self.plot_data:
            self.plot_data[channel_name]['line'].set_visible(visible)
            self.channel_visibility[channel_name] = visible
            
            # Update plot appearance
            self.update_plot_appearance()
    
    def apply_plot_data(self, plot_data: Dict[str, Any], overlay_mode: bool = True) -> None:
        """
        Apply pre-calculated plot data to the canvas.
        
        This method runs on the UI thread and applies data prepared by PlottingThread.
        
        Args:
            plot_data: Dictionary containing prepared plot data
            overlay_mode: Whether to overlay channels or create subplots
        """
        self.plot_mutex.lock()
        try:
            channel_name = plot_data['channel_name']
            time_array = plot_data['time_array']
            voltage_data = plot_data['voltage_data']
            color = plot_data['color']
            visible = plot_data['visible']
            
            # Create the plot line
            line = self._create_plot_line(
                time_array, voltage_data, channel_name, color, visible
            )
            
            # Store the plot data
            self._store_plot_data(
                channel_name, line, plot_data['channel_data'], 
                time_array, color, visible, voltage_data
            )
            
            # Queue UI update instead of immediate draw
            self.queue_ui_update()
            
        except Exception as e:
            print(f"Error applying plot data for {plot_data.get('channel_name', 'unknown')}: {e}")
        finally:
            self.plot_mutex.unlock()
    
    def apply_batch_plot_data(self, plot_results: Dict[str, Dict[str, Any]], 
                             overlay_mode: bool = True) -> None:
        """
        Apply multiple plot data entries in a batch to improve performance.
        
        Args:
            plot_results: Dictionary of channel_name -> plot_data mappings
            overlay_mode: Whether to overlay channels or create subplots
        """
        self.plot_mutex.lock()
        try:
            # Process all channels without drawing
            for channel_name, plot_data in plot_results.items():
                time_array = plot_data['time_array']
                voltage_data = plot_data['voltage_data']
                color = plot_data['color']
                visible = plot_data['visible']
                
                # Create the plot line
                line = self._create_plot_line(
                    time_array, voltage_data, channel_name, color, visible
                )
                
                # Store the plot data
                self._store_plot_data(
                    channel_name, line, plot_data['channel_data'],
                    time_array, color, visible, voltage_data
                )
            
            # Update plot appearance once for all channels
            self.update_plot_appearance()
            
        except Exception as e:
            print(f"Error applying batch plot data: {e}")
        finally:
            self.plot_mutex.unlock()
    
    def queue_ui_update(self) -> None:
        """
        Queue a UI update to be processed on the next event loop iteration.
        
        This prevents blocking the UI thread with immediate drawing.
        """
        if not self.batch_update_pending:
            self.batch_update_pending = True
            # Use QTimer.singleShot to defer the update to the next event loop iteration
            QTimer.singleShot(0, self.process_pending_updates)
    
    def process_pending_updates(self) -> None:
        """Process any pending UI updates and redraw the canvas."""
        self.batch_update_pending = False
        self.update_plot_appearance()
    
    def update_plot_appearance(self) -> None:
        """Update the overall plot appearance (legend, title, scaling) and redraw."""
        # Update plot appearance
        self.ax.set_xlabel(PLOT_XLABEL)
        self.ax.set_ylabel(PLOT_YLABEL)
        self.ax.grid(True, alpha=PLOT_GRID_ALPHA)
        
        # Update legend to only show visible channels
        handles = []
        labels = []
        for ch_name, plot_info in self.plot_data.items():
            if self.channel_visibility.get(ch_name, False):
                handles.append(plot_info['line'])
                labels.append(ch_name)
        
        if handles:
            # Use specified location for better performance with large datasets
            self.ax.legend(handles, labels, loc=PLOT_LEGEND_LOCATION)
        else:
            if self.ax.get_legend():
                self.ax.get_legend().set_visible(False)
        
        # Auto-scale the view
        self.ax.relim()
        self.ax.autoscale()
        
        # Update title with channel info
        num_visible = sum(self.channel_visibility.values())
        self.ax.set_title(PLOT_TITLE_TEMPLATE.format(count=num_visible))
        
        # Finally, redraw the canvas
        self.draw()
            
    def remove_channel(self, channel_name: str) -> None:
        """
        Remove a channel from the plot entirely.
        
        Args:
            channel_name: Name of the channel to remove
        """
        if channel_name in self.plot_data:
            # Remove the line from the plot
            self.plot_data[channel_name]['line'].remove()
            
            # Remove from our tracking dictionaries
            del self.plot_data[channel_name]
            if channel_name in self.channel_visibility:
                del self.channel_visibility[channel_name]
            
            if self.plot_data:
                # Update plot appearance
                self.update_plot_appearance()
            else:
                # No channels left, reset to empty state
                self.clear_all_plots()
    
    def get_visible_channels(self) -> List[str]:
        """
        Get list of currently visible channel names.
        
        Returns:
            List of channel names that are currently visible
        """
        return [
            name for name, visible in self.channel_visibility.items() 
            if visible
        ]
    
    def get_channel_count(self) -> int:
        """
        Get the total number of channels in the plot.
        
        Returns:
            Total number of channels
        """
        return len(self.plot_data)
    
    def get_visible_channel_count(self) -> int:
        """
        Get the number of currently visible channels.
        
        Returns:
            Number of visible channels
        """
        return sum(self.channel_visibility.values())
    
    def has_data(self) -> bool:
        """
        Check if the canvas has any plotted data.
        
        Returns:
            True if there is plotted data, False otherwise
        """
        return len(self.plot_data) > 0
    
    # ==================== Interactive Selection Methods ====================
    
    def _on_pick(self, event) -> None:
        """
        Handle matplotlib pick events when user clicks on a line.
        
        Args:
            event: Matplotlib pick event
        """
        try:
            # Ignore pick events that might be caused by scroll events
            if self.scroll_in_progress:
                return
                
            if event.artist and hasattr(event.artist, 'get_gid'):
                channel_name = event.artist.get_gid()
                if channel_name and channel_name in self.plot_data:
                    self.select_channel(channel_name)
        except Exception as e:
            print(f"Error handling pick event: {e}")
    
    def _on_scroll(self, event) -> None:
        """
        Handle mouse wheel scroll events for Y-axis scaling of selected channel.
        
        Args:
            event: Matplotlib scroll event
        """
        try:
            # Debug information
            print(f"Scroll event: selected_channel={self.selected_channel}, inaxes={event.inaxes == self.ax}, step={event.step}")
            
            # Only handle scroll if we have a selected channel and the event is within the plot area
            if (self.selected_channel and 
                event.inaxes == self.ax and 
                event.xdata is not None and 
                event.ydata is not None):
                
                # Set flag to prevent scroll-triggered pick events
                self.scroll_in_progress = True
                
                # Calculate zoom factor (1.1 for zoom in, 0.9 for zoom out)
                zoom_factor = 1.1 if event.step > 0 else 0.9
                print(f"Scaling {self.selected_channel} by factor {zoom_factor}")
                
                self.scale_channel_y(self.selected_channel, zoom_factor)
                
                # Reset the flag after a short delay to allow pick events again
                QTimer.singleShot(100, self._reset_scroll_flag)
            else:
                print(f"Scroll ignored: selected_channel={self.selected_channel}, event.inaxes={event.inaxes}, ax={self.ax}")
                
        except Exception as e:
            print(f"Error handling scroll event: {e}")
    
    def select_channel(self, channel_name: str) -> None:
        """
        Select a channel and provide visual feedback.
        
        Args:
            channel_name: Name of the channel to select
        """
        if channel_name in self.plot_data and self.selected_channel != channel_name:
            # Clear previous selection
            self._unhighlight_all()
            
            # Set new selection
            self.selected_channel = channel_name
            
            # Highlight the selected channel
            self._highlight_channel(channel_name)
            
            # Emit selection signal
            self.channel_selected.emit(channel_name)
            
            # Redraw to show changes
            self.draw()
    
    def clear_selection(self) -> None:
        """
        Clear the current channel selection.
        """
        if self.selected_channel:
            self._unhighlight_all()
            self.selected_channel = None
            self.draw()
    
    def _highlight_channel(self, channel_name: str) -> None:
        """
        Apply visual highlighting to the selected channel.
        
        Args:
            channel_name: Name of the channel to highlight
        """
        if channel_name in self.plot_data:
            line = self.plot_data[channel_name]['line']
            # Make selected line thicker and more opaque
            line.set_linewidth(PLOT_LINE_WIDTH * 2)
            line.set_alpha(1.0)
    
    def _unhighlight_all(self) -> None:
        """
        Remove highlighting from all channels.
        """
        for channel_name, plot_info in self.plot_data.items():
            line = plot_info['line']
            line.set_linewidth(PLOT_LINE_WIDTH)
            line.set_alpha(PLOT_LINE_ALPHA)
    
    def scale_channel_y(self, channel_name: str, scale_factor: float) -> None:
        """
        Scale the Y-axis data for a specific channel.
        
        Args:
            channel_name: Name of the channel to scale
            scale_factor: Factor to multiply current scale by
        """
        if channel_name in self.plot_data and channel_name in self.original_data:
            # Update the cumulative scale factor
            old_scale = self.channel_scale_factors[channel_name]
            self.channel_scale_factors[channel_name] *= scale_factor
            new_scale = self.channel_scale_factors[channel_name]
            
            print(f"Scaling {channel_name}: {old_scale:.3f} -> {new_scale:.3f} (factor: {scale_factor:.3f})")
            
            # Apply scaling to the original data
            original_data = self.original_data[channel_name]
            scaled_data = original_data * self.channel_scale_factors[channel_name]
            
            # Update the plot line data
            line = self.plot_data[channel_name]['line']
            line.set_ydata(scaled_data)
            
            # Update the stored plot data
            self.plot_data[channel_name]['scaled_data'] = scaled_data
            
            # Force plot update but avoid full auto-scaling (which would defeat individual scaling)
            # Just redraw without changing axes limits
            self.draw()
            
            print(f"Channel {channel_name} Y-data range: {scaled_data.min():.6f} to {scaled_data.max():.6f}")
        else:
            print(f"Cannot scale channel {channel_name}: not found in plot_data or original_data")
            print(f"Available channels in plot_data: {list(self.plot_data.keys())}")
            print(f"Available channels in original_data: {list(self.original_data.keys())}")
    
    def reset_channel_scale(self, channel_name: str) -> None:
        """
        Reset the Y-axis scaling for a specific channel to original.
        
        Args:
            channel_name: Name of the channel to reset
        """
        if channel_name in self.plot_data and channel_name in self.original_data:
            # Reset scale factor
            self.channel_scale_factors[channel_name] = 1.0
            
            # Restore original data
            original_data = self.original_data[channel_name]
            line = self.plot_data[channel_name]['line']
            line.set_ydata(original_data)
            
            # Update the stored plot data
            if 'scaled_data' in self.plot_data[channel_name]:
                del self.plot_data[channel_name]['scaled_data']
            
            # Redraw to show changes
            self.ax.relim()
            self.ax.autoscale_view()
            self.draw()
    
    def get_selected_channel(self) -> Optional[str]:
        """
        Get the currently selected channel name.
        
        Returns:
            Name of the selected channel, or None if no selection
        """
        return self.selected_channel
    
    def get_channel_scale_factor(self, channel_name: str) -> float:
        """
        Get the current scale factor for a channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Current scale factor (1.0 = original scale)
        """
        return self.channel_scale_factors.get(channel_name, 1.0)
    
    def _reset_scroll_flag(self) -> None:
        """
        Reset the scroll_in_progress flag to allow pick events again.
        
        This is called after a short delay following a scroll event.
        """
        self.scroll_in_progress = False

#!/usr/bin/env python3
"""
Plot canvas widget for IFD Signal Analysis Utility.

This module contains the PlotCanvas class that provides matplotlib integration
with PyQt6 for waveform visualization and interactive plotting.
"""

import matplotlib
from typing import Dict, Optional, Any, List, Tuple
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
from ..utils.scaling_utils import (
    calculate_engineering_range,
    get_engineering_tick_values,
    should_use_separate_axes,
    group_channels_by_scale,
    calculate_axis_positions,
    format_engineering_value
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
        
        # Optimize figure layout to minimize white space
        self._optimize_figure_layout()
        
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
        
        # Multiple y-axes support
        self.axes_dict: Dict[int, Any] = {0: self.ax}  # Axis index -> matplotlib axes object
        self.channel_to_axis: Dict[str, int] = {}  # Channel name -> axis index mapping
        self.axis_colors: Dict[int, str] = {}  # Axis index -> representative color
        self.axis_ranges: Dict[int, Tuple[float, float]] = {}  # Axis index -> (min, max) range
        
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
        
        # Toolbar reference for proper matplotlib integration
        self.toolbar = None
        
        self.draw()
    
    def set_toolbar(self, toolbar: Any) -> None:
        """
        Set the NavigationToolbar for this canvas and establish proper bidirectional connection.
        
        Args:
            toolbar: NavigationToolbar2QT instance to associate with this canvas
        """
        self.toolbar = toolbar
        if toolbar is not None:
            # Ensure the toolbar points back to this canvas
            toolbar.canvas = self
            toolbar.figure = self.fig
            # Update the toolbar's internal state
            if hasattr(toolbar, '_init_toolbar'):
                try:
                    toolbar._init_toolbar()
                except Exception as e:
                    print(f"Warning: Could not reinitialize toolbar: {e}")
            print(f"Set toolbar for canvas {getattr(self, 'plot_id', 'unknown')}: {toolbar}")
    
    def _optimize_figure_layout(self) -> None:
        """
        Optimize the matplotlib figure layout to minimize white space around plots.
        
        This method configures tight subplot parameters and figure margins to
        reduce the gap between the toolbar and plot area.
        """
        # Use tight_layout with minimal padding
        self.fig.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
        
        # Further optimize with manual subplot parameters for minimal margins
        self.fig.subplots_adjust(
            left=0.08,    # Minimal left margin for y-axis labels
            bottom=0.08,  # Minimal bottom margin for x-axis labels
            right=0.96,   # Minimal right margin
            top=0.94,     # Minimal top margin below title
            wspace=0.02,  # Minimal horizontal space between subplots
            hspace=0.02   # Minimal vertical space between subplots
        )
        
    def _setup_empty_plot(self) -> None:
        """Configure the initial empty plot appearance."""
        self.ax.set_xlabel(PLOT_XLABEL)
        self.ax.set_ylabel(PLOT_YLABEL)
        self.ax.grid(True, alpha=PLOT_GRID_ALPHA)
        self.ax.set_title(DEFAULT_PLOT_TITLE)
        
    def clear_all_plots(self) -> None:
        """Clear all plotted data and reset the canvas."""
        # Clear all axes except the main one
        for axis_idx, axis_obj in list(self.axes_dict.items()):
            if axis_idx != 0:  # Don't remove main axis
                axis_obj.remove()
        
        # Reset main axis
        self.ax.clear()
        self._setup_empty_plot()
        
        # Reset all state variables
        self.plot_data.clear()
        self.channel_visibility.clear()
        self.selected_channel = None
        self.channel_scale_factors.clear()
        self.original_data.clear()
        self.color_index = 0
        
        # Reset multiple axes state
        self.axes_dict = {0: self.ax}
        self.channel_to_axis.clear()
        self.axis_colors.clear()
        self.axis_ranges.clear()
        
        self.draw()
    
    def get_plot_data(self) -> Dict[str, Any]:
        """
        Extract current plot data in a format suitable for signal processing.
        
        Returns:
            Dictionary containing plot data with channels, header info, and metadata
        """
        if not self.plot_data:
            return {
                'channels': {},
                'header': {},
                'source_info': {
                    'plot_id': getattr(self, 'plot_id', 'unknown'),
                    'export_timestamp': self._get_current_timestamp(),
                    'channel_count': 0
                }
            }
        
        channels_data = {}
        
        # Extract data from each plotted channel
        for channel_name, channel_info in self.plot_data.items():
            if channel_name in self.channel_visibility and self.channel_visibility[channel_name]:
                # Get the line data
                line = channel_info.get('line')
                if line and hasattr(line, 'get_data'):
                    time_data, voltage_data = line.get_data()
                    
                    # Convert to numpy arrays if needed
                    if not isinstance(time_data, np.ndarray):
                        time_data = np.array(time_data)
                    if not isinstance(voltage_data, np.ndarray):
                        voltage_data = np.array(voltage_data)
                    
                    channels_data[channel_name] = {
                        'time_array': time_data,
                        'voltage_data': voltage_data,
                        'metadata': {
                            'color': channel_info.get('color', 'blue'),
                            'visible': self.channel_visibility[channel_name],
                            'axis_index': self.channel_to_axis.get(channel_name, 0),
                            'scale_factor': self.channel_scale_factors.get(channel_name, 1.0)
                        }
                    }
        
        return {
            'channels': channels_data,
            'header': getattr(self, '_last_header', {}),
            'source_info': {
                'plot_id': getattr(self, 'plot_id', 'unknown'),
                'export_timestamp': self._get_current_timestamp(),
                'channel_count': len(channels_data),
                'total_channels': len(self.plot_data),
                'visible_channels': len([c for c, v in self.channel_visibility.items() if v])
            }
        }
    
    def set_processed_data(self, processed_data: Dict[str, Any], 
                          source_plot_id: str = None, 
                          processor_info: Dict[str, Any] = None) -> None:
        """
        Display processed data from another plot.
        
        Args:
            processed_data: Dictionary containing processed waveform data
            source_plot_id: ID of the source plot that generated this data
            processor_info: Information about the processing chain applied
        """
        if 'channels' not in processed_data:
            print("Warning: No channels data in processed_data")
            return
        
        channels = processed_data['channels']
        if not channels:
            print("Warning: No channels to display")
            return
        
        # Clear existing data
        self.clear_all_plots()
        
        # Store header information
        if 'header' in processed_data:
            self._last_header = processed_data['header']
        
        # Update plot title to indicate processed data
        title_parts = []
        if source_plot_id:
            title_parts.append(f"From {source_plot_id}")
        if processor_info and 'processor_name' in processor_info:
            title_parts.append(processor_info['processor_name'])
        
        if title_parts:
            plot_title = f"Processed: {' → '.join(title_parts)}"
        else:
            plot_title = "Processed Data"
        
        self.ax.set_title(plot_title)
        
        # Plot each channel
        for channel_name, channel_data in channels.items():
            if 'time_array' in channel_data and 'voltage_data' in channel_data:
                time_array = channel_data['time_array']
                voltage_data = channel_data['voltage_data']
                
                # Get metadata
                metadata = channel_data.get('metadata', {})
                color = metadata.get('color', self.get_next_color())
                
                # Calculate voltage range for axis assignment
                voltage_range = self._calculate_channel_range(voltage_data)
                
                # Determine which axis this channel should use
                target_axis, axis_index = self._get_or_create_axis_for_channel(channel_name, voltage_range)
                
                # Plot the data on the correct axis
                line, = target_axis.plot(
                    time_array, voltage_data, 
                    color=color, 
                    linewidth=PLOT_LINE_WIDTH,
                    alpha=PLOT_LINE_ALPHA,
                    picker=True,
                    pickradius=5,
                    label=channel_name
                )
                
                # Store plot information
                self.plot_data[channel_name] = {
                    'line': line,
                    'color': color,
                    'time_array': time_array,
                    'voltage_data': voltage_data,
                    'voltage_range': voltage_range,
                    'axis_index': axis_index
                }
                
                # Set channel visibility
                self.channel_visibility[channel_name] = metadata.get('visible', True)
                
                # Store original data and scale factor
                self.original_data[channel_name] = voltage_data.copy()
                self.channel_scale_factors[channel_name] = metadata.get('scale_factor', 1.0)
                
                # Assign to the determined axis
                self.channel_to_axis[channel_name] = axis_index
        
        # Recalculate and apply engineering scaling
        self._recalculate_axis_ranges()
        self._apply_engineering_scaling_to_all_axes()
        
        # Update plot appearance
        self.update_plot_appearance()
        
        # Refresh the canvas
        self.draw()
        
        print(f"Set processed data: {len(channels)} channels from {source_plot_id or 'unknown plot'}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        import datetime
        return datetime.datetime.now().isoformat()
        
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
                         channel_name: str, color: str, visible: bool, axis_obj: Any = None) -> Any:
        """
        Create a plot line on the specified axes.
        
        Args:
            time_array: Time values for x-axis
            voltage_data: Voltage values for y-axis
            channel_name: Name for the legend
            color: Color for the line
            visible: Initial visibility state
            axis_obj: Matplotlib axis object to plot on (defaults to main axis)
            
        Returns:
            Matplotlib line object
        """
        # Use specified axis or default to main axis
        target_axis = axis_obj if axis_obj is not None else self.ax
        
        line, = target_axis.plot(
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
        Store plot data for management with intelligent y-axis assignment.
        
        Args:
            channel_name: Name of the channel
            line: Matplotlib line object
            channel_data: Original channel data
            time_array: Time array data
            color: Line color
            visible: Visibility state
            voltage_data: Voltage data array for scaling operations
        """
        # Calculate voltage range for this channel
        voltage_range = self._calculate_channel_range(voltage_data)
        
        # Determine which axis this channel should use
        target_axis, axis_index = self._get_or_create_axis_for_channel(channel_name, voltage_range)
        
        # Store axis assignment
        self.channel_to_axis[channel_name] = axis_index
        
        # Update axis range to include this channel
        if axis_index in self.axis_ranges:
            existing_min, existing_max = self.axis_ranges[axis_index]
            new_min = min(existing_min, voltage_range[0])
            new_max = max(existing_max, voltage_range[1])
            self.axis_ranges[axis_index] = (new_min, new_max)
        else:
            self.axis_ranges[axis_index] = voltage_range
        
        self.plot_data[channel_name] = {
            'line': line,
            'data': channel_data,
            'time': time_array,
            'color': color,
            'voltage_range': voltage_range,
            'axis_index': axis_index
        }
        self.channel_visibility[channel_name] = visible
        
        # Store original data and initialize scale factor
        self.original_data[channel_name] = voltage_data.copy()
        self.channel_scale_factors[channel_name] = 1.0
        
    def _store_plot_data_with_axis(self, channel_name: str, line: Any, channel_data: 'ChannelData',
                                  time_array: 'np.ndarray', color: str, visible: bool, 
                                  voltage_data: 'np.ndarray', voltage_range: Tuple[float, float],
                                  axis_index: int) -> None:
        """
        Store plot data with pre-determined axis assignment.
        
        Args:
            channel_name: Name of the channel
            line: Matplotlib line object
            channel_data: Original channel data
            time_array: Time array data
            color: Line color
            visible: Visibility state
            voltage_data: Voltage data array for scaling operations
            voltage_range: Pre-calculated voltage range
            axis_index: Pre-determined axis index
        """
        # Store axis assignment
        self.channel_to_axis[channel_name] = axis_index
        
        # Update axis range to include this channel
        if axis_index in self.axis_ranges:
            existing_min, existing_max = self.axis_ranges[axis_index]
            new_min = min(existing_min, voltage_range[0])
            new_max = max(existing_max, voltage_range[1])
            self.axis_ranges[axis_index] = (new_min, new_max)
        else:
            self.axis_ranges[axis_index] = voltage_range
        
        self.plot_data[channel_name] = {
            'line': line,
            'data': channel_data,
            'time': time_array,
            'color': color,
            'voltage_range': voltage_range,
            'axis_index': axis_index
        }
        self.channel_visibility[channel_name] = visible
        
        # Store original data and initialize scale factor
        self.original_data[channel_name] = voltage_data.copy()
        self.channel_scale_factors[channel_name] = 1.0
        
    def set_channel_visibility(self, channel_name: str, visible: bool) -> None:
        """
        Toggle visibility of a specific channel and manage axis visibility.
        
        Args:
            channel_name: Name of the channel to modify
            visible: Desired visibility state
        """
        if channel_name in self.plot_data:
            self.plot_data[channel_name]['line'].set_visible(visible)
            self.channel_visibility[channel_name] = visible
            
            # Get the axis this channel is assigned to
            axis_index = self.plot_data[channel_name].get('axis_index', 0)
            
            # Check if this axis should be hidden/shown based on channel visibility
            self._update_axis_visibility(axis_index)
            
            # Recalculate axis ranges when channels are toggled
            self._recalculate_axis_ranges()
            
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
            
            # First determine which axis to use
            voltage_range = self._calculate_channel_range(voltage_data)
            target_axis, axis_index = self._get_or_create_axis_for_channel(channel_name, voltage_range)
            
            print(f"Assigning {channel_name} to axis {axis_index} with range {voltage_range}")
            
            # Create the plot line on the correct axis
            line = self._create_plot_line(
                time_array, voltage_data, channel_name, color, visible, target_axis
            )
            
            # Store the plot data with axis information
            self._store_plot_data_with_axis(
                channel_name, line, plot_data['channel_data'], 
                time_array, color, visible, voltage_data, 
                voltage_range, axis_index
            )
            
            # Queue UI update instead of immediate draw
            self.queue_ui_update()
            
        except Exception as e:
            print(f"Error applying plot data for {plot_data.get('channel_name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.plot_mutex.unlock()
    
    def apply_batch_plot_data(self, plot_results: Dict[str, Dict[str, Any]], 
                             overlay_mode: bool = True) -> None:
        """
        Apply multiple plot data entries in a batch with optimal axis assignment.
        
        This method uses intelligent grouping to determine optimal y-axis assignments
        upfront, creating a better initial display with proper engineering scaling.
        
        Args:
            plot_results: Dictionary of channel_name -> plot_data mappings
            overlay_mode: Whether to overlay channels or create subplots
        """
        self.plot_mutex.lock()
        try:
            if not plot_results:
                return
            
            # Phase 1: Collect all channel voltage ranges
            channel_ranges = {}
            channel_data_dict = {}
            
            for channel_name, plot_data in plot_results.items():
                voltage_data = plot_data['voltage_data']
                voltage_range = self._calculate_channel_range(voltage_data)
                channel_ranges[channel_name] = voltage_range
                channel_data_dict[channel_name] = plot_data
            
            # Phase 2: Use group_channels_by_scale to determine optimal axis assignments
            # Convert channel_ranges to the expected format for group_channels_by_scale
            channels_data_for_grouping = {
                channel_name: {'voltage_range': voltage_range}
                for channel_name, voltage_range in channel_ranges.items()
            }
            axis_groups = group_channels_by_scale(channels_data_for_grouping)
            
            # Phase 3: Create necessary y-axes upfront
            self._create_axes_for_groups(axis_groups, channel_data_dict)
            
            # Phase 4: Plot channels on their assigned axes
            for axis_idx, channel_list in axis_groups.items():
                for channel_name in channel_list:
                    plot_data = plot_results[channel_name]
                    time_array = plot_data['time_array']
                    voltage_data = plot_data['voltage_data']
                    color = plot_data['color']
                    visible = plot_data['visible']
                    
                    # Get the axis for this channel
                    target_axis = self.axes_dict[axis_idx]
                    
                    # Create the plot line on the assigned axis
                    line = self._create_plot_line(
                        time_array, voltage_data, channel_name, color, visible, target_axis
                    )
                    
                    # Store the plot data with predetermined axis assignment
                    voltage_range = channel_ranges[channel_name]
                    self._store_plot_data_with_axis(
                        channel_name, line, plot_data['channel_data'],
                        time_array, color, visible, voltage_data,
                        voltage_range, axis_idx
                    )
            
            # Phase 5: Apply engineering scaling to all axes with optimal screen utilization
            self._apply_engineering_scaling_to_all_axes()
            
            # Phase 6: Update plot appearance once for all channels
            self.update_plot_appearance()
            
        except Exception as e:
            print(f"Error applying batch plot data: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Update legend to only show visible channels from all axes
        handles = []
        labels = []
        
        # Collect all visible lines from all axes, organized by axis for better grouping
        for axis_idx in sorted(self.axes_dict.keys()):
            axis_handles = []
            axis_labels = []
            
            for ch_name, plot_info in self.plot_data.items():
                if (self.channel_visibility.get(ch_name, False) and 
                    plot_info.get('axis_index', 0) == axis_idx):
                    axis_handles.append(plot_info['line'])
                    axis_labels.append(ch_name)
            
            # Add this axis's channels to the overall legend
            handles.extend(axis_handles)
            labels.extend(axis_labels)
        
        if handles:
            # Create legend on main axis but include all visible channels from all axes
            self.ax.legend(handles, labels, loc=PLOT_LEGEND_LOCATION)
        else:
            if self.ax.get_legend():
                self.ax.get_legend().set_visible(False)
        
        # Ensure all secondary axes don't have their own legends to avoid conflicts
        for axis_idx, axis_obj in self.axes_dict.items():
            if axis_idx != 0 and axis_obj.get_legend():
                axis_obj.get_legend().set_visible(False)
        
        # Apply engineering scaling and visual styling to each axis
        for axis_idx, voltage_range in self.axis_ranges.items():
            if axis_idx in self.axes_dict:
                axis_obj = self.axes_dict[axis_idx]
                self._apply_engineering_scaling_to_axis(axis_obj, voltage_range)
                self._style_axis_appearance(axis_obj, axis_idx)
                print(f"Applied engineering scaling to axis {axis_idx}: range {voltage_range}")
        
        # If no specific axis ranges, fall back to auto-scaling for main axis only
        if not self.axis_ranges:
            self.ax.relim()
            self.ax.autoscale()
        
        # Update title with channel info
        num_visible = sum(self.channel_visibility.values())
        self.ax.set_title(PLOT_TITLE_TEMPLATE.format(count=num_visible))
        
        # Finally, redraw the canvas
        self.draw()
            
    def remove_channel(self, channel_name: str) -> None:
        """
        Remove a channel from the plot entirely with axes cleanup.
        
        Args:
            channel_name: Name of the channel to remove
        """
        if channel_name in self.plot_data:
            # Get the axis this channel was using before removal
            axis_index = self.plot_data[channel_name].get('axis_index', 0)
            
            # Remove the line from the plot
            self.plot_data[channel_name]['line'].remove()
            
            # Remove from our tracking dictionaries
            del self.plot_data[channel_name]
            if channel_name in self.channel_visibility:
                del self.channel_visibility[channel_name]
            if channel_name in self.channel_to_axis:
                del self.channel_to_axis[channel_name]
            if channel_name in self.channel_scale_factors:
                del self.channel_scale_factors[channel_name]
            if channel_name in self.original_data:
                del self.original_data[channel_name]
            
            # Clean up unused axes
            self._cleanup_unused_axes()
            
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
        
    def get_all_channel_names(self) -> List[str]:
        """
        Get list of all channel names (both visible and hidden).
        
        Returns:
            List of all channel names in the plot
        """
        return list(self.plot_data.keys())
        
    def get_channel_visibility_state(self, channel_name: str) -> bool:
        """
        Get the visibility state of a specific channel.
        
        Args:
            channel_name: Name of the channel to check
            
        Returns:
            True if the channel is visible, False otherwise
        """
        return self.channel_visibility.get(channel_name, False)
        
    def get_all_channels_visibility(self) -> Dict[str, bool]:
        """
        Get visibility states for all channels.
        
        Returns:
            Dictionary mapping channel names to their visibility states
        """
        return self.channel_visibility.copy()
    
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
    
    # ==================== Multiple Y-Axis Support Methods ====================
    
    def _create_additional_yaxis(self, axis_index: int, color: str) -> Any:
        """
        Create an additional y-axis using matplotlib's twinx().
        
        Args:
            axis_index: Index for the new axis
            color: Representative color for the axis
            
        Returns:
            New matplotlib axes object
        """
        # Create twin axis
        new_ax = self.ax.twinx()
        
        # Position axis based on index
        positions = calculate_axis_positions(axis_index + 1)
        if axis_index < len(positions):
            side, offset = positions[axis_index]
            if side == 'right' and offset > 0:
                # Move spine to avoid overlap
                new_ax.spines['right'].set_position(('outward', offset * 60))
            elif side == 'left' and offset != 0:
                new_ax.spines['left'].set_position(('outward', abs(offset) * 60))
                new_ax.yaxis.set_label_position('left')
                new_ax.yaxis.tick_left()
        
        # Store the axis and its color
        self.axes_dict[axis_index] = new_ax
        self.axis_colors[axis_index] = color
        
        return new_ax
    
    def _get_or_create_axis_for_channel(self, channel_name: str, 
                                       voltage_range: Tuple[float, float]) -> Tuple[Any, int]:
        """
        Determine which axis a channel should use or create a new one if needed.
        
        Args:
            channel_name: Name of the channel
            voltage_range: (min, max) voltage range for the channel
            
        Returns:
            Tuple of (matplotlib_axis, axis_index)
        """
        # Check if we can use an existing axis
        for axis_idx, axis_range in self.axis_ranges.items():
            existing_ranges = [axis_range, voltage_range]
            if not should_use_separate_axes(existing_ranges):
                # Can share this axis
                return self.axes_dict[axis_idx], axis_idx
        
        # Need new axis if we have multiple ranges that require separation
        if len(self.axis_ranges) > 0:
            all_ranges = list(self.axis_ranges.values()) + [voltage_range]
            if should_use_separate_axes(all_ranges):
                # Create new axis
                new_axis_idx = max(self.axes_dict.keys()) + 1
                # Use the first channel's color for this axis
                color = self.get_next_color()
                new_axis = self._create_additional_yaxis(new_axis_idx, color)
                self.axis_ranges[new_axis_idx] = voltage_range
                return new_axis, new_axis_idx
        
        # Use existing axis 0 (default)
        if 0 not in self.axis_ranges:
            self.axis_ranges[0] = voltage_range
        return self.axes_dict[0], 0
    
    def _calculate_channel_range(self, voltage_data: 'np.ndarray') -> Tuple[float, float]:
        """
        Calculate the voltage range for a channel's data.
        
        Args:
            voltage_data: Numpy array of voltage values
            
        Returns:
            Tuple of (min_value, max_value)
        """
        if len(voltage_data) == 0:
            return (0.0, 1.0)
        
        min_val = float(np.min(voltage_data))
        max_val = float(np.max(voltage_data))
        
        # Handle case where all values are the same
        if min_val == max_val:
            if abs(min_val) < 1e-12:
                return (-1.0, 1.0)
            else:
                margin = abs(min_val) * 0.1
                return (min_val - margin, max_val + margin)
        
        return (min_val, max_val)
    
    def _apply_engineering_scaling_to_axis(self, axis_obj: Any, 
                                          voltage_range: Tuple[float, float]) -> None:
        """
        Apply engineering-friendly scaling to a specific axis.
        
        Args:
            axis_obj: Matplotlib axis object
            voltage_range: (min, max) range for the axis
        """
        min_val, max_val = voltage_range
        
        # Calculate engineering-friendly range with improved centering
        y_min, y_max = calculate_engineering_range(min_val, max_val)
        
        # Debug output
        print(f"Engineering scaling for axis: data range [{min_val:.6f}, {max_val:.6f}] -> axis range [{y_min:.6f}, {y_max:.6f}]")
        
        # Set axis limits
        axis_obj.set_ylim(y_min, y_max)
        
        # Set engineering-friendly tick values
        tick_values = get_engineering_tick_values(y_min, y_max, target_ticks=8)
        axis_obj.set_yticks(tick_values)
        print(f"Tick values: {tick_values}")
        
        # Format tick labels with engineering notation if beneficial
        max_abs_val = max(abs(y_min), abs(y_max))
        print(f"Max absolute value for formatting decision: {max_abs_val}")
        
        if max_abs_val >= 1000 or max_abs_val < 1.0:
            tick_labels = [format_engineering_value(val) for val in tick_values]
            axis_obj.set_yticklabels(tick_labels)
            print(f"Applied engineering formatted tick labels: {tick_labels}")
        else:
            print(f"Using default tick labels (no engineering formatting needed for range {max_abs_val})")
    
    def _style_axis_appearance(self, axis_obj: Any, axis_index: int) -> None:
        """
        Apply visual styling to a specific axis including color coding and labels.
        
        Args:
            axis_obj: Matplotlib axis object
            axis_index: Index of the axis for styling reference
        """
        # Get representative color for this axis
        axis_color = self.axis_colors.get(axis_index, 'black')
        
        # Get the first channel using this axis to determine its color
        first_channel_color = None
        for channel_name, plot_info in self.plot_data.items():
            if plot_info.get('axis_index') == axis_index and self.channel_visibility.get(channel_name, False):
                first_channel_color = plot_info['color']
                break
        
        # Use the first channel's color if available, otherwise use stored axis color
        display_color = first_channel_color if first_channel_color else axis_color
        
        # Color-code the y-axis tick labels and spine
        axis_obj.tick_params(axis='y', colors=display_color, labelsize=9)
        
        # Get all channels assigned to this axis
        channels_for_axis = self._get_channels_for_axis(axis_index)
        visible_channels = [ch for ch in channels_for_axis if self.channel_visibility.get(ch, False)]
        
        # Create channel-based label
        if visible_channels:
            # Always join with comma for consistency, whether single or multiple channels
            axis_label = ", ".join(sorted(visible_channels))
        else:
            # No visible channels - generic label
            axis_label = "Voltage"
        
        # Add appropriate units based on the axis range
        if axis_index in self.axis_ranges:
            min_val, max_val = self.axis_ranges[axis_index]
            max_abs_val = max(abs(min_val), abs(max_val))
            
            if max_abs_val >= 1.0:
                unit_suffix = " (V)"
            elif max_abs_val >= 0.001:
                unit_suffix = " (mV)"
            elif max_abs_val >= 0.000001:
                unit_suffix = " (μV)"
            else:
                unit_suffix = " (nV)"
        else:
            unit_suffix = " (V)"
        
        final_label = axis_label + unit_suffix
        
        # Set y-axis label with appropriate color
        if axis_index == 0:
            # Main (left) axis
            axis_obj.set_ylabel(final_label, color=display_color, fontsize=10)
            axis_obj.spines['left'].set_color(display_color)
        else:
            # Secondary (right) axes
            axis_obj.set_ylabel(final_label, color=display_color, fontsize=10)
            axis_obj.spines['right'].set_color(display_color)
            
            # For right axes, make sure the spine is visible and colored
            axis_obj.spines['right'].set_visible(True)
            axis_obj.spines['right'].set_linewidth(1.2)
        
        # Ensure appropriate spine visibility
        if axis_index == 0:
            # Main axis - show left spine
            axis_obj.spines['left'].set_visible(True)
            axis_obj.spines['left'].set_linewidth(1.2)
        
        # Hide top and bottom spines for cleaner look
        axis_obj.spines['top'].set_visible(False)
        axis_obj.spines['bottom'].set_visible(False)
    
    def _get_channels_for_axis(self, axis_index: int) -> List[str]:
        """
        Get list of channel names assigned to a specific axis.
        
        Args:
            axis_index: Index of the axis
            
        Returns:
            List of channel names using the specified axis
        """
        return [
            channel_name for channel_name, plot_info in self.plot_data.items()
            if plot_info.get('axis_index') == axis_index
        ]
    
    def get_axis_for_channel(self, channel_name: str) -> Optional[Any]:
        """
        Get the matplotlib axis object for a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Matplotlib axis object, or None if channel not found
        """
        axis_idx = self.channel_to_axis.get(channel_name)
        if axis_idx is not None:
            return self.axes_dict.get(axis_idx)
        return None
    
    def get_channel_axis_index(self, channel_name: str) -> Optional[int]:
        """
        Get the axis index for a specific channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Axis index, or None if channel not found
        """
        return self.channel_to_axis.get(channel_name)
    
    def _update_axis_visibility(self, axis_index: int) -> None:
        """
        Update the visibility of an axis based on its assigned channels.
        
        Args:
            axis_index: Index of the axis to check
        """
        if axis_index in self.axes_dict:
            axis_obj = self.axes_dict[axis_index]
            
            # Check if any channels on this axis are visible
            has_visible_channels = False
            for channel_name, plot_info in self.plot_data.items():
                if (plot_info.get('axis_index') == axis_index and 
                    self.channel_visibility.get(channel_name, False)):
                    has_visible_channels = True
                    break
            
            # Show/hide the axis based on channel visibility
            if axis_index == 0:
                # Main axis is always visible (but we can gray out its components)
                if has_visible_channels:
                    axis_obj.tick_params(axis='y', colors='black', labelsize=9)
                    axis_obj.spines['left'].set_visible(True)
                else:
                    axis_obj.tick_params(axis='y', colors='lightgray', labelsize=9)
                    axis_obj.spines['left'].set_color('lightgray')
            else:
                # Secondary axes can be fully hidden
                if has_visible_channels:
                    axis_obj.set_visible(True)
                    axis_obj.spines['right'].set_visible(True)
                else:
                    axis_obj.set_visible(False)
    
    def _recalculate_axis_ranges(self) -> None:
        """
        Recalculate axis ranges based on currently visible channels.
        """
        # Clear existing ranges
        new_axis_ranges = {}
        
        # Recalculate range for each axis based on visible channels
        for axis_index in self.axes_dict.keys():
            visible_channels = [
                channel_name for channel_name, plot_info in self.plot_data.items()
                if (plot_info.get('axis_index') == axis_index and 
                    self.channel_visibility.get(channel_name, False))
            ]
            
            if visible_channels:
                # Calculate combined range for all visible channels on this axis
                all_mins = []
                all_maxs = []
                
                for channel_name in visible_channels:
                    voltage_range = self.plot_data[channel_name]['voltage_range']
                    all_mins.append(voltage_range[0])
                    all_maxs.append(voltage_range[1])
                
                if all_mins and all_maxs:
                    new_axis_ranges[axis_index] = (min(all_mins), max(all_maxs))
        
        # Update the axis ranges
        self.axis_ranges = new_axis_ranges
    
    def _cleanup_unused_axes(self) -> None:
        """
        Remove unused y-axes and reorganize indices to prevent clutter.
        """
        # Find which axes are still in use
        axes_in_use = set()
        for plot_info in self.plot_data.values():
            axis_idx = plot_info.get('axis_index', 0)
            axes_in_use.add(axis_idx)
        
        # Remove unused axes (except axis 0, which is always kept)
        axes_to_remove = []
        for axis_idx, axis_obj in self.axes_dict.items():
            if axis_idx != 0 and axis_idx not in axes_in_use:
                axes_to_remove.append(axis_idx)
                axis_obj.remove()  # Remove from matplotlib figure
                print(f"Removed unused axis {axis_idx}")
        
        # Clean up dictionaries
        for axis_idx in axes_to_remove:
            if axis_idx in self.axes_dict:
                del self.axes_dict[axis_idx]
            if axis_idx in self.axis_colors:
                del self.axis_colors[axis_idx]
            if axis_idx in self.axis_ranges:
                del self.axis_ranges[axis_idx]
        
        # Reorganize axis indices to maintain sequential ordering
        if len(axes_to_remove) > 0:
            self._reorganize_axis_indices()
    
    def _reorganize_axis_indices(self) -> None:
        """
        Reorganize axis indices to maintain sequential ordering after cleanup.
        
        This ensures axes are numbered 0, 1, 2, ... without gaps.
        """
        # Get current axis indices sorted
        current_indices = sorted(self.axes_dict.keys())
        
        # Create mapping from old indices to new sequential indices
        index_mapping = {}
        new_index = 0
        for old_index in current_indices:
            index_mapping[old_index] = new_index
            new_index += 1
        
        # Update all dictionaries with new indices
        if len(index_mapping) > 1:  # Only reorganize if there are multiple axes
            # Reorganize axes_dict
            new_axes_dict = {}
            for old_idx, new_idx in index_mapping.items():
                new_axes_dict[new_idx] = self.axes_dict[old_idx]
            self.axes_dict = new_axes_dict
            
            # Reorganize axis_colors
            new_axis_colors = {}
            for old_idx, new_idx in index_mapping.items():
                if old_idx in self.axis_colors:
                    new_axis_colors[new_idx] = self.axis_colors[old_idx]
            self.axis_colors = new_axis_colors
            
            # Reorganize axis_ranges
            new_axis_ranges = {}
            for old_idx, new_idx in index_mapping.items():
                if old_idx in self.axis_ranges:
                    new_axis_ranges[new_idx] = self.axis_ranges[old_idx]
            self.axis_ranges = new_axis_ranges
            
            # Update channel_to_axis mappings
            for channel_name in self.channel_to_axis:
                old_axis_idx = self.channel_to_axis[channel_name]
                if old_axis_idx in index_mapping:
                    self.channel_to_axis[channel_name] = index_mapping[old_axis_idx]
            
            # Update plot_data axis indices
            for plot_info in self.plot_data.values():
                old_axis_idx = plot_info.get('axis_index', 0)
                if old_axis_idx in index_mapping:
                    plot_info['axis_index'] = index_mapping[old_axis_idx]
            
            print(f"Reorganized axes: {dict(index_mapping)} -> sequential indices")
    
    def _create_axes_for_groups(self, axis_groups: Dict[int, List[str]], 
                               channel_data_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Create necessary y-axes upfront based on grouping results.
        
        Args:
            axis_groups: Dictionary mapping axis indices to lists of channel names
            channel_data_dict: Dictionary containing all channel plot data
        """
        for axis_idx, channel_list in axis_groups.items():
            if axis_idx == 0:
                # Axis 0 always exists (main axis)
                continue
            
            # Get representative color from first channel in this group
            if channel_list:
                first_channel_name = channel_list[0]
                if first_channel_name in channel_data_dict:
                    color = channel_data_dict[first_channel_name]['color']
                else:
                    color = self.get_next_color()
            else:
                color = self.get_next_color()
            
            # Create the axis
            new_axis = self._create_additional_yaxis(axis_idx, color)
            print(f"Created axis {axis_idx} for channels: {channel_list}")
    
    def _apply_engineering_scaling_to_all_axes(self) -> None:
        """
        Apply engineering scaling to all axes with optimal screen utilization.
        
        This method calculates combined ranges for each axis and applies
        engineering-friendly scaling with 90% screen utilization.
        """
        print("Applying engineering scaling to all axes")
        
        # Calculate combined ranges for each axis
        for axis_idx in self.axes_dict.keys():
            # Get all channels assigned to this axis
            channels_for_axis = [
                name for name, plot_info in self.plot_data.items()
                if plot_info.get('axis_index', 0) == axis_idx
            ]
            
            if channels_for_axis:
                # Calculate combined voltage range for this axis
                all_mins = []
                all_maxs = []
                
                for channel_name in channels_for_axis:
                    if channel_name in self.plot_data:
                        voltage_range = self.plot_data[channel_name]['voltage_range']
                        all_mins.append(voltage_range[0])
                        all_maxs.append(voltage_range[1])
                
                if all_mins and all_maxs:
                    combined_range = (min(all_mins), max(all_maxs))
                    self.axis_ranges[axis_idx] = combined_range
                    
                    print(f"Axis {axis_idx} combined range: {combined_range} for channels {channels_for_axis}")
        
        print(f"Final axis ranges: {self.axis_ranges}")
    
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
    
    def zoom_to_fit_all_axes(self) -> None:
        """
        Auto-scale all y-axes to fit all visible data with engineering-friendly scaling.
        
        This method recalculates the optimal ranges for all axes based on currently
        visible channels and applies engineering scaling to each axis independently.
        """
        print("Zoom to fit all axes - called from auto-range for processed data")
        
        if not self.plot_data:
            print("No plot data available for zoom to fit")
            return
        
        # Recalculate axis ranges based on visible channels
        self._recalculate_axis_ranges()
        
        # Apply engineering scaling to all axes
        self._apply_engineering_scaling_to_all_axes()
        
        # Update plot appearance
        self.update_plot_appearance()
        
        print(f"Zoom to fit complete. Axis ranges: {self.axis_ranges}")
    
    def reset_all_axes_scales(self) -> None:
        """
        Reset all Y-axes scaling to auto-scaled engineering values.
        
        This method resets all channel scaling factors to 1.0, restores original data,
        and recalculates optimal engineering-friendly axis ranges for all axes.
        """
        print("Reset all axes scales")
        
        if not self.plot_data:
            print("No plot data available for scale reset")
            return
        
        # Reset all channel scale factors to 1.0
        for channel_name in self.channel_scale_factors:
            self.channel_scale_factors[channel_name] = 1.0
        
        # Restore original data for all channels
        for channel_name, plot_info in self.plot_data.items():
            if channel_name in self.original_data:
                original_data = self.original_data[channel_name]
                line = plot_info['line']
                line.set_ydata(original_data)
                
                # Recalculate voltage range based on original data
                voltage_range = self._calculate_channel_range(original_data)
                plot_info['voltage_range'] = voltage_range
                
                # Remove any stored scaled data
                if 'scaled_data' in plot_info:
                    del plot_info['scaled_data']
        
        # Recalculate axis ranges based on reset data
        self._recalculate_axis_ranges()
        
        # Apply engineering scaling to all axes
        self._apply_engineering_scaling_to_all_axes()
        
        # Update plot appearance
        self.update_plot_appearance()
        
        print(f"Scale reset complete. Axis ranges: {self.axis_ranges}")

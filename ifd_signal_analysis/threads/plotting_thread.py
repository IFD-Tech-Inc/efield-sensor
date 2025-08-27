#!/usr/bin/env python3
"""
Background plotting thread for IFD Signal Analysis Utility.

This module contains the PlottingThread class responsible for preparing waveform
plot data in the background to prevent GUI freezing during rendering operations.
"""

import traceback
from typing import Dict, Optional, Any

from PyQt6.QtCore import QThread, pyqtSignal

from ..utils.constants import (
    AVAILABLE_PLOT_COLORS, 
    PLOT_PROGRESS_STAGES,
    WARNING_NO_ENABLED_CHANNELS,
    ChannelDataDict
)

# Import will be handled at runtime to avoid circular dependencies
try:
    from siglent_parser import ChannelData
    SIGLENT_PARSER_AVAILABLE = True
except ImportError:
    SIGLENT_PARSER_AVAILABLE = False
    ChannelData = None


class PlottingThread(QThread):
    """
    Background thread for plotting waveform data to prevent GUI freezing during rendering.
    
    This thread handles matplotlib operations off the main UI thread by preparing all
    necessary plot data (time arrays, colors, metadata) in the background. The prepared
    data is then sent back to the main thread for actual rendering.
    
    Attributes:
        progress: Signal emitting status messages during processing
        progress_percentage: Signal emitting progress percentage and messages  
        channel_plotted: Signal emitted when individual channel data is ready
        finished: Signal emitted when all processing is complete
        error: Signal emitted when an error occurs
    """
    
    # Signals for communication with main thread
    progress = pyqtSignal(str)  # Status message
    progress_percentage = pyqtSignal(int, str)  # Progress percentage and message
    channel_plotted = pyqtSignal(str, dict)  # channel_name, plot_data_dict
    finished = pyqtSignal(dict)  # Final plotting results
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, channels_data: ChannelDataDict, parser_header: Any, 
                 overlay_mode: bool = True) -> None:
        """
        Initialize the plotting thread.
        
        Args:
            channels_data: Dictionary mapping channel names to ChannelData objects
            parser_header: SiglentBinaryHeader object containing timing information
            overlay_mode: Whether to prepare data for overlay or separate plots
        """
        super().__init__()
        self.channels_data = channels_data
        self.parser_header = parser_header
        self.overlay_mode = overlay_mode
        self._cancelled = False
        
        # Color management for consistent coloring
        self.colors = AVAILABLE_PLOT_COLORS.copy()
        self.color_index = 0
        
    def cancel(self) -> None:
        """Cancel the plotting operation."""
        self._cancelled = True
        
    def is_cancelled(self) -> bool:
        """
        Check if the operation was cancelled.
        
        Returns:
            True if the operation was cancelled, False otherwise
        """
        return self._cancelled
        
    def get_next_color(self) -> str:
        """
        Get the next color in the cycle.
        
        Returns:
            Color name from the matplotlib color cycle
        """
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        return color
        
    def run(self) -> None:
        """
        Execute the plotting preparation in the background.
        
        This method processes each enabled channel to prepare plot data including
        time arrays, voltage data, colors, and metadata. Emits progress signals
        and handles cancellation gracefully.
        """
        try:
            if not self.channels_data:
                self.error.emit("No channel data provided for plotting")
                return
                
            # Filter enabled channels with data
            enabled_channels = {
                name: data for name, data in self.channels_data.items()
                if data.enabled and len(data.voltage_data) > 0
            }
            
            if not enabled_channels:
                self.error.emit(WARNING_NO_ENABLED_CHANNELS)
                return
                
            total_channels = len(enabled_channels)
            processed_channels = 0
            plot_results = {}
            
            self.progress_percentage.emit(
                PLOT_PROGRESS_STAGES['START'], 
                "Preparing plot data..."
            )
            self.progress.emit("Starting plot data preparation...")
            
            # Process each channel
            for channel_name, channel_data in enabled_channels.items():
                if self._cancelled:
                    return
                    
                processed_channels += 1
                
                # Calculate progress between PROCESS_MIN and PROCESS_MAX
                progress_range = (PLOT_PROGRESS_STAGES['PROCESS_MAX'] - 
                                PLOT_PROGRESS_STAGES['PROCESS_MIN'])
                progress = int((processed_channels / total_channels) * progress_range) + \
                          PLOT_PROGRESS_STAGES['PROCESS_MIN']
                
                self.progress_percentage.emit(
                    progress, 
                    f"Processing channel {channel_name} ({processed_channels}/{total_channels})..."
                )
                self.progress.emit(f"Processing channel {channel_name}...")
                
                # Prepare plot data for this channel
                try:
                    plot_data = self._prepare_channel_plot_data(channel_name, channel_data)
                    if plot_data:
                        plot_results[channel_name] = plot_data
                        
                        # Emit individual channel completion for progressive rendering
                        self.channel_plotted.emit(channel_name, plot_data)
                        
                except Exception as e:
                    print(f"Warning: Failed to process channel {channel_name}: {e}")
                    continue
                    
            if self._cancelled:
                return
                
            # Finalize results
            self.progress_percentage.emit(
                PLOT_PROGRESS_STAGES['FINALIZE'], 
                "Finalizing plot data..."
            )
            
            result = {
                'plot_data': plot_results,
                'total_channels': len(plot_results),
                'overlay_mode': self.overlay_mode
            }
            
            self.progress_percentage.emit(
                PLOT_PROGRESS_STAGES['COMPLETE'], 
                "Plot data preparation completed"
            )
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = (f"Failed to prepare plot data: {str(e)}\n"
                        f"{traceback.format_exc()}")
            self.error.emit(error_msg)
            
    def _prepare_channel_plot_data(self, channel_name: str, channel_data: Any) -> Optional[Dict[str, Any]]:
        """
        Prepare plot data for a single channel without actual rendering.
        
        This method performs the computationally expensive operations (like time array
        calculation) in the background thread, preparing all data needed for plotting
        in the main thread.
        
        Args:
            channel_name: Name of the channel (e.g., 'C1', 'C2')
            channel_data: ChannelData object containing waveform data
            
        Returns:
            Dictionary containing prepared plot data, or None if preparation failed
        """
        if len(channel_data.voltage_data) == 0:
            return None
            
        # Calculate time array - this is often the expensive operation
        try:
            time_array = channel_data.get_time_array(
                self.parser_header.time_div,
                self.parser_header.time_delay, 
                self.parser_header.sample_rate,
                self.parser_header.hori_div_num
            )
        except Exception as e:
            print(f"Error calculating time array for {channel_name}: {e}")
            return None
        
        if len(time_array) == 0:
            return None
            
        # Get color for this channel
        color = self.get_next_color()
        
        # Prepare all data needed for plotting
        plot_data = {
            'channel_name': channel_name,
            'time_array': time_array,
            'voltage_data': channel_data.voltage_data,
            'color': color,
            'visible': True,
            'channel_data': channel_data  # Keep reference for info display
        }
        
        return plot_data

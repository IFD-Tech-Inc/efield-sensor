#!/usr/bin/env python3
"""
Channel list widget for IFD Signal Analysis Utility.

This module contains the ChannelListWidget class for managing channel visibility
and properties in the user interface.
"""

from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QWidget, QHBoxLayout, 
    QCheckBox, QLabel, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette

from ..utils.constants import (
    CHANNEL_ITEM_MARGINS,
    CHANNEL_FONT_FAMILY,
    CHANNEL_FONT_SIZE
)

# Import will be handled at runtime to avoid circular dependencies
try:
    from siglent_parser import ChannelData
    SIGLENT_PARSER_AVAILABLE = True
except ImportError:
    SIGLENT_PARSER_AVAILABLE = False
    ChannelData = None


class ChannelListWidget(QListWidget):
    """
    Custom QListWidget for managing channel visibility and properties.
    
    This widget displays a list of loaded channels with checkboxes for visibility
    control and context menu options for channel operations. Each item contains
    channel information including voltage scale and sample count.
    
    Attributes:
        channel_visibility_changed: Signal emitted when channel visibility changes
        channel_removed: Signal emitted when a channel is removed
    """
    
    # Signals for communication with parent components
    channel_visibility_changed = pyqtSignal(str, bool)  # channel_name, visible
    channel_removed = pyqtSignal(str)  # channel_name
    channel_selected = pyqtSignal(str)  # channel_name
    channel_inverted = pyqtSignal(str, bool)  # channel_name, inverted
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the channel list widget.
        
        Args:
            parent: Parent widget, if any
        """
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Channel selection state
        self.selected_channel: Optional[str] = None
        
        # Channel inversion state tracking
        self.inverted_channels: Dict[str, bool] = {}
        
        # Connect item clicks to selection handler
        self.itemClicked.connect(self._on_item_clicked)
        
    def add_channel(self, channel_name: str, channel_data: 'ChannelData', 
                   visible: bool = True) -> None:
        """
        Add a new channel to the list with checkbox for visibility control.
        
        Creates a custom widget containing a checkbox and channel information
        label, then adds it as a list item.
        
        Args:
            channel_name: Name of the channel (e.g., 'C1', 'C2')
            channel_data: ChannelData object containing channel information
            visible: Initial visibility state of the channel
        """
        # Create the list item
        item = QListWidgetItem(self)
        
        # Create a custom widget for the item
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(*CHANNEL_ITEM_MARGINS)
        
        # Checkbox for visibility control
        checkbox = QCheckBox()
        checkbox.setChecked(visible)
        checkbox.stateChanged.connect(
            lambda state, ch=channel_name: self.channel_visibility_changed.emit(
                ch, state == Qt.CheckState.Checked.value
            )
        )
        
        # Channel information label
        info_text = self._format_channel_info(channel_name, channel_data)
        label = QLabel(info_text)
        label.setFont(QFont(CHANNEL_FONT_FAMILY, CHANNEL_FONT_SIZE))
        
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
        
    def _format_channel_info(self, channel_name: str, channel_data: 'ChannelData') -> str:
        """
        Format channel information for display in the list.
        
        Args:
            channel_name: Name of the channel
            channel_data: ChannelData object containing channel information
            
        Returns:
            Formatted string containing channel information
        """
        # Add inversion indicator if channel is inverted
        is_inverted = self.inverted_channels.get(channel_name, False)
        prefix = "⇅ " if is_inverted else ""
        
        info_text = f"{prefix}{channel_name}"
        
        # Add voltage division information if available
        if hasattr(channel_data, 'volt_div_val') and channel_data.volt_div_val:
            try:
                volt_div = channel_data.volt_div_val.get_scaled_value()
                unit = channel_data.volt_div_val.get_unit_string()
                info_text += f" ({volt_div:.3f}{unit}/div)"
            except Exception:
                # Gracefully handle any formatting errors
                pass
                
        # Add sample count information
        if hasattr(channel_data, 'voltage_data') and len(channel_data.voltage_data) > 0:
            sample_count = len(channel_data.voltage_data)
            info_text += f" [{sample_count} samples]"
        
        # Add inversion indicator text if channel is inverted
        if is_inverted:
            info_text += " (inverted)"
            
        return info_text
        
    def clear_all_channels(self) -> None:
        """Remove all channels from the list."""
        self.clear()
        # Clear inversion state tracking
        self.inverted_channels.clear()
        
    def _show_context_menu(self, position: 'QPoint') -> None:
        """
        Show right-click context menu for channel operations.
        
        Args:
            position: Position where the context menu was requested
        """
        item = self.itemAt(position)
        if item is not None:
            channel_name = item.data(Qt.ItemDataRole.UserRole)
            is_inverted = self.inverted_channels.get(channel_name, False)
            
            menu = QMenu(self)
            
            # Inversion toggle action
            invert_text = "Un-invert Channel" if is_inverted else "Invert Channel"
            invert_action = menu.addAction(invert_text)
            invert_action.triggered.connect(
                lambda: self._toggle_channel_inversion(channel_name)
            )
            
            menu.addSeparator()
            
            # Remove action
            remove_action = menu.addAction(f"Remove {channel_name}")
            remove_action.triggered.connect(
                lambda: self._remove_channel_item(channel_name)
            )
            
            menu.exec(self.mapToGlobal(position))
            
    def _toggle_channel_inversion(self, channel_name: str) -> None:
        """
        Toggle the inversion state of a channel and update its voltage data.
        
        Args:
            channel_name: Name of the channel to invert/un-invert
        """
        try:
            # Toggle the inversion state
            current_state = self.inverted_channels.get(channel_name, False)
            new_state = not current_state
            self.inverted_channels[channel_name] = new_state
            
            # Get the channel data from parent's loaded_data
            # We need to access the parent window's data
            parent_window = self.parent()
            while parent_window and not hasattr(parent_window, 'loaded_data'):
                parent_window = parent_window.parent()
                
            if parent_window and hasattr(parent_window, 'loaded_data'):
                # Try to find channel in loaded_data first (original channels)
                channel_data = None
                for file_key, channels in parent_window.loaded_data.items():
                    if channel_name in channels:
                        channel_data = channels[channel_name]
                        break
                
                if channel_data is not None:
                    # Found in loaded_data - invert the original channel data
                    channel_data.voltage_data *= -1
                    print(f"Inverted original channel {channel_name}")
                else:
                    # Not found in loaded_data - might be a processed channel
                    # Try to find it in the active plot canvas
                    if hasattr(parent_window, 'plot_canvas') and parent_window.plot_canvas:
                        canvas = parent_window.plot_canvas
                        if hasattr(canvas, 'plot_data') and channel_name in canvas.plot_data:
                            plot_info = canvas.plot_data[channel_name]
                            if 'voltage_data' in plot_info:
                                # Invert the plot data directly
                                plot_info['voltage_data'] *= -1
                                print(f"Inverted processed channel {channel_name} in plot data")
                            elif hasattr(plot_info.get('line'), 'get_ydata'):
                                # Get data from matplotlib line, invert it, and update
                                line = plot_info['line']
                                ydata = line.get_ydata()
                                inverted_data = ydata * -1
                                line.set_ydata(inverted_data)
                                # Also update the stored plot data if available
                                if 'voltage_data' in plot_info:
                                    plot_info['voltage_data'] = inverted_data
                                print(f"Inverted processed channel {channel_name} via matplotlib line")
                            else:
                                print(f"Warning: Could not find voltage data for processed channel {channel_name}")
                                return
                        else:
                            print(f"Warning: Could not find channel {channel_name} in plot data")
                            return
                    else:
                        print(f"Warning: Could not find plot canvas for processed channel {channel_name}")
                        return
                
                # Update visual indicator
                self._update_channel_visual(channel_name)
                
                # Emit signal to notify parent components
                self.channel_inverted.emit(channel_name, new_state)
            else:
                print("Warning: Could not find parent window with loaded_data")
                
        except Exception as e:
            print(f"Error in _toggle_channel_inversion for {channel_name}: {e}")
    
    def _remove_channel_item(self, channel_name: str) -> None:
        """
        Remove a specific channel from the list.
        
        Args:
            channel_name: Name of the channel to remove
        """
        for i in range(self.count()):
            item = self.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == channel_name:
                self.takeItem(i)
                # Clean up inversion state
                if channel_name in self.inverted_channels:
                    del self.inverted_channels[channel_name]
                self.channel_removed.emit(channel_name)
                break
                
    def get_visible_channels(self) -> list:
        """
        Get list of currently visible channel names.
        
        Returns:
            List of channel names that are currently visible
        """
        visible_channels = []
        for i in range(self.count()):
            item = self.item(i)
            if item:
                widget = self.itemWidget(item)
                if widget:
                    checkbox = widget.findChild(QCheckBox)
                    if checkbox and checkbox.isChecked():
                        channel_name = item.data(Qt.ItemDataRole.UserRole)
                        visible_channels.append(channel_name)
        return visible_channels
    
    def set_channel_visibility(self, channel_name: str, visible: bool) -> None:
        """
        Programmatically set the visibility of a specific channel.
        
        Args:
            channel_name: Name of the channel to modify
            visible: Desired visibility state
        """
        for i in range(self.count()):
            item = self.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == channel_name:
                widget = self.itemWidget(item)
                if widget:
                    checkbox = widget.findChild(QCheckBox)
                    if checkbox:
                        checkbox.setChecked(visible)
                break
    
    def get_channel_count(self) -> int:
        """
        Get the total number of channels in the list.
        
        Returns:
            Total number of channels
        """
        return self.count()
    
    def get_visible_channel_count(self) -> int:
        """
        Get the number of currently visible channels.
        
        Returns:
            Number of visible channels
        """
        return len(self.get_visible_channels())
    
    # ==================== Channel Selection Methods ====================
    
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """
        Handle item clicks for channel selection.
        
        Args:
            item: The clicked list item
        """
        if item:
            channel_name = item.data(Qt.ItemDataRole.UserRole)
            self.select_channel(channel_name)
    
    def select_channel(self, channel_name: str) -> None:
        """
        Select a specific channel and update visual feedback.
        
        Args:
            channel_name: Name of the channel to select
        """
        if self.selected_channel != channel_name:
            # Clear previous selection visual feedback
            self._clear_selection_visual()
            
            # Set new selection
            self.selected_channel = channel_name
            
            # Apply visual feedback to new selection
            self._apply_selection_visual(channel_name)
            
            # Emit selection signal
            self.channel_selected.emit(channel_name)
    
    def get_selected_channel(self) -> Optional[str]:
        """
        Get the currently selected channel name.
        
        Returns:
            Name of the selected channel, or None if no selection
        """
        return self.selected_channel
    
    def clear_selection(self) -> None:
        """
        Clear the current channel selection.
        """
        if self.selected_channel is not None:
            self._clear_selection_visual()
            self.selected_channel = None
    
    def _apply_selection_visual(self, channel_name: str) -> None:
        """
        Apply visual feedback for the selected channel.
        
        Args:
            channel_name: Name of the channel to highlight
        """
        item = self._find_item_by_channel(channel_name)
        if item:
            widget = self.itemWidget(item)
            if widget:
                # Set bold font for the label
                label = widget.findChild(QLabel)
                if label:
                    font = label.font()
                    font.setBold(True)
                    label.setFont(font)
                    
                # Set background color to indicate selection
                widget.setStyleSheet(
                    "QWidget { background-color: palette(highlight); }"
                    "QLabel { color: palette(highlighted-text); }"
                )
    
    def _clear_selection_visual(self) -> None:
        """
        Clear visual feedback from the previously selected channel.
        """
        if self.selected_channel:
            item = self._find_item_by_channel(self.selected_channel)
            if item:
                widget = self.itemWidget(item)
                if widget:
                    # Reset font weight for the label
                    label = widget.findChild(QLabel)
                    if label:
                        font = label.font()
                        font.setBold(False)
                        label.setFont(font)
                        
                    # Clear custom styling
                    widget.setStyleSheet("")
    
    def _find_item_by_channel(self, channel_name: str) -> Optional[QListWidgetItem]:
        """
        Find the list item corresponding to a channel name.
        
        Args:
            channel_name: Name of the channel to find
            
        Returns:
            The QListWidgetItem if found, None otherwise
        """
        for i in range(self.count()):
            item = self.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == channel_name:
                return item
        return None
    
    def _update_channel_visual(self, channel_name: str) -> None:
        """
        Update the visual appearance of a channel to reflect its current state.
        
        Args:
            channel_name: Name of the channel to update
        """
        item = self._find_item_by_channel(channel_name)
        if item:
            widget = self.itemWidget(item)
            if widget:
                label = widget.findChild(QLabel)
                if label:
                    # Get the channel data to regenerate the label text
                    parent_window = self.parent()
                    while parent_window and not hasattr(parent_window, 'loaded_data'):
                        parent_window = parent_window.parent()
                        
                    if parent_window and hasattr(parent_window, 'loaded_data'):
                        # Search through all files for the channel data
                        channel_data = None
                        for file_key, channels in parent_window.loaded_data.items():
                            if channel_name in channels:
                                channel_data = channels[channel_name]
                                break
                        
                        if channel_data is not None:
                            new_text = self._format_channel_info(channel_name, channel_data)
                            label.setText(new_text)
                            
                            # Apply italic font style to inverted channels
                            is_inverted = self.inverted_channels.get(channel_name, False)
                            font = label.font()
                            font.setItalic(is_inverted)
                            label.setFont(font)

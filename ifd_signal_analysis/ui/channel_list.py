#!/usr/bin/env python3
"""
Channel list widget for IFD Signal Analysis Utility.

This module contains the ChannelListWidget class for managing channel visibility
and properties in the user interface.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QListWidget, QListWidgetItem, QWidget, QHBoxLayout, 
    QCheckBox, QLabel, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

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
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the channel list widget.
        
        Args:
            parent: Parent widget, if any
        """
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
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
        info_text = f"{channel_name}"
        
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
            
        return info_text
        
    def clear_all_channels(self) -> None:
        """Remove all channels from the list."""
        self.clear()
        
    def _show_context_menu(self, position: 'QPoint') -> None:
        """
        Show right-click context menu for channel operations.
        
        Args:
            position: Position where the context menu was requested
        """
        item = self.itemAt(position)
        if item is not None:
            channel_name = item.data(Qt.ItemDataRole.UserRole)
            
            menu = QMenu(self)
            remove_action = menu.addAction(f"Remove {channel_name}")
            remove_action.triggered.connect(
                lambda: self._remove_channel_item(channel_name)
            )
            
            menu.exec(self.mapToGlobal(position))
            
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

#!/usr/bin/env python3
"""
Multi-Plot Manager for IFD Signal Analysis Utility.

This module provides the MultiPlotManager class that coordinates multiple
PlotCanvas instances, manages plot lifecycle, and handles inter-plot communication.
"""

from typing import Dict, Optional, List, Any, Tuple
import uuid
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QPalette

from .plot_canvas import PlotCanvas


class PlotLayoutMode(Enum):
    """Enum for different plot layout arrangements."""
    SINGLE = (1, 1)
    HORIZONTAL_2 = (1, 2)
    VERTICAL_2 = (2, 1)
    GRID_2x2 = (2, 2)
    GRID_2x3 = (2, 3)
    GRID_3x2 = (3, 2)


class PlotInfo:
    """Information container for a managed plot."""
    
    def __init__(self, plot_id: str, canvas: PlotCanvas, title: str = "", toolbar: Optional[Any] = None):
        self.plot_id = plot_id
        self.canvas = canvas
        self.title = title or f"Plot {plot_id}"
        self.is_active = True
        self.created_timestamp = uuid.uuid4().hex  # For ordering
        self.toolbar = toolbar  # NavigationToolbar2QT instance for this plot
        
    def __str__(self) -> str:
        return f"PlotInfo({self.plot_id}, {self.title})"


class MultiPlotManager(QObject):
    """
    Manager for coordinating multiple PlotCanvas instances.
    
    This class provides centralized management of multiple plots including:
    - Plot creation and deletion with unique IDs
    - Layout management for optimal screen utilization
    - Inter-plot communication and data sharing
    - Event coordination between plots
    
    Signals:
        plot_added: Emitted when a new plot is created (plot_id, canvas)
        plot_removed: Emitted when a plot is removed (plot_id)
        plot_selected: Emitted when a plot gains focus (plot_id)
        layout_changed: Emitted when plot layout changes (layout_mode)
    """
    
    # Signals for plot management events
    plot_added = pyqtSignal(str, object)  # plot_id, PlotCanvas
    plot_removed = pyqtSignal(str)  # plot_id
    plot_selected = pyqtSignal(str)  # plot_id
    layout_changed = pyqtSignal(object)  # PlotLayoutMode
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the multi-plot manager.
        
        Args:
            parent: Parent widget for the plot container
        """
        super().__init__()
        self.parent_widget = parent
        
        # Plot storage and management
        self.plots: Dict[str, PlotInfo] = {}
        self.max_plots = 6
        self.active_plot_id: Optional[str] = None
        self.plot_counter = 0
        
        # Layout management
        self.current_layout = PlotLayoutMode.SINGLE
        self.container_widget: Optional[QWidget] = None
        self.plot_layout: Optional[QGridLayout] = None
        
        # Create the container widget
        self._create_container()
        
    def _create_container(self) -> None:
        """Create the main container widget for holding plots."""
        self.container_widget = QWidget(self.parent_widget)
        self.container_widget.setObjectName("MultiPlotContainer")
        
        # Create main layout
        main_layout = QVBoxLayout(self.container_widget)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        
        # Create the plot grid layout
        self.plot_layout = QGridLayout()
        self.plot_layout.setSpacing(2)
        main_layout.addLayout(self.plot_layout)
        
        # Set size policy
        self.container_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        
    def get_container(self) -> QWidget:
        """
        Get the container widget for embedding in parent UI.
        
        Returns:
            QWidget containing all managed plots
        """
        return self.container_widget
        
    def create_plot(self, plot_title: Optional[str] = None) -> Optional[str]:
        """
        Create a new plot canvas.
        
        Args:
            plot_title: Optional custom title for the plot
            
        Returns:
            Plot ID string if successful, None if maximum plots reached
        """
        if len(self.plots) >= self.max_plots:
            return None
            
        # Generate unique plot ID
        self.plot_counter += 1
        plot_id = f"Plot{self.plot_counter}"
        
        # Create plot canvas with ID
        canvas = PlotCanvas(parent=self.container_widget)
        canvas.plot_id = plot_id  # Add plot_id attribute
        
        # Set custom title or default
        title = plot_title or f"Plot {self.plot_counter}"
        
        # Create plot info
        plot_info = PlotInfo(plot_id, canvas, title)
        self.plots[plot_id] = plot_info
        
        # Connect canvas signals with plot identification
        canvas.channel_selected.connect(
            lambda channel: self._on_plot_channel_selected(plot_id, channel)
        )
        
        # Add to layout
        self._add_plot_to_layout(plot_info)
        
        # Update active plot if this is the first
        if self.active_plot_id is None:
            self.active_plot_id = plot_id
            
        # Emit signal
        self.plot_added.emit(plot_id, canvas)
        
        return plot_id
        
    def remove_plot(self, plot_id: str) -> bool:
        """
        Remove a plot canvas.
        
        Args:
            plot_id: ID of the plot to remove
            
        Returns:
            True if plot was removed, False if not found
        """
        if plot_id not in self.plots:
            return False
            
        plot_info = self.plots[plot_id]
        
        # Remove from layout
        self.plot_layout.removeWidget(plot_info.canvas)
        
        # Clean up canvas
        plot_info.canvas.setParent(None)
        plot_info.canvas.deleteLater()
        
        # Remove from tracking
        del self.plots[plot_id]
        
        # Update active plot if needed
        if self.active_plot_id == plot_id:
            if self.plots:
                # Set first available plot as active
                self.active_plot_id = next(iter(self.plots.keys()))
            else:
                self.active_plot_id = None
                
        # Re-arrange layout
        self._update_layout()
        
        # Emit signal
        self.plot_removed.emit(plot_id)
        
        return True
        
    def get_plot_canvas(self, plot_id: str) -> Optional[PlotCanvas]:
        """
        Get the PlotCanvas instance for a specific plot ID.
        
        Args:
            plot_id: ID of the plot
            
        Returns:
            PlotCanvas instance or None if not found
        """
        plot_info = self.plots.get(plot_id)
        return plot_info.canvas if plot_info else None
        
    def get_plot_toolbar(self, plot_id: str) -> Optional[Any]:
        """
        Get the NavigationToolbar instance for a specific plot ID.
        
        Args:
            plot_id: ID of the plot
            
        Returns:
            NavigationToolbar instance or None if not found
        """
        plot_info = self.plots.get(plot_id)
        return plot_info.toolbar if plot_info else None
        
    def set_plot_toolbar(self, plot_id: str, toolbar: Any) -> bool:
        """
        Set the NavigationToolbar instance for a specific plot ID.
        
        Args:
            plot_id: ID of the plot
            toolbar: NavigationToolbar instance
            
        Returns:
            True if successful, False if plot not found
        """
        plot_info = self.plots.get(plot_id)
        if plot_info:
            plot_info.toolbar = toolbar
            return True
        return False
        
    def get_plot_ids(self) -> List[str]:
        """
        Get list of all current plot IDs.
        
        Returns:
            List of plot ID strings
        """
        return list(self.plots.keys())
        
    def get_plot_count(self) -> int:
        """
        Get the current number of plots.
        
        Returns:
            Number of active plots
        """
        return len(self.plots)
        
    def get_active_plot_id(self) -> Optional[str]:
        """
        Get the currently active plot ID.
        
        Returns:
            Active plot ID or None if no plots
        """
        return self.active_plot_id
        
    def set_active_plot(self, plot_id: str) -> bool:
        """
        Set the active plot.
        
        Args:
            plot_id: ID of the plot to make active
            
        Returns:
            True if successful, False if plot not found
        """
        if plot_id in self.plots:
            self.active_plot_id = plot_id
            self.plot_selected.emit(plot_id)
            return True
        return False
        
    def get_max_plots(self) -> int:
        """
        Get the maximum number of plots allowed.
        
        Returns:
            Maximum plot count
        """
        return self.max_plots
        
    def can_add_plot(self) -> bool:
        """
        Check if another plot can be added.
        
        Returns:
            True if under the limit, False otherwise
        """
        return len(self.plots) < self.max_plots
        
    def clear_all_plots(self) -> None:
        """Remove all plots and reset the manager."""
        plot_ids = list(self.plots.keys())
        for plot_id in plot_ids:
            self.remove_plot(plot_id)
            
        self.plot_counter = 0
        self.active_plot_id = None
        
    def get_plot_info(self, plot_id: str) -> Optional[PlotInfo]:
        """
        Get detailed information about a specific plot.
        
        Args:
            plot_id: ID of the plot
            
        Returns:
            PlotInfo object or None if not found
        """
        return self.plots.get(plot_id)
        
    def _add_plot_to_layout(self, plot_info: PlotInfo) -> None:
        """
        Add a plot to the current layout.
        
        Args:
            plot_info: Plot information container
        """
        # Calculate position based on current number of plots
        num_plots = len(self.plots)
        
        # Update layout mode based on number of plots
        self._determine_optimal_layout(num_plots)
        
        # Add to grid layout
        rows, cols = self.current_layout.value
        
        # Calculate grid position
        plot_index = num_plots - 1  # 0-based index
        row = plot_index // cols
        col = plot_index % cols
        
        # Add canvas to layout
        self.plot_layout.addWidget(plot_info.canvas, row, col)
        
        # Emit layout change
        self.layout_changed.emit(self.current_layout)
        
    def _update_layout(self) -> None:
        """Update the layout arrangement for remaining plots."""
        # Clear current layout
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(self.container_widget)
                
        # Re-add all plots
        num_plots = len(self.plots)
        if num_plots == 0:
            self.current_layout = PlotLayoutMode.SINGLE
            return
            
        # Determine optimal layout
        self._determine_optimal_layout(num_plots)
        rows, cols = self.current_layout.value
        
        # Add plots back to layout
        for i, (plot_id, plot_info) in enumerate(self.plots.items()):
            row = i // cols
            col = i % cols
            self.plot_layout.addWidget(plot_info.canvas, row, col)
            
        # Emit layout change
        self.layout_changed.emit(self.current_layout)
        
    def _determine_optimal_layout(self, num_plots: int) -> None:
        """
        Determine the best layout arrangement for given number of plots.
        
        Args:
            num_plots: Number of plots to arrange
        """
        if num_plots <= 1:
            self.current_layout = PlotLayoutMode.SINGLE
        elif num_plots == 2:
            self.current_layout = PlotLayoutMode.HORIZONTAL_2
        elif num_plots <= 4:
            self.current_layout = PlotLayoutMode.GRID_2x2
        elif num_plots <= 6:
            self.current_layout = PlotLayoutMode.GRID_2x3
        else:
            # Fallback for more than 6 plots (shouldn't happen due to limit)
            self.current_layout = PlotLayoutMode.GRID_3x2
            
    def _on_plot_channel_selected(self, plot_id: str, channel_name: str) -> None:
        """
        Handle channel selection from a specific plot.
        
        Args:
            plot_id: ID of the plot where selection occurred
            channel_name: Name of the selected channel
        """
        # Update active plot
        self.set_active_plot(plot_id)
        
        # Could emit additional signals here for inter-plot coordination
        
    def get_layout_description(self) -> str:
        """
        Get a human-readable description of the current layout.
        
        Returns:
            Layout description string
        """
        layout_descriptions = {
            PlotLayoutMode.SINGLE: "Single Plot",
            PlotLayoutMode.HORIZONTAL_2: "2 Plots (Horizontal)",
            PlotLayoutMode.VERTICAL_2: "2 Plots (Vertical)", 
            PlotLayoutMode.GRID_2x2: "4 Plots (2×2 Grid)",
            PlotLayoutMode.GRID_2x3: "6 Plots (2×3 Grid)",
            PlotLayoutMode.GRID_3x2: "6 Plots (3×2 Grid)"
        }
        return layout_descriptions.get(self.current_layout, "Unknown Layout")


class PlotContainer(QWidget):
    """
    Widget container for the MultiPlotManager.
    
    This widget serves as a wrapper around MultiPlotManager to provide
    a clean interface for embedding multiple plots in the main UI.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the plot container.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create the plot manager
        self.plot_manager = MultiPlotManager(self)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Add the plot manager's container
        layout.addWidget(self.plot_manager.get_container())
        
        # Connect signals for external access
        self.plot_manager.plot_added.connect(self._on_plot_added)
        self.plot_manager.plot_removed.connect(self._on_plot_removed)
        self.plot_manager.plot_selected.connect(self._on_plot_selected)
        
    def get_manager(self) -> MultiPlotManager:
        """
        Get the underlying plot manager.
        
        Returns:
            MultiPlotManager instance
        """
        return self.plot_manager
        
    def _on_plot_added(self, plot_id: str, canvas: PlotCanvas) -> None:
        """Handle plot addition (placeholder for future features)."""
        pass
        
    def _on_plot_removed(self, plot_id: str) -> None:
        """Handle plot removal (placeholder for future features)."""
        pass
        
    def _on_plot_selected(self, plot_id: str) -> None:
        """Handle plot selection (placeholder for future features)."""
        pass

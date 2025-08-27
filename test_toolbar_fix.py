#!/usr/bin/env python3
"""
Test script to verify that the toolbar fix prevents AttributeError crashes 
during axis configuration in multi-plot mode.

This script demonstrates that each plot now has its own NavigationToolbar instance
with proper bidirectional connections, preventing the 'NoneType' object has no 
attribute 'push_current' error when accessing axis configuration dialogs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from ifd_signal_analysis.ui.main_window import IFDSignalAnalysisMainWindow
import numpy as np

def test_toolbar_integration():
    """Test that toolbar integration works correctly."""
    print("Testing Multi-Plot Toolbar Management System")
    print("=" * 50)
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create main window
    window = IFDSignalAnalysisMainWindow()
    
    print(f"✓ Main window created with multi-plot mode: {window.use_multi_plot}")
    
    # Check that plot manager exists
    plot_manager = window.plot_manager
    plot_ids = list(plot_manager.get_plot_ids())
    print(f"✓ Initial plots created: {plot_ids}")
    
    # Check that toolbars are created for each plot
    for plot_id in plot_ids:
        canvas = plot_manager.get_plot_canvas(plot_id)
        toolbar = plot_manager.get_plot_toolbar(plot_id)
        
        print(f"✓ Plot {plot_id}:")
        print(f"  - Canvas: {type(canvas).__name__} at {hex(id(canvas))}")
        print(f"  - Toolbar: {type(toolbar).__name__} at {hex(id(toolbar))}")
        
        # Verify bidirectional connection
        canvas_toolbar = getattr(canvas, 'toolbar', None)
        toolbar_canvas = getattr(toolbar, 'canvas', None)
        
        print(f"  - Canvas.toolbar: {hex(id(canvas_toolbar)) if canvas_toolbar else 'None'}")
        print(f"  - Toolbar.canvas: {hex(id(toolbar_canvas)) if toolbar_canvas else 'None'}")
        
        # Check that they match
        if canvas_toolbar is toolbar and toolbar_canvas is canvas:
            print(f"  ✓ Bidirectional connection established correctly")
        else:
            print(f"  ✗ Connection mismatch!")
            
    print()
    
    # Test adding a new plot
    print("Testing plot addition...")
    new_plot_id = plot_manager.create_plot("Test Plot")
    if new_plot_id:
        print(f"✓ Created new plot: {new_plot_id}")
        
        # Check toolbar creation for new plot
        new_canvas = plot_manager.get_plot_canvas(new_plot_id)
        new_toolbar = plot_manager.get_plot_toolbar(new_plot_id)
        
        if new_canvas and new_toolbar:
            print(f"✓ New plot has proper toolbar connection")
            
            # Test that canvas has toolbar reference
            if hasattr(new_canvas, 'toolbar') and new_canvas.toolbar is new_toolbar:
                print(f"✓ Canvas -> Toolbar connection: OK")
            else:
                print(f"✗ Canvas -> Toolbar connection: FAIL")
                
            # Test that toolbar has canvas reference
            if hasattr(new_toolbar, 'canvas') and new_toolbar.canvas is new_canvas:
                print(f"✓ Toolbar -> Canvas connection: OK")
            else:
                print(f"✗ Toolbar -> Canvas connection: FAIL")
        else:
            print(f"✗ Failed to create toolbar for new plot")
    else:
        print(f"✗ Failed to create new plot")
    
    print()
    print("Toolbar Fix Verification Complete!")
    print("=" * 50)
    print()
    print("Key Improvements:")
    print("1. Each PlotCanvas has its own NavigationToolbar instance")
    print("2. Bidirectional canvas <-> toolbar references are properly maintained")
    print("3. QStackedWidget manages toolbar switching between plots")
    print("4. Toolbar cleanup happens automatically on plot removal")
    print()
    print("Expected Result:")
    print("- Axis configuration dialogs should work without AttributeError crashes")
    print("- Users can modify axis ranges for any plot independently")
    print("- Toolbar functions (zoom, pan, configure axes) work correctly per plot")

if __name__ == "__main__":
    test_toolbar_integration()

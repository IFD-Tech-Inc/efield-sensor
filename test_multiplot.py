#!/usr/bin/env python3
"""
Test script to verify multi-plot functionality.

This script creates the main window with multiple plots and adds some test data
to make the plots clearly visible.
"""

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

try:
    from ifd_signal_analysis.ui.main_window import IFDSignalAnalysisMainWindow
    
    def add_test_data_to_plot(canvas, plot_id):
        """Add some test data to a plot canvas to make it visible."""
        # Generate test waveform data
        t = np.linspace(0, 1, 1000)
        
        if plot_id == "Plot1":
            # Sine wave
            y = np.sin(2 * np.pi * 5 * t)
            color = 'blue'
        elif plot_id == "Plot2":
            # Cosine wave
            y = np.cos(2 * np.pi * 3 * t)
            color = 'red'
        elif plot_id == "Plot3":
            # Square wave
            y = np.sign(np.sin(2 * np.pi * 2 * t))
            color = 'green'
        elif plot_id == "Plot4":
            # Sawtooth wave
            y = 2 * (t * 4 - np.floor(t * 4 + 0.5))
            color = 'orange'
        else:
            # Random noise
            y = np.random.normal(0, 0.5, len(t))
            color = 'purple'
        
        # Plot the test data
        canvas.ax.clear()
        canvas.ax.plot(t, y, color=color, linewidth=2, label=f'{plot_id} Test Data')
        canvas.ax.set_xlabel('Time (s)')
        canvas.ax.set_ylabel('Amplitude')
        canvas.ax.set_title(f'{plot_id} - Test Waveform')
        canvas.ax.legend()
        canvas.ax.grid(True, alpha=0.3)
        canvas.draw()
    
    def main():
        app = QApplication(sys.argv)
        
        try:
            # Create main window
            window = IFDSignalAnalysisMainWindow()
            print("✅ Main window created successfully")
            
            # Add additional plots
            window.add_new_plot()  # Plot2
            window.add_new_plot()  # Plot3
            window.add_new_plot()  # Plot4
            
            plot_count = window.plot_manager.get_plot_count()
            plot_ids = window.plot_manager.get_plot_ids()
            print(f"✅ Created {plot_count} plots: {plot_ids}")
            
            # Get layout info
            layout_desc = window.plot_manager.get_layout_description()
            print(f"✅ Layout: {layout_desc}")
            
            # Add test data to each plot to make them visible
            for plot_id in plot_ids:
                canvas = window.plot_manager.get_plot_canvas(plot_id)
                if canvas:
                    add_test_data_to_plot(canvas, plot_id)
                    print(f"✅ Added test data to {plot_id}")
            
            # Show the window
            window.show()
            window.resize(1200, 800)  # Make it large enough to see all plots
            
            # Show a message to the user
            msg = QMessageBox()
            msg.setWindowTitle("Multi-Plot Test")
            msg.setText(f"Multi-plot system is working!\n\n"
                       f"Created {plot_count} plots with test data.\n"
                       f"Layout: {layout_desc}\n\n"
                       f"You should see {plot_count} different colored waveforms "
                       f"arranged in a grid.\n\n"
                       f"Click OK to continue or close the window to exit.")
            msg.setIcon(QMessageBox.Icon.Information)
            
            # Center the message box on the main window
            msg.exec()
            
            # If we get here, user clicked OK, so start the event loop
            print("🚀 Starting application event loop...")
            print("📊 You should see multiple plots in the main window")
            print("🔧 Try using the toolbar buttons to add/remove plots")
            print("⚙️  Try the 'Configure Processing' button when you have 2+ plots")
            
            sys.exit(app.exec())
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = QMessageBox()
            error_msg.setWindowTitle("Error")
            error_msg.setText(f"An error occurred:\n\n{str(e)}")
            error_msg.setIcon(QMessageBox.Icon.Critical)
            error_msg.exec()
            
            return 1
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the efield-sensor directory")
    print("and that PyQt6 is installed in your virtual environment.")

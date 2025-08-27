#!/usr/bin/env python3
"""
Entry point for IFD Signal Analysis Utility.

This module serves as the main entry point when running the package with
`python -m ifd_signal_analysis`. It handles application initialization,
splash screen display, and main window creation.
"""

import sys
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

from .utils.constants import (
    APP_NAME, APP_VERSION, APP_ORGANIZATION, APP_DOMAIN,
    SPLASH_SCREEN_DISPLAY_TIME, SPLASH_SCREEN_STEPS,
    ERROR_SPLASH_SCREEN_UNAVAILABLE
)

# Import progress dialog availability check
try:
    from progressdialog import LoadingProgressDialog
    PROGRESS_DIALOG_AVAILABLE = True
except ImportError:
    PROGRESS_DIALOG_AVAILABLE = False

# Import splash screen with availability check
try:
    from splashscreen import IFDSplashScreen
    SPLASH_SCREEN_AVAILABLE = True
except ImportError:
    SPLASH_SCREEN_AVAILABLE = False
    print(ERROR_SPLASH_SCREEN_UNAVAILABLE)


def main() -> None:
    """
    Main application entry point.
    
    Initializes the Qt application, displays splash screen (if available),
    creates the main window, and starts the event loop.
    """
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)
    app.setOrganizationDomain(APP_DOMAIN)
    
    # Set application style and theme
    app.setStyle('Fusion')  # Modern cross-platform style
    
    # Create and show splash screen if available
    splash = None
    if SPLASH_SCREEN_AVAILABLE:
        try:
            splash = IFDSplashScreen()
            splash.show()
            splash.show_message("Loading application...")
            app.processEvents()
            
            # Allow splash to be visible for a moment
            QTimer.singleShot(SPLASH_SCREEN_DISPLAY_TIME, lambda: None)
            for i in range(SPLASH_SCREEN_STEPS):  # Brief delay to show splash
                app.processEvents()
                QTimer.singleShot(100, lambda: None)
                
        except Exception as e:
            print(f"Warning: Could not display splash screen: {e}")
            splash = None
    
    try:
        # Import main window here to avoid circular dependencies
        from .ui.main_window import IFDSignalAnalysisMainWindow
        
        # Create main window
        if splash:
            splash.show_message("Initializing main window...")
            app.processEvents()
            
        window = IFDSignalAnalysisMainWindow()
        
        # Close splash and show main window
        if splash:
            splash.finish_loading(window)
        
        window.show()
        
        # Run the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        # Close splash if there's an error
        if splash:
            splash.close()
            
        # Handle any uncaught exceptions
        error_msg = (
            f'An unexpected error occurred:\n\n{str(e)}\n\n'
            f'{traceback.format_exc()}'
        )
        
        QMessageBox.critical(
            None, 
            'Application Error', 
            error_msg
        )
        sys.exit(1)


if __name__ == '__main__':
    main()

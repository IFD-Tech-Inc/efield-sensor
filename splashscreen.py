#!/usr/bin/env python3
"""
Splash Screen for IFD Signal Analysis Utility

A custom splash screen that displays during application startup with
the application name and development notice.

Author: Assistant
Version: 1.0.0
Dependencies: PyQt6
"""

from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QFont, QFontMetrics, QColor, QLinearGradient, QBrush, QPen
from PyQt6.QtCore import QRect


class IFDSplashScreen(QSplashScreen):
    """
    Custom splash screen for the IFD Signal Analysis Utility.
    
    Creates a branded splash screen with gradient background, main title,
    and development notice that displays during application startup.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create the splash screen pixmap
        pixmap = self._create_splash_pixmap()
        self.setPixmap(pixmap)
        
        # Set window flags to stay on top and have no frame
        self.setWindowFlags(
            Qt.WindowType.SplashScreen | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # Center the splash screen on the screen
        self.center_on_screen()
        
        # Timer to close splash screen after specified duration
        self.timer = QTimer()
        self.timer.timeout.connect(self.close)
        
    def _create_splash_pixmap(self):
        """Create a pixmap with gradient background and text overlay."""
        width = 500
        height = 300
        
        # Create pixmap
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        # Create painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0.0, QColor(45, 85, 125))    # Dark blue
        gradient.setColorAt(0.5, QColor(65, 105, 145))   # Medium blue  
        gradient.setColorAt(1.0, QColor(25, 65, 105))    # Darker blue
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 0, width, height)
        
        # Add border
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(1, 1, width-2, height-2)
        
        # Set up fonts
        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        subtitle_font = QFont("Arial", 14, QFont.Weight.Normal)
        notice_font = QFont("Arial", 11, QFont.Weight.Normal)
        
        # Main title
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(title_font)
        title_text = "IFD Signal Analysis Utility"
        title_rect = QRect(0, height//2 - 60, width, 40)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, title_text)
        
        # Version subtitle
        painter.setFont(subtitle_font)
        version_text = "Version 1.0.0"
        version_rect = QRect(0, height//2 - 15, width, 25)
        painter.drawText(version_rect, Qt.AlignmentFlag.AlignCenter, version_text)
        
        # Development notice
        painter.setPen(QColor(220, 220, 150))  # Light yellow
        painter.setFont(notice_font)
        notice_text = "For internal development use only"
        notice_rect = QRect(0, height//2 + 35, width, 20)
        painter.drawText(notice_rect, Qt.AlignmentFlag.AlignCenter, notice_text)
        
        # Add loading text area at bottom
        painter.setPen(QColor(200, 200, 200))
        loading_font = QFont("Arial", 10)
        painter.setFont(loading_font)
        loading_text = "Initializing application..."
        loading_rect = QRect(0, height - 40, width, 20)
        painter.drawText(loading_rect, Qt.AlignmentFlag.AlignCenter, loading_text)
        
        painter.end()
        return pixmap
        
    def center_on_screen(self):
        """Center the splash screen on the primary screen."""
        from PyQt6.QtWidgets import QApplication
        
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            splash_geometry = self.geometry()
            
            x = (screen_geometry.width() - splash_geometry.width()) // 2
            y = (screen_geometry.height() - splash_geometry.height()) // 2
            
            self.move(x, y)
            
    def show_for_duration(self, duration_ms=3000):
        """
        Show the splash screen for the specified duration.
        
        Args:
            duration_ms: Duration to show splash screen in milliseconds (default: 3000)
        """
        self.show()
        self.timer.start(duration_ms)
        
        # Process events to ensure splash screen is rendered
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
    def show_message(self, message, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter):
        """
        Display a message on the splash screen.
        
        Args:
            message: Text message to display
            alignment: Text alignment (default: bottom center)
        """
        self.showMessage(message, alignment, QColor(200, 200, 200))
        
        # Process events to update display
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
    def finish_loading(self, main_window):
        """
        Close the splash screen and show the main window.
        
        Args:
            main_window: Main application window to show after splash closes
        """
        self.finish(main_window)

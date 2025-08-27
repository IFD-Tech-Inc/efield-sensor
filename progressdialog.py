#!/usr/bin/env python3
"""
Progress Dialog for IFD Signal Analysis Utility

A custom progress dialog that provides visual feedback during waveform
data loading operations with cancellation support.

Author: Assistant
Version: 1.0.0
Dependencies: PyQt6
"""

from PyQt6.QtWidgets import QProgressDialog, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class LoadingProgressDialog(QProgressDialog):
    """
    Custom progress dialog for waveform loading operations.
    
    Provides visual feedback with progress percentage, status messages,
    and cancellation capability during file/directory loading.
    """
    
    # Signal emitted when user cancels the operation
    cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_dialog()
        self.reset_progress()
        
        # Connect cancel signal
        self.canceled.connect(self.on_cancelled)
        
    def setup_dialog(self):
        """Configure the progress dialog appearance and behavior."""
        
        # Dialog configuration
        self.setWindowTitle("Loading Waveform Data")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAutoClose(True)
        self.setAutoReset(False)
        
        # Progress range (0-100 for percentage)
        self.setRange(0, 100)
        
        # Minimum duration before showing dialog (500ms)
        self.setMinimumDuration(500)
        
        # Dialog size and appearance
        self.setMinimumWidth(400)
        
        # Set font for better appearance
        font = QFont("Arial", 9)
        self.setFont(font)
        
        # Cancel button text
        self.setCancelButtonText("Cancel Loading")
        
        # Default labels
        self.setLabelText("Preparing to load waveform data...")
        
    def reset_progress(self):
        """Reset progress dialog to initial state."""
        self.setValue(0)
        self.setLabelText("Preparing to load waveform data...")
        
    def start_loading(self, operation_description="Loading waveform data"):
        """
        Start the loading operation and show progress dialog.
        
        Args:
            operation_description: Description of the loading operation
        """
        self.reset_progress()
        self.setLabelText(f"{operation_description}...")
        self.show()
        
        # Process events to ensure dialog appears
        QApplication.processEvents()
        
    def update_progress(self, percentage, message=""):
        """
        Update progress percentage and optional message.
        
        Args:
            percentage: Progress percentage (0-100)
            message: Optional status message to display
        """
        if self.wasCanceled():
            return False
            
        # Clamp percentage to valid range
        percentage = max(0, min(100, percentage))
        self.setValue(percentage)
        
        if message:
            self.setLabelText(message)
            
        # Process events to update display
        QApplication.processEvents()
        
        return not self.wasCanceled()
        
    def update_message(self, message):
        """
        Update the status message without changing progress.
        
        Args:
            message: Status message to display
        """
        if not self.wasCanceled():
            self.setLabelText(message)
            QApplication.processEvents()
            
    def finish_loading(self, success_message="Loading completed successfully"):
        """
        Complete the loading operation and close dialog.
        
        Args:
            success_message: Message to display briefly before closing
        """
        if not self.wasCanceled():
            self.setValue(100)
            self.setLabelText(success_message)
            QApplication.processEvents()
            
            # Brief delay to show completion message
            QTimer.singleShot(500, self.close)
        else:
            self.close()
            
    def on_cancelled(self):
        """Handle cancellation signal."""
        self.cancelled.emit()
        
    def is_cancelled(self):
        """Check if the operation was cancelled."""
        return self.wasCanceled()


class EnhancedLoadingProgressDialog(LoadingProgressDialog):
    """
    Enhanced version of LoadingProgressDialog with stage-based progress.
    
    Provides more granular progress tracking through predefined stages
    of the loading operation.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Loading stages with their relative weights
        self.stages = {
            'initializing': {'weight': 10, 'message': 'Initializing parser...'},
            'reading_header': {'weight': 20, 'message': 'Reading file header...'},
            'parsing_channels': {'weight': 40, 'message': 'Parsing channel data...'},
            'processing_data': {'weight': 20, 'message': 'Processing waveform data...'},
            'finalizing': {'weight': 10, 'message': 'Finalizing...'}
        }
        
        self.current_stage = None
        self.stage_progress = 0
        self.base_progress = 0
        
    def start_stage(self, stage_name):
        """
        Start a new loading stage.
        
        Args:
            stage_name: Name of the stage to start (must be in self.stages)
        """
        if stage_name not in self.stages:
            return
            
        # Calculate base progress from completed stages
        if self.current_stage:
            completed_stages = list(self.stages.keys())[:completed_stages.index(self.current_stage) + 1]
            self.base_progress = sum(self.stages[stage]['weight'] for stage in completed_stages[:-1])
        else:
            self.base_progress = 0
            
        self.current_stage = stage_name
        self.stage_progress = 0
        
        # Update display
        stage_info = self.stages[stage_name]
        self.update_message(stage_info['message'])
        self.update_progress(self.base_progress, stage_info['message'])
        
    def update_stage_progress(self, stage_percentage, custom_message=None):
        """
        Update progress within the current stage.
        
        Args:
            stage_percentage: Progress within current stage (0-100)
            custom_message: Optional custom message to override stage message
        """
        if not self.current_stage:
            return
            
        self.stage_progress = max(0, min(100, stage_percentage))
        
        # Calculate overall progress
        stage_weight = self.stages[self.current_stage]['weight']
        stage_contribution = (stage_weight * self.stage_progress) / 100
        total_progress = self.base_progress + stage_contribution
        
        # Update display
        if custom_message:
            message = custom_message
        else:
            stage_info = self.stages[self.current_stage]
            message = f"{stage_info['message']} ({self.stage_progress:.0f}%)"
            
        return self.update_progress(int(total_progress), message)
        
    def complete_stage(self):
        """Mark the current stage as complete."""
        if self.current_stage:
            self.update_stage_progress(100)
            
    def set_file_count_info(self, current_file, total_files, filename=""):
        """
        Update progress with file count information for directory loading.
        
        Args:
            current_file: Current file number (1-based)
            total_files: Total number of files
            filename: Optional filename being processed
        """
        if total_files > 1:
            file_progress = ((current_file - 1) / total_files) * 100
            message = f"Processing file {current_file} of {total_files}"
            if filename:
                message += f": {filename}"
            self.update_stage_progress(file_progress, message)

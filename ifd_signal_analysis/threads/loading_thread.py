#!/usr/bin/env python3
"""
Background loading thread for IFD Signal Analysis Utility.

This module contains the LoadWaveformThread class responsible for loading
waveform data from Siglent binary files in the background to prevent GUI freezing.
"""

import traceback
from pathlib import Path
from typing import Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal

from ..utils.constants import (
    LOAD_PROGRESS_STAGES,
    ERROR_SIGLENT_PARSER_UNAVAILABLE,
    LoadResultDict
)

# Import Siglent parser with availability check
try:
    from siglent_parser import SiglentBinaryParser
    SIGLENT_PARSER_AVAILABLE = True
except ImportError as e:
    SIGLENT_PARSER_AVAILABLE = False
    SiglentBinaryParser = None
    print(f"Warning: Siglent parser not available: {e}")


class LoadWaveformThread(QThread):
    """
    Background thread for loading waveform data to prevent GUI freezing.
    
    This thread handles file I/O operations and waveform parsing in the background,
    emitting progress signals and results to the main UI thread. Supports both
    single file and directory loading operations.
    
    Attributes:
        progress: Signal emitting status messages during loading
        progress_percentage: Signal emitting progress percentage and messages
        finished: Signal emitted when loading is complete with results
        error: Signal emitted when an error occurs during loading
    """
    
    # Signals for communication with main thread
    progress = pyqtSignal(str)  # Status message
    progress_percentage = pyqtSignal(int, str)  # Progress percentage and message
    finished = pyqtSignal(dict)  # Channel data dictionary
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, file_path: str, is_directory: bool = False) -> None:
        """
        Initialize the loading thread.
        
        Args:
            file_path: Path to file or directory to load
            is_directory: True if loading from directory, False for single file
        """
        super().__init__()
        self.file_path = file_path
        self.is_directory = is_directory
        
    def run(self) -> None:
        """
        Execute the waveform loading in the background.
        
        This method handles the complete loading process including parser initialization,
        file/directory scanning, data parsing, and result preparation. Progress is
        reported through signals at each stage.
        """
        try:
            if not SIGLENT_PARSER_AVAILABLE:
                self.error.emit(ERROR_SIGLENT_PARSER_UNAVAILABLE)
                return
                
            # Stage 1: Initialize parser
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['INIT'], 
                "Initializing parser..."
            )
            self.progress.emit("Initializing parser...")
            parser = SiglentBinaryParser()
            
            if self.is_directory:
                channels = self._load_from_directory(parser)
            else:
                channels = self._load_from_file(parser)
            
            if channels is None:
                return  # Error already emitted in helper method
            
            # Stage 5: Finalize data processing
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['FINALIZE'], 
                "Finalizing data processing..."
            )
            
            # Prepare result dictionary
            result: LoadResultDict = {
                'channels': channels,
                'parser': parser,
                'source_path': self.file_path,
                'is_directory': self.is_directory
            }
            
            # Complete loading
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['COMPLETE'], 
                "Loading completed successfully"
            )
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = (f"Failed to load waveform data: {str(e)}\n"
                        f"{traceback.format_exc()}")
            self.error.emit(error_msg)
    
    def _load_from_directory(self, parser: 'SiglentBinaryParser') -> Dict[str, Any]:
        """
        Load waveform data from all files in a directory.
        
        Args:
            parser: Initialized SiglentBinaryParser instance
            
        Returns:
            Dictionary mapping channel names to ChannelData objects,
            or None if loading failed
        """
        try:
            # Stage 2: Scanning directory
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['SCAN_DIR'], 
                f"Scanning directory: {Path(self.file_path).name}"
            )
            self.progress.emit(f"Scanning directory: {self.file_path}")
            
            # Stage 3: Parsing files
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['PARSE'], 
                "Parsing waveform files..."
            )
            channels = parser.parse_directory(self.file_path)
            
            # Stage 4: Processing data
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['PROCESS'], 
                f"Processing {len(channels)} channels..."
            )
            self.progress.emit(f"Loaded {len(channels)} channels from directory")
            
            return channels
            
        except Exception as e:
            error_msg = f"Failed to load directory {self.file_path}: {str(e)}"
            self.error.emit(error_msg)
            return None
    
    def _load_from_file(self, parser: 'SiglentBinaryParser') -> Dict[str, Any]:
        """
        Load waveform data from a single file.
        
        Args:
            parser: Initialized SiglentBinaryParser instance
            
        Returns:
            Dictionary mapping channel names to ChannelData objects,
            or None if loading failed
        """
        try:
            # Stage 2: Reading file
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['READ_FILE'], 
                f"Reading file: {Path(self.file_path).name}"
            )
            self.progress.emit(f"Loading file: {Path(self.file_path).name}")
            
            # Stage 3: Parsing data
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['PARSE'], 
                "Parsing waveform data..."
            )
            channels = parser.parse_file(self.file_path)
            
            # Stage 4: Processing channels
            self.progress_percentage.emit(
                LOAD_PROGRESS_STAGES['PROCESS'], 
                f"Processing {len(channels)} channels..."
            )
            self.progress.emit(f"Loaded {len(channels)} channels from file")
            
            return channels
            
        except Exception as e:
            error_msg = f"Failed to load file {self.file_path}: {str(e)}"
            self.error.emit(error_msg)
            return None

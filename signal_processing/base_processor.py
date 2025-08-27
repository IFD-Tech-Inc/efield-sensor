#!/usr/bin/env python3
"""
Base Signal Processor for IFD Signal Analysis Utility.

This module defines the abstract base class for all signal processing modules,
providing a standardized interface for processing waveform data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np


class ProcessingError(Exception):
    """Exception raised for errors in signal processing operations."""
    pass


class ParameterValidationError(ProcessingError):
    """Exception raised for parameter validation errors."""
    pass


class SignalProcessor(ABC):
    """
    Abstract base class for all signal processing modules.
    
    This class defines the standard interface that all signal processors must implement.
    It provides methods for:
    - Module identification (name, description)
    - Parameter definition and validation
    - Signal processing operations
    - Error handling
    
    All signal processing modules should inherit from this class and implement
    the required abstract methods.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the display name of this signal processor.
        
        This name will be shown in the user interface for module selection.
        It should be concise but descriptive.
        
        Returns:
            str: Display name of the processor (e.g., "Voltage Multiplier")
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a detailed description of what this processor does.
        
        This description will be shown as a tooltip or help text in the UI.
        It should explain the purpose and behavior of the processor clearly.
        
        Returns:
            str: Detailed description of the processor's functionality
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameter definitions for this processor.
        
        Returns a dictionary where keys are parameter names and values are
        parameter definition dictionaries containing:
        - 'type': Parameter data type (int, float, bool, str, list)
        - 'default': Default value for the parameter
        - 'description': Human-readable description of the parameter
        - Additional type-specific fields:
          - For numeric types: 'min', 'max' (optional)
          - For string types: 'choices' (optional list of valid values)
          - For list types: 'item_type' (type of list items)
        
        Example:
        {
            'multiplier': {
                'type': float,
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Factor to multiply voltage values by'
            },
            'invert_signal': {
                'type': bool,
                'default': False,
                'description': 'Whether to invert the signal polarity'
            }
        }
        
        Returns:
            Dict[str, Dict[str, Any]]: Parameter definitions
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data using the specified parameters.
        
        This is the main processing method that performs the actual signal
        processing operation on the input data.
        
        Args:
            input_data: Dictionary containing input waveform data with structure:
                {
                    'channels': {
                        'channel_name': {
                            'time_array': np.ndarray,  # Time values
                            'voltage_data': np.ndarray,  # Voltage values
                            'metadata': dict  # Channel metadata (optional)
                        }
                    },
                    'header': dict,  # File header information (optional)
                    'source_info': dict  # Source plot/file info (optional)
                }
            
            parameters: Dictionary containing parameter values as defined by get_parameters()
        
        Returns:
            Dict[str, Any]: Processed waveform data in the same format as input_data
            
        Raises:
            ProcessingError: If processing fails due to invalid input or other errors
            ParameterValidationError: If parameters are invalid
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate the provided parameters against the processor's requirements.
        
        This method checks that all required parameters are present and have
        valid values according to the parameter definitions from get_parameters().
        
        Args:
            parameters: Dictionary of parameter values to validate
            
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
                           Keys are parameter names, values are error messages
        """
        errors = {}
        param_defs = self.get_parameters()
        
        # Check for required parameters and validate types/ranges
        for param_name, param_def in param_defs.items():
            if param_name not in parameters:
                # Use default value if parameter not provided
                parameters[param_name] = param_def['default']
                continue
                
            value = parameters[param_name]
            expected_type = param_def['type']
            
            # Type validation
            if not isinstance(value, expected_type):
                errors[param_name] = f"Expected {expected_type.__name__}, got {type(value).__name__}"
                continue
            
            # Range validation for numeric types
            if expected_type in (int, float):
                if 'min' in param_def and value < param_def['min']:
                    errors[param_name] = f"Value {value} is below minimum {param_def['min']}"
                elif 'max' in param_def and value > param_def['max']:
                    errors[param_name] = f"Value {value} is above maximum {param_def['max']}"
            
            # Choice validation for string types
            elif expected_type == str and 'choices' in param_def:
                if value not in param_def['choices']:
                    errors[param_name] = f"Value '{value}' not in allowed choices: {param_def['choices']}"
        
        return errors
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Validate the structure and content of input data.
        
        Args:
            input_data: Input data dictionary to validate
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check basic structure
        if not isinstance(input_data, dict):
            errors.append("Input data must be a dictionary")
            return errors
            
        if 'channels' not in input_data:
            errors.append("Input data must contain 'channels' key")
            return errors
            
        channels = input_data['channels']
        if not isinstance(channels, dict):
            errors.append("Channels must be a dictionary")
            return errors
            
        if len(channels) == 0:
            errors.append("At least one channel must be provided")
            return errors
        
        # Validate each channel
        for channel_name, channel_data in channels.items():
            if not isinstance(channel_data, dict):
                errors.append(f"Channel '{channel_name}' data must be a dictionary")
                continue
                
            # Check required fields
            required_fields = ['time_array', 'voltage_data']
            for field in required_fields:
                if field not in channel_data:
                    errors.append(f"Channel '{channel_name}' missing required field '{field}'")
                    continue
                    
                # Check that arrays are numpy arrays
                if not isinstance(channel_data[field], np.ndarray):
                    errors.append(f"Channel '{channel_name}' field '{field}' must be numpy array")
                    continue
            
            # Check array dimensions match
            if ('time_array' in channel_data and 'voltage_data' in channel_data and
                isinstance(channel_data['time_array'], np.ndarray) and
                isinstance(channel_data['voltage_data'], np.ndarray)):
                
                time_len = len(channel_data['time_array'])
                voltage_len = len(channel_data['voltage_data'])
                
                if time_len != voltage_len:
                    errors.append(
                        f"Channel '{channel_name}' time_array ({time_len}) and "
                        f"voltage_data ({voltage_len}) lengths don't match"
                    )
        
        return errors
    
    def get_version(self) -> str:
        """
        Get the version of this processor.
        
        Returns:
            str: Version string (default: "1.0")
        """
        return "1.0"
    
    def get_author(self) -> str:
        """
        Get the author of this processor.
        
        Returns:
            str: Author name (default: "Unknown")
        """
        return "Unknown"
    
    def supports_multi_channel(self) -> bool:
        """
        Check if this processor can handle multiple channels simultaneously.
        
        Returns:
            bool: True if processor can handle multiple channels, False otherwise
        """
        return True
    
    def requires_time_domain(self) -> bool:
        """
        Check if this processor requires time-domain data.
        
        Some processors might work on frequency domain or other representations.
        
        Returns:
            bool: True if time-domain data is required (default: True)
        """
        return True
    
    def get_output_channel_suffix(self) -> str:
        """
        Get the suffix to append to output channel names.
        
        This helps identify processed channels in the UI.
        
        Returns:
            str: Suffix for processed channel names (default: "_processed")
        """
        return "_processed"

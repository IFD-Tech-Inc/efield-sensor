#!/usr/bin/env python3
"""
Signal Processing Package for IFD Signal Analysis Utility.

This package provides a framework for creating and managing signal processing modules.
It includes:
- Base classes for signal processors
- Dynamic module discovery and loading
- Parameter validation and type checking
- A registry system for processor management

Usage:
    from signal_processing import get_available_processors, get_processor, process_with_processor
    
    # Get list of available processors
    processors = get_available_processors()
    
    # Get a specific processor
    multiplier = get_processor("Voltage Multiplier")
    
    # Process data directly
    result = process_with_processor("Voltage Multiplier", data, {"multiplier": 2.0})
"""

import importlib.util
import os
from typing import List, Optional, Dict, Any

# Import core classes
from .base_processor import SignalProcessor, ProcessingError, ParameterValidationError
from .processor_registry import (
    ProcessorRegistry,
    get_registry,
    get_available_processors,
    get_processor,
    process_with_processor
)

__version__ = "1.0.0"
__author__ = "IFD Signal Analysis Team"

# Package-level exports
__all__ = [
    # Core classes
    'SignalProcessor',
    'ProcessingError',
    'ParameterValidationError',
    'ProcessorRegistry',
    
    # Registry functions
    'get_registry',
    'get_available_processors',
    'get_processor',
    'process_with_processor',
    
    # Convenience functions
    'initialize_processors',
    'reload_processors',
    'get_processor_info',
    'validate_processor_input'
]


def initialize_processors(auto_discover: bool = True) -> None:
    """
    Initialize the signal processing system.
    
    This function sets up the processor registry and discovers available processors.
    It should be called once during application startup.
    
    Args:
        auto_discover: Whether to automatically discover processors in the package directory
    """
    if auto_discover:
        registry = get_registry()
        registry.discover_processors()
        print(f"Signal processing system initialized with {len(get_available_processors())} processors")


def reload_processors() -> None:
    """
    Reload all signal processors from disk.
    
    This is useful during development when processor modules are modified.
    It will clear the registry cache and rediscover all processors.
    """
    registry = get_registry()
    registry.reload_processors()
    print(f"Reloaded {len(get_available_processors())} signal processors")


def get_processor_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific processor.
    
    Args:
        name: Display name of the processor
        
    Returns:
        Dictionary containing processor metadata or None if not found
    """
    registry = get_registry()
    return registry.get_processor_info(name)


def get_all_processor_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all registered processors.
    
    Returns:
        Dictionary mapping processor names to their metadata
    """
    registry = get_registry()
    return registry.get_all_processor_info()


def validate_processor_input(input_data: Dict[str, Any]) -> List[str]:
    """
    Validate input data structure for signal processing.
    
    This is a convenience function that performs basic validation
    of input data structure without requiring a specific processor.
    
    Args:
        input_data: Input data dictionary to validate
        
    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    # Create a temporary processor instance for validation
    # We'll use the base class validation logic
    from .base_processor import SignalProcessor
    
    class TempValidator(SignalProcessor):
        def get_name(self): return "Validator"
        def get_description(self): return "Temporary validator"
        def get_parameters(self): return {}
        def process(self, input_data, parameters): return input_data
    
    validator = TempValidator()
    return validator.validate_input_data(input_data)


def is_valid_processor_module(file_path: str) -> bool:
    """
    Check if a file is a valid processor module.
    
    Args:
        file_path: Path to the Python file to check
        
    Returns:
        bool: True if the file appears to be a valid processor module
    """
    if not file_path.endswith('_processor.py'):
        return False
        
    if os.path.basename(file_path) == 'base_processor.py':
        return False
        
    # Check if file exists and is readable
    if not os.path.isfile(file_path):
        return False
        
    try:
        # Try to create a module spec (basic syntax check)
        spec = importlib.util.spec_from_file_location("temp", file_path)
        return spec is not None
    except Exception:
        return False


def create_processor_template(name: str, output_path: str) -> None:
    """
    Create a new processor module from template.
    
    This function creates a new processor file with the basic structure
    filled in, ready for customization.
    
    Args:
        name: Name of the new processor (will be converted to class name)
        output_path: Path where the new processor file should be created
        
    Raises:
        ValueError: If name or output_path is invalid
        FileExistsError: If output file already exists
    """
    if not name or not name.replace('_', '').replace(' ', '').isalnum():
        raise ValueError("Processor name must contain only alphanumeric characters, spaces, and underscores")
        
    if os.path.exists(output_path):
        raise FileExistsError(f"File already exists: {output_path}")
    
    # Convert name to class name (CamelCase)
    class_name = ''.join(word.capitalize() for word in name.replace('_', ' ').split())
    if not class_name.endswith('Processor'):
        class_name += 'Processor'
    
    # Template content
    template_content = f'''#!/usr/bin/env python3
"""
{name} Signal Processor for IFD Signal Analysis Utility.

This module implements a signal processor for {name.lower()} operations.
"""

import numpy as np
from typing import Dict, Any

from .base_processor import SignalProcessor, ProcessingError


class {class_name}(SignalProcessor):
    """
    Signal processor for {name.lower()} operations.
    
    This processor performs {name.lower()} on input waveform data.
    """
    
    def get_name(self) -> str:
        """Get the display name of this processor."""
        return "{name}"
    
    def get_description(self) -> str:
        """Get a detailed description of what this processor does."""
        return "Performs {name.lower()} operations on waveform voltage data."
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameter definitions for this processor."""
        return {{
            'parameter1': {{
                'type': float,
                'default': 1.0,
                'min': 0.0,
                'max': 10.0,
                'description': 'First parameter for {name.lower()}'
            }},
            'parameter2': {{
                'type': bool,
                'default': False,
                'description': 'Boolean parameter for {name.lower()}'
            }}
        }}
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data.
        
        Args:
            input_data: Dictionary containing input waveform data
            parameters: Dictionary containing processing parameters
            
        Returns:
            Dictionary containing processed waveform data
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract parameters
            param1 = parameters.get('parameter1', 1.0)
            param2 = parameters.get('parameter2', False)
            
            # Create output data structure
            output_data = {{
                'channels': {{}},
                'header': input_data.get('header', {{}}),
                'source_info': input_data.get('source_info', {{}})
            }}
            
            # Process each channel
            for channel_name, channel_data in input_data['channels'].items():
                time_array = channel_data['time_array']
                voltage_data = channel_data['voltage_data']
                
                # TODO: Implement your processing logic here
                # This is a placeholder - replace with actual processing
                processed_voltage = voltage_data * param1
                if param2:
                    processed_voltage = -processed_voltage
                
                # Create output channel name
                output_channel_name = channel_name + self.get_output_channel_suffix()
                
                # Store processed data
                output_data['channels'][output_channel_name] = {{
                    'time_array': time_array.copy(),
                    'voltage_data': processed_voltage,
                    'metadata': channel_data.get('metadata', {{}})
                }}
            
            return output_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to process data: {{e}}")
    
    def get_output_channel_suffix(self) -> str:
        """Get the suffix for output channel names."""
        return "_{name.lower().replace(' ', '_')}"
'''
    
    # Write template to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    print(f"Created new processor template: {output_path}")
    print(f"Class name: {class_name}")
    print(f"Remember to implement the actual processing logic in the process() method!")


# Initialize processors when package is imported (but don't auto-discover yet)
# This allows the package to be imported without side effects
# Call initialize_processors() explicitly when ready
_initialized = False


def ensure_initialized():
    """Ensure processors are initialized (called automatically by registry functions)."""
    global _initialized
    if not _initialized:
        initialize_processors()
        _initialized = True

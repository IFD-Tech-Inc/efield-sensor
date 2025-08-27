#!/usr/bin/env python3
"""
Signal Processor Registry for IFD Signal Analysis Utility.

This module provides dynamic loading and management of signal processing modules.
It automatically discovers and registers processor classes from the signal_processing directory.

PROCESSOR FILTERING:
The registry automatically excludes template and example processors from the production UI.
These processors (defined in template_processor.py) are intended for developer reference
and should not appear in the user interface. The filtering is based on class names and
display names defined in the EXCLUDED_PROCESSOR_CLASSES and EXCLUDED_PROCESSOR_NAMES
constants at the top of this file.

Template processors remain accessible to developers for:
- Reference when creating new processors
- Copy/paste starting points for new implementations  
- Learning implementation patterns and best practices

To add new exclusions, simply add the class name and display name to the respective
sets defined at the module level.
"""

import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, List, Type, Optional, Any
from pathlib import Path

from .base_processor import SignalProcessor, ProcessingError

# Configuration: Processors to exclude from production UI
# 
# These are template and example processors intended for developer reference only.
# They provide code templates and implementation examples for creating new processors
# but should not appear in the production user interface.
#
# ADDING NEW EXCLUSIONS:
# To exclude additional template or example processors, add them to both sets below:
# 1. Add the class name to EXCLUDED_PROCESSOR_CLASSES
# 2. Add the display name (from get_name() method) to EXCLUDED_PROCESSOR_NAMES
#
# TEMPLATE FILE ACCESS:
# The template_processor.py file remains accessible to developers who can:
# - Use it as reference for creating new processors
# - Copy and modify the template classes for new implementations
# - Study the implementation patterns and best practices
#
EXCLUDED_PROCESSOR_CLASSES = {
    'TemplateProcessor',                    # Main template processor
    'ExampleFrequencyDomainProcessor',      # FFT processing example  
    'ExampleParameterValidationProcessor'   # Parameter validation example
}

EXCLUDED_PROCESSOR_NAMES = {
    'Template Processor',       # Display name for TemplateProcessor
    'Example FFT Processor',    # Display name for ExampleFrequencyDomainProcessor
    'Example Validation Processor'  # Display name for ExampleParameterValidationProcessor
}


class ProcessorRegistry:
    """
    Registry for managing signal processor modules.
    
    This class handles:
    - Automatic discovery of processor modules
    - Dynamic loading and instantiation
    - Validation of processor implementations
    - Caching of processor instances
    """
    
    def __init__(self):
        """Initialize the processor registry."""
        self._processors: Dict[str, Type[SignalProcessor]] = {}
        self._instances: Dict[str, SignalProcessor] = {}
        self._module_info: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        
    def discover_processors(self, directory: Optional[str] = None) -> None:
        """
        Discover and load all processor modules from the specified directory.
        
        Args:
            directory: Directory path to search for processors. If None,
                      uses the signal_processing package directory.
        """
        if directory is None:
            # Use the signal_processing directory
            directory = os.path.dirname(__file__)
            
        self._processors.clear()
        self._instances.clear()
        self._module_info.clear()
        
        # Get all Python files in the directory
        for file_path in Path(directory).glob("*_processor.py"):
            if file_path.name == "base_processor.py":
                continue  # Skip the base class
                
            try:
                self._load_processor_module(file_path, directory)
            except Exception as e:
                print(f"Warning: Failed to load processor from {file_path}: {e}")
                
        self._loaded = True
        print(f"Discovered {len(self._processors)} signal processors")
        
    def _load_processor_module(self, file_path: Path, base_directory: str) -> None:
        """
        Load a processor module from a file path.
        
        Args:
            file_path: Path to the Python module file
            base_directory: Base directory for relative imports
        """
        module_name = file_path.stem
        
        # Create module spec and load
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find SignalProcessor classes in the module
        processor_classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # Check if this is a SignalProcessor subclass by checking the method resolution order
                # This is more robust than issubclass when dealing with dynamic imports
                if self._is_signal_processor_subclass(obj):
                    processor_classes.append(obj)
        
        # Register found processors
        for processor_class in processor_classes:
            self._register_processor_class(processor_class, module_name, file_path)
            
    def _register_processor_class(self, processor_class: Type[SignalProcessor], 
                                 module_name: str, file_path: Path) -> None:
        """
        Register a processor class.
        
        Args:
            processor_class: The processor class to register
            module_name: Name of the module containing the processor
            file_path: Path to the module file
        """
        try:
            # Skip processors that are excluded from production UI
            # These are templates and examples intended for developer reference
            class_name = processor_class.__name__
            if class_name in EXCLUDED_PROCESSOR_CLASSES:
                print(f"Skipping template/example processor: {class_name}")
                return
            
            # Create temporary instance to get metadata
            temp_instance = processor_class()
            processor_name = temp_instance.get_name()
            
            # Double-check exclusion by display name (in case class name doesn't match)
            if processor_name in EXCLUDED_PROCESSOR_NAMES:
                print(f"Skipping template/example processor: {processor_name}")
                return
            
            # Validate the processor implementation
            self._validate_processor_class(processor_class)
            
            # Store processor information
            self._processors[processor_name] = processor_class
            self._module_info[processor_name] = {
                'class_name': processor_class.__name__,
                'module_name': module_name,
                'file_path': str(file_path),
                'description': temp_instance.get_description(),
                'version': temp_instance.get_version(),
                'author': temp_instance.get_author(),
                'parameters': temp_instance.get_parameters()
            }
            
            print(f"Registered processor: {processor_name}")
            
        except Exception as e:
            raise ProcessingError(f"Failed to register processor {processor_class.__name__}: {e}")
    
    def _is_signal_processor_subclass(self, obj) -> bool:
        """
        Check if a class is a SignalProcessor subclass.
        
        This method is more robust than issubclass() when dealing with
        dynamically loaded modules that may have different instances
        of the same base class.
        
        Args:
            obj: The class object to check
            
        Returns:
            bool: True if the class appears to be a SignalProcessor subclass
        """
        if not inspect.isclass(obj):
            return False
            
        # Skip the base SignalProcessor class itself
        if obj.__name__ == 'SignalProcessor' and hasattr(obj, 'process'):
            return False
            
        # Check if the class has the required SignalProcessor methods
        required_methods = ['get_name', 'get_description', 'get_parameters', 'process']
        
        for method_name in required_methods:
            if not hasattr(obj, method_name):
                return False
                
        # Check if any of the base classes are named 'SignalProcessor'
        for base in inspect.getmro(obj):
            if base.__name__ == 'SignalProcessor' and hasattr(base, 'process'):
                return True
                
        return False
            
    def _validate_processor_class(self, processor_class: Type[SignalProcessor]) -> None:
        """
        Validate that a processor class properly implements the required interface.
        
        Args:
            processor_class: The processor class to validate
            
        Raises:
            ProcessingError: If the processor class is invalid
        """
        try:
            instance = processor_class()
            
            # Check required methods return appropriate types
            name = instance.get_name()
            if not isinstance(name, str) or not name.strip():
                raise ProcessingError("get_name() must return a non-empty string")
                
            description = instance.get_description()
            if not isinstance(description, str) or not description.strip():
                raise ProcessingError("get_description() must return a non-empty string")
                
            parameters = instance.get_parameters()
            if not isinstance(parameters, dict):
                raise ProcessingError("get_parameters() must return a dictionary")
                
            # Validate parameter definitions
            for param_name, param_def in parameters.items():
                if not isinstance(param_def, dict):
                    raise ProcessingError(f"Parameter '{param_name}' definition must be a dictionary")
                    
                required_fields = ['type', 'default', 'description']
                for field in required_fields:
                    if field not in param_def:
                        raise ProcessingError(f"Parameter '{param_name}' missing required field '{field}'")
                        
        except Exception as e:
            raise ProcessingError(f"Processor validation failed: {e}")
            
    def get_processor_names(self) -> List[str]:
        """
        Get list of all registered processor names.
        
        Returns:
            List[str]: List of processor display names
        """
        if not self._loaded:
            self.discover_processors()
        return list(self._processors.keys())
        
    def get_processor(self, name: str) -> Optional[SignalProcessor]:
        """
        Get a processor instance by name.
        
        Args:
            name: Display name of the processor
            
        Returns:
            SignalProcessor instance or None if not found
        """
        if not self._loaded:
            self.discover_processors()
            
        if name not in self._processors:
            return None
            
        # Return cached instance or create new one
        if name not in self._instances:
            processor_class = self._processors[name]
            self._instances[name] = processor_class()
            
        return self._instances[name]
        
    def get_processor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a processor.
        
        Args:
            name: Display name of the processor
            
        Returns:
            Dictionary containing processor metadata or None if not found
        """
        if not self._loaded:
            self.discover_processors()
        return self._module_info.get(name)
        
    def get_all_processor_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered processors.
        
        Returns:
            Dictionary mapping processor names to their metadata
        """
        if not self._loaded:
            self.discover_processors()
        return self._module_info.copy()
        
    def is_processor_available(self, name: str) -> bool:
        """
        Check if a processor is available.
        
        Args:
            name: Display name of the processor
            
        Returns:
            bool: True if processor is registered and available
        """
        if not self._loaded:
            self.discover_processors()
        return name in self._processors
        
    def reload_processors(self) -> None:
        """
        Reload all processors from disk.
        
        This is useful during development when processor modules are modified.
        """
        # Clear existing instances (classes will be reloaded)
        self._instances.clear()
        self.discover_processors()
        
    def process_data(self, processor_name: str, input_data: Dict[str, Any], 
                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using the specified processor.
        
        This is a convenience method that handles processor lookup and execution.
        
        Args:
            processor_name: Name of the processor to use
            input_data: Input waveform data
            parameters: Processing parameters
            
        Returns:
            Processed waveform data
            
        Raises:
            ProcessingError: If processor not found or processing fails
        """
        processor = self.get_processor(processor_name)
        if processor is None:
            raise ProcessingError(f"Processor '{processor_name}' not found")
            
        # Validate parameters
        validation_errors = processor.validate_parameters(parameters)
        if validation_errors:
            error_msg = "; ".join([f"{k}: {v}" for k, v in validation_errors.items()])
            raise ProcessingError(f"Parameter validation failed: {error_msg}")
            
        # Validate input data
        input_errors = processor.validate_input_data(input_data)
        if input_errors:
            error_msg = "; ".join(input_errors)
            raise ProcessingError(f"Input data validation failed: {error_msg}")
            
        # Process the data
        return processor.process(input_data, parameters)
        

# Global registry instance
_registry = ProcessorRegistry()


def get_registry() -> ProcessorRegistry:
    """
    Get the global processor registry instance.
    
    Returns:
        ProcessorRegistry: Global registry instance
    """
    return _registry


def get_available_processors() -> List[str]:
    """
    Get list of available processor names.
    
    Returns:
        List[str]: List of processor display names
    """
    return get_registry().get_processor_names()


def get_processor(name: str) -> Optional[SignalProcessor]:
    """
    Get a processor instance by name.
    
    Args:
        name: Display name of the processor
        
    Returns:
        SignalProcessor instance or None if not found
    """
    return get_registry().get_processor(name)


def process_with_processor(processor_name: str, input_data: Dict[str, Any], 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process data using the specified processor.
    
    Args:
        processor_name: Name of the processor to use
        input_data: Input waveform data
        parameters: Processing parameters
        
    Returns:
        Processed waveform data
        
    Raises:
        ProcessingError: If processor not found or processing fails
    """
    return get_registry().process_data(processor_name, input_data, parameters)

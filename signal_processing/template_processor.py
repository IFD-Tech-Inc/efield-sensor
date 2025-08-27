#!/usr/bin/env python3
"""
Template for creating new signal processing modules.

This file serves as a template and reference for creating new signal processing modules
for the IFD Signal Analysis Utility. Copy this file and modify it for your specific
processing needs.

INSTRUCTIONS FOR USE:
1. Copy this file to a new name ending with '_processor.py'
2. Rename the class from TemplateProcessor to YourProcessorName
3. Update the get_name() method to return your processor's display name
4. Update the get_description() method with what your processor does
5. Define your parameters in get_parameters()
6. Implement your processing logic in the process() method
7. Test your processor thoroughly before use

The processor will be automatically discovered and registered by the system
when the application starts.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import copy

try:
    from .base_processor import SignalProcessor, ProcessingError
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from base_processor import SignalProcessor, ProcessingError


class TemplateProcessor(SignalProcessor):
    """
    Template signal processor showing the required structure.
    
    This processor serves as a template and example for creating new
    signal processing modules. It demonstrates proper implementation
    of all required methods and best practices.
    
    Key Implementation Notes:
    - Always inherit from SignalProcessor
    - Implement all abstract methods
    - Use descriptive names and docstrings
    - Handle errors gracefully with ProcessingError
    - Validate inputs thoroughly
    - Preserve data structure and metadata
    """
    
    def get_name(self) -> str:
        """
        Get the display name of this processor.
        
        This name appears in the UI dropdown for processor selection.
        Make it descriptive but concise.
        
        Returns:
            str: Display name shown in the user interface
        """
        return "Template Processor"
    
    def get_description(self) -> str:
        """
        Get a detailed description of what this processor does.
        
        This description is used as tooltip text and help documentation.
        Explain clearly what the processor does, what inputs it expects,
        and what outputs it produces.
        
        Returns:
            str: Detailed description of processor functionality
        """
        return (
            "Template showing required structure for signal processors. "
            "This example processor demonstrates parameter handling, input validation, "
            "and proper data processing patterns. It can multiply voltage values "
            "by a configurable factor and optionally invert the signal polarity."
        )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Define the parameters this processor accepts.
        
        Each parameter is defined as a dictionary with these required fields:
        - 'type': Python type (int, float, bool, str, list)
        - 'default': Default value for the parameter
        - 'description': Human-readable description
        
        Optional fields for different types:
        - For numeric types: 'min', 'max' for range validation
        - For string types: 'choices' for dropdown selection
        - For list types: 'item_type' for list element type
        
        Returns:
            Dict[str, Dict[str, Any]]: Parameter definitions
        """
        return {
            'multiplier': {
                'type': float,
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Factor to multiply voltage values by (0.1 to 10.0)'
            },
            'invert_signal': {
                'type': bool,
                'default': False,
                'description': 'Whether to invert the signal polarity (multiply by -1)'
            },
            'add_offset': {
                'type': float,
                'default': 0.0,
                'min': -5.0,
                'max': 5.0,
                'description': 'DC offset to add to the signal in volts'
            },
            'processing_mode': {
                'type': str,
                'default': 'linear',
                'choices': ['linear', 'absolute', 'squared'],
                'description': 'Processing mode: linear, absolute value, or squared'
            }
        }
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data using the specified parameters.
        
        This is the main processing method where you implement your signal
        processing algorithm. The method receives validated input data and
        parameters, and must return processed data in the same structure.
        
        Args:
            input_data: Dictionary containing input waveform data with structure:
                {
                    'channels': {
                        'channel_name': {
                            'time_array': np.ndarray,    # Time values (seconds)
                            'voltage_data': np.ndarray,  # Voltage values (volts)
                            'metadata': dict             # Channel metadata (optional)
                        },
                        # ... more channels
                    },
                    'header': dict,      # File header information (optional)
                    'source_info': dict  # Source plot/file info (optional)
                }
            
            parameters: Dictionary containing validated parameter values
            
        Returns:
            Dict[str, Any]: Processed data in same format as input_data
            
        Raises:
            ProcessingError: If processing fails for any reason
        """
        try:
            # Extract and validate parameters
            multiplier = parameters.get('multiplier', 1.0)
            invert_signal = parameters.get('invert_signal', False)
            add_offset = parameters.get('add_offset', 0.0)
            processing_mode = parameters.get('processing_mode', 'linear')
            
            # Create output data structure (deep copy to avoid modifying input)
            output_data = {
                'channels': {},
                'header': copy.deepcopy(input_data.get('header', {})),
                'source_info': copy.deepcopy(input_data.get('source_info', {}))
            }
            
            # Add processing information to source_info
            if 'processing_history' not in output_data['source_info']:
                output_data['source_info']['processing_history'] = []
            
            output_data['source_info']['processing_history'].append({
                'processor': self.get_name(),
                'parameters': parameters.copy(),
                'timestamp': self._get_timestamp()
            })
            
            # Process each channel independently
            for channel_name, channel_data in input_data['channels'].items():
                try:
                    processed_channel = self._process_channel(
                        channel_name, channel_data, 
                        multiplier, invert_signal, add_offset, processing_mode
                    )
                    
                    # Generate output channel name
                    output_channel_name = channel_name + self.get_output_channel_suffix()
                    output_data['channels'][output_channel_name] = processed_channel
                    
                except Exception as e:
                    # Log error but continue with other channels
                    print(f"Warning: Failed to process channel {channel_name}: {e}")
                    continue
            
            # Verify we processed at least one channel
            if not output_data['channels']:
                raise ProcessingError("No channels were successfully processed")
            
            return output_data
            
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ProcessingError
            raise ProcessingError(f"Processing failed: {str(e)}")
    
    def _process_channel(self, channel_name: str, channel_data: Dict[str, Any],
                        multiplier: float, invert_signal: bool, add_offset: float,
                        processing_mode: str) -> Dict[str, Any]:
        """
        Process a single channel's data.
        
        This helper method processes one channel at a time. Breaking this out
        into a separate method makes the code more readable and easier to test.
        
        Args:
            channel_name: Name of the channel being processed
            channel_data: Channel data dictionary
            multiplier: Multiplication factor
            invert_signal: Whether to invert signal
            add_offset: DC offset to add
            processing_mode: Processing mode string
            
        Returns:
            Dict[str, Any]: Processed channel data
            
        Raises:
            ProcessingError: If channel processing fails
        """
        # Extract data arrays
        time_array = channel_data['time_array']
        voltage_data = channel_data['voltage_data']
        
        # Validate arrays
        if len(time_array) == 0 or len(voltage_data) == 0:
            raise ProcessingError(f"Channel {channel_name} contains empty arrays")
            
        if len(time_array) != len(voltage_data):
            raise ProcessingError(
                f"Channel {channel_name} time and voltage arrays have different lengths"
            )
        
        # Apply processing based on mode
        if processing_mode == 'linear':
            # Standard linear processing
            processed_voltage = voltage_data * multiplier
        elif processing_mode == 'absolute':
            # Take absolute value then multiply
            processed_voltage = np.abs(voltage_data) * multiplier
        elif processing_mode == 'squared':
            # Square the values then multiply
            processed_voltage = np.square(voltage_data) * multiplier
        else:
            raise ProcessingError(f"Unknown processing mode: {processing_mode}")
        
        # Apply signal inversion if requested
        if invert_signal:
            processed_voltage = -processed_voltage
            
        # Add DC offset
        if add_offset != 0.0:
            processed_voltage = processed_voltage + add_offset
        
        # Create output channel data (preserve metadata)
        output_channel = {
            'time_array': time_array.copy(),  # Time array unchanged
            'voltage_data': processed_voltage,
            'metadata': copy.deepcopy(channel_data.get('metadata', {}))
        }
        
        # Add processing information to metadata
        if 'processing' not in output_channel['metadata']:
            output_channel['metadata']['processing'] = {}
            
        output_channel['metadata']['processing'].update({
            'processor': self.get_name(),
            'original_channel': channel_name,
            'multiplier': multiplier,
            'inverted': invert_signal,
            'offset': add_offset,
            'mode': processing_mode
        })
        
        return output_channel
    
    def get_output_channel_suffix(self) -> str:
        """
        Get the suffix to append to output channel names.
        
        This suffix helps identify processed channels in the UI.
        Make it descriptive but short.
        
        Returns:
            str: Suffix for processed channel names
        """
        return "_template"
    
    def get_version(self) -> str:
        """
        Get the version of this processor.
        
        Returns:
            str: Version string
        """
        return "1.0"
    
    def get_author(self) -> str:
        """
        Get the author of this processor.
        
        Returns:
            str: Author name
        """
        return "Template Author"
    
    def supports_multi_channel(self) -> bool:
        """
        Indicate that this processor can handle multiple channels.
        
        Returns:
            bool: True (this processor handles multiple channels)
        """
        return True
    
    def requires_time_domain(self) -> bool:
        """
        Indicate that this processor requires time-domain data.
        
        Returns:
            bool: True (this processor works on time-domain data)
        """
        return True
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp for processing history.
        
        Returns:
            str: ISO format timestamp
        """
        import datetime
        return datetime.datetime.now().isoformat()


# ADDITIONAL EXAMPLES AND PATTERNS:

class ExampleFrequencyDomainProcessor(SignalProcessor):
    """
    Example of a processor that works in frequency domain.
    
    This demonstrates how to implement a processor that transforms
    data to frequency domain, processes it, and transforms back.
    """
    
    def get_name(self) -> str:
        return "Example FFT Processor"
    
    def get_description(self) -> str:
        return "Example processor that applies FFT-based filtering"
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'cutoff_freq': {
                'type': float,
                'default': 1000.0,
                'min': 1.0,
                'max': 100000.0,
                'description': 'Cutoff frequency in Hz'
            }
        }
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Example FFT-based processing (placeholder implementation)."""
        # This is just a template - implement actual FFT processing here
        raise ProcessingError("FFT processor not implemented - this is just an example")
    
    def requires_time_domain(self) -> bool:
        return True  # Still needs time domain input for FFT
    
    def get_output_channel_suffix(self) -> str:
        return "_fft_filtered"


class ExampleParameterValidationProcessor(SignalProcessor):
    """
    Example showing advanced parameter validation.
    """
    
    def get_name(self) -> str:
        return "Example Validation Processor"
    
    def get_description(self) -> str:
        return "Example showing advanced parameter validation patterns"
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'window_size': {
                'type': int,
                'default': 100,
                'min': 10,
                'max': 10000,
                'description': 'Window size for processing (must be even)'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Example of custom parameter validation.
        
        This shows how to add custom validation beyond the standard
        type and range checking.
        """
        # Call parent validation first
        errors = super().validate_parameters(parameters)
        
        # Add custom validation
        window_size = parameters.get('window_size', 100)
        if window_size % 2 != 0:
            errors['window_size'] = "Window size must be even"
            
        return errors
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        raise ProcessingError("Validation processor not implemented - this is just an example")
    
    def get_output_channel_suffix(self) -> str:
        return "_validated"


"""
IMPLEMENTATION CHECKLIST:

When creating a new processor, make sure to:

✓ Inherit from SignalProcessor
✓ Implement get_name() with a unique, descriptive name
✓ Implement get_description() with clear explanation
✓ Implement get_parameters() with all required fields
✓ Implement process() with proper error handling
✓ Return data in the same structure as input
✓ Preserve metadata and add processing information
✓ Use appropriate output channel suffix
✓ Handle multiple channels correctly
✓ Validate inputs thoroughly
✓ Use ProcessingError for all error conditions
✓ Test with various input data scenarios

COMMON PATTERNS:

1. Parameter Extraction:
   param_value = parameters.get('param_name', default_value)

2. Input Validation:
   if len(voltage_data) == 0:
       raise ProcessingError("Empty voltage data")

3. Processing Each Channel:
   for channel_name, channel_data in input_data['channels'].items():
       # Process channel_data
       output_data['channels'][new_name] = processed_data

4. Error Handling:
   try:
       # Processing code
   except Exception as e:
       raise ProcessingError(f"Processing failed: {e}")

5. Metadata Preservation:
   output_metadata = copy.deepcopy(input_metadata)
   output_metadata['processing'] = processing_info
"""

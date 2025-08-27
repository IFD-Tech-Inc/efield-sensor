#!/usr/bin/env python3
"""
Voltage Multiplier Signal Processor for IFD Signal Analysis Utility.

This module implements a signal processor that multiplies waveform voltage data
by a configurable scalar value. It's a simple but useful processor for signal
scaling and amplitude adjustment operations.
"""

import numpy as np
from typing import Dict, Any
import copy

try:
    from .base_processor import SignalProcessor, ProcessingError
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from base_processor import SignalProcessor, ProcessingError


class VoltageMultiplierProcessor(SignalProcessor):
    """
    Signal processor for voltage multiplication operations.
    
    This processor multiplies all voltage values in the input waveforms
    by a configurable scalar multiplier. It preserves the time axis and
    all metadata while scaling the amplitude of the signals.
    
    Use cases:
    - Signal amplitude scaling
    - Unit conversion (e.g., mV to V)
    - Signal gain simulation
    - Normalization operations
    """
    
    def get_name(self) -> str:
        """Get the display name of this processor."""
        return "Voltage Multiplier"
    
    def get_description(self) -> str:
        """Get a detailed description of what this processor does."""
        return (
            "Multiplies voltage values by a configurable scalar factor. "
            "Useful for signal scaling, unit conversion, and amplitude adjustment. "
            "The time axis and all metadata are preserved unchanged."
        )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameter definitions for this processor."""
        return {
            'multiplier': {
                'type': float,
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'description': 'Scalar multiplier for voltage values (0.1 to 10.0)'
            }
        }
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data by multiplying voltage values.
        
        Args:
            input_data: Dictionary containing input waveform data
            parameters: Dictionary containing processing parameters
            
        Returns:
            Dictionary containing processed waveform data with scaled voltages
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract multiplier parameter
            multiplier = parameters.get('multiplier', 1.0)
            
            # Validate multiplier
            if not isinstance(multiplier, (int, float)):
                raise ProcessingError(f"Multiplier must be a number, got {type(multiplier)}")
            
            if multiplier <= 0:
                raise ProcessingError(f"Multiplier must be positive, got {multiplier}")
            
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
                'parameters': {'multiplier': multiplier},
                'timestamp': self._get_current_timestamp()
            })
            
            # Track processing statistics
            channels_processed = 0
            total_samples = 0
            
            # Process each channel
            for channel_name, channel_data in input_data['channels'].items():
                try:
                    processed_channel = self._multiply_channel(
                        channel_name, channel_data, multiplier
                    )
                    
                    # Generate output channel name
                    if multiplier == 1.0:
                        # No change - use original name with suffix
                        output_channel_name = channel_name + "_x1.0"
                    else:
                        # Include multiplier in name for clarity
                        output_channel_name = f"{channel_name}_x{multiplier:.1f}"
                    
                    output_data['channels'][output_channel_name] = processed_channel
                    channels_processed += 1
                    total_samples += len(channel_data.get('voltage_data', []))
                    
                except Exception as e:
                    # Log warning but continue with other channels
                    print(f"Warning: Failed to process channel {channel_name}: {e}")
                    continue
            
            # Verify we processed at least one channel
            if channels_processed == 0:
                raise ProcessingError("No channels were successfully processed")
            
            # Add summary statistics to source_info
            output_data['source_info']['processing_summary'] = {
                'channels_processed': channels_processed,
                'total_channels': len(input_data['channels']),
                'total_samples_processed': total_samples,
                'multiplier_applied': multiplier
            }
            
            print(f"Successfully processed {channels_processed} channels with multiplier {multiplier}")
            return output_data
            
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ProcessingError
            raise ProcessingError(f"Voltage multiplication failed: {str(e)}")
    
    def _multiply_channel(self, channel_name: str, channel_data: Dict[str, Any],
                         multiplier: float) -> Dict[str, Any]:
        """
        Multiply voltage data for a single channel.
        
        Args:
            channel_name: Name of the channel being processed
            channel_data: Channel data dictionary
            multiplier: Multiplication factor
            
        Returns:
            Dict[str, Any]: Processed channel data
            
        Raises:
            ProcessingError: If channel processing fails
        """
        # Extract required arrays
        if 'time_array' not in channel_data:
            raise ProcessingError(f"Channel {channel_name} missing time_array")
            
        if 'voltage_data' not in channel_data:
            raise ProcessingError(f"Channel {channel_name} missing voltage_data")
        
        time_array = channel_data['time_array']
        voltage_data = channel_data['voltage_data']
        
        # Validate arrays
        if not isinstance(time_array, np.ndarray):
            raise ProcessingError(f"Channel {channel_name} time_array must be numpy array")
            
        if not isinstance(voltage_data, np.ndarray):
            raise ProcessingError(f"Channel {channel_name} voltage_data must be numpy array")
        
        if len(time_array) == 0 or len(voltage_data) == 0:
            raise ProcessingError(f"Channel {channel_name} contains empty arrays")
            
        if len(time_array) != len(voltage_data):
            raise ProcessingError(
                f"Channel {channel_name} time and voltage arrays have mismatched lengths: "
                f"{len(time_array)} vs {len(voltage_data)}"
            )
        
        # Check for non-finite values
        if not np.all(np.isfinite(voltage_data)):
            print(f"Warning: Channel {channel_name} contains non-finite voltage values")
        
        # Perform multiplication
        try:
            multiplied_voltage = voltage_data * multiplier
        except Exception as e:
            raise ProcessingError(f"Failed to multiply voltage data for {channel_name}: {e}")
        
        # Verify result is valid
        if not isinstance(multiplied_voltage, np.ndarray):
            raise ProcessingError(f"Multiplication result is not a numpy array for {channel_name}")
        
        if len(multiplied_voltage) != len(voltage_data):
            raise ProcessingError(f"Multiplication changed array length for {channel_name}")
        
        # Create output channel data (preserve metadata)
        output_channel = {
            'time_array': time_array.copy(),  # Time array unchanged
            'voltage_data': multiplied_voltage,
            'metadata': copy.deepcopy(channel_data.get('metadata', {}))
        }
        
        # Add processing information to metadata
        if 'processing' not in output_channel['metadata']:
            output_channel['metadata']['processing'] = {}
            
        output_channel['metadata']['processing'].update({
            'processor': self.get_name(),
            'original_channel': channel_name,
            'multiplier': multiplier,
            'original_stats': {
                'min': float(np.min(voltage_data)),
                'max': float(np.max(voltage_data)),
                'mean': float(np.mean(voltage_data)),
                'std': float(np.std(voltage_data))
            },
            'processed_stats': {
                'min': float(np.min(multiplied_voltage)),
                'max': float(np.max(multiplied_voltage)),
                'mean': float(np.mean(multiplied_voltage)),
                'std': float(np.std(multiplied_voltage))
            }
        })
        
        return output_channel
    
    def get_output_channel_suffix(self) -> str:
        """Get the suffix for output channel names."""
        return "_multiplied"
    
    def get_version(self) -> str:
        """Get the version of this processor."""
        return "1.0"
    
    def get_author(self) -> str:
        """Get the author of this processor."""
        return "IFD Signal Analysis Team"
    
    def supports_multi_channel(self) -> bool:
        """This processor can handle multiple channels simultaneously."""
        return True
    
    def requires_time_domain(self) -> bool:
        """This processor works on time-domain data."""
        return True
    
    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp for processing history.
        
        Returns:
            str: ISO format timestamp
        """
        import datetime
        return datetime.datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Custom parameter validation for the multiplier processor.
        
        Args:
            parameters: Dictionary of parameter values to validate
            
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
        """
        # Call parent validation first
        errors = super().validate_parameters(parameters)
        
        # Add custom validation for multiplier
        if 'multiplier' in parameters:
            multiplier = parameters['multiplier']
            
            # Check if it's a reasonable value for practical use
            if multiplier < 0.01:
                errors['multiplier'] = "Multiplier too small (minimum practical value is 0.01)"
            elif multiplier > 100.0:
                errors['multiplier'] = "Multiplier too large (maximum practical value is 100.0)"
            
            # Warn about very small or large multipliers
            if 0.01 <= multiplier < 0.1:
                print(f"Warning: Small multiplier {multiplier} may result in very small signals")
            elif 10.0 < multiplier <= 100.0:
                print(f"Warning: Large multiplier {multiplier} may result in very large signals")
        
        return errors


# Additional utility functions for the multiplier processor

def calculate_optimal_multiplier(voltage_data: np.ndarray, target_range: tuple = (-1.0, 1.0)) -> float:
    """
    Calculate optimal multiplier to scale voltage data to target range.
    
    Args:
        voltage_data: Input voltage data array
        target_range: Desired (min, max) range for output
        
    Returns:
        float: Optimal multiplier value
    """
    if len(voltage_data) == 0:
        return 1.0
    
    data_min, data_max = np.min(voltage_data), np.max(voltage_data)
    data_range = data_max - data_min
    
    if data_range == 0:
        return 1.0  # All values are the same
    
    target_min, target_max = target_range
    target_range_size = target_max - target_min
    
    return target_range_size / data_range


def preview_multiplication(voltage_data: np.ndarray, multiplier: float) -> Dict[str, Any]:
    """
    Preview the effects of multiplication without actually processing.
    
    Args:
        voltage_data: Input voltage data
        multiplier: Multiplication factor
        
    Returns:
        Dict containing preview statistics
    """
    if len(voltage_data) == 0:
        return {'error': 'Empty voltage data'}
    
    try:
        result = voltage_data * multiplier
        
        return {
            'original_range': (float(np.min(voltage_data)), float(np.max(voltage_data))),
            'processed_range': (float(np.min(result)), float(np.max(result))),
            'scale_factor': multiplier,
            'original_rms': float(np.sqrt(np.mean(voltage_data**2))),
            'processed_rms': float(np.sqrt(np.mean(result**2))),
            'samples_count': len(voltage_data)
        }
    except Exception as e:
        return {'error': f'Preview failed: {e}'}

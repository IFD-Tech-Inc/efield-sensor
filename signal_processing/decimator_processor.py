#!/usr/bin/env python3
"""
Decimator Signal Processor for IFD Signal Analysis Utility.

This module implements a signal processor that decimates waveform data by
reducing the sampling rate through selection of every Nth sample. Decimation
is a fundamental digital signal processing operation used to reduce data size,
computational load, and storage requirements while maintaining signal characteristics
within the new Nyquist frequency limits.

The processor performs simple decimation (down-sampling) without anti-aliasing
filtering. Users should be aware of potential aliasing effects if the signal
contains frequency components above the new Nyquist frequency (Fs_new/2).

Key Features:
- Configurable decimation factor (2 to 1000)
- Preserves signal characteristics within new bandwidth
- Reduces data storage and processing requirements
- Maintains temporal relationships between channels
- Provides detailed processing statistics and metadata

Use Cases:
- Data reduction for storage efficiency
- Computational load reduction for real-time processing
- Signal conditioning for downstream processing
- Multi-rate signal processing workflows
- Long-term data analysis where high sample rates aren't needed

Warning: Decimation without anti-aliasing filtering can introduce aliasing artifacts
if the original signal contains frequency components above Fs_new/2. Consider
applying appropriate low-pass filtering before decimation if needed.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import copy

try:
    from .base_processor import SignalProcessor, ProcessingError
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from base_processor import SignalProcessor, ProcessingError


class DecimatorProcessor(SignalProcessor):
    """
    Signal processor for waveform decimation operations.
    
    This processor reduces the sampling rate of input waveforms by keeping
    only every Nth sample, where N is the decimation factor. Both time and
    voltage arrays are decimated consistently to maintain proper temporal
    relationships.
    
    The decimation is performed using simple down-sampling without anti-aliasing
    filtering. This approach is computationally efficient but may introduce
    aliasing if the original signal contains frequency components above the
    new Nyquist frequency.
    
    Mathematical Operation:
    - New sampling rate: Fs_new = Fs_original / decimation_factor
    - New Nyquist frequency: Fn_new = Fs_new / 2
    - New sample count: N_new = N_original / decimation_factor (rounded down)
    
    Use Cases:
    - Reducing data size for storage or transmission
    - Computational load reduction for real-time processing
    - Preparing signals for multi-rate processing systems
    - Long-term trend analysis where high resolution isn't needed
    """
    
    def get_name(self) -> str:
        """Get the display name of this processor."""
        return "Signal Decimator"
    
    def get_description(self) -> str:
        """Get a detailed description of what this processor does."""
        return (
            "Decimates (down-samples) waveform data by keeping every Nth sample, "
            "where N is the configurable decimation factor. Reduces sampling rate, "
            "data size, and computational requirements. Preserves temporal "
            "relationships between channels. Warning: May introduce aliasing if "
            "signal contains frequencies above the new Nyquist frequency."
        )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameter definitions for this processor."""
        return {
            'decimation_factor': {
                'type': int,
                'default': 2,
                'min': 2,
                'max': 1000,
                'description': 'Decimation factor N: keep every Nth sample (2 to 1000)'
            }
        }
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data by decimating the sampling rate.
        
        Args:
            input_data: Dictionary containing input waveform data
            parameters: Dictionary containing processing parameters
            
        Returns:
            Dictionary containing decimated waveform data
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract decimation factor parameter
            decimation_factor = parameters.get('decimation_factor', 2)
            
            # Validate decimation factor
            if not isinstance(decimation_factor, int):
                raise ProcessingError(f"Decimation factor must be an integer, got {type(decimation_factor)}")
            
            if decimation_factor < 2:
                raise ProcessingError(f"Decimation factor must be at least 2, got {decimation_factor}")
            
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
                'parameters': {'decimation_factor': decimation_factor},
                'timestamp': self._get_current_timestamp()
            })
            
            # Track processing statistics
            channels_processed = 0
            total_samples_original = 0
            total_samples_decimated = 0
            
            # Process each channel
            for channel_name, channel_data in input_data['channels'].items():
                try:
                    processed_channel, original_count, decimated_count = self._decimate_channel(
                        channel_name, channel_data, decimation_factor
                    )
                    
                    # Generate output channel name
                    output_channel_name = f"{channel_name}_dec{decimation_factor}"
                    
                    output_data['channels'][output_channel_name] = processed_channel
                    channels_processed += 1
                    total_samples_original += original_count
                    total_samples_decimated += decimated_count
                    
                except Exception as e:
                    # Log warning but continue with other channels
                    print(f"Warning: Failed to decimate channel {channel_name}: {e}")
                    continue
            
            # Verify we processed at least one channel
            if channels_processed == 0:
                raise ProcessingError("No channels were successfully processed")
            
            # Calculate overall statistics
            reduction_ratio = total_samples_decimated / total_samples_original if total_samples_original > 0 else 0
            compression_ratio = total_samples_original / total_samples_decimated if total_samples_decimated > 0 else 0
            
            # Add summary statistics to source_info
            output_data['source_info']['processing_summary'] = {
                'channels_processed': channels_processed,
                'total_channels': len(input_data['channels']),
                'decimation_factor': decimation_factor,
                'total_samples_original': total_samples_original,
                'total_samples_decimated': total_samples_decimated,
                'data_reduction_ratio': reduction_ratio,
                'compression_ratio': compression_ratio,
                'new_effective_sample_rate_ratio': 1.0 / decimation_factor
            }
            
            print(f"Successfully decimated {channels_processed} channels by factor {decimation_factor}")
            print(f"Data reduction: {total_samples_original} → {total_samples_decimated} samples "
                  f"({reduction_ratio:.1%} of original)")
            
            return output_data
            
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ProcessingError
            raise ProcessingError(f"Signal decimation failed: {str(e)}")
    
    def _decimate_channel(self, channel_name: str, channel_data: Dict[str, Any],
                         decimation_factor: int) -> Tuple[Dict[str, Any], int, int]:
        """
        Decimate voltage and time data for a single channel.
        
        Args:
            channel_name: Name of the channel being processed
            channel_data: Channel data dictionary
            decimation_factor: Decimation factor (keep every Nth sample)
            
        Returns:
            Tuple of (processed_channel_data, original_sample_count, decimated_sample_count)
            
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
        
        original_sample_count = len(voltage_data)
        
        # Check if decimation factor is reasonable for this data
        if decimation_factor >= original_sample_count:
            raise ProcessingError(
                f"Channel {channel_name} decimation factor {decimation_factor} "
                f"exceeds sample count {original_sample_count}"
            )
        
        # Check for non-finite values
        if not np.all(np.isfinite(voltage_data)):
            print(f"Warning: Channel {channel_name} contains non-finite voltage values")
        
        if not np.all(np.isfinite(time_array)):
            print(f"Warning: Channel {channel_name} contains non-finite time values")
        
        # Perform decimation (select every Nth sample starting from index 0)
        try:
            decimated_indices = np.arange(0, original_sample_count, decimation_factor)
            decimated_time = time_array[decimated_indices]
            decimated_voltage = voltage_data[decimated_indices]
        except Exception as e:
            raise ProcessingError(f"Failed to decimate data for {channel_name}: {e}")
        
        decimated_sample_count = len(decimated_voltage)
        
        # Verify results are valid
        if not isinstance(decimated_time, np.ndarray) or not isinstance(decimated_voltage, np.ndarray):
            raise ProcessingError(f"Decimation result is not numpy arrays for {channel_name}")
        
        if len(decimated_time) != len(decimated_voltage):
            raise ProcessingError(f"Decimated arrays have mismatched lengths for {channel_name}")
        
        if decimated_sample_count == 0:
            raise ProcessingError(f"Decimation resulted in empty arrays for {channel_name}")
        
        # Calculate statistics for original data
        original_stats = {
            'min': float(np.min(voltage_data)),
            'max': float(np.max(voltage_data)),
            'mean': float(np.mean(voltage_data)),
            'std': float(np.std(voltage_data)),
            'sample_count': original_sample_count
        }
        
        # Calculate statistics for decimated data
        decimated_stats = {
            'min': float(np.min(decimated_voltage)),
            'max': float(np.max(decimated_voltage)),
            'mean': float(np.mean(decimated_voltage)),
            'std': float(np.std(decimated_voltage)),
            'sample_count': decimated_sample_count
        }
        
        # Calculate timing statistics
        original_duration = float(time_array[-1] - time_array[0]) if len(time_array) > 1 else 0.0
        decimated_duration = float(decimated_time[-1] - decimated_time[0]) if len(decimated_time) > 1 else 0.0
        
        # Calculate effective sample rates (approximations)
        original_sample_rate = (original_sample_count - 1) / original_duration if original_duration > 0 else 0.0
        decimated_sample_rate = original_sample_rate / decimation_factor
        
        # Create output channel data (preserve original metadata)
        output_channel = {
            'time_array': decimated_time,
            'voltage_data': decimated_voltage,
            'metadata': copy.deepcopy(channel_data.get('metadata', {}))
        }
        
        # Add processing information to metadata
        if 'processing' not in output_channel['metadata']:
            output_channel['metadata']['processing'] = {}
            
        output_channel['metadata']['processing'].update({
            'processor': self.get_name(),
            'original_channel': channel_name,
            'decimation_factor': decimation_factor,
            'original_stats': original_stats,
            'decimated_stats': decimated_stats,
            'timing_info': {
                'original_duration': original_duration,
                'decimated_duration': decimated_duration,
                'original_sample_rate_approx': original_sample_rate,
                'decimated_sample_rate_approx': decimated_sample_rate,
                'time_scaling_preserved': True
            },
            'data_reduction': {
                'reduction_ratio': decimated_sample_count / original_sample_count,
                'compression_ratio': original_sample_count / decimated_sample_count,
                'samples_removed': original_sample_count - decimated_sample_count
            }
        })
        
        return output_channel, original_sample_count, decimated_sample_count
    
    def get_output_channel_suffix(self) -> str:
        """Get the suffix for output channel names."""
        return "_decimated"
    
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
        Custom parameter validation for the decimator processor.
        
        Args:
            parameters: Dictionary of parameter values to validate
            
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
        """
        # Call parent validation first
        errors = super().validate_parameters(parameters)
        
        # Add custom validation for decimation_factor
        if 'decimation_factor' in parameters:
            decimation_factor = parameters['decimation_factor']
            
            # Check for reasonable practical limits
            if decimation_factor < 1:
                errors['decimation_factor'] = "Decimation factor must be at least 1"
            elif decimation_factor == 1:
                errors['decimation_factor'] = "Decimation factor of 1 provides no benefit (no decimation)"
            elif decimation_factor > 10000:
                errors['decimation_factor'] = "Decimation factor too large (maximum practical value is 10000)"
            
            # Provide warnings for potentially problematic values
            if 100 < decimation_factor <= 1000:
                print(f"Warning: Large decimation factor {decimation_factor} will significantly reduce data resolution")
            elif decimation_factor > 1000:
                print(f"Warning: Very large decimation factor {decimation_factor} may lose important signal information")
        
        return errors


# Additional utility functions for the decimator processor

def calculate_optimal_decimation(original_sample_count: int, target_sample_count: int) -> int:
    """
    Calculate optimal decimation factor to achieve target sample count.
    
    Args:
        original_sample_count: Number of samples in original signal
        target_sample_count: Desired number of samples after decimation
        
    Returns:
        int: Optimal decimation factor
    """
    if target_sample_count <= 0 or original_sample_count <= 0:
        return 2  # Default safe value
    
    if target_sample_count >= original_sample_count:
        return 1  # No decimation needed
    
    factor = max(2, int(np.ceil(original_sample_count / target_sample_count)))
    return min(factor, 1000)  # Cap at reasonable maximum


def calculate_decimation_for_duration(original_duration: float, original_sample_count: int,
                                     target_duration: float) -> int:
    """
    Calculate decimation factor to achieve a specific time duration reduction.
    
    Args:
        original_duration: Original signal duration in seconds
        original_sample_count: Number of samples in original signal  
        target_duration: Target duration for decimated signal
        
    Returns:
        int: Decimation factor to achieve target duration
    """
    if target_duration <= 0 or original_duration <= 0 or original_sample_count <= 0:
        return 2
    
    if target_duration >= original_duration:
        return 1
    
    # Calculate target sample count based on duration ratio
    duration_ratio = target_duration / original_duration
    target_samples = int(original_sample_count * duration_ratio)
    
    return calculate_optimal_decimation(original_sample_count, target_samples)


def preview_decimation(time_array: np.ndarray, voltage_data: np.ndarray, 
                      decimation_factor: int) -> Dict[str, Any]:
    """
    Preview the effects of decimation without actually processing.
    
    Args:
        time_array: Original time data
        voltage_data: Original voltage data
        decimation_factor: Decimation factor to preview
        
    Returns:
        Dict containing preview statistics
    """
    if len(time_array) == 0 or len(voltage_data) == 0:
        return {'error': 'Empty input data'}
    
    if len(time_array) != len(voltage_data):
        return {'error': 'Time and voltage arrays have different lengths'}
    
    if decimation_factor < 1:
        return {'error': 'Decimation factor must be at least 1'}
    
    try:
        original_count = len(voltage_data)
        decimated_count = len(range(0, original_count, decimation_factor))
        
        # Calculate timing information
        original_duration = float(time_array[-1] - time_array[0]) if len(time_array) > 1 else 0.0
        original_sample_rate = (original_count - 1) / original_duration if original_duration > 0 else 0.0
        new_sample_rate = original_sample_rate / decimation_factor
        new_nyquist_freq = new_sample_rate / 2.0
        
        return {
            'original_sample_count': original_count,
            'decimated_sample_count': decimated_count,
            'data_reduction_ratio': decimated_count / original_count,
            'compression_ratio': original_count / decimated_count,
            'samples_removed': original_count - decimated_count,
            'original_duration': original_duration,
            'original_sample_rate_approx': original_sample_rate,
            'new_sample_rate_approx': new_sample_rate,
            'new_nyquist_frequency_approx': new_nyquist_freq,
            'decimation_factor': decimation_factor,
            'memory_savings_approx': f"{(1 - decimated_count/original_count)*100:.1f}%"
        }
    except Exception as e:
        return {'error': f'Preview calculation failed: {e}'}


def estimate_anti_alias_cutoff(original_sample_rate: float, decimation_factor: int) -> float:
    """
    Estimate the anti-aliasing filter cutoff frequency for safe decimation.
    
    Args:
        original_sample_rate: Original sampling rate in Hz
        decimation_factor: Decimation factor
        
    Returns:
        float: Recommended anti-aliasing cutoff frequency in Hz
    """
    if original_sample_rate <= 0 or decimation_factor < 2:
        return 0.0
    
    new_sample_rate = original_sample_rate / decimation_factor
    new_nyquist = new_sample_rate / 2.0
    
    # Recommend cutoff at ~80% of new Nyquist frequency to allow for filter rolloff
    recommended_cutoff = new_nyquist * 0.8
    
    return recommended_cutoff


def check_aliasing_risk(voltage_data: np.ndarray, time_array: np.ndarray, 
                       decimation_factor: int) -> Dict[str, Any]:
    """
    Analyze potential aliasing risk from decimation without anti-aliasing filter.
    
    Args:
        voltage_data: Original voltage data
        time_array: Original time data
        decimation_factor: Proposed decimation factor
        
    Returns:
        Dict containing aliasing risk assessment
    """
    try:
        if len(voltage_data) == 0 or len(time_array) == 0:
            return {'error': 'Empty input data'}
        
        # Calculate approximate sample rate
        duration = float(time_array[-1] - time_array[0]) if len(time_array) > 1 else 0.0
        original_fs = (len(time_array) - 1) / duration if duration > 0 else 0.0
        
        if original_fs <= 0:
            return {'error': 'Cannot determine original sample rate'}
        
        # Calculate frequency domain characteristics
        fft = np.fft.fft(voltage_data)
        freqs = np.fft.fftfreq(len(voltage_data), 1.0/original_fs)
        power_spectrum = np.abs(fft)**2
        
        # Find new Nyquist frequency after decimation
        new_fs = original_fs / decimation_factor
        new_nyquist = new_fs / 2.0
        
        # Check energy above new Nyquist frequency
        above_nyquist_mask = np.abs(freqs) > new_nyquist
        total_power = np.sum(power_spectrum)
        above_nyquist_power = np.sum(power_spectrum[above_nyquist_mask])
        
        aliasing_risk_ratio = above_nyquist_power / total_power if total_power > 0 else 0.0
        
        # Risk assessment
        if aliasing_risk_ratio < 0.01:
            risk_level = "Low"
        elif aliasing_risk_ratio < 0.05:
            risk_level = "Moderate"
        elif aliasing_risk_ratio < 0.15:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            'original_sample_rate': original_fs,
            'new_sample_rate': new_fs,
            'original_nyquist': original_fs / 2.0,
            'new_nyquist': new_nyquist,
            'decimation_factor': decimation_factor,
            'aliasing_risk_ratio': aliasing_risk_ratio,
            'aliasing_risk_percentage': aliasing_risk_ratio * 100,
            'risk_level': risk_level,
            'recommended_anti_alias_cutoff': estimate_anti_alias_cutoff(original_fs, decimation_factor),
            'analysis_notes': {
                'method': 'FFT-based power spectrum analysis',
                'assumption': 'Uniform time sampling assumed for FFT analysis'
            }
        }
        
    except Exception as e:
        return {'error': f'Aliasing analysis failed: {e}'}

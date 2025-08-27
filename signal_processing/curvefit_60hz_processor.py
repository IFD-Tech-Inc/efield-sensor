#!/usr/bin/env python3
"""
60 Hz Curve Fitting Signal Processor for IFD Signal Analysis Utility.

This module implements a signal processor that fits input waveform data to a 60 Hz
sine wave model using curve fitting algorithms. It extracts amplitude, phase, and 
DC offset parameters and provides both the fitted signal and parameter channels.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Any, Tuple, List
import copy
import warnings

try:
    from .base_processor import SignalProcessor, ProcessingError
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from base_processor import SignalProcessor, ProcessingError


class CurveFit60HzProcessor(SignalProcessor):
    """
    Signal processor for fitting data to 60 Hz sine wave model.
    
    This processor fits input waveform data to a 60 Hz sine wave of the form:
    y(t) = A * sin(2π * 60 * t + φ) + offset
    
    Where:
    - A is the amplitude
    - φ is the phase shift (radians)
    - offset is the DC offset
    
    The processor outputs:
    1. The fitted 60 Hz sine wave signal
    2. Separate channels for amplitude, phase, and offset parameters
    3. Fit quality metrics in logs and metadata
    
    Features:
    - Robust curve fitting with fallback handling
    - Quality metrics (R-squared, residual analysis)
    - Multi-channel processing support
    - Comprehensive error handling
    """
    
    def get_name(self) -> str:
        """Get the display name of this processor."""
        return "Curve Fit: 60Hz"
    
    def get_description(self) -> str:
        """Get a detailed description of what this processor does."""
        return (
            "Fits input waveform data to a 60 Hz sine wave model using curve fitting. "
            "Extracts amplitude, phase shift, and DC offset parameters. Outputs the "
            "fitted 60 Hz sine wave along with separate parameter channels. "
            "Includes fit quality metrics and handles poor fits gracefully."
        )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the parameter definitions for this processor."""
        return {
            'normalize_signals': {
                'type': bool,
                'default': False,
                'description': 'Normalize fitted waveform output to unit amplitude for comparison purposes'
            },
            'remove_offset': {
                'type': bool,
                'default': False,
                'description': 'Remove DC offset from fitted waveform output for comparison purposes'
            },
            'output_fitted_signal': {
                'type': bool,
                'default': True,
                'description': 'Output the fitted 60 Hz sine wave signal'
            },
            'output_parameters': {
                'type': bool,
                'default': True,
                'description': 'Output separate channels for amplitude, phase, and offset'
            },
            'min_fit_quality': {
                'type': float,
                'default': 0.0,
                'min': 0.0,
                'max': 1.0,
                'description': 'Minimum R-squared threshold for acceptable fits (0.0 = no threshold)'
            },
            'max_iterations': {
                'type': int,
                'default': 5000,
                'min': 100,
                'max': 50000,
                'description': 'Maximum iterations for curve fitting algorithm'
            }
        }
    
    def process(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input waveform data by fitting to 60 Hz sine wave model.
        
        Args:
            input_data: Dictionary containing input waveform data
            parameters: Dictionary containing processing parameters
            
        Returns:
            Dictionary containing processed waveform data with fitted signals and parameters
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract parameters
            normalize_signals = parameters.get('normalize_signals', False)
            remove_offset = parameters.get('remove_offset', False)
            output_fitted_signal = parameters.get('output_fitted_signal', True)
            output_parameters = parameters.get('output_parameters', True)
            min_fit_quality = parameters.get('min_fit_quality', 0.0)
            max_iterations = parameters.get('max_iterations', 5000)
            
            # Create output data structure
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
            
            # Track processing statistics
            channels_processed = 0
            total_samples = 0
            fit_quality_stats = []
            
    # Process each channel
            for channel_name, channel_data in input_data['channels'].items():
                try:
                    # Skip channels with insufficient data
                    if len(channel_data.get('time_array', [])) < 10 or len(channel_data.get('voltage_data', [])) < 10:
                        print(f"Warning: Channel {channel_name} has insufficient data points (<10) for curve fitting")
                        # For short signals, create a flat output if possible
                        if len(channel_data.get('voltage_data', [])) > 0 and output_parameters:
                            # Create simplified parameter channels with basic stats
                            simple_results = self._create_simplified_results(channel_name, channel_data)
                            param_channels = self._create_parameter_channels(
                                channel_name, channel_data, simple_results
                            )
                            output_data['channels'].update(param_channels)
                            channels_processed += 1
                        continue
                    
                    # Skip channels with mismatched array lengths
                    if len(channel_data.get('time_array', [])) != len(channel_data.get('voltage_data', [])):
                        print(f"Warning: Channel {channel_name} has mismatched array lengths - skipping")
                        continue
                    
                    fit_results = self._fit_channel_to_60hz(
                        channel_name, channel_data, max_iterations
                    )
                    
                    channels_processed += 1
                    total_samples += len(channel_data.get('voltage_data', []))
                    fit_quality_stats.append(fit_results['r_squared'])
                    
                    # Create output channels based on parameters
                    if output_fitted_signal:
                        fitted_channel = self._create_fitted_signal_channel(
                            channel_name, channel_data, fit_results, normalize_signals, remove_offset
                        )
                        output_data['channels'][f"{channel_name}_60hz_fit"] = fitted_channel
                    
                    if output_parameters:
                        param_channels = self._create_parameter_channels(
                            channel_name, channel_data, fit_results
                        )
                        output_data['channels'].update(param_channels)
                    
                    # Log fit results
                    self._log_fit_results(channel_name, fit_results, min_fit_quality)
                    
                except Exception as e:
                    print(f"Warning: Failed to process channel {channel_name}: {e}")
                    continue
            
            # Verify we processed at least one channel
            if channels_processed == 0:
                raise ProcessingError("No channels were successfully processed")
            
            # Add summary statistics
            output_data['source_info']['processing_summary'] = {
                'channels_processed': channels_processed,
                'total_channels': len(input_data['channels']),
                'total_samples_processed': total_samples,
                'average_fit_quality': np.mean(fit_quality_stats) if fit_quality_stats else 0.0,
                'min_fit_quality': np.min(fit_quality_stats) if fit_quality_stats else 0.0,
                'max_fit_quality': np.max(fit_quality_stats) if fit_quality_stats else 0.0,
                'parameters_used': {
                    'normalize_signals': normalize_signals,
                    'remove_offset': remove_offset,
                    'output_fitted_signal': output_fitted_signal,
                    'output_parameters': output_parameters,
                    'min_fit_quality': min_fit_quality,
                    'max_iterations': max_iterations
                }
            }
            
            print(f"Successfully processed {channels_processed} channels with 60Hz curve fitting")
            print(f"Average fit quality (R²): {np.mean(fit_quality_stats):.4f}")
            
            return output_data
            
        except ProcessingError:
            # Re-raise ProcessingError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ProcessingError
            raise ProcessingError(f"60Hz curve fitting failed: {str(e)}")
    
    def _fit_channel_to_60hz(self, channel_name: str, channel_data: Dict[str, Any], 
                           max_iterations: int) -> Dict[str, Any]:
        """
        Fit a single channel's data to 60 Hz sine wave model.
        
        Args:
            channel_name: Name of the channel being processed
            channel_data: Channel data dictionary
            max_iterations: Maximum iterations for fitting
            
        Returns:
            Dict containing fit results: amplitude, phase, offset, r_squared, etc.
            
        Raises:
            ProcessingError: If channel processing fails critically
        """
        # Extract and validate arrays
        time_array = channel_data['time_array']
        voltage_data = channel_data['voltage_data']
        
        if len(time_array) == 0 or len(voltage_data) == 0:
            raise ProcessingError(f"Channel {channel_name} contains empty arrays")
            
        if len(time_array) != len(voltage_data):
            raise ProcessingError(
                f"Channel {channel_name} time and voltage arrays have mismatched lengths: "
                f"{len(time_array)} vs {len(voltage_data)}"
            )
        
        if len(time_array) < 10:
            raise ProcessingError(f"Channel {channel_name} has insufficient data points for fitting")
        
        # Store original data for reference
        original_voltage_data = voltage_data.copy()
        
        # Calculate initial parameter guesses (no preprocessing)
        initial_guess = self._calculate_initial_guess(time_array, voltage_data)
        
        # Attempt curve fitting
        try:
            # Suppress scipy optimization warnings for cleaner output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    self._sine_60hz_model, 
                    time_array, 
                    voltage_data,
                    p0=initial_guess,
                    maxfev=max_iterations,
                    method='lm'  # Levenberg-Marquardt algorithm
                )
                
                fitted_amplitude, fitted_phase, fitted_offset = popt
                
            # Calculate parameter uncertainties from covariance matrix
                if pcov is not None:
                    # Handle cases where covariance matrix has issues (inf, nan, etc.)
                    diag_cov = np.diag(pcov)
                    param_errors = []
                    for val in diag_cov:
                        if np.isfinite(val) and val >= 0:
                            param_errors.append(np.sqrt(val))
                        else:
                            param_errors.append(0.0)  # Use 0 for invalid error estimates
                else:
                    param_errors = [0.0, 0.0, 0.0]
                
                converged = True
                
        except Exception as fit_error:
            # Fitting failed - use fallback parameters
            print(f"Warning: Curve fit failed for {channel_name}: {fit_error}")
            
            # Fallback: zero amplitude with mean as offset
            fitted_amplitude = 0.0
            fitted_phase = 0.0
            fitted_offset = np.mean(voltage_data) if len(voltage_data) > 0 else 0.0
            param_errors = [0, 0, 0]
            converged = False
        
        # Generate fitted signal
        fitted_signal = self._sine_60hz_model(time_array, fitted_amplitude, fitted_phase, fitted_offset)
        
        # Calculate fit quality metrics
        r_squared = self._calculate_r_squared(voltage_data, fitted_signal)
        rmse = np.sqrt(np.mean((voltage_data - fitted_signal)**2))
        
        return {
            'amplitude': fitted_amplitude,
            'phase': fitted_phase,
            'offset': fitted_offset,
            'amplitude_error': param_errors[0],
            'phase_error': param_errors[1],
            'offset_error': param_errors[2],
            'fitted_signal': fitted_signal,
            'r_squared': r_squared,
            'rmse': rmse,
            'converged': converged,
            'iterations_used': max_iterations,
            'original_signal': original_voltage_data,
            'time_array': time_array.copy()
        }
    
    def _sine_60hz_model(self, t: np.ndarray, amplitude: float, phase: float, offset: float) -> np.ndarray:
        """
        60 Hz sine wave model function.
        
        Args:
            t: Time array
            amplitude: Sine wave amplitude
            phase: Phase shift in radians
            offset: DC offset
            
        Returns:
            Sine wave values: amplitude * sin(2π * 60 * t + phase) + offset
        """
        return amplitude * np.sin(2 * np.pi * 60.0 * t + phase) + offset
    
    def _calculate_initial_guess(self, time_array: np.ndarray, voltage_data: np.ndarray) -> List[float]:
        """
        Calculate reasonable initial parameter guesses for curve fitting.
        
        Args:
            time_array: Time values
            voltage_data: Voltage values
            
        Returns:
            List of [amplitude_guess, phase_guess, offset_guess]
        """
        # Initial amplitude guess: use range/2 as approximation
        voltage_range = np.max(voltage_data) - np.min(voltage_data)
        amplitude_guess = voltage_range / 2.0 if voltage_range > 0 else 1.0
        
        # Initial offset guess: use mean value
        offset_guess = np.mean(voltage_data)
        
        # Initial phase guess: try to estimate from first few cycles
        # For simplicity, start with 0 and let optimizer find best phase
        phase_guess = 0.0
        
        return [amplitude_guess, phase_guess, offset_guess]
    
    def _calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination) for fit quality.
        
        Args:
            actual: Actual data values
            predicted: Fitted/predicted values
            
        Returns:
            R-squared value (0 to 1, higher is better)
        """
        # Handle edge cases
        if len(actual) == 0 or len(predicted) == 0:
            return 0.0
            
        if len(actual) != len(predicted):
            return 0.0
        
        # Calculate sums of squares
        ss_res = np.sum((actual - predicted) ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        
        # Calculate R-squared
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
            return max(0.0, r_squared)  # Ensure non-negative
    
    def _create_fitted_signal_channel(self, channel_name: str, channel_data: Dict[str, Any], 
                                    fit_results: Dict[str, Any], normalize_signals: bool = False,
                                    remove_offset: bool = False) -> Dict[str, Any]:
        """Create output channel for the fitted 60 Hz sine wave with optional post-processing.
        
        Args:
            channel_name: Original channel name
            channel_data: Original channel data
            fit_results: Results from curve fitting
            normalize_signals: If True, normalize fitted signal to unit amplitude
            remove_offset: If True, remove DC offset from fitted signal
            
        Returns:
            Channel data dictionary for fitted signal
        """
        # Start with the original fitted signal
        fitted_signal = fit_results['fitted_signal'].copy()
        time_array = fit_results['time_array']
        
        # Apply post-processing if requested
        postprocessing_applied = {'normalize': False, 'remove_offset': False}
        
        # Remove DC offset from fitted signal if requested
        if remove_offset:
            dc_offset = np.mean(fitted_signal)
            fitted_signal = fitted_signal - dc_offset
            postprocessing_applied['remove_offset'] = True
            print(f"  Post-processing: Removed DC offset from fitted signal: {dc_offset:.4f} V")
        
        # Normalize fitted signal amplitude if requested  
        if normalize_signals:
            signal_range = np.max(fitted_signal) - np.min(fitted_signal)
            if signal_range > 1e-10:  # Avoid division by zero
                signal_center = np.mean(fitted_signal)
                fitted_signal = (fitted_signal - signal_center) / signal_range + signal_center
                postprocessing_applied['normalize'] = True
                print(f"  Post-processing: Normalized fitted signal amplitude (range: {signal_range:.6f} V)")
            else:
                print(f"  Post-processing: Cannot normalize constant fitted signal (range = {signal_range:.2e})")
        
        output_channel = {
            'time_array': time_array,
            'voltage_data': fitted_signal,
            'metadata': copy.deepcopy(channel_data.get('metadata', {}))
        }
        
        # Add processing metadata
        if 'processing' not in output_channel['metadata']:
            output_channel['metadata']['processing'] = {}
            
        output_channel['metadata']['processing'].update({
            'processor': self.get_name(),
            'original_channel': channel_name,
            'fit_parameters': {
                'amplitude': fit_results['amplitude'],
                'phase': fit_results['phase'],
                'offset': fit_results['offset'],
                'frequency_hz': 60.0
            },
            'fit_quality': {
                'r_squared': fit_results['r_squared'],
                'rmse': fit_results['rmse'],
                'converged': fit_results['converged']
            }
        })
        
        return output_channel
    
    def _create_parameter_channels(self, channel_name: str, channel_data: Dict[str, Any], 
                                 fit_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create output channels for fitted parameters (amplitude, phase, offset).
        
        Args:
            channel_name: Original channel name
            channel_data: Original channel data
            fit_results: Results from curve fitting
            
        Returns:
            Dictionary of parameter channels
        """
        time_array = fit_results['time_array']
        param_channels = {}
        
        # Create constant-value arrays for each parameter
        amplitude_array = np.full_like(time_array, fit_results['amplitude'])
        phase_array = np.full_like(time_array, fit_results['phase'])
        offset_array = np.full_like(time_array, fit_results['offset'])
        
        # Base metadata for parameter channels
        base_metadata = copy.deepcopy(channel_data.get('metadata', {}))
        if 'processing' not in base_metadata:
            base_metadata['processing'] = {}
            
        base_metadata['processing'].update({
            'processor': self.get_name(),
            'original_channel': channel_name,
            'parameter_extraction': True,
            'fit_quality': {
                'r_squared': fit_results['r_squared'],
                'rmse': fit_results['rmse'],
                'converged': fit_results['converged']
            }
        })
        
        # Amplitude parameter channel
        amplitude_metadata = copy.deepcopy(base_metadata)
        amplitude_metadata['processing']['parameter_type'] = 'amplitude'
        amplitude_metadata['processing']['units'] = 'volts'
        amplitude_metadata['processing']['parameter_error'] = fit_results['amplitude_error']
        
        param_channels[f"{channel_name}_amplitude"] = {
            'time_array': time_array.copy(),
            'voltage_data': amplitude_array,
            'metadata': amplitude_metadata
        }
        
        # Phase parameter channel
        phase_metadata = copy.deepcopy(base_metadata)
        phase_metadata['processing']['parameter_type'] = 'phase'
        phase_metadata['processing']['units'] = 'radians'
        phase_metadata['processing']['parameter_error'] = fit_results['phase_error']
        
        param_channels[f"{channel_name}_phase"] = {
            'time_array': time_array.copy(),
            'voltage_data': phase_array,
            'metadata': phase_metadata
        }
        
        # Offset parameter channel
        offset_metadata = copy.deepcopy(base_metadata)
        offset_metadata['processing']['parameter_type'] = 'offset'
        offset_metadata['processing']['units'] = 'volts'
        offset_metadata['processing']['parameter_error'] = fit_results['offset_error']
        
        param_channels[f"{channel_name}_offset"] = {
            'time_array': time_array.copy(),
            'voltage_data': offset_array,
            'metadata': offset_metadata
        }
        
        return param_channels
    
    def _log_fit_results(self, channel_name: str, fit_results: Dict[str, Any], 
                       min_fit_quality: float) -> None:
        """
        Log curve fitting results and quality metrics.
        
        Args:
            channel_name: Name of the processed channel
            fit_results: Curve fitting results
            min_fit_quality: Minimum acceptable R-squared value
        """
        r_squared = fit_results['r_squared']
        amplitude = fit_results['amplitude']
        phase = fit_results['phase']
        offset = fit_results['offset']
        converged = fit_results['converged']
        rmse = fit_results['rmse']
        
        # Detailed fit results
        print(f"Channel {channel_name} - 60Hz Curve Fit Results:")
        print(f"  Amplitude: {amplitude:.4f} ± {fit_results['amplitude_error']:.4f} V")
        print(f"  Phase: {phase:.4f} ± {fit_results['phase_error']:.4f} rad ({np.degrees(phase):.1f}°)")
        print(f"  DC Offset: {offset:.4f} ± {fit_results['offset_error']:.4f} V")
        print(f"  R-squared: {r_squared:.6f}")
        print(f"  RMSE: {rmse:.6f} V")
        print(f"  Converged: {'Yes' if converged else 'No'}")
        
        # Quality assessment
        if r_squared < min_fit_quality and min_fit_quality > 0:
            print(f"  WARNING: Fit quality below threshold ({r_squared:.4f} < {min_fit_quality:.4f})")
        elif r_squared > 0.95:
            print(f"  Excellent fit quality (R² = {r_squared:.6f})")
        elif r_squared > 0.80:
            print(f"  Good fit quality (R² = {r_squared:.6f})")
        elif r_squared > 0.50:
            print(f"  Moderate fit quality (R² = {r_squared:.6f})")
        else:
            print(f"  Poor fit quality (R² = {r_squared:.6f}) - signal may not be 60Hz")
        
        if not converged:
            print(f"  WARNING: Curve fitting did not converge for {channel_name}")
        
        print()  # Empty line for readability
    
    def get_output_channel_suffix(self) -> str:
        """Get the suffix for output channel names."""
        return "_60hz"
    
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
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp for processing history.
        
        Returns:
            str: ISO format timestamp
        """
        import datetime
        return datetime.datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Custom parameter validation for the curve fitting processor.
        
        Args:
            parameters: Dictionary of parameter values to validate
            
        Returns:
            Dict[str, str]: Dictionary of validation errors (empty if valid)
        """
        # Call parent validation first
        errors = super().validate_parameters(parameters)
        
        # Add custom validation
        if 'min_fit_quality' in parameters:
            min_quality = parameters['min_fit_quality']
            if min_quality < 0.0 or min_quality > 1.0:
                errors['min_fit_quality'] = "Minimum fit quality must be between 0.0 and 1.0"
        
        if 'max_iterations' in parameters:
            max_iter = parameters['max_iterations']
            if max_iter < 100:
                errors['max_iterations'] = "Maximum iterations should be at least 100 for reliable fitting"
            elif max_iter > 50000:
                errors['max_iterations'] = "Maximum iterations too high (may cause performance issues)"
        
        # Ensure at least one output type is selected
        output_fitted = parameters.get('output_fitted_signal', True)
        output_params = parameters.get('output_parameters', True)
        if not output_fitted and not output_params:
            errors['output_fitted_signal'] = "At least one output type must be enabled"
            errors['output_parameters'] = "At least one output type must be enabled"
        
        return errors


    def _create_simplified_results(self, channel_name: str, channel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create simplified results for channels with insufficient data for curve fitting.
        
        This provides basic metadata for parameter channels when a full curve fit cannot be performed.
        
        Args:
            channel_name: Name of the channel
            channel_data: Channel data dictionary
            
        Returns:
            Dict with simplified fit results
        """
        # Extract what data we can
        voltage_data = channel_data.get('voltage_data', np.array([0.0]))
        time_array = channel_data.get('time_array', np.array([0.0]))
        
        # Create simple results with basic statistics
        offset = np.mean(voltage_data) if len(voltage_data) > 0 else 0.0
        amplitude = (np.max(voltage_data) - np.min(voltage_data)) / 2.0 if len(voltage_data) > 1 else 0.0
        
        return {
            'amplitude': amplitude,
            'phase': 0.0,
            'offset': offset,
            'amplitude_error': 0.0,
            'phase_error': 0.0,
            'offset_error': 0.0,
            'fitted_signal': np.full_like(voltage_data, offset) if len(voltage_data) > 0 else np.array([offset]),
            'r_squared': 0.0,
            'rmse': 0.0,
            'converged': False,
            'iterations_used': 0,
            'original_signal': voltage_data.copy() if len(voltage_data) > 0 else np.array([0.0]),
            'time_array': time_array.copy() if len(time_array) > 0 else np.array([0.0]),
            'insufficient_data': True
        }


# Utility functions for advanced curve fitting analysis


def analyze_60hz_content(time_array: np.ndarray, voltage_data: np.ndarray) -> Dict[str, float]:
    """
    Analyze how much 60 Hz content is present in a signal using FFT.
    
    Args:
        time_array: Time values
        voltage_data: Voltage values
        
    Returns:
        Dictionary with 60Hz analysis results
    """
    try:
        # Calculate sampling frequency
        if len(time_array) < 2:
            return {'error': 'Insufficient data points'}
        
        dt = np.mean(np.diff(time_array))
        fs = 1.0 / dt
        
        # Perform FFT
        n = len(voltage_data)
        fft_result = np.fft.fft(voltage_data)
        freqs = np.fft.fftfreq(n, dt)
        
        # Find 60 Hz component
        freq_60hz_idx = np.argmin(np.abs(freqs - 60.0))
        freq_60hz_power = np.abs(fft_result[freq_60hz_idx])
        
        # Calculate total power (excluding DC component)
        dc_idx = 0
        fft_no_dc = fft_result.copy()
        fft_no_dc[dc_idx] = 0  # Remove DC component 
        total_power = np.sum(np.abs(fft_no_dc))
        
        # Calculate 60 Hz power percentage
        if total_power > 0:
            power_60hz_percent = (freq_60hz_power / total_power) * 100
        else:
            power_60hz_percent = 0.0
        
        return {
            'sampling_frequency': fs,
            'freq_60hz_power': freq_60hz_power,
            'total_power': total_power,
            'power_60hz_percent': power_60hz_percent,
            'actual_freq_at_peak': freqs[freq_60hz_idx]
        }
        
    except Exception as e:
        return {'error': f'FFT analysis failed: {e}'}


def preview_60hz_fit(time_array: np.ndarray, voltage_data: np.ndarray) -> Dict[str, Any]:
    """
    Preview 60 Hz curve fitting results without full processing.
    
    Args:
        time_array: Time values
        voltage_data: Voltage values
        
    Returns:
        Dictionary with preview results
    """
    try:
        processor = CurveFit60HzProcessor()
        
        # Simulate channel data structure
        channel_data = {
            'time_array': time_array,
            'voltage_data': voltage_data
        }
        
        # Perform fitting
        fit_results = processor._fit_channel_to_60hz('preview', channel_data, 5000)
        
        return {
            'amplitude': fit_results['amplitude'],
            'phase': fit_results['phase'],
            'phase_degrees': np.degrees(fit_results['phase']),
            'offset': fit_results['offset'],
            'r_squared': fit_results['r_squared'],
            'rmse': fit_results['rmse'],
            'converged': fit_results['converged'],
            'fit_quality_description': _describe_fit_quality(fit_results['r_squared'])
        }
        
    except Exception as e:
        return {'error': f'Preview failed: {e}'}


def _describe_fit_quality(r_squared: float) -> str:
    """
    Provide human-readable description of fit quality.
    
    Args:
        r_squared: R-squared value
        
    Returns:
        Quality description string
    """
    if r_squared > 0.95:
        return "Excellent (R² > 0.95)"
    elif r_squared > 0.80:
        return "Good (0.80 < R² ≤ 0.95)"
    elif r_squared > 0.50:
        return "Moderate (0.50 < R² ≤ 0.80)"
    elif r_squared > 0.20:
        return "Poor (0.20 < R² ≤ 0.50)"
    else:
        return "Very Poor (R² ≤ 0.20)"

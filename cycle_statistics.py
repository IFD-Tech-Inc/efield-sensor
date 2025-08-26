"""
Cycle-by-cycle statistical analysis module for E-field sensor data.

This module provides comprehensive statistical analysis of voltage and phase measurements
on a cycle-by-cycle basis, including:
- Average voltage and phase per cycle
- Standard deviations for voltage and phase
- Peak-to-peak voltage analysis
- Frequency stability metrics
- Total Harmonic Distortion (THD) estimation
- Cycle-to-cycle variation analysis

Author: E-field Sensor Analysis Tool
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings


class CycleDetector:
    """Detects individual cycles in sinusoidal signals."""
    
    def __init__(self, signal: np.ndarray, time: np.ndarray, expected_frequency: float = 60.0):
        """
        Initialize cycle detector.
        
        Args:
            signal: The input signal array
            time: Time array corresponding to signal
            expected_frequency: Expected fundamental frequency in Hz (default: 60 Hz)
        """
        self.signal = signal
        self.time = time
        self.dt = np.mean(np.diff(time))
        self.fs = 1.0 / self.dt
        self.expected_frequency = expected_frequency
        self.cycles = []
        
    def detect_zero_crossings(self, signal: np.ndarray, hysteresis_factor: float = 0.05) -> List[int]:
        """
        Detect zero crossing indices in the signal with hysteresis to reduce noise sensitivity.
        
        Args:
            signal: Input signal array
            hysteresis_factor: Hysteresis threshold as fraction of signal range (default: 0.05 = 5%)
            
        Returns:
            List of indices where zero crossings occur
        """
        # Remove DC offset
        signal_centered = signal - np.mean(signal)
        
        # Calculate hysteresis thresholds to avoid noise-induced false crossings
        signal_range = np.max(signal_centered) - np.min(signal_centered)
        hysteresis_threshold = hysteresis_factor * signal_range
        
        # State machine for hysteresis-based zero crossing detection
        crossings = []
        state = 0  # 0: unknown, 1: positive, -1: negative
        
        for i in range(len(signal_centered)):
            value = signal_centered[i]
            
            if state == 0:  # Initialize state
                if value > hysteresis_threshold:
                    state = 1
                elif value < -hysteresis_threshold:
                    state = -1
            elif state == 1:  # Currently positive
                if value < -hysteresis_threshold:
                    # Found negative-going zero crossing
                    # Find more precise crossing point by interpolating backwards
                    for j in range(i, max(0, i-10), -1):
                        if signal_centered[j] > 0 >= signal_centered[j+1] if j+1 < len(signal_centered) else False:
                            crossings.append(j)
                            break
                    state = -1
            elif state == -1:  # Currently negative
                if value > hysteresis_threshold:
                    # Found positive-going zero crossing
                    # Find more precise crossing point by interpolating backwards
                    for j in range(i, max(0, i-10), -1):
                        if signal_centered[j] < 0 <= signal_centered[j+1] if j+1 < len(signal_centered) else False:
                            crossings.append(j)
                            break
                    state = 1
        
        return crossings
    
    def detect_cycles_zero_crossing(self) -> List[Dict]:
        """
        Detect cycles using zero-crossing method.
        Each cycle is from one positive zero crossing to the next.
        
        Returns:
            List of dictionaries containing cycle information
        """
        zero_crossings = self.detect_zero_crossings(self.signal)
        
        if len(zero_crossings) < 2:
            return []
        
        cycles = []
        
        # Filter for positive-going zero crossings (rising edge)
        positive_crossings = []
        signal_centered = self.signal - np.mean(self.signal)
        
        for crossing in zero_crossings:
            if crossing < len(signal_centered) - 1:
                # Check if this is a positive-going crossing
                if signal_centered[crossing + 1] > signal_centered[crossing]:
                    positive_crossings.append(crossing)
        
        # Create cycles from consecutive positive crossings
        for i in range(len(positive_crossings) - 1):
            start_idx = positive_crossings[i]
            end_idx = positive_crossings[i + 1]
            
            if end_idx > start_idx:  # Valid cycle
                cycle_data = {
                    'cycle_number': i + 1,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': self.time[start_idx],
                    'end_time': self.time[end_idx],
                    'duration': self.time[end_idx] - self.time[start_idx],
                    'signal_segment': self.signal[start_idx:end_idx + 1],
                    'time_segment': self.time[start_idx:end_idx + 1]
                }
                
                cycle_data['frequency'] = 1.0 / cycle_data['duration'] if cycle_data['duration'] > 0 else 0
                cycles.append(cycle_data)
        
        return cycles
    
    def detect_cycles_peak_to_peak(self) -> List[Dict]:
        """
        Alternative cycle detection using peak-to-peak method.
        Each cycle is from one positive peak to the next.
        
        Returns:
            List of dictionaries containing cycle information
        """
        # Find positive peaks with prominence filtering
        min_prominence = 0.1 * (np.max(self.signal) - np.min(self.signal))
        peaks, _ = find_peaks(self.signal, prominence=min_prominence)
        
        if len(peaks) < 2:
            return []
        
        cycles = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            cycle_data = {
                'cycle_number': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': self.time[start_idx],
                'end_time': self.time[end_idx],
                'duration': self.time[end_idx] - self.time[start_idx],
                'signal_segment': self.signal[start_idx:end_idx + 1],
                'time_segment': self.time[start_idx:end_idx + 1]
            }
            
            cycle_data['frequency'] = 1.0 / cycle_data['duration'] if cycle_data['duration'] > 0 else 0
            cycles.append(cycle_data)
        
        return cycles


class CycleStatistics:
    """Calculate comprehensive statistics for detected cycles."""
    
    def __init__(self, cycles: List[Dict]):
        """
        Initialize with detected cycles.
        
        Args:
            cycles: List of cycle dictionaries from CycleDetector
        """
        self.cycles = cycles
        self.stats = {}
        
    def calculate_voltage_statistics(self) -> Dict:
        """
        Calculate voltage-related statistics for each cycle.
        
        Returns:
            Dictionary containing voltage statistics
        """
        if not self.cycles:
            return {}
        
        voltage_stats = {
            'cycle_averages': [],
            'cycle_rms': [],
            'cycle_peak_to_peak': [],
            'cycle_max': [],
            'cycle_min': [],
            'cycle_std': []
        }
        
        for cycle in self.cycles:
            signal = cycle['signal_segment']
            
            # Basic voltage statistics
            voltage_stats['cycle_averages'].append(np.mean(signal))
            voltage_stats['cycle_rms'].append(np.sqrt(np.mean(signal**2)))
            voltage_stats['cycle_peak_to_peak'].append(np.max(signal) - np.min(signal))
            voltage_stats['cycle_max'].append(np.max(signal))
            voltage_stats['cycle_min'].append(np.min(signal))
            voltage_stats['cycle_std'].append(np.std(signal))
        
        # Overall statistics across all cycles
        voltage_stats['overall'] = {
            'mean_average': np.mean(voltage_stats['cycle_averages']),
            'std_average': np.std(voltage_stats['cycle_averages']),
            'mean_rms': np.mean(voltage_stats['cycle_rms']),
            'std_rms': np.std(voltage_stats['cycle_rms']),
            'mean_peak_to_peak': np.mean(voltage_stats['cycle_peak_to_peak']),
            'std_peak_to_peak': np.std(voltage_stats['cycle_peak_to_peak']),
            'mean_cycle_std': np.mean(voltage_stats['cycle_std']),
            'std_cycle_std': np.std(voltage_stats['cycle_std'])
        }
        
        return voltage_stats
    
    def calculate_phase_statistics(self, reference_signal: Optional[np.ndarray] = None,
                                 reference_time: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate phase-related statistics for each cycle.
        
        Args:
            reference_signal: Optional reference signal for phase comparison
            reference_time: Time array for reference signal
            
        Returns:
            Dictionary containing phase statistics
        """
        if not self.cycles:
            return {}
        
        phase_stats = {
            'cycle_phases': [],
            'cycle_phase_unwrapped': [],
            'instantaneous_phases': []
        }
        
        for cycle in self.cycles:
            signal = cycle['signal_segment']
            time_seg = cycle['time_segment']
            
            # Use Hilbert transform to get instantaneous phase
            try:
                analytic_signal = hilbert(signal - np.mean(signal))
                instantaneous_phase = np.angle(analytic_signal)
                
                # Unwrap phase to avoid discontinuities
                phase_unwrapped = np.unwrap(instantaneous_phase)
                
                # Store phase information
                phase_stats['cycle_phases'].append(instantaneous_phase)
                phase_stats['cycle_phase_unwrapped'].append(phase_unwrapped)
                phase_stats['instantaneous_phases'].extend(instantaneous_phase)
                
            except Exception as e:
                warnings.warn(f"Phase calculation failed for cycle {cycle['cycle_number']}: {e}")
                phase_stats['cycle_phases'].append([])
                phase_stats['cycle_phase_unwrapped'].append([])
        
        # Calculate phase statistics if we have valid data
        if phase_stats['instantaneous_phases']:
            valid_phases = [p for p in phase_stats['instantaneous_phases'] if not np.isnan(p)]
            
            if valid_phases:
                # Convert to degrees for easier interpretation
                phases_deg = np.degrees(valid_phases)
                
                phase_stats['overall'] = {
                    'mean_phase_deg': np.mean(phases_deg),
                    'std_phase_deg': np.std(phases_deg),
                    'phase_range_deg': np.max(phases_deg) - np.min(phases_deg)
                }
        
        return phase_stats
    
    def calculate_frequency_statistics(self) -> Dict:
        """
        Calculate frequency stability statistics across cycles.
        
        Returns:
            Dictionary containing frequency statistics
        """
        if not self.cycles:
            return {}
        
        frequencies = [cycle['frequency'] for cycle in self.cycles if cycle['frequency'] > 0]
        
        if not frequencies:
            return {}
        
        freq_stats = {
            'cycle_frequencies': frequencies,
            'mean_frequency': np.mean(frequencies),
            'std_frequency': np.std(frequencies),
            'min_frequency': np.min(frequencies),
            'max_frequency': np.max(frequencies),
            'frequency_stability_ppm': (np.std(frequencies) / np.mean(frequencies)) * 1e6 if np.mean(frequencies) > 0 else 0,
            'frequency_drift': np.polyfit(range(len(frequencies)), frequencies, 1)[0] if len(frequencies) > 1 else 0
        }
        
        return freq_stats
    
    def calculate_thd_estimate(self) -> Dict:
        """
        Estimate Total Harmonic Distortion for each cycle.
        
        Returns:
            Dictionary containing THD estimates
        """
        if not self.cycles:
            return {}
        
        thd_stats = {
            'cycle_thd': [],
            'mean_thd': 0,
            'std_thd': 0
        }
        
        for cycle in self.cycles:
            signal = cycle['signal_segment']
            time_seg = cycle['time_segment']
            
            if len(signal) < 10:  # Need minimum points for FFT
                continue
                
            try:
                # Perform FFT
                dt = np.mean(np.diff(time_seg))
                freqs = fftfreq(len(signal), dt)
                fft_vals = fft(signal - np.mean(signal))
                
                # Find fundamental frequency peak
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft_vals[:len(freqs)//2])
                
                if len(positive_fft) > 1:
                    fundamental_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC
                    fundamental_power = positive_fft[fundamental_idx]**2
                    
                    # Calculate harmonic power (simple estimation)
                    total_power = np.sum(positive_fft[1:]**2)  # Exclude DC
                    harmonic_power = total_power - fundamental_power
                    
                    if fundamental_power > 0:
                        thd = np.sqrt(harmonic_power / fundamental_power) * 100
                        thd_stats['cycle_thd'].append(thd)
            
            except Exception as e:
                warnings.warn(f"THD calculation failed for cycle {cycle['cycle_number']}: {e}")
        
        if thd_stats['cycle_thd']:
            thd_stats['mean_thd'] = np.mean(thd_stats['cycle_thd'])
            thd_stats['std_thd'] = np.std(thd_stats['cycle_thd'])
        
        return thd_stats


def analyze_cycles(signal: np.ndarray, time: np.ndarray, 
                  expected_freq: float = 60.0, 
                  detection_method: str = 'zero_crossing') -> Dict:
    """
    Perform comprehensive cycle-by-cycle analysis of a signal.
    
    Args:
        signal: Input signal array
        time: Time array corresponding to signal
        expected_freq: Expected fundamental frequency in Hz
        detection_method: 'zero_crossing' or 'peak_to_peak'
        
    Returns:
        Dictionary containing comprehensive cycle analysis results
    """
    # Initialize detector
    detector = CycleDetector(signal, time, expected_freq)
    
    # Detect cycles
    if detection_method == 'zero_crossing':
        cycles = detector.detect_cycles_zero_crossing()
    else:
        cycles = detector.detect_cycles_peak_to_peak()
    
    if not cycles:
        return {
            'error': 'No cycles detected',
            'num_cycles': 0,
            'detection_method': detection_method
        }
    
    # Calculate statistics
    stats_calculator = CycleStatistics(cycles)
    
    results = {
        'num_cycles': len(cycles),
        'detection_method': detection_method,
        'analysis_duration': time[-1] - time[0],
        'voltage_stats': stats_calculator.calculate_voltage_statistics(),
        'phase_stats': stats_calculator.calculate_phase_statistics(),
        'frequency_stats': stats_calculator.calculate_frequency_statistics(),
        'thd_stats': stats_calculator.calculate_thd_estimate(),
        'cycles': cycles  # Include raw cycle data for debugging
    }
    
    return results


def print_cycle_analysis_summary(results: Dict, channel_name: str = "Signal"):
    """
    Print a formatted summary of cycle analysis results.
    
    Args:
        results: Results dictionary from analyze_cycles()
        channel_name: Name of the channel being analyzed
    """
    print(f"\n{'='*60}")
    print(f"CYCLE-BY-CYCLE ANALYSIS - {channel_name}")
    print(f"{'='*60}")
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"Detection Method: {results['detection_method']}")
    print(f"Analysis Duration: {results['analysis_duration']:.3f} seconds")
    print(f"Number of Complete Cycles: {results['num_cycles']}")
    
    if results['num_cycles'] == 0:
        return
    
    # Voltage Statistics
    print(f"\n--- VOLTAGE STATISTICS ---")
    v_stats = results['voltage_stats']['overall']
    print(f"Average Voltage per Cycle:")
    print(f"  Mean: {v_stats['mean_average']:.4f} V")
    print(f"  Std Dev: {v_stats['std_average']:.4f} V")
    print(f"  Variation: {(v_stats['std_average']/abs(v_stats['mean_average'])*100) if v_stats['mean_average'] != 0 else 0:.2f}%")
    
    print(f"RMS Voltage per Cycle:")
    print(f"  Mean: {v_stats['mean_rms']:.4f} V")
    print(f"  Std Dev: {v_stats['std_rms']:.4f} V")
    print(f"  Variation: {(v_stats['std_rms']/v_stats['mean_rms']*100) if v_stats['mean_rms'] != 0 else 0:.2f}%")
    
    print(f"Peak-to-Peak Voltage:")
    print(f"  Mean: {v_stats['mean_peak_to_peak']:.4f} V")
    print(f"  Std Dev: {v_stats['std_peak_to_peak']:.4f} V")
    print(f"  Variation: {(v_stats['std_peak_to_peak']/v_stats['mean_peak_to_peak']*100) if v_stats['mean_peak_to_peak'] != 0 else 0:.2f}%")
    
    print(f"Intra-Cycle Voltage Std Dev:")
    print(f"  Mean: {v_stats['mean_cycle_std']:.4f} V")
    print(f"  Std Dev: {v_stats['std_cycle_std']:.4f} V")
    
    # Frequency Statistics  
    if 'frequency_stats' in results and results['frequency_stats']:
        print(f"\n--- FREQUENCY STATISTICS ---")
        f_stats = results['frequency_stats']
        print(f"Mean Frequency: {f_stats['mean_frequency']:.6f} Hz")
        print(f"Frequency Std Dev: {f_stats['std_frequency']:.6f} Hz")
        print(f"Frequency Range: {f_stats['min_frequency']:.6f} - {f_stats['max_frequency']:.6f} Hz")
        print(f"Frequency Stability: {f_stats['frequency_stability_ppm']:.1f} ppm")
        print(f"Frequency Drift: {f_stats['frequency_drift']:.2e} Hz/cycle")
    
    # Phase Statistics
    if 'phase_stats' in results and 'overall' in results['phase_stats']:
        print(f"\n--- PHASE STATISTICS ---")
        p_stats = results['phase_stats']['overall']
        print(f"Average Phase: {p_stats['mean_phase_deg']:.2f}°")
        print(f"Phase Std Dev: {p_stats['std_phase_deg']:.2f}°")
        print(f"Phase Range: {p_stats['phase_range_deg']:.2f}°")
    
    # THD Statistics
    if 'thd_stats' in results and results['thd_stats']['cycle_thd']:
        print(f"\n--- HARMONIC DISTORTION ---")
        thd_stats = results['thd_stats']
        print(f"Mean THD: {thd_stats['mean_thd']:.3f}%")
        print(f"THD Std Dev: {thd_stats['std_thd']:.3f}%")
    
    print(f"\n{'='*60}")


def compare_cycle_statistics(results1: Dict, results2: Dict, 
                           name1: str = "Signal 1", name2: str = "Signal 2"):
    """
    Compare cycle statistics between two signals (e.g., E-field vs Mains).
    
    Args:
        results1: Cycle analysis results for first signal
        results2: Cycle analysis results for second signal  
        name1: Name for first signal
        name2: Name for second signal
    """
    print(f"\n{'='*70}")
    print(f"COMPARATIVE CYCLE ANALYSIS: {name1} vs {name2}")
    print(f"{'='*70}")
    
    if 'error' in results1 or 'error' in results2:
        print("Cannot compare - one or both analyses failed")
        return
    
    # Voltage comparison
    if results1.get('voltage_stats') and results2.get('voltage_stats'):
        v1 = results1['voltage_stats']['overall']
        v2 = results2['voltage_stats']['overall']
        
        print(f"VOLTAGE COMPARISON:")
        print(f"{'Metric':<25} {name1:<15} {name2:<15} {'Ratio':<10}")
        print(f"{'-'*70}")
        print(f"{'Mean RMS (V)':<25} {v1['mean_rms']:<15.4f} {v2['mean_rms']:<15.4f} {v1['mean_rms']/v2['mean_rms'] if v2['mean_rms'] != 0 else float('inf'):<10.3f}")
        print(f"{'RMS Stability (%)':<25} {v1['std_rms']/v1['mean_rms']*100 if v1['mean_rms'] != 0 else 0:<15.2f} {v2['std_rms']/v2['mean_rms']*100 if v2['mean_rms'] != 0 else 0:<15.2f}")
        print(f"{'Mean P-P (V)':<25} {v1['mean_peak_to_peak']:<15.4f} {v2['mean_peak_to_peak']:<15.4f} {v1['mean_peak_to_peak']/v2['mean_peak_to_peak'] if v2['mean_peak_to_peak'] != 0 else float('inf'):<10.3f}")
    
    # Frequency comparison
    if results1.get('frequency_stats') and results2.get('frequency_stats'):
        f1 = results1['frequency_stats']
        f2 = results2['frequency_stats']
        
        print(f"\nFREQUENCY COMPARISON:")
        print(f"{'Mean Frequency (Hz)':<25} {f1['mean_frequency']:<15.6f} {f2['mean_frequency']:<15.6f}")
        print(f"{'Freq Stability (ppm)':<25} {f1['frequency_stability_ppm']:<15.1f} {f2['frequency_stability_ppm']:<15.1f}")
        print(f"{'Freq Coherence':<25} {abs(f1['mean_frequency'] - f2['mean_frequency']) / max(f1['mean_frequency'], f2['mean_frequency']) * 100:<15.3f}% difference")
    
    print(f"{'='*70}")

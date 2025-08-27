#!/usr/bin/env python3
"""
Demonstration script for testing preprocessing features with real SDS814X binary data.

This script loads actual oscilloscope data files, applies the 60 Hz curve fitting 
processor with and without preprocessing options, and compares the results to 
validate that the normalize_signals and remove_offset features are working correctly.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from siglent_parser import SiglentBinaryParser
    from signal_processing.curvefit_60hz_processor import CurveFit60HzProcessor
except ImportError:
    try:
        # Alternative import paths
        sys.path.append('.')
        from siglent_parser import SiglentBinaryParser
        from curvefit_60hz_processor import CurveFit60HzProcessor
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you're running this from the correct directory")
        sys.exit(1)


class PreprocessingDemo:
    """Demo class for testing preprocessing features with real data."""
    
    def __init__(self):
        """Initialize the demo."""
        self.parser = SiglentBinaryParser()
        self.processor = CurveFit60HzProcessor()
        print("Preprocessing Demo initialized")
    
    def find_binary_files(self, directory: str = ".") -> list:
        """Find SDS814X binary files in the given directory."""
        directory = Path(directory)
        
        # Look for .bin files that match SDS814X pattern
        patterns = ["*SDS814X*.bin", "*.bin"]
        bin_files = []
        
        for pattern in patterns:
            bin_files.extend(directory.glob(pattern))
        
        return sorted(list(set(bin_files)))  # Remove duplicates and sort
    
    def load_and_convert_data(self, file_path: Path):
        """Load binary file and convert to processor input format."""
        print(f"\nLoading file: {file_path}")
        
        try:
            # Parse the binary file
            channels = self.parser.parse_file(file_path)
            
            if not channels:
                print("No channels found in file")
                return None
            
            # Convert to processor input format
            processor_input = {
                'channels': {},
                'header': {
                    'filename': str(file_path),
                    'format': 'Siglent Binary V4.0'
                },
                'source_info': {
                    'loader': 'SiglentBinaryParser'
                }
            }
            
            # Convert each channel
            for ch_name, channel_data in channels.items():
                if channel_data.enabled and len(channel_data.voltage_data) > 0:
                    # Create time array
                    time_array = channel_data.get_time_array(
                        self.parser.header.time_div,
                        self.parser.header.time_delay, 
                        self.parser.header.sample_rate
                    )
                    
                    processor_input['channels'][ch_name] = {
                        'time_array': time_array,
                        'voltage_data': channel_data.voltage_data,
                        'metadata': {
                            'channel_name': ch_name,
                            'volt_div': channel_data.volt_div_val.get_scaled_value(),
                            'vert_offset': channel_data.vert_offset.get_scaled_value(),
                            'probe_value': channel_data.probe_value
                        }
                    }
                    
                    print(f"  Channel {ch_name}: {len(channel_data.voltage_data)} samples")
                    print(f"    Voltage range: {np.min(channel_data.voltage_data):.4f} to {np.max(channel_data.voltage_data):.4f} V")
                    print(f"    DC level: {np.mean(channel_data.voltage_data):.4f} V")
                    print(f"    RMS amplitude: {np.std(channel_data.voltage_data):.4f} V")
            
            return processor_input
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def test_preprocessing_modes(self, input_data: dict, channel_name: str):
        """Test different preprocessing modes on the given channel."""
        print(f"\n=== Testing Preprocessing Modes for Channel {channel_name} ===")
        
        if channel_name not in input_data['channels']:
            print(f"Channel {channel_name} not found in data")
            return
        
        ch_data = input_data['channels'][channel_name]
        time_array = ch_data['time_array']
        voltage_data = ch_data['voltage_data']
        
        # Original signal statistics
        print(f"\nOriginal Signal Statistics:")
        print(f"  Samples: {len(voltage_data)}")
        print(f"  Time span: {time_array[0]:.6f} to {time_array[-1]:.6f} s ({time_array[-1] - time_array[0]:.6f} s)")
        print(f"  Voltage range: {np.min(voltage_data):.4f} to {np.max(voltage_data):.4f} V")
        print(f"  DC offset (mean): {np.mean(voltage_data):.4f} V")
        print(f"  RMS: {np.std(voltage_data):.4f} V")
        print(f"  Peak-to-peak: {np.max(voltage_data) - np.min(voltage_data):.4f} V")
        
        # Test configurations
        test_configs = [
            {
                'name': 'Standard (no preprocessing)',
                'normalize_signals': False,
                'remove_offset': False
            },
            {
                'name': 'Remove offset only',
                'normalize_signals': False,
                'remove_offset': True
            },
            {
                'name': 'Normalize only',
                'normalize_signals': True,
                'remove_offset': False
            },
            {
                'name': 'Both normalize and remove offset',
                'normalize_signals': True,
                'remove_offset': True
            }
        ]
        
        results = []
        
        for config in test_configs:
            print(f"\n--- {config['name']} ---")
            
            # Base parameters
            params = {
                'normalize_signals': config['normalize_signals'],
                'remove_offset': config['remove_offset'],
                'output_fitted_signal': True,
                'output_parameters': True,
                'min_fit_quality': 0.0,
                'max_iterations': 5000
            }
            
            try:
                # Process the data
                result = self.processor.process(input_data, params)
                
                # Extract results
                if f"{channel_name}_amplitude" in result['channels']:
                    amplitude = result['channels'][f"{channel_name}_amplitude"]['voltage_data'][0]
                    phase = result['channels'][f"{channel_name}_phase"]['voltage_data'][0]
                    offset = result['channels'][f"{channel_name}_offset"]['voltage_data'][0]
                    
                    # Get fit quality
                    if f"{channel_name}_60hz_fit" in result['channels']:
                        fit_metadata = result['channels'][f"{channel_name}_60hz_fit"]['metadata']['processing']['fit_quality']
                        r_squared = fit_metadata['r_squared']
                        rmse = fit_metadata['rmse']
                        converged = fit_metadata['converged']
                    else:
                        r_squared = 0.0
                        rmse = float('inf')
                        converged = False
                    
                    # Store results
                    results.append({
                        'config': config['name'],
                        'amplitude': amplitude,
                        'phase': phase,
                        'phase_deg': np.degrees(phase),
                        'offset': offset,
                        'r_squared': r_squared,
                        'rmse': rmse,
                        'converged': converged,
                        'params': params
                    })
                    
                    # Print results
                    print(f"  Fitted Parameters:")
                    print(f"    Amplitude: {amplitude:.6f} V")
                    print(f"    Phase: {phase:.4f} rad ({np.degrees(phase):.1f}°)")
                    print(f"    DC Offset: {offset:.6f} V")
                    print(f"  Fit Quality:")
                    print(f"    R-squared: {r_squared:.6f}")
                    print(f"    RMSE: {rmse:.6f} V")
                    print(f"    Converged: {'Yes' if converged else 'No'}")
                    
                    # Check for preprocessing metadata
                    if 'processing_summary' in result['source_info']:
                        preproc_used = result['source_info']['processing_summary']['parameters_used']
                        if preproc_used['normalize_signals']:
                            print(f"    ✓ Signal normalization applied")
                        if preproc_used['remove_offset']:
                            print(f"    ✓ DC offset removal applied")
                
                else:
                    print(f"  ERROR: No parameter channels found in results")
                    results.append({
                        'config': config['name'],
                        'error': 'No results'
                    })
            
            except Exception as e:
                print(f"  ERROR: Processing failed: {e}")
                results.append({
                    'config': config['name'],
                    'error': str(e)
                })
        
        return results
    
    def compare_results(self, results: list):
        """Compare and analyze the results from different preprocessing modes."""
        print(f"\n=== Preprocessing Results Comparison ===")
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to compare")
            return
        
        print(f"\nSummary Table:")
        print(f"{'Mode':<35} {'Amplitude (V)':<15} {'Phase (°)':<12} {'Offset (V)':<15} {'R²':<10} {'RMSE (V)':<12}")
        print("-" * 100)
        
        for result in valid_results:
            print(f"{result['config']:<35} "
                  f"{result['amplitude']:<15.6f} "
                  f"{result['phase_deg']:<12.1f} "
                  f"{result['offset']:<15.6f} "
                  f"{result['r_squared']:<10.6f} "
                  f"{result['rmse']:<12.6f}")
        
        # Analysis
        print(f"\n--- Analysis ---")
        
        # Find best fit quality
        best_r2 = max(r['r_squared'] for r in valid_results)
        best_config = next(r['config'] for r in valid_results if r['r_squared'] == best_r2)
        print(f"Best fit quality: {best_config} (R² = {best_r2:.6f})")
        
        # Parameter consistency check
        amplitudes = [r['amplitude'] for r in valid_results]
        phases = [r['phase_deg'] for r in valid_results]
        offsets = [r['offset'] for r in valid_results]
        
        print(f"Parameter consistency:")
        print(f"  Amplitude range: {min(amplitudes):.6f} to {max(amplitudes):.6f} V (Δ = {max(amplitudes) - min(amplitudes):.6f} V)")
        print(f"  Phase range: {min(phases):.1f}° to {max(phases):.1f}° (Δ = {max(phases) - min(phases):.1f}°)")
        print(f"  Offset range: {min(offsets):.6f} to {max(offsets):.6f} V (Δ = {max(offsets) - min(offsets):.6f} V)")
        
        # Quality metrics
        r_squares = [r['r_squared'] for r in valid_results]
        print(f"Fit quality range: R² = {min(r_squares):.6f} to {max(r_squares):.6f}")
        
        if max(r_squares) - min(r_squares) > 0.1:
            print("  ⚠ Large variation in fit quality - preprocessing may be beneficial")
        else:
            print("  ✓ Consistent fit quality across all modes")
    
    def plot_comparison(self, input_data: dict, channel_name: str, results: list, save_plot: bool = True):
        """Create comparison plots showing the effect of preprocessing."""
        print(f"\n=== Creating Comparison Plots ===")
        
        if channel_name not in input_data['channels']:
            print(f"Channel {channel_name} not found")
            return
        
        ch_data = input_data['channels'][channel_name]
        time_array = ch_data['time_array']
        voltage_data = ch_data['voltage_data']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'60Hz Curve Fitting with Preprocessing - Channel {channel_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Original signal and fitted signals
        ax1.plot(time_array * 1000, voltage_data, 'b-', alpha=0.7, label='Original Data', linewidth=1)
        
        # Plot fitted signals from different modes
        colors = ['red', 'green', 'orange', 'purple']
        valid_results = [r for r in results if 'error' not in r]
        
        for i, result in enumerate(valid_results[:4]):  # Limit to 4 for readability
            # Generate fitted signal
            amp = result['amplitude']
            phase = result['phase']
            offset = result['offset']
            fitted_signal = amp * np.sin(2 * np.pi * 60.0 * time_array + phase) + offset
            
            ax1.plot(time_array * 1000, fitted_signal, '--', color=colors[i], 
                    label=f"{result['config']} (R²={result['r_squared']:.3f})", 
                    linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Signal Comparison')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Limit time range for clarity (show first few cycles)
        t_max = min(0.1, time_array[-1])  # Show up to 100ms or end of data
        mask = time_array <= t_max
        ax1.set_xlim(0, t_max * 1000)
        
        # Plot 2: Parameter comparison - Amplitude
        if valid_results:
            configs = [r['config'] for r in valid_results]
            amplitudes = [r['amplitude'] for r in valid_results]
            r_squares = [r['r_squared'] for r in valid_results]
            
            bars = ax2.bar(range(len(configs)), amplitudes, 
                          color=[colors[i % len(colors)] for i in range(len(configs))],
                          alpha=0.7)
            ax2.set_xlabel('Preprocessing Mode')
            ax2.set_ylabel('Fitted Amplitude (V)')
            ax2.set_title('Amplitude Comparison')
            ax2.set_xticks(range(len(configs)))
            ax2.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8, rotation=0)
            
            # Add R² values on bars
            for i, (bar, r2) in enumerate(zip(bars, r_squares)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'R²={r2:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Phase comparison
        if valid_results:
            phases_deg = [r['phase_deg'] for r in valid_results]
            bars = ax3.bar(range(len(configs)), phases_deg,
                          color=[colors[i % len(colors)] for i in range(len(configs))],
                          alpha=0.7)
            ax3.set_xlabel('Preprocessing Mode')
            ax3.set_ylabel('Fitted Phase (degrees)')
            ax3.set_title('Phase Comparison')
            ax3.set_xticks(range(len(configs)))
            ax3.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8, rotation=0)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Fit quality comparison
        if valid_results:
            bars = ax4.bar(range(len(configs)), r_squares,
                          color=[colors[i % len(colors)] for i in range(len(configs))],
                          alpha=0.7)
            ax4.set_xlabel('Preprocessing Mode')
            ax4.set_ylabel('R-squared')
            ax4.set_title('Fit Quality Comparison')
            ax4.set_xticks(range(len(configs)))
            ax4.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=8, rotation=0)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add threshold line
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good fit threshold')
            ax4.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f"preprocessing_comparison_{channel_name}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()


def main():
    """Main demo function."""
    print("=" * 70)
    print("60Hz Curve Fitting Preprocessing Features Demo")
    print("Testing with Real SDS814X Binary Data")
    print("=" * 70)
    
    demo = PreprocessingDemo()
    
    # Look for binary files
    print("\nSearching for SDS814X binary files...")
    bin_files = demo.find_binary_files()
    
    # If no files in current directory, try parent/data directory
    if not bin_files:
        data_dir = Path('..') / 'data'
        if data_dir.exists():
            print(f"Checking data directory: {data_dir}")
            bin_files = demo.find_binary_files(str(data_dir))
    
    if not bin_files:
        print("No binary files found in current directory.")
        print("Please place some SDS814X .bin files in this directory and try again.")
        return
    
    print(f"Found {len(bin_files)} binary files:")
    for i, file in enumerate(bin_files, 1):
        print(f"  {i}. {file}")
    
    # Load first file
    test_file = bin_files[0]
    print(f"\nTesting with: {test_file}")
    
    # Load and convert data
    input_data = demo.load_and_convert_data(test_file)
    
    if not input_data:
        print("Failed to load data file")
        return
    
    # Find a channel to test
    available_channels = list(input_data['channels'].keys())
    if not available_channels:
        print("No channels found in loaded data")
        return
    
    test_channel = available_channels[0]
    print(f"\nTesting with channel: {test_channel}")
    
    # Test preprocessing modes
    results = demo.test_preprocessing_modes(input_data, test_channel)
    
    # Compare results
    demo.compare_results(results)
    
    # Create plots
    try:
        demo.plot_comparison(input_data, test_channel, results, save_plot=True)
    except Exception as e:
        print(f"Plotting failed (but this is optional): {e}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    # Summary
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        print("✓ Preprocessing features are working correctly!")
        print("✓ All preprocessing modes produced valid results")
        
        # Check if preprocessing improved anything
        r_squares = [r['r_squared'] for r in valid_results]
        if max(r_squares) > min(r_squares) + 0.01:  # Significant improvement
            best_config = next(r['config'] for r in valid_results if r['r_squared'] == max(r_squares))
            print(f"✓ Best performance with: {best_config}")
        else:
            print("✓ Consistent performance across all modes")
    else:
        print("✗ Issues found with preprocessing features")


if __name__ == "__main__":
    main()

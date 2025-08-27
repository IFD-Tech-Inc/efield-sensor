#!/usr/bin/env python3
"""
Enhanced Demonstration script for the 60 Hz Curve Fitting Signal Processor.

This script shows how to use the CurveFit60HzProcessor with the new
normalize signals and remove offset preprocessing options.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from curvefit_60hz_processor import CurveFit60HzProcessor, preview_60hz_fit, analyze_60hz_content


def create_test_signals():
    """Create various test signals for demonstration including challenging cases."""
    # Time array: 1 seconds at 1000 Hz sampling rate
    time = np.linspace(0, 1, 1000)
    
    signals = {}
    
    # 1. Perfect 60 Hz sine wave with large DC offset
    amplitude1, phase1, offset1 = 2.0, 0.5, 5.0  # Large DC offset
    signals['perfect_60hz_offset'] = {
        'time_array': time,
        'voltage_data': amplitude1 * np.sin(2 * np.pi * 60 * time + phase1) + offset1,
        'expected_params': {'amplitude': amplitude1, 'phase': phase1, 'offset': offset1},
        'description': f'Perfect 60Hz with large offset: {amplitude1}V amplitude, {offset1}V offset'
    }
    
    # 2. Small amplitude 60 Hz signal  
    amplitude2, phase2, offset2 = 0.1, 0.0, 0.05  # Very small amplitude
    signals['small_amplitude'] = {
        'time_array': time,
        'voltage_data': amplitude2 * np.sin(2 * np.pi * 60 * time + phase2) + offset2,
        'expected_params': {'amplitude': amplitude2, 'phase': phase2, 'offset': offset2},
        'description': f'Small amplitude 60Hz: {amplitude2}V amplitude, {offset2}V offset'
    }
    
    # 3. Large amplitude 60 Hz signal
    amplitude3, phase3, offset3 = 50.0, -1.2, -10.0  # Large amplitude and negative offset
    signals['large_amplitude'] = {
        'time_array': time,
        'voltage_data': amplitude3 * np.sin(2 * np.pi * 60 * time + phase3) + offset3,
        'expected_params': {'amplitude': amplitude3, 'phase': phase3, 'offset': offset3},
        'description': f'Large amplitude 60Hz: {amplitude3}V amplitude, {offset3}V offset'
    }
    
    # 4. Noisy signal with DC bias
    amplitude4, phase4, offset4 = 1.5, 0.8, 3.0
    np.random.seed(42)
    clean_signal = amplitude4 * np.sin(2 * np.pi * 60 * time + phase4) + offset4
    noise = 0.3 * np.random.normal(size=len(time))
    signals['noisy_with_bias'] = {
        'time_array': time,
        'voltage_data': clean_signal + noise,
        'expected_params': {'amplitude': amplitude4, 'phase': phase4, 'offset': offset4},
        'description': f'Noisy 60Hz with DC bias: {amplitude4}V amplitude, {offset4}V offset + noise'
    }
    
    return signals


def format_input_data(signal_name, signal_data):
    """Format signal data for processor input."""
    return {
        'channels': {
            signal_name: {
                'time_array': signal_data['time_array'],
                'voltage_data': signal_data['voltage_data'],
                'metadata': {
                    'description': signal_data['description'],
                    'test_signal': True
                }
            }
        },
        'header': {'demo_data': True},
        'source_info': {'demo_script': True}
    }


def demonstrate_preprocessing_features():
    """Demonstrate the normalize and remove offset preprocessing features."""
    print("=" * 80)
    print("60 Hz Curve Fitting - Preprocessing Features Demonstration")
    print("=" * 80)
    
    # Create processor instance
    processor = CurveFit60HzProcessor()
    print(f"Processor: {processor.get_name()}")
    print(f"Version: {processor.get_version()}")
    print()
    
    # Create test signals
    print("Creating challenging test signals...")
    signals = create_test_signals()
    print(f"Generated {len(signals)} test signals")
    print()
    
    # Define parameter combinations to test
    parameter_combinations = [
        {'normalize_signals': False, 'remove_offset': False, 'name': 'Standard Processing'},
        {'normalize_signals': True, 'remove_offset': False, 'name': 'With Normalization'},
        {'normalize_signals': False, 'remove_offset': True, 'name': 'With Offset Removal'},
        {'normalize_signals': True, 'remove_offset': True, 'name': 'Full Preprocessing'}
    ]
    
    base_parameters = {
        'output_fitted_signal': True,
        'output_parameters': True,
        'min_fit_quality': 0.0,
        'max_iterations': 5000
    }
    
    all_results = {}
    
    # Test each signal with each parameter combination
    for signal_name, signal_data in signals.items():
        print(f"\n{'='*60}")
        print(f"TESTING SIGNAL: {signal_name}")
        print(f"Description: {signal_data['description']}")
        print(f"{'='*60}")
        
        signal_results = {}
        
        for param_combo in parameter_combinations:
            print(f"\n--- {param_combo['name']} ---")
            
            # Create parameters for this test
            test_params = base_parameters.copy()
            test_params['normalize_signals'] = param_combo['normalize_signals']
            test_params['remove_offset'] = param_combo['remove_offset']
            
            # Format input data
            input_data = format_input_data(signal_name, signal_data)
            
            try:
                # Process the signal
                result = processor.process(input_data, test_params)
                
                # Extract results
                channels = result['channels']
                amplitude = channels[f'{signal_name}_amplitude']['voltage_data'][0]
                phase = channels[f'{signal_name}_phase']['voltage_data'][0]
                offset = channels[f'{signal_name}_offset']['voltage_data'][0]
                
                # Get fit quality
                fit_metadata = channels[f'{signal_name}_60hz_fit']['metadata']['processing']['fit_quality']
                r_squared = fit_metadata['r_squared']
                rmse = fit_metadata['rmse']
                
                # Store results
                signal_results[param_combo['name']] = {
                    'amplitude': amplitude,
                    'phase': phase,
                    'offset': offset,
                    'r_squared': r_squared,
                    'rmse': rmse
                }
                
                # Calculate parameter errors
                expected = signal_data['expected_params']
                amp_error = abs(amplitude - expected['amplitude'])
                phase_error = abs(phase - expected['phase'])
                offset_error = abs(offset - expected['offset'])
                
                print(f"Results:")
                print(f"  Amplitude: {amplitude:.4f} V (expected: {expected['amplitude']:.4f}, error: {amp_error:.4f})")
                print(f"  Phase: {phase:.4f} rad (expected: {expected['phase']:.4f}, error: {phase_error:.4f})")
                print(f"  Offset: {offset:.4f} V (expected: {expected['offset']:.4f}, error: {offset_error:.4f})")
                print(f"  R-squared: {r_squared:.6f}")
                print(f"  RMSE: {rmse:.6f} V")
                
                # Assess improvement
                if param_combo['name'] != 'Standard Processing':
                    standard_r2 = signal_results.get('Standard Processing', {}).get('r_squared', 0)
                    if r_squared > standard_r2:
                        improvement = ((r_squared - standard_r2) / max(standard_r2, 1e-10)) * 100
                        print(f"  Improvement over standard: +{improvement:.2f}% R-squared")
                    elif r_squared < standard_r2:
                        degradation = ((standard_r2 - r_squared) / max(standard_r2, 1e-10)) * 100
                        print(f"  Change from standard: -{degradation:.2f}% R-squared")
                
            except Exception as e:
                print(f"ERROR: Failed to process with {param_combo['name']}: {e}")
        
        all_results[signal_name] = signal_results
        print()
    
    # Summary analysis
    print("\n" + "=" * 80)
    print("PREPROCESSING EFFECTIVENESS SUMMARY")
    print("=" * 80)
    
    for signal_name, signal_results in all_results.items():
        if not signal_results:
            continue
            
        print(f"\n{signal_name.upper()}:")
        print(f"Signal: {signals[signal_name]['description']}")
        
        # Find best performing method
        best_method = None
        best_r2 = -1
        
        for method_name, results in signal_results.items():
            r2 = results['r_squared']
            if r2 > best_r2:
                best_r2 = r2
                best_method = method_name
        
        print(f"Best method: {best_method} (R² = {best_r2:.6f})")
        
        # Show comparison table
        print("Method                   | R-squared | Amplitude Error | Phase Error | Offset Error")
        print("-------------------------|-----------|-----------------|-------------|-------------")
        
        expected = signals[signal_name]['expected_params']
        for method_name, results in signal_results.items():
            amp_err = abs(results['amplitude'] - expected['amplitude'])
            phase_err = abs(results['phase'] - expected['phase'])
            offset_err = abs(results['offset'] - expected['offset'])
            
            print(f"{method_name:24s} | {results['r_squared']:8.6f} | "
                  f"{amp_err:14.6f} | {phase_err:10.6f} | {offset_err:11.6f}")
    
    # General recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("• Use 'Remove Offset' for signals with large, unknown DC offsets")
    print("• Use 'Normalization' for very small or very large amplitude signals")
    print("• Combine both for optimal results when signal amplitude and offset are unknown")
    print("• Standard processing works best when signal characteristics are well-known")
    print("• Monitor R-squared values to assess effectiveness of preprocessing")
    print()
    
    return all_results


def create_comparison_visualization(results):
    """Create visualization comparing preprocessing methods."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping visualization")
        return
    
    print("Creating preprocessing comparison visualization...")
    
    # Prepare data for plotting
    methods = ['Standard Processing', 'With Normalization', 'With Offset Removal', 'Full Preprocessing']
    signals = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, signal_name in enumerate(signals[:4]):  # Plot up to 4 signals
        if i >= len(axes):
            break
            
        ax = axes[i]
        signal_results = results[signal_name]
        
        r_squared_values = []
        method_names = []
        
        for method in methods:
            if method in signal_results:
                r_squared_values.append(signal_results[method]['r_squared'])
                method_names.append(method)
        
        # Create bar plot
        bars = ax.bar(range(len(method_names)), r_squared_values, 
                     color=['blue', 'green', 'orange', 'red'][:len(method_names)])
        
        # Customize plot
        ax.set_xlabel('Preprocessing Method')
        ax.set_ylabel('R-squared')
        ax.set_title(f'{signal_name}: R² Comparison')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in method_names], rotation=0, ha='center')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, r_squared_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved preprocessing comparison as 'preprocessing_comparison.png'")
    print()


def main():
    """Main demonstration function."""
    try:
        # Run the preprocessing demonstration
        results = demonstrate_preprocessing_features()
        
        # Create visualization if possible
        if MATPLOTLIB_AVAILABLE:
            create_comparison_visualization(results)
        
        print("Enhanced demonstration completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("✓ Normalize Signals - Handles very small/large amplitude signals")
        print("✓ Remove Offset - Handles signals with unknown DC bias")
        print("✓ Combined Processing - Best results for challenging signals")
        print("✓ Performance Comparison - Shows effectiveness of each method")
        print()
        print("The enhanced processor is ready for production use!")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

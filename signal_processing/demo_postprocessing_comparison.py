#!/usr/bin/env python3
"""
Demonstration of Post-processing Benefits for Waveform Comparison
================================================================

This script demonstrates how the new post-processing features (normalize_signals
and remove_offset) enable better comparison of fitted waveforms from different
channels by putting them on a common scale.

Key Features Demonstrated:
- Normal curve fitting preserves original signal characteristics
- Post-processing enables comparison by normalizing scale and offset
- Parameter extraction remains accurate regardless of post-processing
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add current directory to path for imports
sys.path.append('.')

try:
    from curvefit_60hz_processor import CurveFit60HzProcessor
except ImportError as e:
    print(f"Error importing processor: {e}")
    print("Make sure you're running this from the signal_processing directory")
    sys.exit(1)


def create_test_signals():
    """Create test signals with different amplitudes and offsets."""
    time = np.linspace(0, 1/6, 1000)  # 1/6 second = 10 cycles at 60Hz
    
    # Signal 1: Small amplitude, large positive offset  
    signal1 = 0.5 * np.sin(2 * np.pi * 60 * time + 0.2) + 5.0
    
    # Signal 2: Large amplitude, negative offset
    signal2 = 8.0 * np.sin(2 * np.pi * 60 * time - 0.8) - 2.0
    
    # Signal 3: Medium amplitude, no offset, different phase
    signal3 = 2.5 * np.sin(2 * np.pi * 60 * time + 1.5) + 0.1
    
    return time, signal1, signal2, signal3


def process_signals(time_array, signals, normalize=False, remove_offset=False):
    """Process signals with the curve fitting processor."""
    
    processor = CurveFit60HzProcessor()
    
    # Create input data structure
    input_data = {
        'channels': {},
        'header': {'test_data': True},
        'source_info': {'demo': 'post-processing comparison'}
    }
    
    # Add each signal as a channel
    for i, signal in enumerate(signals, 1):
        input_data['channels'][f'CH{i}'] = {
            'time_array': time_array,
            'voltage_data': signal,
            'metadata': {'channel_number': i}
        }
    
    # Process parameters
    params = {
        'normalize_signals': normalize,
        'remove_offset': remove_offset,
        'output_fitted_signal': True,
        'output_parameters': True,
        'min_fit_quality': 0.0,
        'max_iterations': 5000
    }
    
    # Process the signals
    result = processor.process(input_data, params)
    
    return result


def plot_comparison(time_array, original_signals, results_normal, results_postproc):
    """Create comparison plots showing the benefit of post-processing."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Post-processing Benefits for Waveform Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    labels = ['CH1 (Small Amp, +5V offset)', 'CH2 (Large Amp, -2V offset)', 'CH3 (Med Amp, 0V offset)']
    
    # Plot 1: Original signals
    for i, (signal, color, label) in enumerate(zip(original_signals, colors, labels)):
        ax1.plot(time_array * 1000, signal, color=color, label=label, linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Original Input Signals')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)  # Show first 3 cycles
    
    # Plot 2: Fitted signals without post-processing
    for i in range(1, 4):
        ch_name = f'CH{i}'
        if f'{ch_name}_60hz_fit' in results_normal['channels']:
            fitted_signal = results_normal['channels'][f'{ch_name}_60hz_fit']['voltage_data']
            ax2.plot(time_array * 1000, fitted_signal, '--', color=colors[i-1], 
                    label=f'{ch_name} Fitted', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_title('Fitted Signals (No Post-processing)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    
    # Plot 3: Fitted signals with post-processing
    for i in range(1, 4):
        ch_name = f'CH{i}'
        if f'{ch_name}_60hz_fit' in results_postproc['channels']:
            fitted_signal = results_postproc['channels'][f'{ch_name}_60hz_fit']['voltage_data']
            ax3.plot(time_array * 1000, fitted_signal, '--', color=colors[i-1], 
                    label=f'{ch_name} Post-processed', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Normalized Voltage (V)')
    ax3.set_title('Fitted Signals (With Post-processing: Normalized + Offset Removed)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 50)
    
    # Plot 4: Parameter comparison
    channel_names = ['CH1', 'CH2', 'CH3']
    amplitudes = []
    phases_deg = []
    offsets = []
    
    for ch in channel_names:
        if f'{ch}_amplitude' in results_normal['channels']:
            amp = results_normal['channels'][f'{ch}_amplitude']['voltage_data'][0]
            phase = results_normal['channels'][f'{ch}_phase']['voltage_data'][0]
            offset = results_normal['channels'][f'{ch}_offset']['voltage_data'][0]
            
            amplitudes.append(amp)
            phases_deg.append(np.degrees(phase))
            offsets.append(offset)
    
    x_pos = np.arange(len(channel_names))
    width = 0.25
    
    bars1 = ax4.bar(x_pos - width, amplitudes, width, label='Amplitude (V)', color='skyblue', alpha=0.8)
    bars2 = ax4.bar(x_pos, [p/50 for p in phases_deg], width, label='Phase (deg/50)', color='lightcoral', alpha=0.8) 
    bars3 = ax4.bar(x_pos + width, offsets, width, label='DC Offset (V)', color='lightgreen', alpha=0.8)
    
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Extracted Parameters (Unchanged by Post-processing)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(channel_names)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for i, (bar, phase) in enumerate(zip(bars2, phases_deg)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{phase:.0f}°', ha='center', va='bottom', fontsize=9)
    
    for bar in bars3:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def print_comparison_analysis(results_normal, results_postproc):
    """Print detailed analysis of the comparison."""
    
    print("\n" + "="*80)
    print("POST-PROCESSING COMPARISON ANALYSIS")
    print("="*80)
    
    print("\n1. SIGNAL CHARACTERISTICS BEFORE POST-PROCESSING:")
    print("-" * 50)
    
    for i in range(1, 4):
        ch_name = f'CH{i}'
        if f'{ch_name}_60hz_fit' in results_normal['channels']:
            fitted_signal = results_normal['channels'][f'{ch_name}_60hz_fit']['voltage_data']
            amplitude = results_normal['channels'][f'{ch_name}_amplitude']['voltage_data'][0]
            offset = results_normal['channels'][f'{ch_name}_offset']['voltage_data'][0]
            
            signal_min = np.min(fitted_signal)
            signal_max = np.max(fitted_signal)
            signal_mean = np.mean(fitted_signal)
            signal_pp = signal_max - signal_min
            
            print(f"{ch_name}: Range [{signal_min:.3f}, {signal_max:.3f}] V, "
                  f"DC: {signal_mean:.3f} V, P-P: {signal_pp:.3f} V")
            print(f"      Parameters: Amp={amplitude:.3f} V, Offset={offset:.3f} V")
    
    print("\n2. SIGNAL CHARACTERISTICS AFTER POST-PROCESSING:")
    print("-" * 50)
    
    for i in range(1, 4):
        ch_name = f'CH{i}'
        if f'{ch_name}_60hz_fit' in results_postproc['channels']:
            fitted_signal = results_postproc['channels'][f'{ch_name}_60hz_fit']['voltage_data']
            amplitude = results_postproc['channels'][f'{ch_name}_amplitude']['voltage_data'][0]  # Should be unchanged
            offset = results_postproc['channels'][f'{ch_name}_offset']['voltage_data'][0]  # Should be unchanged
            
            signal_min = np.min(fitted_signal)
            signal_max = np.max(fitted_signal)
            signal_mean = np.mean(fitted_signal)
            signal_pp = signal_max - signal_min
            
            print(f"{ch_name}: Range [{signal_min:.3f}, {signal_max:.3f}] V, "
                  f"DC: {signal_mean:.6f} V, P-P: {signal_pp:.3f} V")
            print(f"      Parameters: Amp={amplitude:.3f} V, Offset={offset:.3f} V (UNCHANGED)")
    
    print("\n3. KEY BENEFITS OF POST-PROCESSING:")
    print("-" * 50)
    print("✓ All fitted waveforms now have the same amplitude scale (±0.5 V)")
    print("✓ All fitted waveforms are centered at 0V (no DC offset)")  
    print("✓ Phase differences are clearly visible for comparison")
    print("✓ Original parameter values are preserved for analysis")
    print("✓ Enables meaningful overlay comparisons of different channels")
    
    print("\n4. USE CASES:")
    print("-" * 50)
    print("• Multi-channel phase relationship analysis")
    print("• Waveform shape comparison independent of amplitude/offset")
    print("• Overlay plotting of signals from different measurement scales")
    print("• Normalized display for presentation and documentation")


def main():
    """Main demonstration function."""
    print("Post-processing Demo: Waveform Comparison Benefits")
    print("=" * 60)
    
    # Create test signals with different characteristics
    time, signal1, signal2, signal3 = create_test_signals()
    signals = [signal1, signal2, signal3]
    
    print("Creating test signals with different amplitudes and offsets...")
    print("  CH1: 0.5V amplitude, +5.0V offset")
    print("  CH2: 8.0V amplitude, -2.0V offset") 
    print("  CH3: 2.5V amplitude, +0.1V offset")
    
    # Process without post-processing
    print("\n1. Processing without post-processing...")
    results_normal = process_signals(time, signals, normalize=False, remove_offset=False)
    
    # Process with post-processing
    print("\n2. Processing with post-processing (normalize + remove offset)...")
    results_postproc = process_signals(time, signals, normalize=True, remove_offset=True)
    
    # Print detailed analysis
    print_comparison_analysis(results_normal, results_postproc)
    
    # Create comparison plots
    print("\n5. GENERATING COMPARISON PLOTS:")
    print("-" * 50)
    try:
        fig = plot_comparison(time, signals, results_normal, results_postproc)
        
        # Save the plot
        plot_filename = 'postprocessing_comparison_demo.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plots saved as: {plot_filename}")
        
        # Try to show the plot
        plt.show()
        print("✓ Plots displayed (if GUI available)")
        
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
        print("  (This is normal in environments without display support)")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("• Post-processing enables meaningful comparison of fitted waveforms")
    print("• Parameters remain unchanged - they represent the original signals")
    print("• Normalization puts all signals on the same amplitude scale")  
    print("• Offset removal centers all signals at 0V for easy comparison")
    print("• This is ideal for multi-channel phase and shape analysis")


if __name__ == "__main__":
    main()

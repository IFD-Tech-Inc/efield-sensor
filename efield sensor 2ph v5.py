import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import textwrap
import math

# --- Configuration ---
CSV_FILE = r"C:\Users\DavidLin\Desktop\E-field Data\Data\123.csv"
AMPLIFICATION_CH2 = 20  # Amplify channel 2 (E-field) x10, adjust as needed
AMPLIFICATION_CH3 = 20  # Amplify channel 3 (E-field) x10, adjust as needed
DOWNSAMPLE_FACTOR = 10  # Downsample factor to speed processing; set to 1 to disable
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = None  # None = auto cutoff at fs/10, or specify as needed

# --- Helper functions ---

def read_oscilloscope_csv(file_path):
    """
    Read oscilloscope CSV, skipping metadata lines until data header line 'Second'.
    Returns pandas DataFrame with time and voltage columns.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find index where data actually starts (line starting with "Second")
    for i, line in enumerate(lines):
        if line.startswith("Second"):
            start_line = i
            break

    # Read CSV from the data header line onward
    df = pd.read_csv(file_path, skiprows=start_line)
    df.columns = df.columns.str.strip()  # Clean header whitespace
    return df

def amplify_signal(signal, factor):
    """Amplify signal by a constant factor."""
    return signal * factor

def downsample_signal(signal, factor):
    """Downsample signal by integer factor."""
    if factor <= 1:
        return signal
    return signal[::factor]

def butter_lowpass(cutoff, fs, order=4):
    """Create a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def zero_phase_filter(signal, fs, cutoff=None, order=4):
    """
    Apply zero-phase Butterworth lowpass filter using filtfilt,
    with cutoff default to fs/10 if None.
    """
    if cutoff is None:
        cutoff = fs / 10
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, signal)

def sine_function(t, A, f, phi, offset):
    """Sinusoidal model: A*sin(2*pi*f*t + phi) + offset."""
    return A * np.sin(2 * np.pi * f * t + phi) + offset

def fit_sine_to_signal(time, signal):
    """
    Fit sinusoidal function to the input signal.
    Returns dictionary of amplitude, frequency, phase, offset, and fit success flag.
    """
    A_guess = (np.max(signal) - np.min(signal)) / 2
    offset_guess = np.mean(signal)

    # FFT to estimate frequency
    fft = np.fft.rfft(signal - offset_guess)
    freqs = np.fft.rfftfreq(len(signal), d=time[1] - time[0])
    f_guess = freqs[np.argmax(np.abs(fft))]
    phi_guess = 0

    try:
        popt, _ = curve_fit(sine_function, time, signal,
                            p0=[A_guess, f_guess, phi_guess, offset_guess],
                            maxfev=10000)
        A, f, phi, offset = popt
        fit_success = True
    except RuntimeError:
        A, f, phi, offset = np.nan, np.nan, np.nan, np.nan
        fit_success = False

    return {'amplitude': A, 'frequency': f, 'phase': phi, 'offset': offset, 'fit_success': fit_success}

def rms(signal):
    """Calculate RMS value of a signal."""
    return np.sqrt(np.mean(signal ** 2))

def plot_signals(time, raw_signals, filtered_signals, fitted_params, downsample_factor):
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    channels = ['Channel 1', 'Channel 2', 'Channel 3']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Amplify channels 2 and 3 for combined plots
    raw_signals_amp = [
        raw_signals[0],                       # Channel 1 unchanged
        raw_signals[1] * AMPLIFICATION_CH2,  # Channel 2 amplified
        raw_signals[2] * AMPLIFICATION_CH3   # Channel 3 amplified
    ]

    # 1st subplot: All three raw signals (with channel 2 and 3 amplified)
    for i in range(3):
        axs[0].plot(time, raw_signals_amp[i], color=colors[i], label=channels[i])
    axs[0].set_title('All Raw Channels (Ch2 & Ch3 Amplified)')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].legend()
    axs[0].grid(True)


    # For fitted sine curves, generate fitted curves with amplification on Ch2 & Ch3
    fitted_curves_amp = []
    for i in range(3):
        if fitted_params[i]['fit_success']:
            # Amplify amplitude of fitted sine for channels 2 and 3
            amplitude = fitted_params[i]['amplitude']
            if i == 1:
                amplitude *= AMPLIFICATION_CH2
            elif i == 2:
                amplitude *= AMPLIFICATION_CH3
            fitted_curve = sine_function(time,
                                        amplitude,
                                        fitted_params[i]['frequency'],
                                        fitted_params[i]['phase'],
                                        fitted_params[i]['offset'])
            fitted_curves_amp.append(fitted_curve)
        else:
            # For channels with failed fit, append nan array
            fitted_curves_amp.append(np.full_like(time, np.nan))

    # 2nd subplot: All three fitted sine signals (with amplification on Ch2 & Ch3)
    for i in range(3):
        axs[1].plot(time, fitted_curves_amp[i], color=colors[i], label=channels[i])
    axs[1].set_title('All Fitted Sine Channels (Ch2 & Ch3 Amplified)')
    axs[1].set_ylabel('Voltage (V)')
    axs[1].legend()
    axs[1].grid(True)

    # Subplots 3,4,5: Individual channel detailed plots (raw, filtered, fitted)
    for i in range(3):
        axs[i+2].plot(time, raw_signals[i], color=colors[i], alpha=0.5, label='Raw')
        axs[i+2].plot(time, filtered_signals[i], color=colors[i], linestyle='--', label='Filtered')
        if fitted_params[i]['fit_success']:
            fit_curve = sine_function(time,
                                      fitted_params[i]['amplitude'],
                                      fitted_params[i]['frequency'],
                                      fitted_params[i]['phase'],
                                      fitted_params[i]['offset'])
            axs[i+2].plot(time, fit_curve, color=colors[i], linestyle='-', alpha=0.7, label='Fitted Sine')
        axs[i+2].set_ylabel(f"{channels[i]} (V)")
        axs[i+2].legend()
        axs[i+2].grid(True)
        axs[i+2].set_box_aspect(0.3)

    axs[-1].set_xlabel(f'Time (s) - Downsample factor {downsample_factor}')

    # Create textual summary with fitted results and metrics
    text_str = ""
    for i, ch in enumerate(channels):
        p = fitted_params[i]
        if p['phase'] is not None:
            phase_deg = math.degrees(p['phase'])
        else:
            phase_deg = float('nan')

        text_str += (f"{ch}:\n\n"
                    f"Freq: {p['frequency']:.3f} Hz\n\n"
                    f"Amplitude: {p['amplitude']:.3f} V\n\n"
                    f"Phase: {phase_deg:.3f} °\n\n"
                    f"RMS: {rms(filtered_signals[i]):.3f} V\n\n"
                    f"Fit success: {p['fit_success']}\n\n")

    # Phase differences relative to channel 1
    if all(p['fit_success'] for p in fitted_params):
        phase_diff_2 = fitted_params[1]['phase'] - fitted_params[0]['phase']
        phase_diff_3 = fitted_params[2]['phase'] - fitted_params[0]['phase']
        phase_diff_4 = fitted_params[2]['phase'] - fitted_params[1]['phase']
        phase_diff_2_deg = math.degrees(phase_diff_2)
        phase_diff_3_deg = math.degrees(phase_diff_3)
        phase_diff_4_deg = math.degrees(phase_diff_4)
        text_str += f"Phase diff (Ch2 - Ch1): {phase_diff_2_deg:.3f}°\n"
        text_str += f"Phase diff (Ch3 - Ch1): {phase_diff_3_deg:.3f}°\n"
        text_str += f"Phase diff (Ch3 - Ch2): {phase_diff_4_deg:.3f}°\n"

    # Wrap the text to ~40 chars per line for readability
    wrapper = textwrap.TextWrapper(width=40)
    wrapped_text = wrapper.fill(text_str)

    # Place wrapped text inside figure with monospace font for alignment
    plt.gcf().text(0.75, 0.5, wrapped_text, fontsize=10, va='center', ha='left',
                    fontfamily='monospace',
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# --- Main processing ---

def main():
    # 1. Read data from CSV
    data = read_oscilloscope_csv(CSV_FILE)

    # Extract time and signals as floats from columns; note channel columns may need adjusting
    time = data['Second'].astype(float).values
    ch1 = data['Volt'].astype(float).values   # Channel 1 voltage
    ch2 = data.iloc[:, 2].astype(float).values  # Channel 2 voltage (E-field)
    ch3 = data.iloc[:, 3].astype(float).values  # Channel 3 voltage (E-field)

    # 2. Amplify E-field signals (channels 2 and 3)
    ch2_amp = amplify_signal(ch2, AMPLIFICATION_CH2)
    ch3_amp = amplify_signal(ch3, AMPLIFICATION_CH3)

    # 3. Downsample time and signals to speed processing
    time_ds = downsample_signal(time, DOWNSAMPLE_FACTOR)
    ch1_ds = downsample_signal(ch1, DOWNSAMPLE_FACTOR)
    ch2_ds = downsample_signal(ch2_amp, DOWNSAMPLE_FACTOR)
    ch3_ds = downsample_signal(ch3_amp, DOWNSAMPLE_FACTOR)

    # Sampling frequency from downsampled time vector
    fs = 1 / np.mean(np.diff(time_ds))

    # 4. Apply zero-phase Butterworth lowpass filter to remove noise
    ch1_filt = zero_phase_filter(ch1_ds, fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER)
    ch2_filt = zero_phase_filter(ch2_ds, fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER)
    ch3_filt = zero_phase_filter(ch3_ds, fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER)

    # 5. Fit sine function to each filtered signal to find amplitude, freq, phase
    params_ch1 = fit_sine_to_signal(time_ds, ch1_filt)
    params_ch2 = fit_sine_to_signal(time_ds, ch2_filt)
    params_ch3 = fit_sine_to_signal(time_ds, ch3_filt)

    # 6. Plot raw, filtered, and fitted signals; show numerical summary with text wrapping
    plot_signals(time_ds,
                 [ch1_ds, ch2_ds, ch3_ds],
                 [ch1_filt, ch2_filt, ch3_filt],
                 [params_ch1, params_ch2, params_ch3],
                 DOWNSAMPLE_FACTOR)

if __name__ == "__main__":
    main()

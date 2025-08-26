# Siglent Waveform Viewer

A comprehensive PyQt6-based GUI application for visualizing and analyzing oscilloscope waveform data from Siglent Binary Format V4.0 files.

## Features

### ✅ Core Functionality Complete
- **File Loading**: Open single `.bin` files or entire directories
- **Interactive Plotting**: Matplotlib integration with zoom, pan, and navigation
- **Channel Management**: Toggle channel visibility with checkboxes
- **Overlay Mode**: Display multiple waveforms on the same plot
- **Export Capabilities**: Save plots as PNG, PDF, or SVG files
- **Background Loading**: Non-blocking file loading with progress updates
- **Settings Persistence**: Window geometry and preferences are saved

### 🔮 Future Features (Extensible Architecture)
- **Separate Plot Mode**: Individual subplots for each channel
- **Math Operations**: Add, subtract, multiply, and custom operations on waveforms
- **Measurement Tools**: Cursors, automatic measurements (RMS, peak-to-peak, etc.)
- **Advanced Filtering**: Interactive filter design and application

## Usage

### Starting the Application
```bash
# Make sure you're in the virtual environment
python main.py
```

### Loading Data
1. **Single File**: `File → Open File...` or press `Ctrl+O`
   - Select a Siglent `.bin` file
   - All enabled channels will be loaded and displayed

2. **Directory**: `File → Open Directory...` or press `Ctrl+D`
   - Select a directory containing multiple `.bin` files
   - All files will be processed and channels combined

### Managing Channels
- **Visibility**: Use checkboxes in the "Loaded Channels" panel
- **Remove Channel**: Right-click on a channel and select "Remove"
- **Channel Info**: View sample count and voltage scale in the channel list

### Plot Controls
- **Navigation Toolbar**: Standard matplotlib controls (zoom, pan, home, save)
- **Zoom to Fit**: `View → Zoom to Fit` or press `Ctrl+F`
- **Save Plot**: Use the "💾 Save Plot..." button or toolbar save icon

### Keyboard Shortcuts
- `Ctrl+O`: Open File
- `Ctrl+D`: Open Directory
- `Ctrl+N`: Clear All Data
- `Ctrl+F`: Zoom to Fit
- `Ctrl+Q`: Exit Application

## Technical Architecture

### File Structure
- `main.py`: Main PyQt6 application
- `siglent_parser.py`: Binary format parsing library
- `main.py.backup`: Original signal processing script (preserved)
- `requirements.txt`: Updated dependencies including PyQt6

### Key Classes

#### `WaveformViewerMainWindow`
Main application window with menus, toolbars, and layout management.

#### `PlotCanvas`
Custom matplotlib canvas integrated with PyQt6:
- Handles waveform plotting and visualization
- Manages channel visibility and color cycling
- Provides extensible architecture for future math operations

#### `ChannelListWidget`
Custom QListWidget for channel management:
- Checkboxes for visibility control
- Context menus for channel operations
- Channel information display

#### `LoadWaveformThread`
Background worker thread:
- Prevents GUI freezing during file loading
- Provides progress updates and error handling
- Supports both single files and directories

### Data Flow
```
Siglent .bin file → SiglentBinaryParser → ChannelData objects → PlotCanvas → matplotlib display
```

## Dependencies

### Required
- **PyQt6**: Modern GUI framework
- **matplotlib**: Plotting and visualization (Qt backend)
- **numpy**: Numerical computing
- **scipy**: Signal processing

### Installation
```bash
pip install -r requirements.txt
```

## Extensibility for Math Operations

The architecture is designed to easily support math operations on waveforms:

### Current Foundation
- Channel data is stored in `ChannelData` objects with voltage and time arrays
- Plot management allows easy addition/removal of calculated channels
- Color cycling and legend management are automatic

### Future Math Operations Implementation
1. **Add Math Menu**: Create math operations in menu bar
2. **Operation Dialogs**: Channel selection and parameter input
3. **Calculated Channels**: Create virtual channels for math results
4. **Real-time Updates**: Recalculate when input channels change

Example future implementation:
```python
def add_channels(self, ch1_name, ch2_name):
    """Add two channels and create a new calculated channel."""
    ch1_data = self.plot_canvas.plot_data[ch1_name]
    ch2_data = self.plot_canvas.plot_data[ch2_name]
    
    # Perform math operation
    result_voltage = ch1_data['data'].voltage_data + ch2_data['data'].voltage_data
    
    # Create virtual channel and plot
    calc_channel_name = f"{ch1_name}+{ch2_name}"
    self.plot_calculated_channel(calc_channel_name, ch1_data['time'], result_voltage)
```

## Error Handling

The application includes comprehensive error handling:
- **File Format Validation**: Only Siglent Binary Format V4.0 supported
- **Missing Dependencies**: Graceful degradation if siglent_parser unavailable
- **Background Loading Errors**: User-friendly error dialogs
- **Exception Handling**: Application-level error recovery

## Settings Persistence

Application settings are automatically saved:
- Window geometry and position
- View mode preferences (overlay/separate)
- Recent files (future enhancement)

## Troubleshooting

### Common Issues

1. **"Siglent parser module not available"**
   - Ensure `siglent_parser.py` is in the same directory as `main.py`

2. **Application won't start**
   - Check Python version (3.7+ recommended)
   - Verify PyQt6 installation: `pip list | findstr PyQt6`

3. **No data displayed after loading**
   - Check if channels are enabled in the original oscilloscope capture
   - Verify file format is Siglent Binary Format V4.0

### Debug Mode
For debugging, you can run with additional output:
```python
# Add this to the beginning of main() in main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

- **Large Files**: Background loading prevents GUI freezing
- **Memory Usage**: Efficient NumPy arrays for waveform data
- **Rendering**: Hardware-accelerated matplotlib with Qt backend
- **Responsiveness**: Asynchronous file operations

---

**Version**: 1.0.0  
**Author**: Assistant  
**License**: Open source (customize as needed)  
**Dependencies**: PyQt6, matplotlib, numpy, scipy

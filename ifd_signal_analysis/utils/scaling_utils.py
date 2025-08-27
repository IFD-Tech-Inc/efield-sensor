#!/usr/bin/env python3
"""
Engineering scaling utilities for IFD Signal Analysis Utility.

This module provides functions for intelligent y-axis scaling with engineering-friendly
values and automatic determination of when multiple y-axes are needed.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def calculate_engineering_range(min_val: float, max_val: float, 
                              fill_percentage: float = 0.9) -> Tuple[float, float]:
    """
    Calculate optimal y-axis range using engineering-friendly values (1-2-5 series).
    
    The function centers the waveforms optimally and uses nice round numbers
    for axis limits, with minimal but reasonable padding.
    
    Args:
        min_val: Minimum data value
        max_val: Maximum data value  
        fill_percentage: Fraction of screen the data should fill (0.9 = 90%)
        
    Returns:
        Tuple of (y_min, y_max) representing optimal axis limits
    """
    if min_val == max_val:
        # Handle constant values - create small range around the value
        center = min_val
        if abs(center) < 1e-15:  # Essentially zero (much smaller threshold)
            return (-0.001, 0.001)  # Default to 1mV range for zero
        else:
            # For small values, use a percentage-based margin
            margin = max(abs(center) * 0.1, 1e-6)  # At least 1μV margin
            return (center - margin, center + margin)
    
    data_range = max_val - min_val
    data_center = (min_val + max_val) / 2
    
    # Calculate the total range needed (data range + padding)
    # Use higher fill percentage (90%) for better space utilization
    total_range = data_range / fill_percentage
    
    # Calculate symmetric padding around the data center
    half_range = total_range / 2
    
    # Preliminary limits centered on the data
    prelim_min = data_center - half_range
    prelim_max = data_center + half_range
    
    # Round to engineering-friendly values using 1-2-5 series
    def round_to_engineering(value: float, direction: str) -> float:
        """Round to engineering-friendly value in 1-2-5 series."""
        if abs(value) < 1e-15:  # Essentially zero (much smaller threshold for small signals)
            return 0.0
        
        # For very small values, prevent underflow issues
        abs_value = abs(value)
        if abs_value < 1e-12:  # Use a minimum scaling to prevent floating point issues
            if direction == 'down':
                result = -1e-12 if value < 0 else 0.0
            else:  # direction == 'up'
                result = 1e-12 if value > 0 else 0.0
            return result
            
        # Get the order of magnitude
        try:
            magnitude = math.floor(math.log10(abs_value))
        except ValueError:  # Handle edge case for very small values
            magnitude = -12  # Default to picovolts
            
        normalized = abs_value / (10 ** magnitude)
        
        # 1-2-5 series values
        if direction == 'down':
            if normalized <= 1.0:
                nice_val = 1.0
            elif normalized <= 2.0:
                nice_val = 1.0  # Round down
            elif normalized <= 5.0:
                nice_val = 2.0  # Round down  
            else:
                nice_val = 5.0  # Round down
        else:  # direction == 'up'
            if normalized <= 1.0:
                nice_val = 1.0
            elif normalized <= 2.0:
                nice_val = 2.0  # Round up
            elif normalized <= 5.0:
                nice_val = 5.0  # Round up
            else:
                nice_val = 10.0  # Round up to next magnitude
                magnitude += 1
        
        result = nice_val * (10 ** magnitude)
        return result if value >= 0 else -result
    
    # Round limits to engineering values
    y_min = round_to_engineering(prelim_min, 'down')
    y_max = round_to_engineering(prelim_max, 'up')
    
    # CRITICAL: Ensure the rounded limits don't clip the actual data
    # If rounding made the limits too restrictive, expand them
    if y_min > min_val:
        # Round down more aggressively to ensure we include all data
        y_min = round_to_engineering(min_val - abs(min_val) * 0.05, 'down')
    
    if y_max < max_val:
        # Round up more aggressively to ensure we include all data
        y_max = round_to_engineering(max_val + abs(max_val) * 0.05, 'up')
    
    # Ensure we don't have identical limits
    if y_min == y_max:
        if abs(y_min) < 1e-15:
            # For essentially zero values, default to a reasonable small range
            y_min, y_max = -0.001, 0.001  # 1mV range
        else:
            # For small values, ensure minimum meaningful range
            margin = max(abs(y_min) * 0.1, 1e-6)  # At least 1μV margin
            y_min -= margin
            y_max += margin
    
    # Final safety check - ensure data is never clipped
    if y_min > min_val or y_max < max_val:
        print(f"Warning: Engineering scaling would clip data. Original: [{min_val:.6f}, {max_val:.6f}], Calculated: [{y_min:.6f}, {y_max:.6f}]")
        # Fallback: use simple symmetric expansion
        data_center = (min_val + max_val) / 2
        data_range = max_val - min_val
        safety_margin = data_range * 0.1  # 10% margin
        y_min = data_center - (data_range / 2 + safety_margin)
        y_max = data_center + (data_range / 2 + safety_margin)
    
    return (y_min, y_max)


def get_engineering_tick_values(y_min: float, y_max: float, 
                               target_ticks: int = 8) -> List[float]:
    """
    Generate engineering-friendly tick mark values for the given range.
    
    Args:
        y_min: Minimum axis value
        y_max: Maximum axis value
        target_ticks: Desired number of tick marks (approximate)
        
    Returns:
        List of tick mark values using engineering-friendly intervals
    """
    if y_min >= y_max:
        return [y_min]
    
    range_val = y_max - y_min
    
    # Calculate rough tick interval
    rough_interval = range_val / (target_ticks - 1)
    
    # Round interval to engineering-friendly value
    if rough_interval <= 0:
        return [y_min]
        
    # Get order of magnitude
    magnitude = math.floor(math.log10(rough_interval))
    normalized = rough_interval / (10 ** magnitude)
    
    # Choose nice interval from 1-2-5 series
    if normalized <= 1.0:
        nice_interval = 1.0
    elif normalized <= 2.0:
        nice_interval = 2.0
    elif normalized <= 5.0:
        nice_interval = 5.0
    else:
        nice_interval = 10.0
    
    tick_interval = nice_interval * (10 ** magnitude)
    
    # Generate tick marks
    # Start from a round number at or below y_min
    first_tick = math.floor(y_min / tick_interval) * tick_interval
    
    ticks = []
    current = first_tick
    while current <= y_max + tick_interval * 0.001:  # Small epsilon for floating point
        if current >= y_min:
            ticks.append(current)
        current += tick_interval
        
        # Safety check to prevent infinite loops
        if len(ticks) > target_ticks * 2:
            break
    
    return ticks


def should_use_separate_axes(ranges_list: List[Tuple[float, float]], 
                           separation_threshold: float = 10.0) -> bool:
    """
    Determine if multiple y-axes are needed based on data ranges.
    
    Args:
        ranges_list: List of (min, max) tuples for each channel
        separation_threshold: Ratio threshold for requiring separate axes
        
    Returns:
        True if separate axes should be used, False if single axis is sufficient
    """
    if len(ranges_list) <= 1:
        return False
    
    # Calculate range magnitudes
    magnitudes = []
    for min_val, max_val in ranges_list:
        range_val = max_val - min_val
        if range_val > 0:
            magnitudes.append(range_val)
    
    if len(magnitudes) <= 1:
        return False
    
    # Check if any ranges differ by more than the threshold
    max_magnitude = max(magnitudes)
    min_magnitude = min(magnitudes)
    
    if max_magnitude / min_magnitude > separation_threshold:
        return True
    
    # Also check if ranges are separated (don't overlap significantly)
    all_mins = [r[0] for r in ranges_list]
    all_maxs = [r[1] for r in ranges_list]
    overall_min = min(all_mins)
    overall_max = max(all_maxs)
    overall_range = overall_max - overall_min
    
    # If the combined range is much larger than individual ranges,
    # separate axes might be beneficial
    for min_val, max_val in ranges_list:
        individual_range = max_val - min_val
        if overall_range / individual_range > separation_threshold * 2:
            return True
    
    return False


def group_channels_by_scale(channels_data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Group channels that can share the same y-axis based on their voltage ranges.
    
    Args:
        channels_data: Dictionary mapping channel names to data ranges
        Expected format: {channel_name: {'voltage_range': (min, max), ...}}
        
    Returns:
        Dictionary mapping axis_index -> list of channel names for that axis
    """
    if not channels_data:
        return {}
    
    # Extract ranges for each channel
    channel_ranges = {}
    for channel_name, data in channels_data.items():
        if 'voltage_range' in data:
            channel_ranges[channel_name] = data['voltage_range']
        else:
            # Fallback - calculate range from voltage data if available
            if 'voltage_data' in data and hasattr(data['voltage_data'], '__len__'):
                voltage_data = data['voltage_data']
                if len(voltage_data) > 0:
                    channel_ranges[channel_name] = (float(np.min(voltage_data)), 
                                                   float(np.max(voltage_data)))
    
    if not channel_ranges:
        return {}
    
    # Start with first channel on axis 0
    axes_groups = {0: []}
    channel_names = list(channel_ranges.keys())
    ranges_for_axes = {0: []}
    
    for channel_name in channel_names:
        channel_range = channel_ranges[channel_name]
        assigned = False
        
        # Try to assign to existing axis
        for axis_idx in axes_groups:
            current_ranges = ranges_for_axes[axis_idx] + [channel_range]
            
            if not should_use_separate_axes(current_ranges):
                # Can share this axis
                axes_groups[axis_idx].append(channel_name)
                ranges_for_axes[axis_idx].append(channel_range)
                assigned = True
                break
        
        if not assigned:
            # Need new axis
            new_axis_idx = max(axes_groups.keys()) + 1
            axes_groups[new_axis_idx] = [channel_name]
            ranges_for_axes[new_axis_idx] = [channel_range]
    
    return axes_groups


def calculate_axis_positions(num_axes: int) -> List[Tuple[str, float]]:
    """
    Calculate positioning for multiple y-axes to avoid overlap.
    
    Args:
        num_axes: Number of y-axes needed
        
    Returns:
        List of (position, offset) tuples for each axis.
        Position is 'left' or 'right', offset is fractional position.
    """
    if num_axes <= 1:
        return [('left', 0.0)]
    
    positions = []
    
    # First axis is always on the left
    positions.append(('left', 0.0))
    
    if num_axes == 2:
        # Second axis on the right
        positions.append(('right', 0.0))
    else:
        # Multiple axes - alternate sides and stack
        right_count = 0
        for i in range(1, num_axes):
            if i % 2 == 1:  # Odd indices go to right
                offset = right_count * 0.1  # 10% spacing
                positions.append(('right', offset))
                right_count += 1
            else:  # Even indices go to left (but offset)
                offset = (i // 2) * 0.1
                positions.append(('left', -offset))
    
    return positions


def format_engineering_value(value: float, precision: int = 3) -> str:
    """
    Format a value using engineering notation with appropriate precision.
    
    Args:
        value: Value to format
        precision: Number of significant digits
        
    Returns:
        Formatted string with engineering notation
    """
    if abs(value) < 1e-12:
        return "0"
    
    # Standard engineering prefixes
    prefixes = {
        -12: 'p', -9: 'n', -6: 'μ', -3: 'm', 0: '', 
        3: 'k', 6: 'M', 9: 'G', 12: 'T'
    }
    
    # Find appropriate prefix
    magnitude = math.floor(math.log10(abs(value)) / 3) * 3
    
    # Clamp to available prefixes
    magnitude = max(-12, min(12, magnitude))
    
    # Scale value
    scaled_value = value / (10 ** magnitude)
    
    # Format with appropriate precision
    if abs(scaled_value) >= 100:
        formatted = f"{scaled_value:.0f}"
    elif abs(scaled_value) >= 10:
        formatted = f"{scaled_value:.1f}"
    else:
        formatted = f"{scaled_value:.2f}"
    
    # Add prefix
    prefix = prefixes.get(magnitude, '')
    return f"{formatted}{prefix}"

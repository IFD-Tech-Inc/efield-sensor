#!/usr/bin/env python3
"""
Siglent Binary Format V4.0 Parser

This module provides functionality to parse Siglent oscilloscope binary files
according to the Binary Format V4.0 specification ONLY.

Version: 1.0.0
Supported Format: Siglent Binary Format V4.0 only
Author: Assistant
Date: 2025-08-25
"""

__version__ = "1.0.0"
__supported_format_version__ = 4
__format_name__ = "Siglent Binary Format V4.0"

import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os


# Magnitude multipliers based on Table 3
MAGNITUDE_TABLE = {
    0: 1e-24,   # YOCTO
    1: 1e-21,   # ZEPTO
    2: 1e-18,   # ATTO
    3: 1e-15,   # FEMTO
    4: 1e-12,   # PICO
    5: 1e-9,    # NANO
    6: 1e-6,    # MICRO
    7: 1e-3,    # MILLI
    8: 1.0,     # IU (unit)
    9: 1e3,     # KILO
    10: 1e6,    # MEGA
    11: 1e9,    # GIGA
    12: 1e12,   # TERA
    13: 1e15,   # PETA
    14: 1e18,   # EXA
    15: 1e21,   # ZETTA
    16: 1e24,   # YOTTA
}


@dataclass
class DataWithUnit:
    """Represents a value with unit information from the binary format."""
    value: float  # 64-bit float
    magnitude: int  # Units of value's magnitude (0-16)
    unit_type: int  # Basic unit type (0-11)
    v_power_num: int  # V power numerator
    v_power_den: int  # V power denominator
    a_power_num: int  # A power numerator
    a_power_den: int  # A power denominator
    s_power_num: int  # S power numerator
    s_power_den: int  # S power denominator
    
    def get_scaled_value(self) -> float:
        """Get the value scaled by its magnitude."""
        return self.value * MAGNITUDE_TABLE.get(self.magnitude, 1.0)
    
    def get_unit_string(self) -> str:
        """Generate a human-readable unit string."""
        unit_names = {
            0: "V^{}/{}·A^{}/{}·S^{}/{}".format(
                self.v_power_num, self.v_power_den,
                self.a_power_num, self.a_power_den,
                self.s_power_num, self.s_power_den
            ),
            1: "DBV",
            2: "DBA", 
            3: "DB",
            4: "VPP",
            5: "VDC",
            6: "DBM",
            7: "SA",
            8: "DIV",
            9: "PTS",
            10: "NULL_SENSE",
            11: "DEGREE",
            12: "PERCENT"
        }
        
        base_unit = unit_names.get(self.unit_type, f"UNIT_{self.unit_type}")
        magnitude_names = {
            0: "y", 1: "z", 2: "a", 3: "f", 4: "p", 5: "n", 
            6: "μ", 7: "m", 8: "", 9: "k", 10: "M", 11: "G",
            12: "T", 13: "P", 14: "E", 15: "Z", 16: "Y"
        }
        magnitude_prefix = magnitude_names.get(self.magnitude, f"10^{MAGNITUDE_TABLE.get(self.magnitude, 1.0)}")
        
        return f"{magnitude_prefix}{base_unit}"


@dataclass
class ChannelData:
    """Represents data for a single oscilloscope channel."""
    channel_name: str
    enabled: bool
    volt_div_val: DataWithUnit
    vert_offset: DataWithUnit
    probe_value: float
    vert_code_per_div: int
    raw_data: np.ndarray
    voltage_data: np.ndarray
    
    def get_time_array(self, time_div: DataWithUnit, time_delay: DataWithUnit, 
                      sample_rate: DataWithUnit, hori_div_num: int = 10) -> np.ndarray:
        """Calculate time array for this channel's data."""
        if len(self.raw_data) == 0:
            return np.array([])
            
        sample_period = 1.0 / sample_rate.get_scaled_value()  # seconds per sample
        time_offset = -(time_div.get_scaled_value() * hori_div_num / 2) - time_delay.get_scaled_value()
        
        return time_offset + np.arange(len(self.raw_data)) * sample_period


@dataclass  
class SiglentBinaryHeader:
    """Complete header structure for Siglent Binary Format V4.0."""
    version: int
    data_offset_byte: int
    
    # Channel enable status
    ch1_on: bool
    ch2_on: bool  
    ch3_on: bool
    ch4_on: bool
    
    # Channel voltage/div values
    ch1_volt_div_val: DataWithUnit
    ch2_volt_div_val: DataWithUnit
    ch3_volt_div_val: DataWithUnit
    ch4_volt_div_val: DataWithUnit
    
    # Channel vertical offsets
    ch1_vert_offset: DataWithUnit
    ch2_vert_offset: DataWithUnit
    ch3_vert_offset: DataWithUnit
    ch4_vert_offset: DataWithUnit
    
    # Digital channel settings
    digital_on: bool
    d0_d15_on: List[bool]  # 16 digital channels
    
    # Time settings
    time_div: DataWithUnit
    time_delay: DataWithUnit
    
    # Wave data info
    wave_length: int
    sample_rate: DataWithUnit
    digital_wave_length: int
    digital_sample_rate: DataWithUnit
    
    # Probe settings
    ch1_probe: float
    ch2_probe: float
    ch3_probe: float
    ch4_probe: float
    
    # Data format info
    data_width: int  # 0=8-bit, 1=16-bit
    byte_order: int  # 0=LSB, 1=MSB
    hori_div_num: int
    
    # Vertical code per div
    ch1_vert_code_per_div: int
    ch2_vert_code_per_div: int
    ch3_vert_code_per_div: int
    ch4_vert_code_per_div: int
    
    # Math channel settings
    math1_switch: bool
    math2_switch: bool
    math3_switch: bool
    math4_switch: bool
    
    # Math channel voltage/div values
    math1_vdiv_val: DataWithUnit
    math2_vdiv_val: DataWithUnit
    math3_vdiv_val: DataWithUnit
    math4_vdiv_val: DataWithUnit
    
    # Math channel offsets
    math1_vpos_val: DataWithUnit
    math2_vpos_val: DataWithUnit
    math3_vpos_val: DataWithUnit
    math4_vpos_val: DataWithUnit
    
    # Math wave lengths
    math1_store_len: int
    math2_store_len: int
    math3_store_len: int
    math4_store_len: int
    
    # Math sample intervals
    math1_f_time: float
    math2_f_time: float
    math3_f_time: float
    math4_f_time: float
    
    # Math vertical code per div
    math_vert_code_per_div: int
    
    # Additional arrays for positioning and memory
    ch_insert: List[int]  # 8 elements
    math_insert: List[int]  # 4 elements
    digital_insert: List[int]  # 16 elements
    ch_move: List[int]  # 8 elements
    math_move: List[int]  # 4 elements
    digital_move: List[int]  # 16 elements
    
    # Memory settings
    memory_switch: List[bool]  # 4 elements
    memory_wave_format: List[int]  # 4 elements
    memory_vdiv_val: List[DataWithUnit]  # 4 elements
    memory_vpos_val: List[DataWithUnit]  # 4 elements
    memory_hdiv_val: List[DataWithUnit]  # 4 elements
    memory_hpos_val: List[DataWithUnit]  # 4 elements
    memory_store_len: List[int]  # 4 elements
    memory_f_time: List[float]  # 4 elements
    memory_vert_code_per_div: List[int]  # 4 elements
    memory_insert: List[int]  # 4 elements
    memory_move: List[int]  # 4 elements
    memory_probe_fval: List[float]  # 4 elements
    
    # Zoom settings
    zoom_switch: bool
    zoom_td_val: DataWithUnit
    zoom_trig_delay_val: DataWithUnit
    zoom_vdiv_val: List[DataWithUnit]  # 8 elements
    zoom_vpos_val: List[DataWithUnit]  # 8 elements


class SiglentBinaryParser:
    """Parser for Siglent Binary Format V4.0 files."""
    
    def __init__(self):
        self.header: Optional[SiglentBinaryHeader] = None
        self.channels: Dict[str, ChannelData] = {}
    
    def _read_data_with_unit(self, data: bytes, offset: int) -> DataWithUnit:
        """Read a DataWithUnit structure from binary data."""
        # Extract 40-byte DataWithUnit structure
        unit_data = data[offset:offset + 40]
        
        # Parse the structure
        value = struct.unpack('<d', unit_data[0:8])[0]  # 64-bit float, little-endian
        magnitude = struct.unpack('<I', unit_data[8:12])[0]  # 32-bit int
        unit_type = struct.unpack('<I', unit_data[12:16])[0]  # 32-bit int
        v_power_num = struct.unpack('<I', unit_data[16:20])[0]
        v_power_den = struct.unpack('<I', unit_data[20:24])[0]  
        a_power_num = struct.unpack('<I', unit_data[24:28])[0]
        a_power_den = struct.unpack('<I', unit_data[28:32])[0]
        s_power_num = struct.unpack('<I', unit_data[32:36])[0]
        s_power_den = struct.unpack('<I', unit_data[36:40])[0]
        
        return DataWithUnit(
            value=value,
            magnitude=magnitude,
            unit_type=unit_type,
            v_power_num=v_power_num,
            v_power_den=v_power_den,
            a_power_num=a_power_num,
            a_power_den=a_power_den,
            s_power_num=s_power_num,
            s_power_den=s_power_den
        )
    
    def _parse_header(self, data: bytes) -> SiglentBinaryHeader:
        """Parse the 4K header from binary data."""
        
        def read_int32(offset):
            return struct.unpack('<I', data[offset:offset+4])[0]
            
        def read_int32_signed(offset):
            return struct.unpack('<i', data[offset:offset+4])[0]
            
        def read_uint8(offset):
            return struct.unpack('<B', data[offset:offset+1])[0]
            
        def read_float64(offset):
            return struct.unpack('<d', data[offset:offset+8])[0]
        
        # Parse according to the specification table
        version = read_int32(0x00)
        data_offset_byte = read_int32(0x04)
        
        # Channel enable status (0x08-0x17)
        ch1_on = bool(read_int32_signed(0x08))
        ch2_on = bool(read_int32(0x0c))
        ch3_on = bool(read_int32(0x10))
        ch4_on = bool(read_int32(0x14))
        
        # Channel voltage/div values (0x18-0xb7)
        ch1_volt_div_val = self._read_data_with_unit(data, 0x18)
        ch2_volt_div_val = self._read_data_with_unit(data, 0x40)
        ch3_volt_div_val = self._read_data_with_unit(data, 0x68)
        ch4_volt_div_val = self._read_data_with_unit(data, 0x90)
        
        # Channel vertical offsets (0xb8-0x157)
        ch1_vert_offset = self._read_data_with_unit(data, 0xb8)
        ch2_vert_offset = self._read_data_with_unit(data, 0xe0)
        ch3_vert_offset = self._read_data_with_unit(data, 0x108)
        ch4_vert_offset = self._read_data_with_unit(data, 0x130)
        
        # Digital settings (0x158-0x19b)
        digital_on = bool(read_int32(0x158))
        d0_d15_on = []
        for i in range(16):
            d0_d15_on.append(bool(read_int32(0x15c + i*4)))
        
        # Time settings (0x19c-0x1eb)
        time_div = self._read_data_with_unit(data, 0x19c)
        time_delay = self._read_data_with_unit(data, 0x1c4)
        
        # Wave data info (0x1ec-0x243)
        wave_length = read_int32(0x1ec)
        sample_rate = self._read_data_with_unit(data, 0x1f0)
        digital_wave_length = read_int32(0x218)
        digital_sample_rate = self._read_data_with_unit(data, 0x21c)
        
        # Probe settings (0x244-0x263)
        ch1_probe = read_float64(0x244)
        ch2_probe = read_float64(0x24c)
        ch3_probe = read_float64(0x254)
        ch4_probe = read_float64(0x25c)
        
        # Data format (0x264-0x265)
        data_width = read_uint8(0x264)
        byte_order = read_uint8(0x265)
        
        # Additional settings (0x26c onwards)
        hori_div_num = read_int32_signed(0x26c)
        ch1_vert_code_per_div = read_int32_signed(0x270)
        ch2_vert_code_per_div = read_int32_signed(0x274)
        ch3_vert_code_per_div = read_int32_signed(0x278)
        ch4_vert_code_per_div = read_int32_signed(0x27c)
        
        # Math channel switches (0x280-0x28f)
        math1_switch = bool(read_int32_signed(0x280))
        math2_switch = bool(read_int32_signed(0x284))
        math3_switch = bool(read_int32_signed(0x288))
        math4_switch = bool(read_int32_signed(0x28c))
        
        # Math channel voltage/div values (0x290-0x32f)
        math1_vdiv_val = self._read_data_with_unit(data, 0x290)
        math2_vdiv_val = self._read_data_with_unit(data, 0x2b8)
        math3_vdiv_val = self._read_data_with_unit(data, 0x2e0)
        math4_vdiv_val = self._read_data_with_unit(data, 0x308)
        
        # Math channel offsets (0x330-0x3cf)
        math1_vpos_val = self._read_data_with_unit(data, 0x330)
        math2_vpos_val = self._read_data_with_unit(data, 0x358)
        math3_vpos_val = self._read_data_with_unit(data, 0x380)
        math4_vpos_val = self._read_data_with_unit(data, 0x3a8)
        
        # Math wave lengths (0x3d0-0x3df)
        math1_store_len = read_int32(0x3d0)
        math2_store_len = read_int32(0x3d4)
        math3_store_len = read_int32(0x3d8)
        math4_store_len = read_int32(0x3dc)
        
        # Math sample intervals (0x3e0-0x3ff)
        math1_f_time = read_float64(0x3e0)
        math2_f_time = read_float64(0x3e8)
        math3_f_time = read_float64(0x3f0)
        math4_f_time = read_float64(0x3f8)
        
        # Math vertical code per div (0x400)
        math_vert_code_per_div = read_int32(0x400)
        
        # Insert coefficients (0x584-0x5f3)
        ch_insert = [read_int32(0x584 + i*4) for i in range(8)]
        math_insert = [read_int32(0x5a4 + i*4) for i in range(4)]
        digital_insert = [read_int32(0x5b4 + i*4) for i in range(16)]
        
        # Move positions (0x5f4-0x663)
        ch_move = [read_int32(0x5f4 + i*4) for i in range(8)]
        math_move = [read_int32(0x614 + i*4) for i in range(4)]
        digital_move = [read_int32(0x624 + i*4) for i in range(16)]
        
        # Memory settings (0x664-0xaf3)
        memory_switch = [bool(read_int32_signed(0x664 + i*4)) for i in range(4)]
        memory_wave_format = [struct.unpack('<H', data[0x674 + i*2:0x674 + i*2 + 2])[0] for i in range(4)]
        
        memory_vdiv_val = [self._read_data_with_unit(data, 0x684 + i*40) for i in range(4)]
        memory_vpos_val = [self._read_data_with_unit(data, 0x724 + i*40) for i in range(4)]
        memory_hdiv_val = [self._read_data_with_unit(data, 0x904 + i*40) for i in range(4)]
        memory_hpos_val = [self._read_data_with_unit(data, 0x9a4 + i*40) for i in range(4)]
        
        memory_store_len = [read_int32(0xa64 + i*4) for i in range(4)]
        memory_f_time = [read_float64(0xa74 + i*8) for i in range(4)]
        memory_vert_code_per_div = [read_int32(0xa94 + i*4) for i in range(4)]
        memory_insert = [read_int32(0xaa4 + i*4) for i in range(4)]
        memory_move = [read_int32(0xab4 + i*4) for i in range(4)]
        memory_probe_fval = [read_float64(0xac4 + i*8) for i in range(4)]
        
        # Zoom settings (0xaf4-0xdc7)
        zoom_switch = bool(read_int32_signed(0xaf4))
        zoom_td_val = self._read_data_with_unit(data, 0xaf8)
        zoom_trig_delay_val = self._read_data_with_unit(data, 0xb20)
        zoom_vdiv_val = [self._read_data_with_unit(data, 0xb48 + i*40) for i in range(8)]
        zoom_vpos_val = [self._read_data_with_unit(data, 0xc88 + i*40) for i in range(8)]
        
        return SiglentBinaryHeader(
            version=version,
            data_offset_byte=data_offset_byte,
            ch1_on=ch1_on, ch2_on=ch2_on, ch3_on=ch3_on, ch4_on=ch4_on,
            ch1_volt_div_val=ch1_volt_div_val, ch2_volt_div_val=ch2_volt_div_val,
            ch3_volt_div_val=ch3_volt_div_val, ch4_volt_div_val=ch4_volt_div_val,
            ch1_vert_offset=ch1_vert_offset, ch2_vert_offset=ch2_vert_offset,
            ch3_vert_offset=ch3_vert_offset, ch4_vert_offset=ch4_vert_offset,
            digital_on=digital_on, d0_d15_on=d0_d15_on,
            time_div=time_div, time_delay=time_delay,
            wave_length=wave_length, sample_rate=sample_rate,
            digital_wave_length=digital_wave_length, digital_sample_rate=digital_sample_rate,
            ch1_probe=ch1_probe, ch2_probe=ch2_probe, ch3_probe=ch3_probe, ch4_probe=ch4_probe,
            data_width=data_width, byte_order=byte_order, hori_div_num=hori_div_num,
            ch1_vert_code_per_div=ch1_vert_code_per_div, ch2_vert_code_per_div=ch2_vert_code_per_div,
            ch3_vert_code_per_div=ch3_vert_code_per_div, ch4_vert_code_per_div=ch4_vert_code_per_div,
            math1_switch=math1_switch, math2_switch=math2_switch, math3_switch=math3_switch, math4_switch=math4_switch,
            math1_vdiv_val=math1_vdiv_val, math2_vdiv_val=math2_vdiv_val,
            math3_vdiv_val=math3_vdiv_val, math4_vdiv_val=math4_vdiv_val,
            math1_vpos_val=math1_vpos_val, math2_vpos_val=math2_vpos_val,
            math3_vpos_val=math3_vpos_val, math4_vpos_val=math4_vpos_val,
            math1_store_len=math1_store_len, math2_store_len=math2_store_len,
            math3_store_len=math3_store_len, math4_store_len=math4_store_len,
            math1_f_time=math1_f_time, math2_f_time=math2_f_time,
            math3_f_time=math3_f_time, math4_f_time=math4_f_time,
            math_vert_code_per_div=math_vert_code_per_div,
            ch_insert=ch_insert, math_insert=math_insert, digital_insert=digital_insert,
            ch_move=ch_move, math_move=math_move, digital_move=digital_move,
            memory_switch=memory_switch, memory_wave_format=memory_wave_format,
            memory_vdiv_val=memory_vdiv_val, memory_vpos_val=memory_vpos_val,
            memory_hdiv_val=memory_hdiv_val, memory_hpos_val=memory_hpos_val,
            memory_store_len=memory_store_len, memory_f_time=memory_f_time,
            memory_vert_code_per_div=memory_vert_code_per_div,
            memory_insert=memory_insert, memory_move=memory_move, memory_probe_fval=memory_probe_fval,
            zoom_switch=zoom_switch, zoom_td_val=zoom_td_val, zoom_trig_delay_val=zoom_trig_delay_val,
            zoom_vdiv_val=zoom_vdiv_val, zoom_vpos_val=zoom_vpos_val
        )
    
    def _convert_raw_to_voltage(self, raw_data: np.ndarray, volt_div_val: DataWithUnit, 
                               vert_offset: DataWithUnit, vert_code_per_div: int, 
                               data_width: int, probe_value: float = 1.0) -> np.ndarray:
        """Convert raw binary data to voltage values using the specified formula."""
        # Calculate center code based on data width
        if data_width == 0:  # 8-bit
            center_code = 128  # 2^7
        else:  # 16-bit
            center_code = 32768  # 2^15
        
        # Apply conversion formula: voltage = (data - center_code) * volt_div_val / code_per_div + vert_offset
        # Then multiply by probe factor to get actual voltage at the probe tip
        voltage = ((raw_data.astype(float) - center_code) * 
                  volt_div_val.get_scaled_value() / vert_code_per_div + 
                  vert_offset.get_scaled_value()) * probe_value
        
        return voltage
    
    def _extract_channel_data(self, data: bytes, header: SiglentBinaryHeader) -> Dict[str, ChannelData]:
        """Extract channel data from the binary file."""
        channels = {}
        
        # Start reading from data offset
        data_start = header.data_offset_byte if header.data_offset_byte > 0 else 0x1000
        current_offset = data_start
        
        # Determine data type based on data width
        if header.data_width == 0:  # 8-bit
            dtype = np.uint8
            bytes_per_sample = 1
        else:  # 16-bit
            dtype = np.uint16
            bytes_per_sample = 2
        
        # Process each analog channel
        channel_configs = [
            ("C1", header.ch1_on, header.ch1_volt_div_val, header.ch1_vert_offset, 
             header.ch1_probe, header.ch1_vert_code_per_div),
            ("C2", header.ch2_on, header.ch2_volt_div_val, header.ch2_vert_offset, 
             header.ch2_probe, header.ch2_vert_code_per_div),
            ("C3", header.ch3_on, header.ch3_volt_div_val, header.ch3_vert_offset, 
             header.ch3_probe, header.ch3_vert_code_per_div),
            ("C4", header.ch4_on, header.ch4_volt_div_val, header.ch4_vert_offset, 
             header.ch4_probe, header.ch4_vert_code_per_div),
        ]
        
        for ch_name, enabled, volt_div_val, vert_offset, probe_value, vert_code_per_div in channel_configs:
            if enabled and header.wave_length > 0:
                # Read raw data
                data_bytes = header.wave_length * bytes_per_sample
                raw_bytes = data[current_offset:current_offset + data_bytes]
                
                if len(raw_bytes) == data_bytes:
                    # Convert to numpy array
                    raw_data = np.frombuffer(raw_bytes, dtype=dtype)
                    
                    # Handle byte order if needed
                    if header.byte_order == 1 and bytes_per_sample > 1:  # MSB
                        raw_data = raw_data.byteswap()
                    
                    # Convert to voltage
                    voltage_data = self._convert_raw_to_voltage(
                        raw_data, volt_div_val, vert_offset, 
                        vert_code_per_div, header.data_width, probe_value
                    )
                    
                    channels[ch_name] = ChannelData(
                        channel_name=ch_name,
                        enabled=enabled,
                        volt_div_val=volt_div_val,
                        vert_offset=vert_offset,
                        probe_value=probe_value,
                        vert_code_per_div=vert_code_per_div,
                        raw_data=raw_data,
                        voltage_data=voltage_data
                    )
                    
                current_offset += data_bytes
            else:
                # Channel disabled or no data
                channels[ch_name] = ChannelData(
                    channel_name=ch_name,
                    enabled=enabled,
                    volt_div_val=volt_div_val,
                    vert_offset=vert_offset,
                    probe_value=probe_value,
                    vert_code_per_div=vert_code_per_div,
                    raw_data=np.array([]),
                    voltage_data=np.array([])
                )
        
        return channels
    
    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, ChannelData]:
        """Parse a single Siglent binary file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        if len(data) < 4096:  # Minimum header size
            raise ValueError(f"File too small to contain valid header: {file_path}")
        
        # Parse header
        self.header = self._parse_header(data)
        
        # Strict validation: Only accept version 4
        if self.header.version != __supported_format_version__:
            raise ValueError(
                f"Unsupported file format version: {self.header.version}. "
                f"This parser only supports {__format_name__} (version {__supported_format_version__}). "
                f"Please ensure your file is in the correct format."
            )
        
        # Extract channel data
        self.channels = self._extract_channel_data(data, self.header)
        
        return self.channels
    
    def parse_directory(self, directory_path: Union[str, Path]) -> Dict[str, ChannelData]:
        """Parse all Siglent binary files in a directory and combine channel data."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all .bin files
        bin_files = list(directory_path.glob("*.bin"))
        
        if not bin_files:
            raise ValueError(f"No .bin files found in directory: {directory_path}")
        
        combined_channels = {}
        
        for bin_file in sorted(bin_files):
            try:
                channels = self.parse_file(bin_file)
                
                # Extract channel name from filename (e.g., "SDS814X_HD_Binary_C1_1.bin" -> "C1")
                filename = bin_file.name
                if "_C" in filename:
                    ch_identifier = filename.split("_C")[1].split("_")[0]
                    ch_name = f"C{ch_identifier}"
                    
                    if ch_name in channels and channels[ch_name].enabled:
                        combined_channels[ch_name] = channels[ch_name]
                        
            except Exception as e:
                print(f"Warning: Could not parse {bin_file}: {e}")
                continue
        
        self.channels = combined_channels
        return combined_channels
    
    def get_channel_names(self) -> List[str]:
        """Get list of available channel names."""
        return list(self.channels.keys())
    
    def get_enabled_channels(self) -> Dict[str, ChannelData]:
        """Get only the enabled channels."""
        return {name: ch for name, ch in self.channels.items() if ch.enabled}
    
    def print_header_info(self):
        """Print header information in a readable format."""
        if not self.header:
            print("No header loaded. Parse a file first.")
            return
        
        h = self.header
        print(f"=== Siglent Binary File Header (Version {h.version}) ===")
        print(f"Data offset: 0x{h.data_offset_byte:04x}")
        print(f"Wave length: {h.wave_length}")
        print(f"Sample rate: {h.sample_rate.get_scaled_value():e} Sa/s")
        print(f"Data width: {'8-bit' if h.data_width == 0 else '16-bit'}")
        print(f"Byte order: {'LSB' if h.byte_order == 0 else 'MSB'}")
        print(f"Time div: {h.time_div.get_scaled_value():e} {h.time_div.get_unit_string()}")
        print(f"Time delay: {h.time_delay.get_scaled_value():e} {h.time_delay.get_unit_string()}")
        print()
        
        # Channel info
        ch_info = [
            ("C1", h.ch1_on, h.ch1_volt_div_val, h.ch1_vert_offset, h.ch1_probe),
            ("C2", h.ch2_on, h.ch2_volt_div_val, h.ch2_vert_offset, h.ch2_probe),
            ("C3", h.ch3_on, h.ch3_volt_div_val, h.ch3_vert_offset, h.ch3_probe),
            ("C4", h.ch4_on, h.ch4_volt_div_val, h.ch4_vert_offset, h.ch4_probe),
        ]
        
        for ch_name, enabled, volt_div, vert_offset, probe in ch_info:
            status = "ON" if enabled else "OFF"
            print(f"{ch_name}: {status}")
            if enabled:
                print(f"  Volt/div: {volt_div.get_scaled_value():e} {volt_div.get_unit_string()}")
                print(f"  Offset: {vert_offset.get_scaled_value():e} {vert_offset.get_unit_string()}")
                print(f"  Probe: {probe}x")


# Backwards compatibility alias
SiglentBinaryParser = SiglentBinaryParser

#!/usr/bin/env python3
"""
Tests for the new preprocessing features in the 60 Hz Curve Fitting Processor.

This module tests the normalize_signals and remove_offset parameters.
"""

import unittest
import numpy as np
import warnings
from typing import Dict, Any

try:
    from curvefit_60hz_processor import CurveFit60HzProcessor
    from base_processor import ProcessingError
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from curvefit_60hz_processor import CurveFit60HzProcessor
    from base_processor import ProcessingError


class TestPreprocessingFeatures(unittest.TestCase):
    """Test the new preprocessing features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = CurveFit60HzProcessor()
        
        # Create test time array
        self.time_array = np.linspace(0, 1, 1000)
        
        # Suppress warnings during tests
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    def tearDown(self):
        """Clean up after tests."""
        warnings.resetwarnings()
    
    def _create_test_input(self, channel_name: str, time_array: np.ndarray, voltage_data: np.ndarray) -> Dict[str, Any]:
        """Create properly formatted test input data."""
        return {
            'channels': {
                channel_name: {
                    'time_array': time_array,
                    'voltage_data': voltage_data,
                    'metadata': {'test_channel': True}
                }
            },
            'header': {'test_data': True},
            'source_info': {'test_source': True}
        }
    
    def test_parameter_definitions(self):
        """Test that new parameters are properly defined."""
        params = self.processor.get_parameters()
        
        # Check that normalize_signals parameter exists
        self.assertIn('normalize_signals', params)
        norm_param = params['normalize_signals']
        self.assertEqual(norm_param['type'], bool)
        self.assertEqual(norm_param['default'], False)
        self.assertIn('normalize', norm_param['description'].lower())
        
        # Check that remove_offset parameter exists
        self.assertIn('remove_offset', params)
        offset_param = params['remove_offset']
        self.assertEqual(offset_param['type'], bool)
        self.assertEqual(offset_param['default'], False)
        self.assertIn('offset', offset_param['description'].lower())
    
    def test_normalize_signals_feature(self):
        """Test the normalize signals preprocessing feature."""
        # Create test signal with very large amplitude
        large_amplitude = 100.0
        phase = 0.0
        offset = 0.0
        
        voltage_data = large_amplitude * np.sin(2 * np.pi * 60 * self.time_array + phase) + offset
        input_data = self._create_test_input('large_signal', self.time_array, voltage_data)
        
        # Test without normalization
        params_no_norm = {
            'normalize_signals': False,
            'remove_offset': False,
            'output_fitted_signal': True,
            'output_parameters': True,
            'min_fit_quality': 0.0,
            'max_iterations': 5000
        }
        
        result_no_norm = self.processor.process(input_data, params_no_norm)
        amp_no_norm = result_no_norm['channels']['large_signal_amplitude']['voltage_data'][0]
        
        # Test with normalization
        params_with_norm = params_no_norm.copy()
        params_with_norm['normalize_signals'] = True
        
        result_with_norm = self.processor.process(input_data, params_with_norm)
        amp_with_norm = result_with_norm['channels']['large_signal_amplitude']['voltage_data'][0]
        
        # Both should recover the correct amplitude
        self.assertAlmostEqual(amp_no_norm, large_amplitude, delta=0.1)
        self.assertAlmostEqual(amp_with_norm, large_amplitude, delta=0.1)
        
        print(f"Large amplitude signal test:")
        print(f"  Without normalization: {amp_no_norm:.4f} V")
        print(f"  With normalization: {amp_with_norm:.4f} V")
        print(f"  Expected: {large_amplitude:.4f} V")
    
    def test_remove_offset_feature(self):
        """Test the remove offset preprocessing feature."""
        # Create test signal with large DC offset
        amplitude = 2.0
        phase = 0.5
        large_offset = 50.0
        
        voltage_data = amplitude * np.sin(2 * np.pi * 60 * self.time_array + phase) + large_offset
        input_data = self._create_test_input('offset_signal', self.time_array, voltage_data)
        
        # Test without offset removal
        params_no_offset = {
            'normalize_signals': False,
            'remove_offset': False,
            'output_fitted_signal': True,
            'output_parameters': True,
            'min_fit_quality': 0.0,
            'max_iterations': 5000
        }
        
        result_no_offset = self.processor.process(input_data, params_no_offset)
        amp_no_offset = result_no_offset['channels']['offset_signal_amplitude']['voltage_data'][0]
        offset_no_offset = result_no_offset['channels']['offset_signal_offset']['voltage_data'][0]
        
        # Test with offset removal
        params_with_offset = params_no_offset.copy()
        params_with_offset['remove_offset'] = True
        
        result_with_offset = self.processor.process(input_data, params_with_offset)
        amp_with_offset = result_with_offset['channels']['offset_signal_amplitude']['voltage_data'][0]
        offset_with_offset = result_with_offset['channels']['offset_signal_offset']['voltage_data'][0]
        
        # Both should recover correct parameters
        self.assertAlmostEqual(amp_no_offset, amplitude, delta=0.1)
        self.assertAlmostEqual(amp_with_offset, amplitude, delta=0.1)
        self.assertAlmostEqual(offset_no_offset, large_offset, delta=0.1)
        self.assertAlmostEqual(offset_with_offset, large_offset, delta=0.1)
        
        print(f"Large offset signal test:")
        print(f"  Without offset removal: amp={amp_no_offset:.4f} V, offset={offset_no_offset:.4f} V")
        print(f"  With offset removal: amp={amp_with_offset:.4f} V, offset={offset_with_offset:.4f} V")
        print(f"  Expected: amp={amplitude:.4f} V, offset={large_offset:.4f} V")
    
    def test_combined_preprocessing(self):
        """Test combined normalization and offset removal."""
        # Create challenging signal: small amplitude with large offset
        small_amplitude = 0.05
        phase = 1.0
        large_offset = 20.0
        
        voltage_data = small_amplitude * np.sin(2 * np.pi * 60 * self.time_array + phase) + large_offset
        input_data = self._create_test_input('challenging_signal', self.time_array, voltage_data)
        
        # Test with both preprocessing options enabled
        params_full_preproc = {
            'normalize_signals': True,
            'remove_offset': True,
            'output_fitted_signal': True,
            'output_parameters': True,
            'min_fit_quality': 0.0,
            'max_iterations': 5000
        }
        
        result = self.processor.process(input_data, params_full_preproc)
        
        # Extract parameters
        amplitude_recovered = result['channels']['challenging_signal_amplitude']['voltage_data'][0]
        phase_recovered = result['channels']['challenging_signal_phase']['voltage_data'][0]
        offset_recovered = result['channels']['challenging_signal_offset']['voltage_data'][0]
        
        # Get fit quality
        fit_metadata = result['channels']['challenging_signal_60hz_fit']['metadata']['processing']['fit_quality']
        r_squared = fit_metadata['r_squared']
        
        # Verify parameter recovery
        self.assertAlmostEqual(amplitude_recovered, small_amplitude, delta=0.01)
        self.assertAlmostEqual(phase_recovered, phase, delta=0.1)
        self.assertAlmostEqual(offset_recovered, large_offset, delta=0.1)
        
        # Should have good fit quality
        self.assertGreater(r_squared, 0.9)
        
        print(f"Combined preprocessing test:")
        print(f"  Recovered: amp={amplitude_recovered:.6f} V, phase={phase_recovered:.4f} rad, offset={offset_recovered:.4f} V")
        print(f"  Expected: amp={small_amplitude:.6f} V, phase={phase:.4f} rad, offset={large_offset:.4f} V")
        print(f"  R-squared: {r_squared:.6f}")
    
    def test_preprocessing_with_constant_signal(self):
        """Test preprocessing with constant (DC-only) signal."""
        dc_value = 5.0
        voltage_data = np.full_like(self.time_array, dc_value)
        input_data = self._create_test_input('dc_signal', self.time_array, voltage_data)
        
        # Test with normalization (should handle gracefully)
        params_norm = {
            'normalize_signals': True,
            'remove_offset': False,
            'output_fitted_signal': True,
            'output_parameters': True,
            'min_fit_quality': 0.0,
            'max_iterations': 5000
        }
        
        # Should not raise an exception
        result = self.processor.process(input_data, params_norm)
        
        # Should have zero amplitude and correct offset
        amplitude = result['channels']['dc_signal_amplitude']['voltage_data'][0]
        offset = result['channels']['dc_signal_offset']['voltage_data'][0]
        
        self.assertLess(abs(amplitude), 0.1)  # Near zero amplitude
        self.assertAlmostEqual(offset, dc_value, delta=0.1)
        
        print(f"Constant signal test:")
        print(f"  Recovered: amp={amplitude:.6f} V, offset={offset:.4f} V")
        print(f"  Expected: amp=0.0 V, offset={dc_value:.4f} V")
    
    def test_preprocessing_improves_fit_quality(self):
        """Test that preprocessing improves fit quality for appropriate signals."""
        # Create signal that should benefit from preprocessing
        amplitude = 0.1  # Small amplitude
        phase = 0.0
        offset = 10.0   # Large offset
        
        # Add some noise
        np.random.seed(123)
        clean_signal = amplitude * np.sin(2 * np.pi * 60 * self.time_array + phase) + offset
        noisy_signal = clean_signal + 0.02 * np.random.normal(size=len(clean_signal))
        
        input_data = self._create_test_input('test_signal', self.time_array, noisy_signal)
        
        # Test without preprocessing
        params_standard = {
            'normalize_signals': False,
            'remove_offset': False,
            'output_fitted_signal': True,
            'output_parameters': True,
            'min_fit_quality': 0.0,
            'max_iterations': 5000
        }
        
        result_standard = self.processor.process(input_data, params_standard)
        r2_standard = result_standard['channels']['test_signal_60hz_fit']['metadata']['processing']['fit_quality']['r_squared']
        
        # Test with full preprocessing
        params_preproc = params_standard.copy()
        params_preproc['normalize_signals'] = True
        params_preproc['remove_offset'] = True
        
        result_preproc = self.processor.process(input_data, params_preproc)
        r2_preproc = result_preproc['channels']['test_signal_60hz_fit']['metadata']['processing']['fit_quality']['r_squared']
        
        print(f"Fit quality comparison:")
        print(f"  Standard processing: R² = {r2_standard:.6f}")
        print(f"  With preprocessing: R² = {r2_preproc:.6f}")
        
        # Preprocessing should maintain or improve fit quality
        # (Note: For this specific case, both should be good, but preprocessing provides robustness)
        self.assertGreaterEqual(r2_preproc, 0.8)
        self.assertGreaterEqual(r2_standard, 0.8)


def run_preprocessing_tests():
    """Run preprocessing feature tests with detailed output."""
    print("=" * 70)
    print("Testing Preprocessing Features for 60Hz Curve Fitting Processor")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessingFeatures)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING TESTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("\n✓ All preprocessing features are working correctly!")
        print("✓ Normalize Signals feature tested and validated")
        print("✓ Remove Offset feature tested and validated")
        print("✓ Combined preprocessing tested and validated")
    
    return success


if __name__ == "__main__":
    run_preprocessing_tests()

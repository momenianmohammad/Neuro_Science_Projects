#!/usr/bin/env python3
"""
Unit Tests for Sleep Analysis Tool
==================================

Comprehensive test suite for the Sleep and Dream Pattern Analysis Tool.

Author: [Your Name]
Institution: [Your University]
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from sleep_analyzer import SleepAnalyzer

class TestSleepAnalyzer(unittest.TestCase):
    """Test cases for SleepAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = SleepAnalyzer(sampling_rate=256, epoch_length=30)
        self.sample_data = self._create_test_data()
    
    def _create_test_data(self):
        """Create sample EEG data for testing."""
        duration = 3600  # 1 hour
        n_samples = duration * self.analyzer.sampling_rate
        
        # Create two channels of synthetic EEG
        channel1 = np.random.randn(n_samples) * 0.5
        channel2 = np.random.randn(n_samples) * 0.5
        
        return {
            'data': np.array([channel1, channel2]),
            'channel_names': ['EEG1', 'EEG2'],
            'sampling_rate': self.analyzer.sampling_rate,
            'duration': duration
        }
    
    def test_initialization(self):
        """Test SleepAnalyzer initialization."""
        self.assertEqual(self.analyzer.sampling_rate, 256)
        self.assertEqual(self.analyzer.epoch_length, 30)
        self.assertEqual(self.analyzer.samples_per_epoch, 256 * 30)
        self.assertEqual(len(self.analyzer.sleep_stages), 5)
        self.assertIn('delta', self.analyzer.frequency_bands)
    
    def test_feature_extraction(self):
        """Test feature extraction from EEG epoch."""
        # Create a single epoch of data
        epoch_data = np.random.randn(2, self.analyzer.samples_per_epoch)
        
        features = self.analyzer.extract_features(epoch_data)
        
        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for expected feature types
        feature_keys = list(features.keys())
        self.assertTrue(any('delta_power' in key for key in feature_keys))
        self.assertTrue(any('theta_power' in key for key in feature_keys))
        self.assertTrue(any('mean' in key for key in feature_keys))
        self.assertTrue(any('std' in key for key in feature_keys))
    
    def test_classifier_training(self):
        """Test classifier training with synthetic data."""
        # Train the classifier
        accuracy = self.analyzer.train_classifier()
        
        # Check that classifier was trained
        self.assertIsNotNone(self.analyzer.classifier)
        self.assertIsNotNone(self.analyzer.scaler)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_sleep_stage_classification(self):
        """Test sleep stage classification."""
        # Train classifier first
        self.analyzer.train_classifier()
        
        # Classify sleep stages
        results = self.analyzer.classify_sleep_stages(self.sample_data)
        
        # Check results structure
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertIn('epoch_times', results)
        self.assertIn('stage_names', results)
        
        # Check predictions are valid
        predictions = results['predictions']
        self.assertTrue(all(0 <= pred <= 4 for pred in predictions))
        
        # Check probabilities sum to 1
        probabilities = results['probabilities']
        for prob in probabilities:
            self.assertAlmostEqual(np.sum(prob), 1.0, places=5)
    
    def test_sleep_metrics_calculation(self):
        """Test sleep metrics calculation."""
        # Create mock sleep stage results
        n_epochs = 120  # 1 hour of 30-second epochs
        mock_results = {
            'predictions': np.random.randint(0, 5, n_epochs),
            'epoch_times': np.arange(n_epochs) * 0.5,  # in minutes
            'stage_names': [self.analyzer.sleep_stages[i] for i in np.random.randint(0, 5, n_epochs)]
        }
        
        metrics = self.analyzer.calculate_sleep_metrics(mock_results)
        
        # Check required metrics are present
        required_metrics = [
            'total_recording_time', 'total_sleep_time', 'sleep_efficiency',
            'sleep_onset_latency', 'rem_latency', 'stage_counts',
            'stage_percentages', 'quality_score'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['sleep_efficiency'], 0)
        self.assertLessEqual(metrics['sleep_efficiency'], 100)
        self.assertGreaterEqual(metrics['quality_score'], 0)
        self.assertLessEqual(metrics['quality_score'], 100)
    
    def test_report_generation(self):
        """Test sleep report generation."""
        # Create mock data
        mock_stages = {
            'predictions': np.array([0, 1, 2, 3, 4, 2, 3, 2]),
            'epoch_times': np.arange(8) * 0.5,
            'stage_names': ['Wake', 'N1', 'N2', 'N3', 'REM', 'N2', 'N3', 'N2']
        }
        
        mock_metrics = {
            'total_recording_time': 60,
            'total_sleep_time': 50,
            'sleep_efficiency': 85.0,
            'sleep_onset_latency': 10.0,
            'rem_latency': 90.0,
            'stage_counts': {'Wake': 1, 'N1': 1, 'N2': 3, 'N3': 2, 'REM': 1},
            'stage_percentages': {'N1': 12.5, 'N2': 37.5, 'N3': 25.0, 'REM': 12.5},
            'transitions': 7,
            'fragmentation_index': 8.4,
            'quality_score': 78.5
        }
        
        patient_info = {'name': 'Test Patient', 'age': '30', 'date': '2024-01-01'}
        
        report = self.analyzer.generate_sleep_report(mock_stages, mock_metrics, patient_info)
        
        # Check report content
        self.assertIsInstance(report, str)
        self.assertIn('SLEEP ANALYSIS REPORT', report)
        self.assertIn('Test Patient', report)
        self.assertIn('Sleep Efficiency: 85.0%', report)
        self.assertIn('Quality Score: 78.5/100', report)
    
    def test_visualization_creation(self):
        """Test visualization creation without displaying."""
        # Mock sleep stages and metrics
        mock_stages = {
            'predictions': np.array([0, 1, 2, 3, 4, 2, 3, 2]),
            'epoch_times': np.arange(8) * 0.5,
            'stage_names': ['Wake', 'N1', 'N2', 'N3', 'REM', 'N2', 'N3', 'N2']
        }
        
        mock_metrics = {
            'stage_counts': {'Wake': 1, 'N1': 1, 'N2': 3, 'N3': 2, 'REM': 1},
            'stage_percentages': {'N1': 12.5, 'N2': 37.5, 'N3': 25.0, 'REM': 12.5},
            'sleep_efficiency': 85.0,
            'quality_score': 78.5
        }
        
        # Test visualization creation (mock plt.show to avoid display)
        with patch('matplotlib.pyplot.show'):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                try:
                    self.analyzer.visualize_sleep_patterns(mock_stages, mock_metrics, tmp.name)
                    self.assertTrue(os.path.exists(tmp.name))
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
    
    def test_complete_analysis_pipeline(self):
        """Test the complete analysis pipeline."""
        patient_info = {'name': 'Test', 'age': '25', 'date': '2024-01-01'}
        
        # Mock plt.show to avoid displaying plots
        with patch('matplotlib.pyplot.show'):
            results = self.analyzer.analyze_sleep(self.sample_data, patient_info, save_plots=False)
        
        # Check all components are present
        self.assertIn('sleep_stages', results)
        self.assertIn('metrics', results)
        self.assertIn('report', results)
        self.assertIn('quality_score', results)
        
        # Check data types
        self.assertIsInstance(results['quality_score'], (int, float))
        self.assertIsInstance(results['report'], str)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short data
        short_data = {
            'data': np.random.randn(2, 1000),  # Very short
            'channel_names': ['EEG1', 'EEG2'],
            'sampling_rate': 256,
            'duration': 1000/256
        }
        
        # Should handle gracefully
        with patch('matplotlib.pyplot.show'):
            results = self.analyzer.analyze_sleep(short_data, save_plots=False)
            self.assertIsNotNone(results)
    
    def test_frequency_band_validation(self):
        """Test frequency band definitions."""
        for band_name, (low, high) in self.analyzer.frequency_bands.items():
            self.assertIsInstance(low, (int, float))
            self.assertIsInstance(high, (int, float))
            self.assertLess(low, high)
            self.assertGreaterEqual(low, 0)
    
    def test_synthetic_data_generation(self):
        """Test synthetic training data generation."""
        X, y = self.analyzer.create_synthetic_training_data(n_samples=100)
        
        # Check data shapes
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(X.shape[0], y.shape[0])
        
        # Check label range
        self.assertTrue(all(0 <= label <= 4 for label in y))
        
        # Check feature extraction worked
        self.assertGreater(X.shape[1], 0)  # Should have features
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with invalid sampling rate
        with self.assertRaises((ValueError, TypeError)):
            SleepAnalyzer(sampling_rate=-1)
        
        # Test with invalid epoch length
        with self.assertRaises((ValueError, TypeError)):
            SleepAnalyzer(epoch_length=0)

class TestSleepMetrics(unittest.TestCase):
    """Test cases for sleep metrics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SleepAnalyzer()
    
    def test_sleep_efficiency_calculation(self):
        """Test sleep efficiency calculation."""
        # All wake epochs should give 0% efficiency
        all_wake = {
            'predictions': np.zeros(100),  # All wake (stage 0)
            'epoch_times': np.arange(100) * 0.5
        }
        metrics = self.analyzer.calculate_sleep_metrics(all_wake)
        self.assertEqual(metrics['sleep_efficiency'], 0.0)
        
        # No wake epochs should give 100% efficiency
        no_wake = {
            'predictions': np.ones(100),  # All N1 (stage 1)
            'epoch_times': np.arange(100) * 0.5
        }
        metrics = self.analyzer.calculate_sleep_metrics(no_wake)
        self.assertEqual(metrics['sleep_efficiency'], 100.0)
    
    def test_stage_percentages(self):
        """Test stage percentage calculations."""
        # Equal distribution test
        equal_stages = {
            'predictions': np.tile([1, 2, 3, 4], 25),  # 100 epochs, equal N1,N2,N3,REM
            'epoch_times': np.arange(100) * 0.5
        }
        
        metrics = self.analyzer.calculate_sleep_metrics(equal_stages)
        
        # Each sleep stage should be 25% (excluding wake)
        for stage in ['N1', 'N2', 'N3', 'REM']:
            self.assertAlmostEqual(metrics['stage_percentages'][stage], 25.0, places=1)
    
    def test_quality_score_bounds(self):
        """Test that quality score stays within valid bounds."""
        # Test multiple random configurations
        for _ in range(10):
            random_stages = {
                'predictions': np.random.randint(0, 5, 200),
                'epoch_times': np.arange(200) * 0.5
            }
            
            metrics = self.analyzer.calculate_sleep_metrics(random_stages)
            
            self.assertGreaterEqual(metrics['quality_score'], 0)
            self.assertLessEqual(metrics['quality_score'], 100)

class TestFeatureExtraction(unittest.TestCase):
    """Test cases for EEG feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SleepAnalyzer()
        self.epoch_data = np.random.randn(2, self.analyzer.samples_per_epoch)
    
    def test_power_features(self):
        """Test power spectral density feature extraction."""
        features = self.analyzer.extract_features(self.epoch_data)
        
        # Check that power features exist for each band
        for band in self.analyzer.frequency_bands.keys():
            power_features = [k for k in features.keys() if f'{band}_power' in k]
            self.assertGreater(len(power_features), 0)
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.analyzer.extract_features(self.epoch_data)
        
        # Check for statistical features
        stat_features = ['mean', 'std', 'skewness', 'kurtosis']
        for stat in stat_features:
            stat_keys = [k for k in features.keys() if stat in k]
            self.assertGreater(len(stat_keys), 0)
    
    def test_hjorth_parameters(self):
        """Test Hjorth parameters calculation."""
        features = self.analyzer.extract_features(self.epoch_data)
        
        # Check for Hjorth parameters
        hjorth_params = ['activity', 'mobility', 'complexity']
        for param in hjorth_params:
            param_keys = [k for k in features.keys() if param in k]
            self.assertGreater(len(param_keys), 0)
    
    def test_feature_values_validity(self):
        """Test that extracted features have valid values."""
        features = self.analyzer.extract_features(self.epoch_data)
        
        for key, value in features.items():
            # Check for NaN or infinite values
            self.assertFalse(np.isnan(value), f"Feature {key} is NaN")
            self.assertFalse(np.isinf(value), f"Feature {key} is infinite")
            # Check that power values are non-negative
            if 'power' in key:
                self.assertGreaterEqual(value, 0, f"Power feature {key} is negative")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SleepAnalyzer()
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test data
        duration = 7200  # 2 hours
        n_samples = duration * self.analyzer.sampling_rate
        
        test_data = {
            'data': np.random.randn(2, n_samples),
            'channel_names': ['C3-A2', 'C4-A1'],
            'sampling_rate': self.analyzer.sampling_rate,
            'duration': duration
        }
        
        patient_info = {
            'name': 'Integration Test',
            'age': '35',
            'date': '2024-01-01'
        }
        
        # Run complete analysis
        with patch('matplotlib.pyplot.show'):
            results = self.analyzer.analyze_sleep(test_data, patient_info, save_plots=False)
        
        # Verify all components
        self.assertIn('sleep_stages', results)
        self.assertIn('metrics', results)
        self.assertIn('report', results)
        self.assertIn('quality_score', results)
        
        # Verify data consistency
        n_epochs = len(results['sleep_stages']['predictions'])
        expected_epochs = duration // self.analyzer.epoch_length
        self.assertEqual(n_epochs, expected_epochs)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same input."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        test_data = {
            'data': np.random.randn(2, 30720),  # 2 minutes of data
            'channel_names': ['EEG1', 'EEG2'],
            'sampling_rate': 256,
            'duration': 120
        }
        
        # Run analysis twice
        with patch('matplotlib.pyplot.show'):
            results1 = self.analyzer.analyze_sleep(test_data, save_plots=False)
            
            # Reset random seed
            np.random.seed(42)
            results2 = self.analyzer.analyze_sleep(test_data, save_plots=False)
        
        # Results should be identical
        np.testing.assert_array_equal(
            results1['sleep_stages']['predictions'],
            results2['sleep_stages']['predictions']
        )

def run_performance_tests():
    """Run performance benchmarks."""
    print("\n🚀 Running Performance Tests")
    print("=" * 30)
    
    analyzer = SleepAnalyzer()
    
    # Test different data sizes
    test_sizes = [
        (1, "1 hour"),
        (4, "4 hours"),
        (8, "8 hours")
    ]
    
    for hours, description in test_sizes:
        duration = hours * 3600
        n_samples = duration * analyzer.sampling_rate
        
        test_data = {
            'data': np.random.randn(2, n_samples),
            'channel_names': ['EEG1', 'EEG2'],
            'sampling_rate': analyzer.sampling_rate,
            'duration': duration
        }
        
        import time
        start_time = time.time()
        
        with patch('matplotlib.pyplot.show'):
            results = analyzer.analyze_sleep(test_data, save_plots=False)
        
        processing_time = time.time() - start_time
        n_epochs = len(results['sleep_stages']['predictions'])
        
        print(f"{description:8} | {n_epochs:4d} epochs | {processing_time:6.2f}s | "
              f"{processing_time/hours:5.2f}s/hour")

def main():
    """Run all tests."""
    print("🧪 Sleep Analysis Tool - Test Suite")
    print("=" * 40)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSleepAnalyzer,
        TestSleepMetrics,
        TestFeatureExtraction,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Performance tests
    run_performance_tests()
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
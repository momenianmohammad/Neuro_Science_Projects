"""
Unit tests for Brain Connectome Analysis System
==============================================

This module contains comprehensive tests for the BrainConnectomeAnalyzer class
to ensure reliability and correctness of network analysis functions.

Author: [Your Name]
Date: 2024
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from brain_connectome_analyzer import BrainConnectomeAnalyzer, create_sample_data

class TestBrainConnectomeAnalyzer(unittest.TestCase):
    """Test cases for BrainConnectomeAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        np.random.seed(42)  # For reproducible tests
        self.connectivity_matrix, self.region_labels = create_sample_data()
        
        # Initialize analyzer
        self.analyzer = BrainConnectomeAnalyzer()
        self.analyzer.load_connectivity_data(self.connectivity_matrix, self.region_labels)
    
    def test_initialization(self):
        """Test proper initialization of the analyzer"""
        self.assertIsNotNone(self.analyzer.connectivity_matrix)
        self.assertIsNotNone(self.analyzer.graph)
        self.assertIsNotNone(self.analyzer.node_labels)
        self.assertEqual(len(self.analyzer.node_labels), self.connectivity_matrix.shape[0])
    
    def test_connectivity_matrix_properties(self):
        """Test connectivity matrix has correct properties"""
        # Should be square matrix
        self.assertEqual(self.connectivity_matrix.shape[0], self.connectivity_matrix.shape[1])
        
        # Should be symmetric (for undirected networks)
        np.testing.assert_array_almost_equal(
            self.connectivity_matrix, 
            self.connectivity_matrix.T,
            decimal=10
        )
        
        # Diagonal should be zeros (no self-connections)
        np.testing.assert_array_equal(
            np.diag(self.connectivity_matrix),
            np.zeros(self.connectivity_matrix.shape[0])
        )
        
        # Values should be non-negative
        self.assertTrue(np.all(self.connectivity_matrix >= 0))
    
    def test_global_metrics_calculation(self):
        """Test calculation of global network metrics"""
        metrics = self.analyzer.calculate_global_metrics()
        
        # Check that all expected metrics are present
        expected_metrics = [
            'num_nodes', 'num_edges', 'density', 
            'average_path_length', 'global_clustering',
            'transitivity', 'small_world_sigma'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges and types
        self.assertIsInstance(metrics['num_nodes'], int)
        self.assertIsInstance(metrics['num_edges'], int)
        self.assertGreater(metrics['num_nodes'], 0)
        self.assertGreater(metrics['num_edges'], 0)
        
        # Density should be between 0 and 1
        self.assertGreaterEqual(metrics['density'], 0)
        self.assertLessEqual(metrics['density'], 1)
        
        # Clustering should be between 0 and 1
        self.assertGreaterEqual(metrics['global_clustering'], 0)
        self.assertLessEqual(metrics['global_clustering'], 1)
        
        # Path length should be positive
        self.assertGreater(metrics['average_path_length'], 0)
    
    def test_nodal_metrics_calculation(self):
        """Test calculation of node-level metrics"""
        nodal_metrics = self.analyzer.calculate_nodal_metrics()
        
        # Check DataFrame structure
        self.assertIsInstance(nodal_metrics, pd.DataFrame)
        self.assertEqual(len(nodal_metrics), len(self.analyzer.node_labels))
        
        # Check that all expected columns are present
        expected_columns = [
            'Region', 'Degree_Centrality', 'Betweenness_Centrality',
            'Closeness_Centrality', 'Eigenvector_Centrality', 'Clustering_Coefficient'
        ]
        
        for column in expected_columns:
            self.assertIn(column, nodal_metrics.columns)
        
        # Check value ranges
        for column in expected_columns[1:]:  # Skip 'Region' column
            values = nodal_metrics[column]
            self.assertTrue(np.all(values >= 0))
            self.assertTrue(np.all(values <= 1))  # Centrality measures are normalized
        
        # Check that region names match
        self.assertEqual(
            set(nodal_metrics['Region']), 
            set(self.analyzer.node_labels)
        )
    
    def test_hub_identification(self):
        """Test hub identification functionality"""
        # Calculate nodal metrics first
        self.analyzer.calculate_nodal_metrics()
        
        # Test different hub identification methods
        methods = ['degree', 'betweenness', 'composite']
        
        for method in methods:
            hubs = self.analyzer.identify_hubs(method=method, threshold_percentile=90)
            
            self.assertIsInstance(hubs, pd.DataFrame)
            self.assertGreater(len(hubs), 0)  # Should identify some hubs
            self.assertLessEqual(len(hubs), len(self.analyzer.node_labels))
            
            # Hubs should have higher values than non-hubs
            if method == 'degree':
                metric_col = 'Degree_Centrality'
            elif method == 'betweenness':
                metric_col = 'Betweenness_Centrality'
            else:  # composite
                metric_col = 'Composite_Hub_Score'
            
            if metric_col in self.analyzer.nodal_metrics.columns:
                hub_values = hubs[metric_col]
                all_values = self.analyzer.nodal_metrics[metric_col]
                self.assertGreater(hub_values.min(), all_values.quantile(0.85))
    
    def test_community_detection(self):
        """Test community detection functionality"""
        communities, modularity = self.analyzer.detect_communities()
        
        # Check return types
        self.assertIsInstance(communities, list)
        self.assertIsInstance(modularity, float)
        
        # Modularity should be between -1 and 1
        self.assertGreaterEqual(modularity, -1)
        self.assertLessEqual(modularity, 1)
        
        # Should detect some communities
        self.assertGreater(len(communities), 0)
        self.assertLess(len(communities), len(self.analyzer.node_labels))
        
        # All nodes should be assigned to exactly one community
        all_nodes_in_communities = set()
        for community in communities:
            all_nodes_in_communities.update(community)
        
        self.assertEqual(all_nodes_in_communities, set(self.analyzer.graph.nodes()))
        
        # Communities should be non-empty
        for community in communities:
            self.assertGreater(len(community), 0)
    
    def test_network_comparison(self):
        """Test network comparison functionality"""
        # Create a slightly different network for comparison
        comparison_matrix = self.connectivity_matrix * 0.9 + np.random.normal(0, 0.01, self.connectivity_matrix.shape)
        comparison_matrix = np.maximum(comparison_matrix, 0)
        comparison_matrix = (comparison_matrix + comparison_matrix.T) / 2
        np.fill_diagonal(comparison_matrix, 0)
        
        # Perform comparison
        comparison_results, comparison_analyzer = self.analyzer.compare_networks(
            comparison_matrix, 
            self.region_labels,
            comparison_name="Test Comparison"
        )
        
        # Check return types
        self.assertIsInstance(comparison_results, dict)
        self.assertIsInstance(comparison_analyzer, BrainConnectomeAnalyzer)
        
        # Check comparison results structure
        for metric_name, values in comparison_results.items():
            self.assertIsInstance(values, dict)
            self.assertIn('current', values)
            self.assertIn('comparison', values)
            self.assertIn('difference', values)
            self.assertIn('percent_change', values)
    
    def test_visualization_data_preparation(self):
        """Test that visualization methods prepare data correctly"""
        # Calculate metrics needed for visualization
        self.analyzer.calculate_nodal_metrics()
        self.analyzer.detect_communities()
        
        # Test connectivity matrix plot data
        fig_matrix = self.analyzer.create_connectivity_matrix_plot()
        self.assertIsNotNone(fig_matrix)
        
        # Test 3D visualization data preparation
        fig_3d = self.analyzer.visualize_network_3d()
        self.assertIsNotNone(fig_3d)
        
        # Check that the figure has data
        self.assertGreater(len(fig_3d.data), 0)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid connectivity matrix (non-square)
        invalid_matrix = np.random.rand(10, 15)
        
        with self.assertRaises((ValueError, AssertionError)):
            invalid_analyzer = BrainConnectomeAnalyzer()
            invalid_analyzer.load_connectivity_data(invalid_matrix)
    
    def test_metrics_consistency(self):
        """Test that metrics are internally consistent"""
        global_metrics = self.analyzer.calculate_global_metrics()
        nodal_metrics = self.analyzer.calculate_nodal_metrics()
        
        # Number of nodes should match
        self.assertEqual(global_metrics['num_nodes'], len(nodal_metrics))
        
        # Average degree centrality should relate to network density
        avg_degree_centrality = nodal_metrics['Degree_Centrality'].mean()
        expected_avg_degree = global_metrics['density']
        
        # They should be reasonably close (within 10%)
        self.assertLess(abs(avg_degree_centrality - expected_avg_degree) / expected_avg_degree, 0.1)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same input"""
        # Calculate metrics twice
        metrics1 = self.analyzer.calculate_global_metrics()
        metrics2 = self.analyzer.calculate_global_metrics()
        
        # Should be identical
        for key in metrics1:
            if isinstance(metrics1[key], float):
                self.assertAlmostEqual(metrics1[key], metrics2[key], places=10)
            else:
                self.assertEqual(metrics1[key], metrics2[key])


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""
    
    def test_sample_data_properties(self):
        """Test that generated sample data has correct properties"""
        connectivity_matrix, region_labels = create_sample_data()
        
        # Check matrix properties
        self.assertEqual(connectivity_matrix.shape[0], connectivity_matrix.shape[1])
        self.assertEqual(len(region_labels), connectivity_matrix.shape[0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(
            connectivity_matrix, 
            connectivity_matrix.T,
            decimal=10
        )
        
        # Check diagonal is zeros
        np.testing.assert_array_equal(
            np.diag(connectivity_matrix),
            np.zeros(connectivity_matrix.shape[0])
        )
        
        # Check non-negative values
        self.assertTrue(np.all(connectivity_matrix >= 0))
        
        # Check reasonable value range
        self.assertTrue(np.all(connectivity_matrix <= 1))
    
    def test_region_labels_format(self):
        """Test that region labels have correct format"""
        _, region_labels = create_sample_data()
        
        # Should have labels for different brain regions
        frontal_count = sum(1 for label in region_labels if 'Frontal' in label)
        parietal_count = sum(1 for label in region_labels if 'Parietal' in label)
        temporal_count = sum(1 for label in region_labels if 'Temporal' in label)
        
        self.assertGreater(frontal_count, 0)
        self.assertGreater(parietal_count, 0)
        self.assertGreater(temporal_count, 0)
        
        # All labels should be unique
        self.assertEqual(len(region_labels), len(set(region_labels)))


class TestNetworkProperties(unittest.TestCase):
    """Test specific network theory properties"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.connectivity_matrix, self.region_labels = create_sample_data()
        self.analyzer = BrainConnectomeAnalyzer()
        self.analyzer.load_connectivity_data(self.connectivity_matrix, self.region_labels)
    
    def test_small_world_properties(self):
        """Test small-world network properties"""
        metrics = self.analyzer.calculate_global_metrics()
        
        # Small-world networks should have:
        # 1. High clustering (> random network)
        # 2. Short path lengths (~ random network)
        # 3. Small-world sigma > 1
        
        self.assertGreater(metrics['global_clustering'], 0.1)  # Should have some clustering
        self.assertGreater(metrics['small_world_sigma'], 0.5)  # Should show some small-world properties
    
    def test_degree_distribution(self):
        """Test degree distribution properties"""
        nodal_metrics = self.analyzer.calculate_nodal_metrics()
        
        degree_centralities = nodal_metrics['Degree_Centrality']
        
        # Should have some variation in degrees
        self.assertGreater(degree_centralities.std(), 0.01)
        
        # Should have reasonable mean degree
        self.assertGreater(degree_centralities.mean(), 0.1)
        self.assertLess(degree_centralities.mean(), 0.9)
    
    def test_centrality_correlations(self):
        """Test correlations between different centrality measures"""
        nodal_metrics = self.analyzer.calculate_nodal_metrics()
        
        # Degree and eigenvector centrality should be positively correlated
        correlation = np.corrcoef(
            nodal_metrics['Degree_Centrality'],
            nodal_metrics['Eigenvector_Centrality']
        )[0, 1]
        
        self.assertGreater(correlation, 0.3)  # Should have moderate positive correlation


def run_performance_tests():
    """Run performance tests for large networks"""
    print("Running performance tests...")
    
    import time
    
    # Test with larger network
    np.random.seed(42)
    n_nodes = 200
    large_matrix = np.random.rand(n_nodes, n_nodes) * 0.3
    large_matrix = (large_matrix + large_matrix.T) / 2
    np.fill_diagonal(large_matrix, 0)
    
    region_labels = [f"Region_{i}" for i in range(n_nodes)]
    
    analyzer = BrainConnectomeAnalyzer()
    
    # Time the analysis
    start_time = time.time()
    analyzer.load_connectivity_data(large_matrix, region_labels)
    load_time = time.time() - start_time
    
    start_time = time.time()
    global_metrics = analyzer.calculate_global_metrics()
    global_time = time.time() - start_time
    
    start_time = time.time()
    nodal_metrics = analyzer.calculate_nodal_metrics()
    nodal_time = time.time() - start_time
    
    start_time = time.time()
    communities, modularity = analyzer.detect_communities()
    community_time = time.time() - start_time
    
    print(f"Performance Results for {n_nodes} nodes:")
    print(f"  Data loading: {load_time:.3f} seconds")
    print(f"  Global metrics: {global_time:.3f} seconds") 
    print(f"  Nodal metrics: {nodal_time:.3f} seconds")
    print(f"  Community detection: {community_time:.3f} seconds")
    print(f"  Total time: {load_time + global_time + nodal_time + community_time:.3f} seconds")


if __name__ == '__main__':
    # Run unit tests
    print("Running Brain Connectome Analysis System Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBrainConnectomeAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestSampleDataGeneration))
    test_suite.addTest(unittest.makeSuite(TestNetworkProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
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
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n✅ All tests passed successfully!")
        
        # Run performance tests
        print("\n" + "=" * 50)
        run_performance_tests()
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print("\n" + "=" * 50)

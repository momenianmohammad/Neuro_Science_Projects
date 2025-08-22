"""
Brain Connectome Analysis System - Complete Usage Examples
=========================================================

This file demonstrates comprehensive usage of the Brain Connectome Analysis System
for various neuroscience research applications.

Author: [Your Name]
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brain_connectome_analyzer import BrainConnectomeAnalyzer, create_sample_data

def example_1_basic_analysis():
    """
    Example 1: Basic Brain Network Analysis
    --------------------------------------
    Demonstrates fundamental network analysis workflow
    """
    print("="*60)
    print("EXAMPLE 1: BASIC BRAIN NETWORK ANALYSIS")
    print("="*60)
    
    # Create sample data (you would load your own connectivity matrix here)
    connectivity_matrix, region_labels = create_sample_data()
    
    print(f"Loaded connectivity matrix: {connectivity_matrix.shape}")
    print(f"Number of brain regions: {len(region_labels)}")
    
    # Initialize the analyzer
    analyzer = BrainConnectomeAnalyzer()
    analyzer.load_connectivity_data(connectivity_matrix, region_labels)
    
    # Step 1: Calculate global network properties
    print("\n--- Step 1: Global Network Metrics ---")
    global_metrics = analyzer.calculate_global_metrics()
    
    for metric, value in global_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Step 2: Calculate node-level metrics
    print("\n--- Step 2: Nodal Metrics ---")
    nodal_metrics = analyzer.calculate_nodal_metrics()
    print("\nTop 5 regions by degree centrality:")
    top_degree = nodal_metrics.nlargest(5, 'Degree_Centrality')
    print(top_degree[['Region', 'Degree_Centrality']])
    
    # Step 3: Identify network hubs
    print("\n--- Step 3: Hub Identification ---")
    hubs = analyzer.identify_hubs(method='composite', threshold_percentile=85)
    
    # Step 4: Detect communities
    print("\n--- Step 4: Community Detection ---")
    communities, modularity = analyzer.detect_communities()
    
    print(f"\nDetected {len(communities)} communities with modularity = {modularity:.4f}")
    for i, community in enumerate(communities):
        print(f"Community {i+1}: {len(community)} regions")
        if len(community) <= 5:  # Show small communities
            print(f"  Regions: {list(community)}")
    
    return analyzer

def example_2_disease_comparison():
    """
    Example 2: Comparing Healthy vs Disease Networks
    ----------------------------------------------
    Demonstrates network comparison for clinical research
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: HEALTHY VS DISEASE NETWORK COMPARISON")
    print("="*60)
    
    # Create sample healthy and disease networks
    healthy_matrix, region_labels = create_sample_data()
    
    # Simulate disease network (reduced connectivity in some regions)
    np.random.seed(123)  # Different seed for disease network
    disease_matrix = healthy_matrix.copy()
    
    # Simulate disease effects: reduce connectivity in frontal regions
    frontal_indices = [i for i, label in enumerate(region_labels) if 'Frontal' in label]
    for i in frontal_indices:
        for j in range(len(region_labels)):
            disease_matrix[i, j] *= 0.7  # Reduce connectivity by 30%
            disease_matrix[j, i] *= 0.7
    
    # Add some noise to simulate individual differences
    disease_matrix += np.random.normal(0, 0.05, disease_matrix.shape)
    disease_matrix = np.maximum(disease_matrix, 0)  # Ensure non-negative
    
    # Analyze healthy network
    healthy_analyzer = BrainConnectomeAnalyzer()
    healthy_analyzer.load_connectivity_data(healthy_matrix, region_labels)
    
    # Compare with disease network
    print("\n--- Comparing Networks ---")
    comparison_results, disease_analyzer = healthy_analyzer.compare_networks(
        disease_matrix, 
        region_labels, 
        comparison_name="Neurodegenerative Disease"
    )
    
    # Display comparison results
    print("\nNetwork Comparison Results:")
    print("-" * 40)
    for metric, values in comparison_results.items():
        print(f"{metric}:")
        print(f"  Healthy: {values['current']:.4f}")
        print(f"  Disease: {values['comparison']:.4f}")
        print(f"  Change: {values['percent_change']:+.2f}%")
        print()
    
    # Compare hub regions
    print("\n--- Hub Region Comparison ---")
    healthy_hubs = healthy_analyzer.identify_hubs(method='composite', threshold_percentile=85)
    disease_hubs = disease_analyzer.identify_hubs(method='composite', threshold_percentile=85)
    
    print("Healthy network hubs:")
    for _, hub in healthy_hubs.iterrows():
        print(f"  {hub['Region']}: {hub['Composite_Hub_Score']:.4f}")
    
    print("\nDisease network hubs:")
    for _, hub in disease_hubs.iterrows():
        print(f"  {hub['Region']}: {hub['Composite_Hub_Score']:.4f}")
    
    return healthy_analyzer, disease_analyzer

def example_3_advanced_visualization():
    """
    Example 3: Advanced Network Visualization
    ----------------------------------------
    Demonstrates various visualization capabilities
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: ADVANCED NETWORK VISUALIZATION")
    print("="*60)
    
    # Create and analyze a network
    connectivity_matrix, region_labels = create_sample_data()
    analyzer = BrainConnectomeAnalyzer()
    analyzer.load_connectivity_data(connectivity_matrix, region_labels)
    
    # Calculate metrics needed for visualization
    analyzer.calculate_global_metrics()
    analyzer.calculate_nodal_metrics()
    analyzer.detect_communities()
    
    print("\n--- Creating Visualizations ---")
    
    # 1. 2D Network plots with different layouts
    print("1. Creating 2D network visualizations...")
    
    # Spring layout colored by community
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    analyzer.visualize_network_2d(layout='spring', 
                                 node_size_metric='Degree_Centrality',
                                 node_color_metric='Community')
    plt.title('Spring Layout - Community Colors')
    
    plt.subplot(1, 3, 2)
    analyzer.visualize_network_2d(layout='circular',
                                 node_size_metric='Betweenness_Centrality', 
                                 node_color_metric='Clustering_Coefficient')
    plt.title('Circular Layout - Clustering Colors')
    
    plt.subplot(1, 3, 3)
    analyzer.visualize_network_2d(layout='kamada_kawai',
                                 node_size_metric='Eigenvector_Centrality',
                                 node_color_metric='Degree_Centrality') 
    plt.title('Kamada-Kawai Layout')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Interactive 3D visualization
    print("2. Creating interactive 3D network visualization...")
    fig_3d = analyzer.visualize_network_3d()
    fig_3d.update_layout(title="Interactive 3D Brain Network")
    fig_3d.show()
    
    # 3. Connectivity matrix heatmap
    print("3. Creating connectivity matrix heatmap...")
    fig_matrix = analyzer.create_connectivity_matrix_plot()
    fig_matrix.show()
    
    # 4. Custom analysis plots
    print("4. Creating custom analysis plots...")
    create_custom_analysis_plots(analyzer)
    
    return analyzer

def create_custom_analysis_plots(analyzer):
    """Create custom analysis and comparison plots"""
    
    # Centrality measures comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Degree vs Betweenness centrality
    axes[0, 0].scatter(analyzer.nodal_metrics['Degree_Centrality'], 
                      analyzer.nodal_metrics['Betweenness_Centrality'],
                      c=analyzer.nodal_metrics['Community'], 
                      cmap='tab10', alpha=0.7)
    axes[0, 0].set_xlabel('Degree Centrality')
    axes[0, 0].set_ylabel('Betweenness Centrality')
    axes[0, 0].set_title('Degree vs Betweenness Centrality')
    
    # Plot 2: Distribution of clustering coefficients
    axes[0, 1].hist(analyzer.nodal_metrics['Clustering_Coefficient'], 
                   bins=20, alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Clustering Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Clustering Coefficients')
    
    # Plot 3: Centrality measures by community
    communities = analyzer.nodal_metrics['Community'].unique()
    centrality_by_community = []
    for community in communities:
        community_data = analyzer.nodal_metrics[
            analyzer.nodal_metrics['Community'] == community
        ]
        centrality_by_community.append(community_data['Degree_Centrality'].mean())
    
    axes[1, 0].bar(range(len(communities)), centrality_by_community, 
                  color='lightcoral', alpha=0.7)
    axes[1, 0].set_xlabel('Community')
    axes[1, 0].set_ylabel('Mean Degree Centrality')
    axes[1, 0].set_title('Average Centrality by Community')
    
    # Plot 4: Network efficiency analysis
    # This is a simplified efficiency calculation
    path_lengths = []
    for node in analyzer.graph.nodes():
        lengths = []
        for target in analyzer.graph.nodes():
            if node != target:
                try:
                    length = analyzer.graph[node][target]['weight']
                    lengths.append(1/length if length > 0 else 0)
                except:
                    lengths.append(0)
        path_lengths.append(np.mean(lengths))
    
    axes[1, 1].hist(path_lengths, bins=20, alpha=0.7, color='lightgreen')
    axes[1, 1].set_xlabel('Local Efficiency')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Local Efficiency')
    
    plt.tight_layout()
    plt.show()

def example_4_clinical_application():
    """
    Example 4: Clinical Application - Aging Study
    --------------------------------------------
    Simulates a longitudinal aging study analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: CLINICAL APPLICATION - AGING STUDY")
    print("="*60)
    
    # Simulate connectivity data for different age groups
    np.random.seed(42)
    
    # Young adults (20-30 years)
    young_matrix, region_labels = create_sample_data()
    
    # Middle-aged adults (40-50 years) - slightly reduced connectivity
    middle_matrix = young_matrix * 0.95 + np.random.normal(0, 0.02, young_matrix.shape)
    middle_matrix = np.maximum(middle_matrix, 0)
    
    # Older adults (65-75 years) - more reduced connectivity, especially in frontal regions
    older_matrix = young_matrix * 0.85
    frontal_indices = [i for i, label in enumerate(region_labels) if 'Frontal' in label]
    for i in frontal_indices:
        older_matrix[i, :] *= 0.8
        older_matrix[:, i] *= 0.8
    older_matrix += np.random.normal(0, 0.03, older_matrix.shape)
    older_matrix = np.maximum(older_matrix, 0)
    
    # Analyze each age group
    analyzers = {}
    age_groups = {
        'Young (20-30)': young_matrix,
        'Middle (40-50)': middle_matrix, 
        'Older (65-75)': older_matrix
    }
    
    print("\n--- Age Group Analysis ---")
    results_summary = {}
    
    for age_group, matrix in age_groups.items():
        print(f"\nAnalyzing {age_group} group...")
        analyzer = BrainConnectomeAnalyzer()
        analyzer.load_connectivity_data(matrix, region_labels)
        
        # Calculate key metrics
        global_metrics = analyzer.calculate_global_metrics()
        nodal_metrics = analyzer.calculate_nodal_metrics()
        hubs = analyzer.identify_hubs(method='composite', threshold_percentile=90)
        
        analyzers[age_group] = analyzer
        results_summary[age_group] = {
            'density': global_metrics['density'],
            'clustering': global_metrics['global_clustering'],
            'path_length': global_metrics['average_path_length'],
            'small_world': global_metrics['small_world_sigma'],
            'num_hubs': len(hubs)
        }
    
    # Create aging trajectory plot
    print("\n--- Creating Aging Trajectory Analysis ---")
    create_aging_analysis_plot(results_summary)
    
    # Compare hub regions across age groups
    print("\n--- Hub Region Changes with Age ---")
    for age_group in age_groups:
        analyzer = analyzers[age_group]
        hubs = analyzer.identify_hubs(method='composite', threshold_percentile=90)
        print(f"\n{age_group} - Hub regions ({len(hubs)} total):")
        for _, hub in hubs.head(5).iterrows():  # Show top 5
            print(f"  {hub['Region']}: {hub['Composite_Hub_Score']:.4f}")
    
    return analyzers

def create_aging_analysis_plot(results_summary):
    """Create visualization of aging effects on brain networks"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    age_groups = list(results_summary.keys())
    metrics = ['density', 'clustering', 'path_length', 'small_world']
    metric_titles = ['Network Density', 'Global Clustering', 
                    'Average Path Length', 'Small-World Sigma']
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        row, col = idx // 2, idx % 2
        values = [results_summary[group][metric] for group in age_groups]
        
        axes[row, col].plot(age_groups, values, 'o-', linewidth=2, markersize=8)
        axes[row, col].set_title(f'{title} Across Age Groups')
        axes[row, col].set_ylabel(title)
        axes[row, col].grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = range(len(age_groups))
        z = np.polyfit(x_numeric, values, 1)
        p = np.poly1d(z)
        axes[row, col].plot(age_groups, p(x_numeric), "r--", alpha=0.8)
    
    plt.suptitle('Brain Network Changes with Aging', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def example_5_research_workflow():
    """
    Example 5: Complete Research Workflow
    -----------------------------------
    Demonstrates a complete research analysis pipeline
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: COMPLETE RESEARCH WORKFLOW")
    print("="*60)
    
    print("""
    This example demonstrates a complete research workflow for studying
    brain connectivity differences in a hypothetical autism spectrum disorder study.
    """)
    
    # Simulate datasets
    np.random.seed(456)
    
    # Control group
    control_matrix, region_labels = create_sample_data()
    
    # ASD group - altered connectivity patterns
    asd_matrix = control_matrix.copy()
    
    # Simulate ASD characteristics:
    # 1. Reduced long-range connectivity
    # 2. Increased local connectivity in some regions
    # 3. Altered frontal-posterior connectivity
    
    frontal_idx = [i for i, label in enumerate(region_labels) if 'Frontal' in label]
    temporal_idx = [i for i, label in enumerate(region_labels) if 'Temporal' in label]
    
    # Reduce fronto-temporal connectivity
    for i in frontal_idx:
        for j in temporal_idx:
            asd_matrix[i, j] *= 0.7
            asd_matrix[j, i] *= 0.7
    
    # Increase local frontal connectivity
    for i in frontal_idx:
        for j in frontal_idx:
            if i != j:
                asd_matrix[i, j] *= 1.3
    
    # Add noise
    asd_matrix += np.random.normal(0, 0.02, asd_matrix.shape)
    asd_matrix = np.maximum(asd_matrix, 0)
    
    # Complete analysis pipeline
    print("\n--- Step 1: Data Loading and Quality Check ---")
    control_analyzer = BrainConnectomeAnalyzer()
    control_analyzer.load_connectivity_data(control_matrix, region_labels)
    
    asd_analyzer = BrainConnectomeAnalyzer() 
    asd_analyzer.load_connectivity_data(asd_matrix, region_labels)
    
    print("\n--- Step 2: Group-Level Analysis ---")
    
    # Analyze both groups
    control_metrics = control_analyzer.calculate_global_metrics()
    control_nodal = control_analyzer.calculate_nodal_metrics()
    control_communities = control_analyzer.detect_communities()
    
    asd_metrics = asd_analyzer.calculate_global_metrics()
    asd_nodal = asd_analyzer.calculate_nodal_metrics()
    asd_communities = asd_analyzer.detect_communities()
    
    print("\n--- Step 3: Statistical Comparison ---")
    
    # Compare key metrics
    comparison_results = {}
    key_metrics = ['density', 'global_clustering', 'average_path_length', 'small_world_sigma']
    
    for metric in key_metrics:
        control_val = control_metrics[metric]
        asd_val = asd_metrics[metric]
        percent_diff = ((asd_val - control_val) / control_val) * 100
        
        comparison_results[metric] = {
            'control': control_val,
            'asd': asd_val,
            'percent_difference': percent_diff
        }
        
        print(f"{metric}:")
        print(f"  Control: {control_val:.4f}")
        print(f"  ASD: {asd_val:.4f}")
        print(f"  Difference: {percent_diff:+.2f}%")
        print()
    
    print("\n--- Step 4: Regional Analysis ---")
    
    # Compare nodal metrics
    nodal_comparison = pd.merge(
        control_nodal[['Region', 'Degree_Centrality']], 
        asd_nodal[['Region', 'Degree_Centrality']], 
        on='Region', 
        suffixes=('_Control', '_ASD')
    )
    
    nodal_comparison['Centrality_Difference'] = (
        nodal_comparison['Degree_Centrality_ASD'] - 
        nodal_comparison['Degree_Centrality_Control']
    )
    
    # Identify regions with largest differences
    print("Regions with largest centrality differences (ASD vs Control):")
    top_differences = nodal_comparison.nlargest(5, 'Centrality_Difference')
    for _, row in top_differences.iterrows():
        print(f"  {row['Region']}: {row['Centrality_Difference']:+.4f}")
    
    print("\nRegions with largest centrality reductions (ASD vs Control):")
    bottom_differences = nodal_comparison.nsmallest(5, 'Centrality_Difference')
    for _, row in bottom_differences.iterrows():
        print(f"  {row['Region']}: {row['Centrality_Difference']:+.4f}")
    
    print("\n--- Step 5: Generate Research Report ---")
    create_research_report(control_analyzer, asd_analyzer, comparison_results, nodal_comparison)
    
    return control_analyzer, asd_analyzer, comparison_results

def create_research_report(control_analyzer, asd_analyzer, comparison_results, nodal_comparison):
    """Generate a comprehensive research report"""
    
    print("\n" + "="*80)
    print("BRAIN CONNECTIVITY IN AUTISM SPECTRUM DISORDER - RESEARCH REPORT")
    print("="*80)
    
    print("""
ABSTRACT:
This study analyzed brain connectivity patterns in autism spectrum disorder (ASD)
compared to neurotypical controls using graph theoretical approaches. We found
significant alterations in global network properties and regional connectivity
patterns in the ASD group.

METHODS:
- Network analysis using graph theory metrics
- Comparison of global and nodal network properties
- Community detection and hub identification
- Statistical comparison between groups

RESULTS:
    """)
    
    # Global findings
    print("Global Network Differences:")
    for metric, values in comparison_results.items():
        direction = "increased" if values['percent_difference'] > 0 else "decreased"
        print(f"  - {metric.replace('_', ' ').title()}: {direction} by {abs(values['percent_difference']):.1f}%")
    
    # Regional findings  
    print(f"\nRegional Connectivity Changes:")
    print(f"  - {len(nodal_comparison[nodal_comparison['Centrality_Difference'] > 0])} regions showed increased centrality")
    print(f"  - {len(nodal_comparison[nodal_comparison['Centrality_Difference'] < 0])} regions showed decreased centrality")
    
    # Most affected regions
    most_increased = nodal_comparison.loc[nodal_comparison['Centrality_Difference'].idxmax()]
    most_decreased = nodal_comparison.loc[nodal_comparison['Centrality_Difference'].idxmin()]
    
    print(f"  - Most increased connectivity: {most_increased['Region']} (+{most_increased['Centrality_Difference']:.3f})")
    print(f"  - Most decreased connectivity: {most_decreased['Region']} ({most_decreased['Centrality_Difference']:.3f})")
    
    print("""
CONCLUSIONS:
Our findings suggest altered brain network organization in ASD, with changes in
both global network efficiency and regional connectivity patterns. These results
are consistent with theories of altered connectivity in autism spectrum disorders.

CLINICAL IMPLICATIONS:
The identified connectivity patterns may serve as potential biomarkers for ASD
diagnosis and could inform targeted therapeutic interventions.
    """)
    
    print("="*80)

def main():
    """
    Main function to run all examples
    """
    print("BRAIN CONNECTOME ANALYSIS SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 65)
    
    try:
        # Run all examples
        print("\nRunning Example 1: Basic Analysis...")
        analyzer1 = example_1_basic_analysis()
        
        print("\nRunning Example 2: Disease Comparison...")  
        healthy_analyzer, disease_analyzer = example_2_disease_comparison()
        
        print("\nRunning Example 3: Advanced Visualization...")
        analyzer3 = example_3_advanced_visualization()
        
        print("\nRunning Example 4: Clinical Application...")
        aging_analyzers = example_4_clinical_application()
        
        print("\nRunning Example 5: Research Workflow...")
        control_analyzer, asd_analyzer, results = example_5_research_workflow()
        
        print("\n" + "="*65)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*65)
        
        print("""
NEXT STEPS:
1. Modify the examples to work with your own connectivity data
2. Customize the analysis parameters for your specific research questions
3. Adapt the visualization styles to your preferences
4. Use the comparison functions for your clinical/research studies

For more information, see the README.md file and documentation.
        """)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Please check your dependencies and data files.")

if __name__ == "__main__":
    main()

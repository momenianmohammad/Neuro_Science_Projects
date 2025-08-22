# Brain Connectome Analysis System
# A comprehensive toolkit for analyzing neural connections and brain networks

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BrainConnectomeAnalyzer:
    """
    A comprehensive system for brain connectome analysis including:
    - Network topology analysis
    - Hub identification
    - Community detection
    - Disease comparison
    - 3D visualization
    """
    
    def __init__(self):
        self.connectivity_matrix = None
        self.graph = None
        self.node_labels = None
        self.metrics = {}
        self.communities = None
        
    def load_connectivity_data(self, connectivity_matrix, node_labels=None):
        """
        Load connectivity matrix and create brain network graph
        
        Args:
            connectivity_matrix (numpy.ndarray): N x N connectivity matrix
            node_labels (list): List of brain region names
        """
        self.connectivity_matrix = np.array(connectivity_matrix)
        n_nodes = self.connectivity_matrix.shape[0]
        
        if node_labels is None:
            self.node_labels = [f"Region_{i+1}" for i in range(n_nodes)]
        else:
            self.node_labels = node_labels
            
        # Create NetworkX graph
        self.graph = nx.from_numpy_array(self.connectivity_matrix)
        
        # Add node labels
        label_mapping = {i: self.node_labels[i] for i in range(n_nodes)}
        self.graph = nx.relabel_nodes(self.graph, label_mapping)
        
        print(f"Loaded brain network with {n_nodes} regions")
        print(f"Network density: {nx.density(self.graph):.4f}")
        
    def calculate_global_metrics(self):
        """Calculate global network metrics"""
        print("Calculating global network metrics...")
        
        # Basic metrics
        self.metrics['num_nodes'] = self.graph.number_of_nodes()
        self.metrics['num_edges'] = self.graph.number_of_edges()
        self.metrics['density'] = nx.density(self.graph)
        
        # Path-based metrics
        if nx.is_connected(self.graph):
            self.metrics['average_path_length'] = nx.average_shortest_path_length(self.graph)
            self.metrics['diameter'] = nx.diameter(self.graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            self.metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
            self.metrics['diameter'] = nx.diameter(subgraph)
            
        # Clustering and efficiency
        self.metrics['global_clustering'] = nx.average_clustering(self.graph)
        self.metrics['transitivity'] = nx.transitivity(self.graph)
        
        # Small-world properties
        # Compare to random network
        random_graph = nx.erdos_renyi_graph(
            self.metrics['num_nodes'], 
            self.metrics['density']
        )
        random_clustering = nx.average_clustering(random_graph)
        random_path_length = nx.average_shortest_path_length(random_graph)
        
        self.metrics['small_world_sigma'] = (
            (self.metrics['global_clustering'] / random_clustering) /
            (self.metrics['average_path_length'] / random_path_length)
        )
        
        return self.metrics
    
    def calculate_nodal_metrics(self):
        """Calculate node-level network metrics"""
        print("Calculating nodal network metrics...")
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
        
        # Clustering coefficient
        clustering_coeff = nx.clustering(self.graph)
        
        # Create DataFrame
        nodal_metrics_df = pd.DataFrame({
            'Region': list(self.graph.nodes()),
            'Degree_Centrality': [degree_centrality[node] for node in self.graph.nodes()],
            'Betweenness_Centrality': [betweenness_centrality[node] for node in self.graph.nodes()],
            'Closeness_Centrality': [closeness_centrality[node] for node in self.graph.nodes()],
            'Eigenvector_Centrality': [eigenvector_centrality[node] for node in self.graph.nodes()],
            'Clustering_Coefficient': [clustering_coeff[node] for node in self.graph.nodes()]
        })
        
        self.nodal_metrics = nodal_metrics_df
        return nodal_metrics_df
    
    def identify_hubs(self, method='degree', threshold_percentile=90):
        """
        Identify hub nodes in the network
        
        Args:
            method (str): Method to identify hubs ('degree', 'betweenness', 'composite')
            threshold_percentile (float): Percentile threshold for hub identification
        """
        if not hasattr(self, 'nodal_metrics'):
            self.calculate_nodal_metrics()
            
        if method == 'degree':
            metric_col = 'Degree_Centrality'
        elif method == 'betweenness':
            metric_col = 'Betweenness_Centrality'
        elif method == 'composite':
            # Create composite hub score
            scaler = StandardScaler()
            centrality_scores = scaler.fit_transform(
                self.nodal_metrics[['Degree_Centrality', 'Betweenness_Centrality', 
                                 'Closeness_Centrality', 'Eigenvector_Centrality']]
            )
            self.nodal_metrics['Composite_Hub_Score'] = np.mean(centrality_scores, axis=1)
            metric_col = 'Composite_Hub_Score'
        
        threshold = np.percentile(self.nodal_metrics[metric_col], threshold_percentile)
        hubs = self.nodal_metrics[self.nodal_metrics[metric_col] >= threshold]
        
        print(f"Identified {len(hubs)} hub regions using {method} method:")
        for _, hub in hubs.iterrows():
            print(f"  - {hub['Region']}: {hub[metric_col]:.4f}")
            
        return hubs
    
    def detect_communities(self, method='modularity'):
        """
        Detect communities in the brain network
        
        Args:
            method (str): Community detection method ('modularity', 'leiden')
        """
        print(f"Detecting communities using {method} method...")
        
        if method == 'modularity':
            # Greedy modularity optimization
            communities = nx.community.greedy_modularity_communities(self.graph)
        elif method == 'leiden':
            # Note: This would require python-igraph and leidenalg packages
            # For now, we'll use Louvain method which is similar
            communities = nx.community.louvain_communities(self.graph)
            
        # Convert to node-community mapping
        community_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_mapping[node] = i
                
        # Add community information to nodal metrics
        if hasattr(self, 'nodal_metrics'):
            self.nodal_metrics['Community'] = [
                community_mapping[region] for region in self.nodal_metrics['Region']
            ]
        
        self.communities = communities
        self.community_mapping = community_mapping
        
        # Calculate modularity
        modularity = nx.community.modularity(self.graph, communities)
        print(f"Detected {len(communities)} communities with modularity: {modularity:.4f}")
        
        return communities, modularity
    
    def compare_networks(self, other_connectivity_matrix, other_labels=None, 
                        comparison_name="Comparison"):
        """
        Compare current network with another network (e.g., disease vs. healthy)
        
        Args:
            other_connectivity_matrix (numpy.ndarray): Other connectivity matrix
            other_labels (list): Labels for other network
            comparison_name (str): Name for the comparison
        """
        print(f"Comparing networks: Current vs. {comparison_name}")
        
        # Create comparison analyzer
        comparison_analyzer = BrainConnectomeAnalyzer()
        comparison_analyzer.load_connectivity_data(other_connectivity_matrix, other_labels)
        
        # Calculate metrics for both networks
        current_metrics = self.calculate_global_metrics()
        comparison_metrics = comparison_analyzer.calculate_global_metrics()
        
        # Compare global metrics
        comparison_results = {}
        for metric in current_metrics:
            if isinstance(current_metrics[metric], (int, float)):
                difference = current_metrics[metric] - comparison_metrics[metric]
                percent_change = (difference / comparison_metrics[metric]) * 100
                comparison_results[metric] = {
                    'current': current_metrics[metric],
                    'comparison': comparison_metrics[metric],
                    'difference': difference,
                    'percent_change': percent_change
                }
        
        return comparison_results, comparison_analyzer
    
    def visualize_network_2d(self, layout='spring', node_size_metric='Degree_Centrality',
                            node_color_metric='Community', figsize=(15, 10)):
        """
        Create 2D network visualization
        
        Args:
            layout (str): Network layout ('spring', 'circular', 'kamada_kawai')
            node_size_metric (str): Metric to determine node sizes
            node_color_metric (str): Metric to determine node colors
        """
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Node sizes
        if hasattr(self, 'nodal_metrics') and node_size_metric in self.nodal_metrics.columns:
            node_sizes = self.nodal_metrics.set_index('Region')[node_size_metric]
            node_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min())
            node_sizes = 100 + node_sizes * 500  # Scale to reasonable size range
        else:
            node_sizes = 200
        
        # Node colors
        if hasattr(self, 'nodal_metrics') and node_color_metric in self.nodal_metrics.columns:
            node_colors = self.nodal_metrics.set_index('Region')[node_color_metric]
        else:
            node_colors = 'lightblue'
        
        # Draw network
        nx.draw(self.graph, pos, 
                node_size=[node_sizes[node] if isinstance(node_sizes, pd.Series) else node_sizes for node in self.graph.nodes()],
                node_color=[node_colors[node] if isinstance(node_colors, pd.Series) else node_colors for node in self.graph.nodes()],
                with_labels=False, 
                edge_color='gray', 
                alpha=0.7,
                cmap=plt.cm.viridis)
        
        plt.title(f"Brain Network Visualization\nLayout: {layout}, Size: {node_size_metric}, Color: {node_color_metric}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_network_3d(self, node_positions=None):
        """
        Create interactive 3D network visualization using Plotly
        
        Args:
            node_positions (dict): 3D positions for nodes (if None, will generate)
        """
        if node_positions is None:
            # Generate 3D positions (simplified brain-like arrangement)
            n_nodes = len(self.graph.nodes())
            # Create a roughly brain-shaped 3D layout
            angles = np.linspace(0, 2*np.pi, n_nodes)
            x = np.cos(angles) + 0.3 * np.random.randn(n_nodes)
            y = np.sin(angles) + 0.3 * np.random.randn(n_nodes)
            z = np.sin(2*angles) * 0.5 + 0.2 * np.random.randn(n_nodes)
            
            node_positions = {
                list(self.graph.nodes())[i]: (x[i], y[i], z[i]) 
                for i in range(n_nodes)
            }
        
        # Extract coordinates
        node_trace_x = [node_positions[node][0] for node in self.graph.nodes()]
        node_trace_y = [node_positions[node][1] for node in self.graph.nodes()]
        node_trace_z = [node_positions[node][2] for node in self.graph.nodes()]
        
        # Create edge traces
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.graph.edges():
            x0, y0, z0 = node_positions[edge[0]]
            x1, y1, z1 = node_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create traces
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.3)', width=1),
            hoverinfo='none',
            name='Connections'
        )
        
        # Node colors based on degree
        node_degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        
        node_trace = go.Scatter3d(
            x=node_trace_x, y=node_trace_y, z=node_trace_z,
            mode='markers',
            marker=dict(
                size=8,
                color=node_degrees,
                colorscale='Viridis',
                colorbar=dict(title="Node Degree"),
                line=dict(width=0.5, color='white')
            ),
            text=list(self.graph.nodes()),
            hovertemplate='<b>%{text}</b><br>Degree: %{marker.color}<extra></extra>',
            name='Brain Regions'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='3D Brain Connectome Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def create_connectivity_matrix_plot(self):
        """Create interactive heatmap of connectivity matrix"""
        fig = go.Figure(data=go.Heatmap(
            z=self.connectivity_matrix,
            x=self.node_labels,
            y=self.node_labels,
            colorscale='RdBu_r',
            zmid=0
        ))
        
        fig.update_layout(
            title='Brain Connectivity Matrix',
            xaxis_title='Brain Regions',
            yaxis_title='Brain Regions',
            width=800,
            height=800
        )
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("="*60)
        print("BRAIN CONNECTOME ANALYSIS REPORT")
        print("="*60)
        
        # Global metrics
        if self.metrics:
            print(f"\nGLOBAL NETWORK METRICS:")
            print(f"  Nodes: {self.metrics['num_nodes']}")
            print(f"  Edges: {self.metrics['num_edges']}")
            print(f"  Density: {self.metrics['density']:.4f}")
            print(f"  Average Path Length: {self.metrics['average_path_length']:.4f}")
            print(f"  Global Clustering: {self.metrics['global_clustering']:.4f}")
            print(f"  Small-World Sigma: {self.metrics['small_world_sigma']:.4f}")
        
        # Nodal metrics summary
        if hasattr(self, 'nodal_metrics'):
            print(f"\nNODAL METRICS SUMMARY:")
            print(self.nodal_metrics.describe())
        
        # Communities
        if self.communities:
            print(f"\nCOMMUNITY STRUCTURE:")
            print(f"  Number of communities: {len(self.communities)}")
            for i, community in enumerate(self.communities):
                print(f"  Community {i+1}: {len(community)} regions")
        
        print("="*60)


def create_sample_data():
    """Create sample connectivity data for demonstration"""
    np.random.seed(42)
    n_regions = 50
    
    # Create a structured connectivity matrix with communities
    connectivity_matrix = np.random.rand(n_regions, n_regions) * 0.3
    
    # Add community structure
    community_size = n_regions // 3
    for i in range(0, n_regions, community_size):
        end_idx = min(i + community_size, n_regions)
        # Strengthen intra-community connections
        connectivity_matrix[i:end_idx, i:end_idx] += np.random.rand(end_idx-i, end_idx-i) * 0.5
    
    # Make symmetric
    connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
    np.fill_diagonal(connectivity_matrix, 0)  # No self-connections
    
    # Create region labels
    regions = [
        f"Frontal_{i}" if i < 15 else 
        f"Parietal_{i-15}" if i < 30 else 
        f"Temporal_{i-30}" for i in range(n_regions)
    ]
    
    return connectivity_matrix, regions


def main():
    """Main demonstration function"""
    print("Brain Connectome Analysis System - Demo")
    print("======================================")
    
    # Create sample data
    connectivity_matrix, region_labels = create_sample_data()
    
    # Initialize analyzer
    analyzer = BrainConnectomeAnalyzer()
    analyzer.load_connectivity_data(connectivity_matrix, region_labels)
    
    # Perform comprehensive analysis
    print("\n1. Calculating global metrics...")
    global_metrics = analyzer.calculate_global_metrics()
    
    print("\n2. Calculating nodal metrics...")
    nodal_metrics = analyzer.calculate_nodal_metrics()
    
    print("\n3. Identifying hub regions...")
    hubs = analyzer.identify_hubs(method='composite', threshold_percentile=85)
    
    print("\n4. Detecting communities...")
    communities, modularity = analyzer.detect_communities()
    
    print("\n5. Generating visualizations...")
    # 2D network plot
    analyzer.visualize_network_2d(layout='spring')
    
    # 3D interactive plot
    fig_3d = analyzer.visualize_network_3d()
    fig_3d.show()
    
    # Connectivity matrix heatmap
    fig_matrix = analyzer.create_connectivity_matrix_plot()
    fig_matrix.show()
    
    print("\n6. Generating comprehensive report...")
    analyzer.generate_report()
    
    print("\nAnalysis complete! Check the generated plots and report.")
    
    return analyzer


if __name__ == "__main__":
    # Run demonstration
    analyzer = main()

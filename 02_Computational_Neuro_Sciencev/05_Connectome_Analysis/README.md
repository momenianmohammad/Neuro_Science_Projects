# Brain Connectome Analysis System

A comprehensive Python toolkit for analyzing neural connections and brain networks in computational neuroscience research. This system provides advanced network analysis capabilities for understanding brain connectivity patterns, identifying critical hubs, and comparing different populations.

## 🧠 Features

### Core Functionality
- **Global Network Analysis**: Calculate comprehensive network metrics including clustering coefficients, path lengths, and small-world properties
- **Nodal Metrics**: Compute centrality measures (degree, betweenness, closeness, eigenvector) for individual brain regions
- **Hub Identification**: Automatically identify critical brain regions using multiple centrality measures
- **Community Detection**: Discover modular organization in brain networks using advanced algorithms
- **Network Comparison**: Compare connectivity patterns between different groups (e.g., healthy vs. disease)
- **Interactive Visualizations**: Generate 2D and 3D network visualizations with customizable layouts

### Advanced Analytics
- Small-world network analysis
- Modularity optimization for community detection
- Statistical comparison between networks
- Composite hub scoring systems
- Multi-modal visualization support

## 🚀 Installation

### Prerequisites
```bash
pip install numpy pandas networkx matplotlib seaborn scipy scikit-learn plotly
```

### Optional Dependencies
For enhanced functionality:
```bash
pip install nibabel  # For neuroimaging data
pip install igraph leidenalg  # For advanced community detection
```

## 📊 Usage

### Basic Usage

```python
from brain_connectome_analyzer import BrainConnectomeAnalyzer
import numpy as np

# Load your connectivity matrix (N x N symmetric matrix)
connectivity_matrix = np.load('your_connectivity_matrix.npy')
region_labels = ['Region_1', 'Region_2', ...]  # Optional

# Initialize analyzer
analyzer = BrainConnectomeAnalyzer()
analyzer.load_connectivity_data(connectivity_matrix, region_labels)

# Perform comprehensive analysis
global_metrics = analyzer.calculate_global_metrics()
nodal_metrics = analyzer.calculate_nodal_metrics()
hubs = analyzer.identify_hubs(method='composite')
communities, modularity = analyzer.detect_communities()

# Generate visualizations
analyzer.visualize_network_2d(layout='spring')
fig_3d = analyzer.visualize_network_3d()
fig_3d.show()

# Create comprehensive report
analyzer.generate_report()
```

### Advanced Analysis

```python
# Compare networks (e.g., healthy vs. disease)
comparison_matrix = np.load('disease_connectivity.npy')
comparison_results, comparison_analyzer = analyzer.compare_networks(
    comparison_matrix, 
    comparison_name="Alzheimer's Disease"
)

# Custom hub identification
hubs_degree = analyzer.identify_hubs(method='degree', threshold_percentile=90)
hubs_betweenness = analyzer.identify_hubs(method='betweenness', threshold_percentile=85)

# Interactive connectivity matrix visualization
fig_matrix = analyzer.create_connectivity_matrix_plot()
fig_matrix.show()
```

## 📈 Key Metrics Calculated

### Global Network Metrics
- **Density**: Overall connectivity strength
- **Average Path Length**: Efficiency of information transfer
- **Global Clustering**: Local connectivity patterns
- **Small-World Sigma**: Balance between segregation and integration
- **Modularity**: Community structure strength

### Nodal Metrics
- **Degree Centrality**: Number of connections
- **Betweenness Centrality**: Importance for information flow
- **Closeness Centrality**: Access to other nodes
- **Eigenvector Centrality**: Influence based on neighbors
- **Clustering Coefficient**: Local connectivity density

## 🎯 Applications

### Clinical Research
- **Alzheimer's Disease**: Identify connectivity disruptions and affected hubs
- **Autism Spectrum Disorders**: Analyze atypical network organization
- **Schizophrenia**: Study altered brain connectivity patterns
- **Depression**: Examine changes in limbic and prefrontal networks

### Cognitive Neuroscience
- **Aging Studies**: Track connectivity changes across lifespan
- **Learning and Memory**: Understand network plasticity
- **Attention Networks**: Analyze task-related connectivity
- **Language Processing**: Study language network organization

### Computational Modeling
- **Network Simulation**: Generate realistic brain network models
- **Intervention Planning**: Identify optimal stimulation targets
- **Drug Development**: Predict therapeutic effects on brain networks

## 📁 File Structure

```
brain-connectome-analysis/
│
├── brain_connectome_analyzer.py    # Main analysis class
├── requirements.txt                # Python dependencies
├── README.md                      # Documentation
├── examples/
│   ├── basic_analysis.py          # Simple usage example
│   ├── disease_comparison.py      # Network comparison demo
│   └── visualization_demo.py      # Visualization examples
├── data/
│   ├── sample_connectivity.npy    # Example connectivity matrix
│   └── region_labels.txt          # Sample region names
└── tests/
    ├── test_analyzer.py           # Unit tests
    └── test_visualizations.py     # Visualization tests
```

## 🔬 Technical Details

### Connectivity Matrix Format
- **Input**: N×N symmetric matrix where N is the number of brain regions
- **Values**: Connection strengths (typically 0-1 or correlation coefficients)
- **Diagonal**: Should be zeros (no self-connections)

### Network Construction
- Undirected weighted graphs using NetworkX
- Optional thresholding for sparse networks
- Support for negative correlations

### Community Detection Algorithms
- **Modularity Optimization**: Greedy algorithm for fast community detection
- **Louvain Method**: High-quality community detection
- **Leiden Algorithm**: Improved resolution and speed (optional)

## 📊 Sample Output

### Global Metrics Report
```
BRAIN CONNECTOME ANALYSIS REPORT
====================================
GLOBAL NETWORK METRICS:
  Nodes: 84
  Edges: 2156
  Density: 0.6201
  Average Path Length: 1.4832
  Global Clustering: 0.7453
  Small-World Sigma: 1.8234

COMMUNITY STRUCTURE:
  Number of communities: 6
  Community 1: 18 regions (Frontal network)
  Community 2: 14 regions (Parietal network)
  ...
```

### Hub Identification
```
Identified 8 hub regions using composite method:
  - Left Superior Frontal: 0.8932
  - Right Angular Gyrus: 0.8547
  - Posterior Cingulate: 0.8234
  ...
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/yourusername/brain-connectome-analysis.git
cd brain-connectome-analysis
pip install -r requirements.txt
python -m pytest tests/
```

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{brain_connectome_analyzer,
  title={Brain Connectome Analysis System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/brain-connectome-analysis},
  version={1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check our [Wiki](../../wiki) for detailed guides
- **Issues**: Report bugs on our [Issues page](../../issues)
- **Discussions**: Join conversations in [Discussions](../../discussions)
- **Email**: contact@yourresearchgroup.com

## 🔗 Related Projects

- [NetworkX](https://networkx.org/): Network analysis library
- [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/): MATLAB toolbox for brain networks
- [Nilearn](https://nilearn.github.io/): Machine learning for neuroimaging

## 🏆 Acknowledgments

- NetworkX development team for the excellent graph analysis library
- Plotly team for interactive visualization capabilities
- The brain connectivity research community for theoretical foundations
- Contributors and beta testers

---

**Developed for computational neuroscience research and clinical applications**

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

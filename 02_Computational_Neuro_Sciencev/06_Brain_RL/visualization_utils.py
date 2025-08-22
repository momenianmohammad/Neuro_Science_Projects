"""
Visualization and Analysis Utilities

This module provides comprehensive visualization and analysis tools
for brain-inspired reinforcement learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.signal import savgol_filter


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curve(
    rewards: List[float], 
    window: int = 100, 
    title: str = "Learning Curve",
    save_path: Optional[str] = None
):
    """Plot learning curve with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw rewards and moving average
    episodes = np.arange(len(rewards))
    moving_avg = pd.Series(rewards).rolling(window=window).mean()
    moving_std = pd.Series(rewards).rolling(window=window).std()
    
    # Plot 1: Learning curve
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
    ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
    ax1.fill_between(episodes, 
                     moving_avg - moving_std, 
                     moving_avg + moving_std, 
                     alpha=0.2, color='red')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward distribution histogram
    ax2.hist(rewards, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--', 
                label=f'Median: {np.median(rewards):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Density')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_dopamine_activity(
    dopamine_signals: List[float],
    value_predictions: List[float],
    title: str = "Dopamine Activity Analysis"
):
    """Plot dopamine signals and value predictions over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(len(dopamine_signals))
    
    # Plot 1: Dopamine signals over time
    axes[0, 0].plot(episodes, dopamine_signals, color='purple', alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Dopamine Signal')
    axes[0, 0].set_title('Dopamine Signals Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Value predictions over time
    axes[0, 1].plot(episodes, value_predictions, color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Value Prediction')
    axes[0, 1].set_title('Value Predictions Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Dopamine distribution
    axes[1, 0].hist(dopamine_signals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(np.mean(dopamine_signals), color='red', linestyle='--',
                       label=f'Mean: {np.mean(dopamine_signals):.3f}')
    axes[1, 0].set_xlabel('Dopamine Signal')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Dopamine Signal Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation between dopamine and value prediction
    axes[1, 1].scatter(value_predictions, dopamine_signals, alpha=0.6, color='green')
    
    # Add correlation line
    if len(value_predictions) > 1:
        z = np.polyfit(value_predictions, dopamine_signals, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(value_predictions, p(value_predictions), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(value_predictions, dopamine_signals)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=axes[1, 1].transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    axes[1, 1].set_xlabel('Value Prediction')
    axes[1, 1].set_ylabel('Dopamine Signal')
    axes[1, 1].set_title('Dopamine vs Value Prediction')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_neural_activity(activations: Dict[str, np.ndarray], title: str = "Neural Activity"):
    """Visualize neural network activations."""
    if not activations:
        print("No activation data available for visualization.")
        return
    
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 6))
    
    if n_layers == 1:
        axes = [axes]
    
    for i, (layer_name, activation) in enumerate(activations.items()):
        # Flatten activation if needed
        if activation.ndim > 2:
            activation = activation.reshape(activation.shape[0], -1)
        
        im = axes[i].imshow(activation.T, cmap='viridis', aspect='auto')
        axes[i].set_title(f'{layer_name}')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Neuron')
        plt.colorbar(im, ax=axes[i])
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_reward_prediction_error_analysis(
    reward_prediction_errors: List[float],
    rewards: List[float],
    title: str = "Reward Prediction Error Analysis"
):
    """Analyze reward prediction errors and their relationship to actual rewards."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(len(reward_prediction_errors))
    
    # Plot 1: RPE over time
    axes[0, 0].plot(episodes, reward_prediction_errors, alpha=0.7, color='red')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward Prediction Error')
    axes[0, 0].set_title('RPE Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: RPE vs actual rewards
    min_len = min(len(rewards), len(reward_prediction_errors))
    axes[0, 1].scatter(rewards[:min_len], reward_prediction_errors[:min_len], 
                       alpha=0.6, color='blue')
    axes[0, 1].set_xlabel('Actual Reward')
    axes[0, 1].set_ylabel('Reward Prediction Error')
    axes[0, 1].set_title('RPE vs Actual Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RPE distribution
    axes[1, 0].hist(reward_prediction_errors, bins=30, alpha=0.7, 
                    color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(reward_prediction_errors), color='red', linestyle='--',
                       label=f'Mean: {np.mean(reward_prediction_errors):.3f}')
    axes[1, 0].set_xlabel('Reward Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('RPE Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: RPE magnitude over time (absolute values)
    rpe_magnitude = np.abs(reward_prediction_errors)
    smoothed_rpe = savgol_filter(rpe_magnitude, 
                                 window_length=min(51, len(rpe_magnitude)//2*2+1), 
                                 polyorder=3)
    
    axes[1, 1].plot(episodes, rpe_magnitude, alpha=0.3, color='gray', label='Raw RPE Magnitude')
    axes[1, 1].plot(episodes, smoothed_rpe, color='purple', linewidth=2, label='Smoothed RPE Magnitude')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('|RPE|')
    axes[1, 1].set_title('RPE Magnitude Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_interactive_dashboard(
    rewards: List[float],
    dopamine_signals: List[float],
    value_predictions: List[float],
    save_html: bool = True
):
    """Create an interactive dashboard using Plotly."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Learning Curve', 'Dopamine Activity', 
                       'Value Predictions', 'Dopamine vs Rewards'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    episodes = list(range(len(rewards)))
    
    # Learning curve
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, mode='lines', name='Episode Rewards',
                   line=dict(color='blue', width=1), opacity=0.6),
        row=1, col=1
    )
    
    # Moving average
    window = 50
    moving_avg = pd.Series(rewards).rolling(window=window).mean().tolist()
    fig.add_trace(
        go.Scatter(x=episodes, y=moving_avg, mode='lines', name='Moving Average',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Dopamine signals
    fig.add_trace(
        go.Scatter(x=episodes[:len(dopamine_signals)], y=dopamine_signals, 
                   mode='lines', name='Dopamine Signals',
                   line=dict(color='purple', width=1)),
        row=1, col=2
    )
    
    # Value predictions
    fig.add_trace(
        go.Scatter(x=episodes[:len(value_predictions)], y=value_predictions, 
                   mode='lines', name='Value Predictions',
                   line=dict(color='orange', width=1)),
        row=2, col=1
    )
    
    # Dopamine vs rewards scatter
    min_len = min(len(rewards), len(dopamine_signals))
    fig.add_trace(
        go.Scatter(x=rewards[:min_len], y=dopamine_signals[:min_len], 
                   mode='markers', name='Dopamine vs Rewards',
                   marker=dict(color='green', size=5, opacity=0.6)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Brain-Inspired RL Agent Dashboard",
        title_x=0.5,
        height=600,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Dopamine Signal", row=1, col=2)
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Value Prediction", row=2, col=1)
    
    fig.update_xaxes(title_text="Reward", row=2, col=2)
    fig.update_yaxes(title_text="Dopamine Signal", row=2, col=2)
    
    if save_html:
        fig.write_html("brain_rl_dashboard.html")
        print("Interactive dashboard saved as 'brain_rl_dashboard.html'")
    
    fig.show()
    return fig


def compare_agents_performance(
    agent_results: Dict[str, List[float]],
    title: str = "Agent Performance Comparison"
):
    """Compare performance of different agents."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot comparison
    agent_names = list(agent_results.keys())
    rewards_data = [agent_results[name] for name in agent_names]
    
    ax1.boxplot(rewards_data, labels=agent_names)
    ax1.set_ylabel('Reward')
    ax1.set_title('Performance Distribution Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Learning curves comparison
    for name, rewards in agent_results.items():
        episodes = np.arange(len(rewards))
        moving_avg = pd.Series(rewards).rolling(window=100).mean()
        ax2.plot(episodes, moving_avg, label=name, linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Moving Average Reward')
    ax2.set_title('Learning Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistical comparison
    print("\nStatistical Comparison:")
    print("=" * 50)
    for name, rewards in agent_results.items():
        print(f"{name}:")
        print(f"  Mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  Median: {np.median(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print()


def plot_state_visitation_heatmap(state_history: List[int], n_states: int):
    """Plot state visitation frequency as a heatmap."""
    # Create visitation matrix over time
    time_windows = 10
    window_size = len(state_history) // time_windows
    
    visitation_matrix = np.zeros((time_windows, n_states))
    
    for i in range(time_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < time_windows - 1 else len(state_history)
        
        window_states = state_history[start_idx:end_idx]
        for state in window_states:
            visitation_matrix[i, state] += 1
    
    # Normalize by window size
    visitation_matrix = visitation_matrix / window_size
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(visitation_matrix, 
                xticklabels=[f'State {i}' for i in range(n_states)],
                yticklabels=[f'Window {i+1}' for i in range(time_windows)],
                cmap='YlOrRd', annot=True, fmt='.2f')
    
    plt.title('State Visitation Frequency Over Time')
    plt.xlabel('States')
    plt.ylabel('Time Windows')
    plt.tight_layout()
    plt.show()


def generate_performance_report(
    rewards: List[float],
    dopamine_signals: List[float],
    agent_name: str = "Brain-Inspired Agent"
) -> Dict:
    """Generate comprehensive performance report."""
    
    report = {
        'agent_name': agent_name,
        'total_episodes': len(rewards),
        'reward_statistics': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'median': np.median(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'q1': np.percentile(rewards, 25),
            'q3': np.percentile(rewards, 75)
        },
        'learning_metrics': {
            'final_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'improvement': np.mean(rewards[-100:]) - np.mean(rewards[:100]) if len(rewards) >= 200 else 0,
            'stability': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
        },
        'dopamine_analysis': {
            'mean_dopamine': np.mean(dopamine_signals),
            'dopamine_variance': np.var(dopamine_signals),
            'positive_dopamine_ratio': np.mean(np.array(dopamine_signals) > 0),
            'dopamine_range': np.max(dopamine_signals) - np.min(dopamine_signals),
            'dopamine_trend': 'increasing' if np.corrcoef(range(len(dopamine_signals)), dopamine_signals)[0,1] > 0.1 else 'stable'
        }
    }
    
    return report


def save_experiment_results(
    results: Dict,
    filename: str = "experiment_results.json"
):
    """Save experiment results to JSON file."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Experiment results saved to {filename}")


class ExperimentTracker:
    """Class to track and analyze experiments over multiple runs."""
    
    def __init__(self):
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, name: str, config: Dict):
        """Start a new experiment."""
        self.current_experiment = {
            'name': name,
            'config': config,
            'rewards': [],
            'dopamine_signals': [],
            'value_predictions': [],
            'timestamps': [],
            'metadata': {}
        }
    
    def log_episode(self, reward: float, dopamine: float, value: float):
        """Log data from an episode."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        import time
        self.current_experiment['rewards'].append(reward)
        self.current_experiment['dopamine_signals'].append(dopamine)
        self.current_experiment['value_predictions'].append(value)
        self.current_experiment['timestamps'].append(time.time())
    
    def end_experiment(self, metadata: Dict = None):
        """End current experiment and store results."""
        if self.current_experiment is None:
            raise ValueError("No active experiment to end.")
        
        if metadata:
            self.current_experiment['metadata'].update(metadata)
        
        # Calculate summary statistics
        rewards = self.current_experiment['rewards']
        self.current_experiment['summary'] = {
            'total_episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'final_performance': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
            'learning_efficiency': self._calculate_learning_efficiency(rewards)
        }
        
        self.experiments.append(self.current_experiment)
        self.current_experiment = None
    
    def _calculate_learning_efficiency(self, rewards: List[float]) -> float:
        """Calculate how quickly the agent learned (area under learning curve)."""
        if len(rewards) < 10:
            return 0.0
        
        # Normalize rewards to [0, 1]
        min_reward, max_reward = np.min(rewards), np.max(rewards)
        if max_reward > min_reward:
            normalized_rewards = (np.array(rewards) - min_reward) / (max_reward - min_reward)
        else:
            normalized_rewards = np.ones_like(rewards)
        
        # Calculate area under curve (higher = faster learning)
        return np.trapz(normalized_rewards) / len(rewards)
    
    def compare_experiments(self):
        """Compare all tracked experiments."""
        if len(self.experiments) < 2:
            print("Need at least 2 experiments for comparison.")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Learning curves
        for exp in self.experiments:
            rewards = exp['rewards']
            moving_avg = pd.Series(rewards).rolling(window=min(50, len(rewards)//4)).mean()
            axes[0, 0].plot(moving_avg, label=exp['name'], linewidth=2)
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Moving Average Reward')
        axes[0, 0].set_title('Learning Curves Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Final performance comparison
        exp_names = [exp['name'] for exp in self.experiments]
        final_perfs = [exp['summary']['final_performance'] for exp in self.experiments]
        
        axes[0, 1].bar(exp_names, final_perfs, color='skyblue', edgecolor='black')
        axes[0, 1].set_ylabel('Final Performance')
        axes[0, 1].set_title('Final Performance Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning efficiency
        learning_effs = [exp['summary']['learning_efficiency'] for exp in self.experiments]
        axes[1, 0].bar(exp_names, learning_effs, color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylabel('Learning Efficiency')
        axes[1, 0].set_title('Learning Efficiency Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Dopamine activity comparison
        for exp in self.experiments:
            dopamine_signals = exp['dopamine_signals']
            if dopamine_signals:
                episodes = np.arange(len(dopamine_signals))
                axes[1, 1].plot(episodes, dopamine_signals, label=exp['name'], alpha=0.7)
        
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Dopamine Signal')
        axes[1, 1].set_title('Dopamine Activity Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical comparison
        print("\nExperiment Comparison Summary:")
        print("=" * 60)
        for exp in self.experiments:
            print(f"\n{exp['name']}:")
            print(f"  Episodes: {exp['summary']['total_episodes']}")
            print(f"  Mean Reward: {exp['summary']['mean_reward']:.3f}")
            print(f"  Final Performance: {exp['summary']['final_performance']:.3f}")
            print(f"  Learning Efficiency: {exp['summary']['learning_efficiency']:.3f}")
    
    def export_results(self, filename: str = "experiment_tracker_results.json"):
        """Export all experiment results."""
        save_experiment_results({'experiments': self.experiments}, filename)


def plot_algorithm_comparison(
    algorithms_data: Dict[str, Dict],
    metrics: List[str] = ['mean_reward', 'final_performance', 'learning_efficiency']
):
    """Create comprehensive comparison plots for different algorithms."""
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    algorithm_names = list(algorithms_data.keys())
    
    for i, metric in enumerate(metrics):
        values = [algorithms_data[alg]['summary'][metric] for alg in algorithm_names]
        
        bars = axes[i].bar(algorithm_names, values, 
                          color=sns.color_palette("husl", len(algorithm_names)),
                          edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0 or minimum value
        y_min = min(values)
        y_max = max(values)
        y_range = y_max - y_min
        axes[i].set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
    
    plt.tight_layout()
    plt.show()


def create_neuroscience_interpretation_plot(
    dopamine_signals: List[float],
    rewards: List[float],
    title: str = "Neuroscientific Interpretation"
):
    """Create plots that relate to neuroscientific findings."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Dopamine response to unexpected rewards
    min_len = min(len(dopamine_signals), len(rewards))
    reward_types = []
    dopamine_responses = []
    
    for i in range(min_len):
        if rewards[i] > 0.5:  # High reward
            reward_types.append('High Reward')
            dopamine_responses.append(dopamine_signals[i])
        elif rewards[i] > 0:  # Low reward
            reward_types.append('Low Reward')
            dopamine_responses.append(dopamine_signals[i])
        else:  # No/negative reward
            reward_types.append('No Reward')
            dopamine_responses.append(dopamine_signals[i])
    
    # Create box plot for different reward types
    reward_categories = ['No Reward', 'Low Reward', 'High Reward']
    dopamine_by_category = [
        [dopamine_responses[i] for i, rt in enumerate(reward_types) if rt == cat]
        for cat in reward_categories
    ]
    
    axes[0, 0].boxplot(dopamine_by_category, labels=reward_categories)
    axes[0, 0].set_ylabel('Dopamine Signal')
    axes[0, 0].set_title('Dopamine Response by Reward Type\n(Mimics VTA Neuron Activity)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction error over learning
    # Smooth dopamine signals to show learning trend
    if len(dopamine_signals) > 50:
        smoothed_dopamine = savgol_filter(np.abs(dopamine_signals), 51, 3)
        episodes = np.arange(len(smoothed_dopamine))
        
        axes[0, 1].plot(episodes, smoothed_dopamine, color='purple', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('|Prediction Error|')
        axes[0, 1].set_title('Prediction Error Magnitude Over Learning\n(Should Decrease as Learning Progresses)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Reward prediction error histogram with normal distribution overlay
    axes[1, 0].hist(dopamine_signals, bins=30, density=True, alpha=0.7, 
                    color='lightblue', edgecolor='black', label='Observed RPE')
    
    # Fit and plot normal distribution
    mu, sigma = stats.norm.fit(dopamine_signals)
    x = np.linspace(min(dopamine_signals), max(dopamine_signals), 100)
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                    label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
    
    axes[1, 0].set_xlabel('Dopamine Signal (RPE)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('RPE Distribution\n(Should approach zero mean with learning)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning phases identification
    if len(rewards) > 100:
        # Divide learning into phases
        n_phases = 4
        phase_size = len(rewards) // n_phases
        phase_rewards = []
        phase_labels = []
        
        for i in range(n_phases):
            start_idx = i * phase_size
            end_idx = (i + 1) * phase_size if i < n_phases - 1 else len(rewards)
            phase_rewards.append(np.mean(rewards[start_idx:end_idx]))
            phase_labels.append(f'Phase {i+1}')
        
        axes[1, 1].plot(phase_labels, phase_rewards, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Learning Phases\n(Performance Across Training)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add phase annotations
        for i, reward in enumerate(phase_rewards):
            axes[1, 1].annotate(f'{reward:.2f}', 
                               (i, reward), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Utility functions for statistical analysis
def perform_statistical_tests(results1: List[float], results2: List[float], alpha: float = 0.05):
    """Perform statistical tests to compare two sets of results."""
    from scipy import stats
    
    # Normality tests
    _, p_norm1 = stats.shapiro(results1[:min(5000, len(results1))])  # Shapiro-Wilk test
    _, p_norm2 = stats.shapiro(results2[:min(5000, len(results2))])
    
    normal1 = p_norm1 > alpha
    normal2 = p_norm2 > alpha
    
    print("Statistical Analysis:")
    print("=" * 40)
    print(f"Normality test (α = {alpha}):")
    print(f"  Group 1: {'Normal' if normal1 else 'Non-normal'} (p = {p_norm1:.4f})")
    print(f"  Group 2: {'Normal' if normal2 else 'Non-normal'} (p = {p_norm2:.4f})")
    
    # Choose appropriate test
    if normal1 and normal2:
        # Both normal: use t-test
        stat, p_value = stats.ttest_ind(results1, results2)
        test_name = "Independent t-test"
    else:
        # At least one non-normal: use Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    
    print(f"\n{test_name}:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < alpha else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1, ddof=1) + 
                          (len(results2) - 1) * np.var(results2, ddof=1)) / 
                         (len(results1) + len(results2) - 2))
    
    if pooled_std > 0:
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"  Effect size interpretation: {effect_interpretation}")
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d if 'cohens_d' in locals() else None
    }
"""
Main training example for Brain-Inspired Reinforcement Learning Platform

This script demonstrates how to use the brain-inspired RL agents
and provides comprehensive analysis of their performance.
"""

import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import json
from datetime import datetime
import os

# Import our custom modules
from src.agents.dopamine_agent import DopamineAgent
from src.environments.reward_learning_env import RewardLearningEnv, DecisionMakingEnv
from src.utils.visualization import (
    plot_learning_curve, plot_dopamine_activity, 
    create_interactive_dashboard, ExperimentTracker,
    generate_performance_report, create_neuroscience_interpretation_plot
)


class BrainRLExperiment:
    """Main experiment class for brain-inspired reinforcement learning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.tracker = ExperimentTracker()
        
        # Set random seeds for reproducibility
        np.random.seed(config.get('seed', 42))
        torch.manual_seed(config.get('seed', 42))
        
        # Create results directory
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Experiment initialized. Results will be saved to: {self.results_dir}")
    
    def run_single_experiment(
        self, 
        agent_type: str = "dopamine",
        env_type: str = "reward_learning",
        n_episodes: int = 1000,
        render_freq: int = 0
    ) -> Dict:
        """Run a single experiment with specified agent and environment."""
        
        print(f"\n=== Running {agent_type.upper()} Agent on {env_type.replace('_', ' ').title()} Environment ===")
        
        # Create environment
        if env_type == "reward_learning":
            env = RewardLearningEnv(
                n_states=self.config.get('n_states', 10),
                reward_probability=self.config.get('reward_probability', 0.8),
                max_steps=self.config.get('max_steps', 200)
            )
        elif env_type == "decision_making":
            env = DecisionMakingEnv(
                n_options=self.config.get('n_options', 5),
                time_pressure=self.config.get('time_pressure', True)
            )
        else:
            # Use OpenAI Gym environment
            env = gym.make(env_type)
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        
        if agent_type == "dopamine":
            agent = DopamineAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.config.get('learning_rate', 0.001),
                dopamine_decay=self.config.get('dopamine_decay', 0.95),
                exploration_rate=self.config.get('exploration_rate', 0.1)
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Training loop
        rewards = []
        episode_lengths = []
        
        self.tracker.start_experiment(
            name=f"{agent_type}_{env_type}",
            config={
                'agent_type': agent_type,
                'env_type': env_type,
                'n_episodes': n_episodes,
                **self.config
            }
        )
        
        for episode in range(n_episodes):
            state, _ = env.reset() if hasattr(env.reset(), '__iter__') and len(env.reset()) == 2 else (env.reset(), {})
            total_reward = 0
            step_count = 0
            done = False
            
            while not done:
                # Select and execute action
                action = agent.select_action(state)
                result = env.step(action)
                
                # Handle different return formats
                if len(result) == 4:
                    next_state, reward, done, info = result
                elif len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected step return format: {len(result)} items")
                
                # Update agent
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                if render_freq > 0 and episode % render_freq == 0:
                    env.render()
            
            rewards.append(total_reward)
            episode_lengths.append(step_count)
            
            # Log to tracker
            self.tracker.log_episode(
                reward=total_reward,
                dopamine=agent.get_dopamine_signal(),
                value=agent.get_value_prediction()
            )
            
            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                avg_dopamine = np.mean(agent.dopamine_history[-100:]) if agent.dopamine_history else 0
                print(f"Episode {episode:4d}: Avg Reward = {avg_reward:7.2f}, "
                      f"Dopamine = {avg_dopamine:6.3f}, "
                      f"Exploration = {agent.exploration_rate:.3f}")
        
        # End experiment tracking
        self.tracker.end_experiment({
            'final_exploration_rate': agent.exploration_rate,
            'total_training_steps': agent.step_count
        })
        
        env.close()
        
        # Prepare results
        experiment_results = {
            'agent_type': agent_type,
            'env_type': env_type,
            'rewards': rewards,
            'episode_lengths': episode_lengths,
            'dopamine_signals': agent.dopamine_history,
            'value_predictions': agent.value_predictions,
            'reward_prediction_errors': agent.reward_prediction_errors,
            'config': self.config
        }
        
        return experiment_results
    
    def run_comparative_study(self) -> Dict:
        """Run comparative study with multiple agents/environments."""
        
        print("\n" + "="*60)
        print("RUNNING COMPARATIVE STUDY")
        print("="*60)
        
        # Define experimental conditions
        conditions = [
            ('dopamine', 'reward_learning'),
            ('dopamine', 'CartPole-v1'),
            ('dopamine', 'LunarLander-v2') if self.config.get('include_gym_envs', True) else None
        ]
        
        # Filter out None conditions
        conditions = [c for c in conditions if c is not None]
        
        comparative_results = {}
        
        for agent_type, env_type in conditions:
            try:
                experiment_name = f"{agent_type}_{env_type}"
                print(f"\nRunning condition: {experiment_name}")
                
                results = self.run_single_experiment(
                    agent_type=agent_type,
                    env_type=env_type,
                    n_episodes=self.config.get('n_episodes', 1000)
                )
                
                comparative_results[experiment_name] = results
                
                # Save individual results
                self.save_results(results, f"{experiment_name}_results.json")
                
            except Exception as e:
                print(f"Error in condition {agent_type}_{env_type}: {e}")
                continue
        
        return comparative_results
    
    def analyze_results(self, results: Dict):
        """Comprehensive analysis of experimental results."""
        
        print("\n" + "="*60)
        print("ANALYZING RESULTS")
        print("="*60)
        
        for exp_name, exp_results in results.items():
            print(f"\n--- Analysis for {exp_name} ---")
            
            rewards = exp_results['rewards']
            dopamine_signals = exp_results['dopamine_signals']
            
            # Generate performance report
            report = generate_performance_report(
                rewards=rewards,
                dopamine_signals=dopamine_signals,
                agent_name=exp_name
            )
            
            # Print summary
            print(f"Total Episodes: {report['total_episodes']}")
            print(f"Mean Reward: {report['reward_statistics']['mean']:.3f} ± {report['reward_statistics']['std']:.3f}")
            print(f"Final Performance: {report['learning_metrics']['final_performance']:.3f}")
            print(f"Learning Improvement: {report['learning_metrics']['improvement']:.3f}")
            print(f"Mean Dopamine: {report['dopamine_analysis']['mean_dopamine']:.3f}")
            print(f"Dopamine Trend: {report['dopamine_analysis']['dopamine_trend']}")
            
            # Save detailed report
            self.save_results(report, f"{exp_name}_report.json")
    
    def create_visualizations(self, results: Dict):
        """Create comprehensive visualizations."""
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        for exp_name, exp_results in results.items():
            print(f"Creating visualizations for {exp_name}...")
            
            rewards = exp_results['rewards']
            dopamine_signals = exp_results['dopamine_signals']
            value_predictions = exp_results['value_predictions']
            
            # 1. Learning curve
            plot_learning_curve(
                rewards=rewards,
                title=f"Learning Curve - {exp_name}",
                save_path=os.path.join(self.results_dir, f"{exp_name}_learning_curve.png")
            )
            
            # 2. Dopamine activity analysis
            plot_dopamine_activity(
                dopamine_signals=dopamine_signals,
                value_predictions=value_predictions,
                title=f"Dopamine Activity - {exp_name}"
            )
            plt.savefig(os.path.join(self.results_dir, f"{exp_name}_dopamine_analysis.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Neuroscience interpretation
            create_neuroscience_interpretation_plot(
                dopamine_signals=dopamine_signals,
                rewards=rewards,
                title=f"Neuroscientific Interpretation - {exp_name}"
            )
            plt.savefig(os.path.join(self.results_dir, f"{exp_name}_neuroscience_plot.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Comparative analysis
        if len(results) > 1:
            print("Creating comparative analysis...")
            self.tracker.compare_experiments()
            plt.savefig(os.path.join(self.results_dir, "comparative_analysis.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Interactive dashboard
        if len(results) > 0:
            first_exp = list(results.values())[0]
            dashboard = create_interactive_dashboard(
                rewards=first_exp['rewards'],
                dopamine_signals=first_exp['dopamine_signals'],
                value_predictions=first_exp['value_predictions'],
                save_html=True
            )
            
            # Move HTML file to results directory
            if os.path.exists("brain_rl_dashboard.html"):
                import shutil
                shutil.move("brain_rl_dashboard.html", 
                          os.path.join(self.results_dir, "interactive_dashboard.html"))
    
    def save_results(self, data: Dict, filename: str):
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        
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
        
        serializable_data = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        print(f"Results saved to: {filepath}")
    
    def generate_final_report(self, results: Dict):
        """Generate final comprehensive report."""
        
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        report = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'results_directory': self.results_dir
            },
            'experiments': {}
        }
        
        for exp_name, exp_results in results.items():
            rewards = exp_results['rewards']
            dopamine_signals = exp_results['dopamine_signals']
            
            exp_report = generate_performance_report(
                rewards=rewards,
                dopamine_signals=dopamine_signals,
                agent_name=exp_name
            )
            
            report['experiments'][exp_name] = exp_report
        
        # Add comparative statistics if multiple experiments
        if len(results) > 1:
            exp_names = list(results.keys())
            final_performances = [
                np.mean(results[name]['rewards'][-100:]) 
                if len(results[name]['rewards']) >= 100 
                else np.mean(results[name]['rewards'])
                for name in exp_names
            ]
            
            best_performer = exp_names[np.argmax(final_performances)]
            
            report['comparative_analysis'] = {
                'best_performer': best_performer,
                'performance_ranking': sorted(
                    zip(exp_names, final_performances), 
                    key=lambda x: x[1], 
                    reverse=True
                ),
                'performance_variance': np.var(final_performances)
            }
        
        # Save final report
        self.save_results(report, "final_experiment_report.json")
        
        # Print summary
        print("\nEXPERIMENT SUMMARY:")
        print("-" * 40)
        for exp_name, exp_data in report['experiments'].items():
            print(f"{exp_name}:")
            print(f"  Final Performance: {exp_data['learning_metrics']['final_performance']:.3f}")
            print(f"  Total Episodes: {exp_data['total_episodes']}")
            print(f"  Mean Dopamine: {exp_data['dopamine_analysis']['mean_dopamine']:.3f}")
        
        if 'comparative_analysis' in report:
            print(f"\nBest Performer: {report['comparative_analysis']['best_performer']}")
        
        print(f"\nAll results saved to: {self.results_dir}")


def create_default_config() -> Dict:
    """Create default configuration for experiments."""
    return {
        # Environment settings
        'n_states': 10,
        'reward_probability': 0.8,
        'max_steps': 200,
        'n_options': 5,
        'time_pressure': True,
        
        # Agent settings
        'learning_rate': 0.001,
        'dopamine_decay': 0.95,
        'exploration_rate': 0.1,
        'memory_size': 10000,
        'batch_size': 32,
        
        # Experiment settings
        'n_episodes': 1000,
        'seed': 42,
        'include_gym_envs': True,
        'render_freq': 0,  # Set to > 0 to render environment during training
    }


def main():
    """Main function to run brain-inspired RL experiments."""
    
    parser = argparse.ArgumentParser(description='Brain-Inspired Reinforcement Learning Platform')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--mode', type=str, default='comparative', 
                       choices=['single', 'comparative'],
                       help='Experiment mode: single experiment or comparative study')
    parser.add_argument('--agent', type=str, default='dopamine',
                       choices=['dopamine'],
                       help='Agent type to use')
    parser.add_argument('--env', type=str, default='reward_learning',
                       help='Environment to use')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['seed'] = args.seed
    config['n_episodes'] = args.episodes
    
    print("Brain-Inspired Reinforcement Learning Platform")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize experiment
    experiment = BrainRLExperiment(config)
    
    try:
        if args.mode == 'single':
            # Run single experiment
            results = {
                f"{args.agent}_{args.env}": experiment.run_single_experiment(
                    agent_type=args.agent,
                    env_type=args.env,
                    n_episodes=args.episodes
                )
            }
        else:
            # Run comparative study
            results = experiment.run_comparative_study()
        
        # Analyze results
        experiment.analyze_results(results)
        
        # Create visualizations
        if not args.no_viz:
            experiment.create_visualizations(results)
        
        # Generate final report
        experiment.generate_final_report(results)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results directory: {experiment.results_dir}")
        print("\nFiles generated:")
        for file in os.listdir(experiment.results_dir):
            print(f"  - {file}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to: {experiment.results_dir}")
    
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"Partial results may be available in: {experiment.results_dir}")


# Example usage functions
def run_quick_demo():
    """Run a quick demonstration of the brain-inspired RL system."""
    
    print("Running Quick Demo...")
    
    config = {
        'n_states': 5,
        'n_episodes': 200,
        'learning_rate': 0.01,
        'seed': 123
    }
    
    experiment = BrainRLExperiment(config)
    
    # Run single experiment
    results = experiment.run_single_experiment(
        agent_type='dopamine',
        env_type='reward_learning',
        n_episodes=200
    )
    
    # Quick analysis
    rewards = results['rewards']
    dopamine_signals = results['dopamine_signals']
    
    print(f"\nDemo Results:")
    print(f"Total Episodes: {len(rewards)}")
    print(f"Final Performance: {np.mean(rewards[-50:]):.3f}")
    print(f"Mean Dopamine Signal: {np.mean(dopamine_signals):.3f}")
    print(f"Learning Improvement: {np.mean(rewards[-50:]) - np.mean(rewards[:50]):.3f}")
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.7)
    plt.plot(pd.Series(rewards).rolling(20).mean(), color='red', linewidth=2)
    plt.title('Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dopamine_signals, alpha=0.7, color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Dopamine Activity')
    plt.xlabel('Episode')
    plt.ylabel('Dopamine Signal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def run_neuroscience_analysis():
    """Run analysis focused on neuroscientific aspects."""
    
    print("Running Neuroscience-Focused Analysis...")
    
    config = create_default_config()
    config['n_episodes'] = 500
    
    experiment = BrainRLExperiment(config)
    
    # Run experiment with brain-inspired agent
    results = experiment.run_single_experiment(
        agent_type='dopamine',
        env_type='reward_learning',
        n_episodes=500
    )
    
    rewards = results['rewards']
    dopamine_signals = results['dopamine_signals']
    
    # Neuroscience-specific analysis
    print("\nNeuroscientific Analysis:")
    print("-" * 30)
    
    # Dopamine response characteristics
    positive_dopamine_ratio = np.mean(np.array(dopamine_signals) > 0)
    print(f"Positive dopamine responses: {positive_dopamine_ratio:.1%}")
    
    # Learning phases
    early_performance = np.mean(rewards[:100])
    late_performance = np.mean(rewards[-100:])
    print(f"Early performance: {early_performance:.3f}")
    print(f"Late performance: {late_performance:.3f}")
    print(f"Learning improvement: {late_performance - early_performance:.3f}")
    
    # Prediction error analysis
    rpe_variance_early = np.var(dopamine_signals[:100])
    rpe_variance_late = np.var(dopamine_signals[-100:])
    print(f"RPE variance (early): {rpe_variance_early:.4f}")
    print(f"RPE variance (late): {rpe_variance_late:.4f}")
    print(f"RPE reduction: {(rpe_variance_early - rpe_variance_late) / rpe_variance_early:.1%}")
    
    # Create neuroscience interpretation plot
    create_neuroscience_interpretation_plot(
        dopamine_signals=dopamine_signals,
        rewards=rewards,
        title="Neuroscientific Analysis of Brain-Inspired RL"
    )
    
    return results


if __name__ == "__main__":
    main()


# Jupyter notebook helper functions
def setup_notebook_environment():
    """Setup environment for Jupyter notebook usage."""
    
    # Enable inline plotting
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print("Notebook environment configured successfully!")
    except:
        print("Not running in Jupyter notebook environment.")
    
    # Import commonly used modules
    import warnings
    warnings.filterwarnings('ignore')
    
    return {
        'quick_demo': run_quick_demo,
        'neuroscience_analysis': run_neuroscience_analysis,
        'default_config': create_default_config(),
        'experiment_class': BrainRLExperiment
    }


def create_tutorial_notebook():
    """Generate a tutorial notebook content."""
    
    notebook_content = """
# Brain-Inspired Reinforcement Learning Tutorial

## Introduction
This notebook demonstrates the Brain-Inspired Reinforcement Learning Platform, 
which implements dopamine-inspired learning algorithms.

## Quick Start

```python
# Import the necessary modules
from main import run_quick_demo, create_default_config, BrainRLExperiment

# Run a quick demonstration
results = run_quick_demo()
```

## Custom Experiment

```python
# Create custom configuration
config = create_default_config()
config['n_episodes'] = 500
config['learning_rate'] = 0.005

# Initialize experiment
experiment = BrainRLExperiment(config)

# Run single experiment
results = experiment.run_single_experiment(
    agent_type='dopamine',
    env_type='reward_learning',
    n_episodes=500
)

# Analyze results
experiment.analyze_results({'custom_experiment': results})
experiment.create_visualizations({'custom_experiment': results})
```

## Neuroscience Analysis

```python
# Run neuroscience-focused analysis
neuro_results = run_neuroscience_analysis()
```

For more examples, see the examples/ directory.
"""
    
    with open('tutorial_notebook.md', 'w') as f:
        f.write(notebook_content)
    
    print("Tutorial notebook content saved to 'tutorial_notebook.md'")
    print("You can convert this to a Jupyter notebook using pandoc or jupytext.")
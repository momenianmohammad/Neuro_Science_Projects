"""
Reward Learning Environment

Custom OpenAI Gym environment for testing brain-inspired RL agents.
This environment simulates reward learning and prediction scenarios.
"""

import gym
from gym import spaces
import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt


class RewardLearningEnv(gym.Env):
    """
    Environment for testing reward learning and prediction capabilities.
    
    The agent must learn to navigate to rewarding states while avoiding
    punishment. Reward contingencies can change over time to test adaptation.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_states: int = 10,
        n_actions: int = 4,
        reward_probability: float = 0.8,
        punishment_probability: float = 0.2,
        reward_magnitude: float = 1.0,
        punishment_magnitude: float = -0.5,
        state_transition_noise: float = 0.1,
        reward_schedule_changes: bool = True,
        max_steps: int = 200
    ):
        super(RewardLearningEnv, self).__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_probability = reward_probability
        self.punishment_probability = punishment_probability
        self.reward_magnitude = reward_magnitude
        self.punishment_magnitude = punishment_magnitude
        self.state_transition_noise = state_transition_noise
        self.reward_schedule_changes = reward_schedule_changes
        self.max_steps = max_steps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_states + 2,), dtype=np.float32
        )  # one-hot state + time + uncertainty
        
        # Initialize environment
        self.current_state = 0
        self.step_count = 0
        self.reward_schedule = self._generate_reward_schedule()
        self.punishment_schedule = self._generate_punishment_schedule()
        
        # Tracking variables
        self.state_visit_counts = np.zeros(n_states)
        self.reward_history = []
        self.state_history = []
        
        # Visualization
        self.fig = None
        self.ax = None
    
    def _generate_reward_schedule(self) -> np.ndarray:
        """Generate reward probabilities for each state."""
        schedule = np.random.beta(2, 2, self.n_states)  # Beta distribution
        schedule = schedule / schedule.max()  # Normalize
        return schedule * self.reward_probability
    
    def _generate_punishment_schedule(self) -> np.ndarray:
        """Generate punishment probabilities for each state."""
        schedule = np.random.beta(1, 3, self.n_states)  # Skewed towards low punishment
        return schedule * self.punishment_probability
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_state = random.randint(0, self.n_states - 1)
        self.step_count = 0
        
        # Occasionally change reward schedule (to test adaptation)
        if self.reward_schedule_changes and random.random() < 0.1:
            self.reward_schedule = self._generate_reward_schedule()
            self.punishment_schedule = self._generate_punishment_schedule()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.step_count += 1
        
        # State transition with some randomness
        if random.random() < self.state_transition_noise:
            # Random transition
            next_state = random.randint(0, self.n_states - 1)
        else:
            # Deterministic transition based on action
            if action == 0:  # Move left
                next_state = max(0, self.current_state - 1)
            elif action == 1:  # Move right
                next_state = min(self.n_states - 1, self.current_state + 1)
            elif action == 2:  # Stay
                next_state = self.current_state
            else:  # Random jump
                next_state = random.randint(0, self.n_states - 1)
        
        self.current_state = next_state
        self.state_visit_counts[self.current_state] += 1
        
        # Calculate reward based on current state
        reward = self._calculate_reward(self.current_state)
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Additional info
        info = {
            'state_visits': self.state_visit_counts.copy(),
            'reward_schedule': self.reward_schedule.copy(),
            'punishment_schedule': self.punishment_schedule.copy(),
            'dopamine_context': {
                'unexpected_reward': reward > 0 and self.reward_schedule[self.current_state] < 0.3,
                'expected_punishment': reward < 0 and self.punishment_schedule[self.current_state] > 0.5,
                'prediction_error_magnitude': abs(reward - self._expected_reward(self.current_state))
            }
        }
        
        # Store history
        self.reward_history.append(reward)
        self.state_history.append(self.current_state)
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, state: int) -> float:
        """Calculate reward for the given state."""
        reward = 0.0
        
        # Reward with probability
        if random.random() < self.reward_schedule[state]:
            reward += self.reward_magnitude
        
        # Punishment with probability
        if random.random() < self.punishment_schedule[state]:
            reward += self.punishment_magnitude
        
        # Small exploration bonus
        visit_bonus = -0.01 * np.log(self.state_visit_counts[state] + 1)
        reward += visit_bonus
        
        return reward
    
    def _expected_reward(self, state: int) -> float:
        """Calculate expected reward for a state."""
        expected = (self.reward_schedule[state] * self.reward_magnitude + 
                   self.punishment_schedule[state] * self.punishment_magnitude)
        return expected
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # One-hot encoding of current state
        state_encoding = np.zeros(self.n_states)
        state_encoding[self.current_state] = 1.0
        
        # Time information (normalized)
        time_info = self.step_count / self.max_steps
        
        # Uncertainty measure (based on visit counts)
        uncertainty = 1.0 / (1.0 + self.state_visit_counts[self.current_state])
        
        observation = np.concatenate([state_encoding, [time_info, uncertainty]])
        return observation.astype(np.float32)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render environment for human viewing."""
        if self.fig is None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plt.ion()
        
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Current state and reward schedule
        states = np.arange(self.n_states)
        self.ax1.bar(states, self.reward_schedule, alpha=0.6, label='Reward Probability', color='green')
        self.ax1.bar(states, -self.punishment_schedule, alpha=0.6, label='Punishment Probability', color='red')
        
        # Highlight current state
        self.ax1.axvline(x=self.current_state, color='blue', linewidth=3, label='Current State')
        
        self.ax1.set_xlabel('State')
        self.ax1.set_ylabel('Probability')
        self.ax1.set_title('Reward/Punishment Schedule')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recent reward history
        if len(self.reward_history) > 0:
            recent_rewards = self.reward_history[-50:]  # Last 50 rewards
            self.ax2.plot(recent_rewards, 'o-', alpha=0.7)
            self.ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            self.ax2.set_xlabel('Recent Steps')
            self.ax2.set_ylabel('Reward')
            self.ax2.set_title('Recent Reward History')
            self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def _render_rgb_array(self):
        """Render environment as RGB array."""
        if self.fig is None:
            self._render_human()
        
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf
    
    def close(self):
        """Clean up rendering."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
    
    def get_state_values(self) -> np.ndarray:
        """Get expected values for all states (for analysis)."""
        return np.array([self._expected_reward(s) for s in range(self.n_states)])
    
    def get_optimal_policy(self) -> np.ndarray:
        """Get optimal policy (for comparison)."""
        state_values = self.get_state_values()
        policy = np.zeros(self.n_states, dtype=int)
        
        for state in range(self.n_states):
            # Simple policy: move towards highest value state
            if state > 0 and state_values[state - 1] > state_values[state]:
                policy[state] = 0  # Move left
            elif state < self.n_states - 1 and state_values[state + 1] > state_values[state]:
                policy[state] = 1  # Move right
            else:
                policy[state] = 2  # Stay
        
        return policy


class DecisionMakingEnv(gym.Env):
    """
    Environment for testing decision-making under uncertainty.
    
    Features multiple objectives, time pressure, and conflicting rewards
    to simulate complex real-world decision scenarios.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        n_options: int = 5,
        n_attributes: int = 3,
        time_pressure: bool = True,
        conflicting_objectives: bool = True,
        uncertainty_level: float = 0.3
    ):
        super(DecisionMakingEnv, self).__init__()
        
        self.n_options = n_options
        self.n_attributes = n_attributes
        self.time_pressure = time_pressure
        self.conflicting_objectives = conflicting_objectives
        self.uncertainty_level = uncertainty_level
        
        # Action space: choose option or gather more information
        self.action_space = spaces.Discrete(n_options + 1)  # +1 for information gathering
        
        # Observation space: option attributes + time + uncertainty
        obs_dim = n_options * n_attributes + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize
        self.reset()
    
    def reset(self):
        """Reset the decision-making scenario."""
        # Generate new decision scenario
        self.options = self._generate_options()
        self.time_remaining = 1.0
        self.information_gathered = np.zeros((self.n_options, self.n_attributes))
        self.step_count = 0
        
        return self._get_observation()
    
    def _generate_options(self):
        """Generate options with different attribute values."""
        options = np.random.rand(self.n_options, self.n_attributes)
        
        if self.conflicting_objectives:
            # Make attributes negatively correlated
            for i in range(self.n_options):
                # High value in one attribute means lower in others
                dominant_attr = np.argmax(options[i])
                for j in range(self.n_attributes):
                    if j != dominant_attr:
                        options[i, j] *= 0.3
        
        return options
    
    def step(self, action):
        """Execute decision or information gathering."""
        self.step_count += 1
        reward = 0
        done = False
        
        # Time pressure
        if self.time_pressure:
            self.time_remaining -= 0.1
            reward -= 0.05  # Cost of time
        
        if action < self.n_options:
            # Decision made
            chosen_option = action
            
            # Calculate reward based on option quality
            true_value = np.mean(self.options[chosen_option])
            
            # Add uncertainty
            noise = np.random.normal(0, self.uncertainty_level)
            observed_value = true_value + noise
            
            reward += observed_value
            
            # Bonus for quick decisions if time pressure
            if self.time_pressure and self.time_remaining > 0.5:
                reward += 0.1
            
            done = True
            
        else:
            # Information gathering
            option_to_investigate = np.random.randint(self.n_options)
            attr_to_investigate = np.random.randint(self.n_attributes)
            
            # Reveal information about the option
            self.information_gathered[option_to_investigate, attr_to_investigate] = 1
            
            # Small cost for information gathering
            reward -= 0.02
        
        # End episode if time runs out
        if self.time_remaining <= 0:
            done = True
            reward -= 0.5  # Penalty for not deciding
        
        info = {
            'true_option_values': np.mean(self.options, axis=1),
            'information_gathered': self.information_gathered.sum(),
            'time_remaining': self.time_remaining
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation including known information."""
        # Revealed option attributes
        obs_options = self.options * self.information_gathered
        
        # Add noise to unknown attributes
        noise_mask = (1 - self.information_gathered) * np.random.normal(
            0, self.uncertainty_level, (self.n_options, self.n_attributes)
        )
        obs_options += noise_mask
        
        # Flatten and add time and information metrics
        obs = obs_options.flatten()
        obs = np.append(obs, [self.time_remaining, self.information_gathered.mean()])
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Simple text rendering."""
        print(f"Step {self.step_count}, Time remaining: {self.time_remaining:.2f}")
        print("Options (revealed attributes):")
        
        for i in range(self.n_options):
            known_attrs = []
            for j in range(self.n_attributes):
                if self.information_gathered[i, j]:
                    known_attrs.append(f"{self.options[i, j]:.2f}")
                else:
                    known_attrs.append("?")
            print(f"  Option {i}: [{', '.join(known_attrs)}]")
        print()


# Utility functions for environment analysis
def analyze_environment_dynamics(env: RewardLearningEnv, n_episodes: int = 100):
    """Analyze the dynamics of the reward learning environment."""
    all_rewards = []
    state_transitions = np.zeros((env.n_states, env.n_states))
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        prev_state = env.current_state
        
        done = False
        while not done:
            action = env.action_space.sample()  # Random policy
            state, reward, done, info = env.step(action)
            
            # Track transitions
            curr_state = env.current_state
            state_transitions[prev_state, curr_state] += 1
            
            episode_rewards.append(reward)
            prev_state = curr_state
        
        all_rewards.extend(episode_rewards)
    
    # Normalize transition matrix
    for i in range(env.n_states):
        row_sum = state_transitions[i].sum()
        if row_sum > 0:
            state_transitions[i] /= row_sum
    
    return {
        'reward_statistics': {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards)
        },
        'transition_matrix': state_transitions,
        'state_values': env.get_state_values(),
        'optimal_policy': env.get_optimal_policy()
    }


def create_curriculum_environments():
    """Create a curriculum of environments with increasing difficulty."""
    environments = []
    
    # Easy: Simple reward structure
    easy_env = RewardLearningEnv(
        n_states=5,
        reward_probability=0.9,
        punishment_probability=0.1,
        state_transition_noise=0.05,
        reward_schedule_changes=False
    )
    environments.append(('easy', easy_env))
    
    # Medium: Some uncertainty and changes
    medium_env = RewardLearningEnv(
        n_states=8,
        reward_probability=0.7,
        punishment_probability=0.3,
        state_transition_noise=0.15,
        reward_schedule_changes=True
    )
    environments.append(('medium', medium_env))
    
    # Hard: High uncertainty and frequent changes
    hard_env = RewardLearningEnv(
        n_states=12,
        reward_probability=0.6,
        punishment_probability=0.4,
        state_transition_noise=0.25,
        reward_schedule_changes=True
    )
    environments.append(('hard', hard_env))
    
    return environments
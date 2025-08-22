"""
Dopamine-Inspired Reinforcement Learning Agent

This module implements a reinforcement learning agent that mimics
dopaminergic reward processing mechanisms found in the brain.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional
import random
from collections import deque


class DopamineNetwork(nn.Module):
    """Neural network that models dopaminergic reward prediction."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(DopamineNetwork, self).__init__()
        
        # Value network (mimics ventral tegmental area)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network (mimics prefrontal cortex)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Dopamine prediction network
        self.dopamine_network = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # state + reward
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Dopamine signal can be positive or negative
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the networks."""
        value = self.value_network(state)
        policy = self.policy_network(state)
        return value, policy
    
    def predict_dopamine(self, state: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """Predict dopamine signal based on state and reward."""
        combined_input = torch.cat([state, reward.unsqueeze(-1)], dim=-1)
        return self.dopamine_network(combined_input)


class DopamineAgent:
    """
    Brain-inspired reinforcement learning agent using dopaminergic mechanisms.
    
    This agent implements temporal difference learning with reward prediction
    error signals that mimic dopaminergic neurons in the brain.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        dopamine_decay: float = 0.95,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dopamine_decay = dopamine_decay
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.network = DopamineNetwork(state_dim, action_dim)
        self.target_network = DopamineNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Tracking variables
        self.dopamine_history = []
        self.value_predictions = []
        self.reward_prediction_errors = []
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy with dopamine modulation."""
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value, policy = self.network(state_tensor)
            action_probs = policy.squeeze()
            
            # Sample from policy distribution
            action = torch.multinomial(action_probs, 1).item()
            
            # Store value prediction for analysis
            self.value_predictions.append(value.item())
            
        return action
    
    def compute_reward_prediction_error(
        self, 
        state: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> float:
        """Compute reward prediction error (mimics dopamine neuron response)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        with torch.no_grad():
            current_value = self.network.value_network(state_tensor).item()
            
            if done:
                target_value = reward
            else:
                next_value = self.target_network.value_network(next_state_tensor).item()
                target_value = reward + self.dopamine_decay * next_value
            
            # Reward prediction error (dopamine signal)
            rpe = target_value - current_value
            
        return rpe
    
    def update(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Update the agent using dopamine-inspired learning."""
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
        # Compute and store reward prediction error
        rpe = self.compute_reward_prediction_error(state, reward, next_state, done)
        self.reward_prediction_errors.append(rpe)
        self.dopamine_history.append(rpe)
        
        # Train if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self._train_networks()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
    
    def _train_networks(self):
        """Train the neural networks using sampled experiences."""
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values and policy
        current_values, current_policies = self.network(states)
        current_q_values = current_values.squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_values, _ = self.target_network(next_states)
            target_q_values = rewards + (self.dopamine_decay * next_values.squeeze() * ~dones)
        
        # Compute losses
        value_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Policy loss (using advantage as dopamine signal)
        advantages = (target_q_values - current_q_values).detach()
        action_probs = current_policies.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(action_probs) * advantages).mean()
        
        # Dopamine prediction loss
        dopamine_predictions = self.network.predict_dopamine(states, rewards)
        actual_dopamine = (target_q_values - current_q_values).detach().unsqueeze(1)
        dopamine_loss = nn.MSELoss()(dopamine_predictions, actual_dopamine)
        
        # Total loss
        total_loss = value_loss + policy_loss + 0.1 * dopamine_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.network.state_dict())
    
    def get_dopamine_signal(self) -> float:
        """Get the most recent dopamine signal."""
        return self.dopamine_history[-1] if self.dopamine_history else 0.0
    
    def get_value_prediction(self) -> float:
        """Get the most recent value prediction."""
        return self.value_predictions[-1] if self.value_predictions else 0.0
    
    def get_activations(self) -> dict:
        """Get current network activations for visualization."""
        if not hasattr(self, '_last_state'):
            return {}
        
        state_tensor = torch.FloatTensor(self._last_state).unsqueeze(0)
        
        activations = {}
        with torch.no_grad():
            # Get intermediate activations
            x = state_tensor
            for i, layer in enumerate(self.network.value_network):
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    activations[f'value_layer_{i}'] = x.numpy()
        
        return activations
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dopamine_history': self.dopamine_history,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dopamine_history = checkpoint['dopamine_history']
        self.step_count = checkpoint['step_count']


class ExperienceReplay:
    """Experience replay buffer with prioritized sampling based on dopamine signals."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, dopamine_signal):
        """Add experience with priority based on dopamine signal."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Priority based on absolute dopamine signal (surprise)
        priority = abs(dopamine_signal) + 1e-6
        if len(self.priorities) < self.capacity:
            self.priorities.append(priority ** self.alpha)
        else:
            self.priorities[self.position] = priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample experiences based on priorities."""
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        return experiences
    
    def __len__(self):
        return len(self.buffer)
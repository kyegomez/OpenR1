"""
Group Relative Policy Optimization (GRPO) implementation in PyTorch.

This module implements GRPO as described in the paper, providing a policy optimization
algorithm that foregoes the critic model and estimates the baseline from group scores.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger

@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm."""
    epsilon: float = 0.2  # Clipping parameter
    beta: float = 0.01    # KL penalty coefficient
    group_size: int = 16  # Number of outputs to sample per question
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

class GRPO:
    """
    Implementation of Group Relative Policy Optimization.
    
    Attributes:
        policy (nn.Module): Policy network that outputs action probabilities
        optimizer (torch.optim.Optimizer): Optimizer for policy network
        config (GRPOConfig): Configuration parameters
        device (torch.device): Device to run computations on
    """
    
    def __init__(
        self,
        policy: nn.Module,
        config: Optional[GRPOConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.policy = policy
        self.config = config or GRPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        logger.info(f"Initialized GRPO with device: {self.device}")
        logger.info(f"Config: {self.config}")

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using group rewards.
        
        Args:
            rewards: Tensor of shape (batch_size,) containing rewards
            group_indices: Tensor of shape (batch_size,) containing group indices
            
        Returns:
            Tensor of shape (batch_size,) containing advantage estimates
        """
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_indices)
        
        for group_idx in unique_groups:
            mask = group_indices == group_idx
            group_rewards = rewards[mask]
            
            # Compute advantage as (reward - mean) / std
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std()
            if std_reward > 0:
                advantages[mask] = (group_rewards - mean_reward) / std_reward
            else:
                advantages[mask] = group_rewards - mean_reward
                
        return advantages

    def compute_kl_divergence(
        self,
        old_probs: torch.Tensor,
        new_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between old and new policy distributions.
        
        Args:
            old_probs: Tensor of shape (batch_size, action_dim) containing old probabilities
            new_probs: Tensor of shape (batch_size, action_dim) containing new probabilities
            
        Returns:
            Scalar tensor containing mean KL divergence
        """
        kl = torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=-1)
        return kl.mean()

    def compute_policy_loss(
        self,
        old_probs: torch.Tensor,
        new_probs: torch.Tensor,
        advantages: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the clipped policy loss.
        
        Args:
            old_probs: Tensor of shape (batch_size, action_dim) with old probabilities
            new_probs: Tensor of shape (batch_size, action_dim) with new probabilities
            advantages: Tensor of shape (batch_size,) with advantages
            actions: Tensor of shape (batch_size,) with taken actions
            
        Returns:
            Scalar tensor containing the policy loss
        """
        # Get probabilities for taken actions
        old_action_probs = torch.gather(old_probs, 1, actions.unsqueeze(1)).squeeze()
        new_action_probs = torch.gather(new_probs, 1, actions.unsqueeze(1)).squeeze()
        
        # Compute probability ratio
        ratio = new_action_probs / (old_action_probs + 1e-10)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
        
        return -torch.min(surr1, surr2).mean()

    def update(
        self, 
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        group_indices: torch.Tensor,
        old_probs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update policy using GRPO algorithm.
        
        Args:
            states: Tensor of shape (batch_size, *state_dim) containing input states
            actions: Tensor of shape (batch_size,) containing taken actions
            rewards: Tensor of shape (batch_size,) containing rewards
            group_indices: Tensor of shape (batch_size,) containing group indices
            old_probs: Tensor of shape (batch_size, action_dim) containing old action probabilities
            
        Returns:
            Dictionary containing training metrics
        """
        # Move inputs to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        group_indices = group_indices.to(self.device)
        old_probs = old_probs.to(self.device)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, group_indices)
        
        # Get new action probabilities
        new_probs = self.policy(states)
        
        # Compute losses
        policy_loss = self.compute_policy_loss(old_probs, new_probs, advantages, actions)
        kl_loss = self.compute_kl_divergence(old_probs, new_probs)
        total_loss = policy_loss + self.config.beta * kl_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        states: torch.Tensor,
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the current policy.
        
        Args:
            states: Tensor of shape (batch_size, *state_dim) containing input states
            num_samples: Optional number of samples per state. If None, returns single sample.
            
        Returns:
            Tuple containing:
                - Tensor of shape (batch_size, num_samples) containing sampled actions
                - Tensor of shape (batch_size, num_samples, action_dim) containing action probabilities
        """
        if num_samples is None:
            num_samples = 1
            
        batch_size = states.shape[0]
        states = states.to(self.device)
        
        # Repeat states for multiple samples
        repeated_states = states.repeat_interleave(num_samples, dim=0)
        
        # Get action probabilities
        probs = self.policy(repeated_states)
        
        # Sample actions
        dist = Categorical(probs)
        actions = dist.sample()
        
        # Reshape outputs
        actions = actions.view(batch_size, num_samples)
        probs = probs.view(batch_size, num_samples, -1)
        
        return actions, probs
    


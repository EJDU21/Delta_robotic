"""Observation wrapper to flatten dict observations for GAIL."""

import gymnasium as gym
import numpy as np
import torch
from typing import Union, Any


class FlattenDictObsWrapper(gym.Wrapper):
    """Wrapper to flatten dictionary observations to 1D arrays.
    
        This wrapper converts dict observations to flattened numpy arrays
        for compatibility with GAIL algorithms that expect vector observations.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        
        # Store original observation space
        self.original_obs_space = env.observation_space
        
        # Create flattened observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            if len(env.observation_space.shape) == 2:
                num_envs, obs_dim = env.observation_space.shape
                self.observation_space = gym.spaces.Box(
                    low=env.observation_space.low[0, 0] if hasattr(env.observation_space.low, '__getitem__') else -np.inf,
                    high=env.observation_space.high[0, 0] if hasattr(env.observation_space.high, '__getitem__') else np.inf,
                    shape=(num_envs, obs_dim),  # Keep batch dimension
                    dtype=env.observation_space.dtype
                )
            else:
                print(f"env.observation_space.shape is not 2")
            
    
    def _flatten_obs(self, obs: Union[dict, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Flatten observation while preserving batch dimension for vectorized envs.
        
        Args:
            obs: Observation which can be a dict, numpy array, or torch tensor.
            
        Returns:
            Flattened observation as numpy array. Shape: (obs_dim,) for single env,
            or (num_envs, obs_dim) for vectorized envs.
        """
        if isinstance(obs, dict):
            # Flatten all values in the dict and concatenate
            flattened_parts = []
            batch_size = None
            
            for key in sorted(obs.keys()):  # Sort for consistency
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                if isinstance(value, np.ndarray):
                    # Handle vectorized observations
                    if len(value.shape) > 1:
                        # This is a batch, flatten each observation but keep batch dimension
                        if batch_size is None:
                            batch_size = value.shape[0]
                        flattened_parts.append(value.reshape(value.shape[0], -1))
                    else:
                        flattened_parts.append(value.flatten())
                else:
                    flattened_parts.append(np.array(value).flatten())
            
            if batch_size is not None:
                # Batch case: concatenate along last dimension, keep batch dimension
                return np.concatenate(flattened_parts, axis=-1)
            else:
                # Single observation case
                return np.concatenate(flattened_parts)
        elif isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
            if len(obs_np.shape) > 2:
                # Batch case: (num_envs, ...) -> (num_envs, flattened_dim)
                return obs_np.reshape(obs_np.shape[0], -1)
            elif len(obs_np.shape) == 2:
                # Already 2D (num_envs, obs_dim), return as is
                return obs_np
            else:
                # 1D: single observation
                return obs_np.flatten()
        else:
            obs_np = np.array(obs)
            if len(obs_np.shape) > 1:
                # Keep batch dimension if present (vectorized env)
                return obs_np.reshape(obs_np.shape[0], -1) if obs_np.shape[0] > 1 else obs_np.flatten()
            return obs_np.flatten()
    
    def reset(self, **kwargs):
        """Reset the environment and flatten observation."""
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        """Step the environment and flatten observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

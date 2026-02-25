"""Frame stacking wrapper for GAIL training.

This wrapper stacks multiple consecutive observations to match expert demonstrations
that include history frames.
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque
from typing import Union


class FrameStackWrapper(gym.Wrapper):
    """Wrapper that stacks consecutive frames to create observation history.
    
    This wrapper maintains a history of the last N observations and concatenates
    them to create a single observation vector. This matches the format of expert
    demonstrations that include history frames.
    
    Example:
        If num_frames=8 and current obs is (23,), the output will be (184,) = (8*23,)
    """
    
    def __init__(self, env: gym.Env, num_frames: int = 8):
        super().__init__(env)
        
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        
        self.num_frames = num_frames

        if not hasattr(env, 'observation_space'):
            raise ValueError("Environment must have observation_space attribute")
        
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
        
        # Check if vectorized by shape dimension
        if len(obs_space.shape) == 2:
            # Vectorized environment: (num_envs, obs_dim)
            num_envs = obs_space.shape[0]
            base_obs_space = gym.spaces.Box(
                low=obs_space.low[0] if hasattr(obs_space, 'low') else -np.inf,
                high=obs_space.high[0] if hasattr(obs_space, 'high') else np.inf,
                shape=obs_space.shape[1:],  # Remove batch dimension
                dtype=obs_space.dtype
            )
        else:
            # Single environment: (obs_dim,)
            num_envs = 1
            base_obs_space = obs_space
        
        # Calculate stacked observation shape
        # base_obs_space is guaranteed to be Box type from above
        base_shape = base_obs_space.shape
        if len(base_shape) == 0:
            base_dim = 1
        else:
            print(f"    [FrameStackWrapper] len of base_shape : {len(base_shape)}")
            base_dim = int(np.prod(base_shape))
            print(f"    [FrameStackWrapper] base_dim: {base_dim}")
        # Stacked shape: (num_frames * base_dim,)
        stacked_dim = num_frames * base_dim
        print(f"    [FrameStackWrapper] stacked_dim: {stacked_dim}")
        # Update observation space
        if num_envs > 1:
            # Vectorized environment: keep batch dimension
            print(f"    [FrameStackWrapper] num_envs : {num_envs}")
            self.observation_space = gym.spaces.Box(
                low=base_obs_space.low.flatten()[0] if hasattr(base_obs_space, 'low') else -np.inf,
                high=base_obs_space.high.flatten()[0] if hasattr(base_obs_space, 'high') else np.inf,
                shape=(num_envs, stacked_dim),
                dtype=base_obs_space.dtype
            )
            
        else:
            print(f"    [FrameStackWrapper] num_envs(<=1) : {num_envs}")
            self.observation_space = gym.spaces.Box(
                low=base_obs_space.low.flatten()[0] if hasattr(base_obs_space, 'low') else -np.inf,
                high=base_obs_space.high.flatten()[0] if hasattr(base_obs_space, 'high') else np.inf,
                shape=(stacked_dim,),
                dtype=base_obs_space.dtype
            )
        
        # Store original observation space for reference
        self.original_obs_space = base_obs_space
        self.base_dim = base_dim
        
        # Initialize frame buffers (one per environment if vectorized)
        self.num_envs = num_envs
        if num_envs > 1:
            # List of deques, one per environment
            self.frames = [deque(maxlen=num_frames) for _ in range(self.num_envs)]
        else:
            # Single deque for single environment
            self.frames = deque(maxlen=num_frames)
    
    def _flatten_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Flatten observation while preserving batch dimension for vectorized envs.
        
        Args:
            obs: Observation which can be torch.Tensor or numpy array.
                 Shape: (obs_dim,) for single env or (num_envs, obs_dim) for vectorized.
        
        Returns:
            Flattened observation. Shape: (obs_dim,) for single env or (num_envs, obs_dim) for vectorized.
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        obs = np.asarray(obs, dtype=np.float32)
        
        # Handle vectorized environment: preserve batch dimension
        if self.num_envs > 1:
            if len(obs.shape) == 1:
                # If input is 1D but we have multiple envs, reshape it
                # This shouldn't happen normally, but handle it gracefully
                if obs.shape[0] == self.num_envs * self.base_dim:
                    obs = obs.reshape(self.num_envs, self.base_dim)
                else:
                    raise ValueError(f"Unexpected obs shape {obs.shape} for {self.num_envs} environments")
            elif len(obs.shape) == 2:
                # Already in (num_envs, obs_dim) format
                if obs.shape[0] != self.num_envs:
                    raise ValueError(f"Batch size mismatch: obs.shape[0]={obs.shape[0]}, num_envs={self.num_envs}")
                # Flatten each environment's observation but keep batch dimension
                return obs.reshape(self.num_envs, -1)
            else:
                # Multi-dimensional: flatten each env's obs but keep batch dimension
                return obs.reshape(self.num_envs, -1)
        else:
            # Single environment: flatten to 1D
            return obs.flatten()
    
    def _stack_frames(self, obs: np.ndarray) -> np.ndarray:
        """Stack frames from buffer and current observation.
        
        Args:
            obs: Current observation (already flattened)
            
        Returns:
            Stacked observation vector
        """
        if self.num_envs > 1:
            # Vectorized environment
            stacked_obs = np.zeros((self.num_envs, self.num_frames * self.base_dim), dtype=np.float32)
            for env_idx in range(self.num_envs):
                # Get frames for this environment
                frames_list = list(self.frames[env_idx])
                
                # Pad with zeros if not enough frames yet
                while len(frames_list) < self.num_frames:
                    frames_list.insert(0, np.zeros(self.base_dim, dtype=np.float32))
                
                # Stack frames: oldest first, newest last
                stacked_obs[env_idx] = np.concatenate(frames_list)
            
            return stacked_obs
        else:
            # Single environment
            frames_list = list(self.frames)
            
            # Pad with zeros if not enough frames yet
            while len(frames_list) < self.num_frames:
                frames_list.insert(0, np.zeros(self.base_dim, dtype=np.float32))
            
            # Stack frames: oldest first, newest last
            return np.concatenate(frames_list)
    
    def reset(self, **kwargs):
        """Reset the environment and initialize frame buffers."""
        obs, info = self.env.reset(**kwargs)
        
        # Flatten observation
        obs_flat = self._flatten_obs(obs)
        
        # Initialize frame buffers
        if self.num_envs > 1:
            # Vectorized: initialize each environment's buffer with current obs
            for env_idx in range(self.num_envs):
                self.frames[env_idx].clear()
                # Fill with current observation (repeated)
                for _ in range(self.num_frames):
                    self.frames[env_idx].append(obs_flat[env_idx])
        else:
            # Single environment: fill buffer with current obs (repeated)
            self.frames.clear()
            for _ in range(self.num_frames):
                self.frames.append(obs_flat)
        
        # Stack frames
        stacked_obs = self._stack_frames(obs_flat)
        return stacked_obs, info
    
    def step(self, action):
        """Step the environment and update frame buffers."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Flatten observation
        obs_flat = self._flatten_obs(obs)
        
        # Update frame buffers
        if self.num_envs > 1:
            # Vectorized: update each environment's buffer
            # Convert terminated/truncated to numpy if needed
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.cpu().numpy()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.cpu().numpy()
            
            for env_idx in range(self.num_envs):
                # Only update if environment is not terminated/truncated
                # If terminated/truncated, keep the last frame
                term_val = terminated[env_idx] if isinstance(terminated, np.ndarray) else bool(terminated)
                trunc_val = truncated[env_idx] if isinstance(truncated, np.ndarray) else bool(truncated)
                if not term_val and not trunc_val:
                    self.frames[env_idx].append(obs_flat[env_idx])
        else:
            # Single environment: update buffer
            # Convert to bool if needed
            if isinstance(terminated, torch.Tensor):
                terminated = bool(terminated.item())
            elif isinstance(terminated, np.ndarray):
                terminated = bool(terminated.item() if terminated.size == 1 else terminated[0])
            
            if isinstance(truncated, torch.Tensor):
                truncated = bool(truncated.item())
            elif isinstance(truncated, np.ndarray):
                truncated = bool(truncated.item() if truncated.size == 1 else truncated[0])
            
            if not terminated and not truncated:
                self.frames.append(obs_flat)
        
        # Stack frames
        stacked_obs = self._stack_frames(obs_flat)
        
        return stacked_obs, reward, terminated, truncated, info

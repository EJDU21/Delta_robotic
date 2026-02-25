"""Wrapper to evaluate vectorized environments as single environment."""

import numpy as np
import torch


class VecEnvEvaluator:
    """Wrapper to evaluate a single environment from a vectorized environment.
    
    This wrapper allows using a vectorized environment for evaluation by
    using only the first environment in the batch.
    """
    
    def __init__(self, vec_env):
        """Initialize the evaluator.
        
        Args:
            vec_env: Vectorized environment to evaluate.
        """
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs if hasattr(vec_env, 'num_envs') else 1
    
    def reset(self, **kwargs):
        """Reset the first environment.
        
        Returns:
            Observation from the first environment.
        """
        obs, info = self.vec_env.reset(**kwargs)
        # Extract first environment's observation
        if isinstance(obs, dict):
            return {k: v[0] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) > 1 else v 
                    for k, v in obs.items()}, info
        elif isinstance(obs, (np.ndarray, torch.Tensor)) and len(obs.shape) > 1:
            return obs[0], info
        else:
            return obs, info
    
    def step(self, action):
        """Step the first environment.
        
        Args:
            action: Action for the first environment.
            
        Returns:
            Observation, reward, done, truncated, info from the first environment.
        """
        # Convert single action to batch action (repeat for all envs)
        if isinstance(action, np.ndarray):
            if len(action.shape) == 1:
                # Single action, repeat for all environments
                batch_action = np.tile(action, (self.num_envs, 1))
            else:
                batch_action = action
        elif isinstance(action, torch.Tensor):
            if len(action.shape) == 1:
                batch_action = action.unsqueeze(0).repeat(self.num_envs, 1)
            else:
                batch_action = action
        else:
            # Convert to numpy and repeat
            action_np = np.array(action)
            if len(action_np.shape) == 1:
                batch_action = np.tile(action_np, (self.num_envs, 1))
            else:
                batch_action = action_np
        
        # Step all environments
        obs, reward, terminated, truncated, info = self.vec_env.step(batch_action)
        
        # Extract first environment's results
        if isinstance(obs, dict):
            obs = {k: v[0] if isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) > 1 else v 
                   for k, v in obs.items()}
        elif isinstance(obs, (np.ndarray, torch.Tensor)) and len(obs.shape) > 1:
            obs = obs[0]
        
        # Extract first environment's reward and done
        if isinstance(reward, (np.ndarray, torch.Tensor)) and len(reward.shape) > 0:
            reward = float(reward[0])
        else:
            reward = float(reward)
        
        if isinstance(terminated, (np.ndarray, torch.Tensor)) and len(terminated.shape) > 0:
            done = bool(terminated[0])
        else:
            done = bool(terminated)
        
        return obs, reward, done, truncated, info
    
    def seed(self, seed):
        """Set seed for the environment."""
        if hasattr(self.vec_env, 'seed'):
            self.vec_env.seed(seed)

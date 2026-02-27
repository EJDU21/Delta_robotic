from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state):
        # Convert state to tensor, handling various input types
        # First, handle tuple return from reset (obs, info)
        if isinstance(state, tuple) and len(state) == 2:
            # If it's a tuple from reset, extract the observation
            state, _ = state
        
        # Now convert state to tensor
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        elif isinstance(state, np.ndarray):
            # Direct numpy array conversion - ensure contiguous and correct dtype
            if not state.flags['C_CONTIGUOUS']:
                state = np.ascontiguousarray(state)
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            # Try to convert to numpy array first
            try:
                # Handle various input types
                if hasattr(state, 'cpu') and hasattr(state, 'numpy'):
                    # It's a torch tensor, convert to numpy first
                    state_np = state.cpu().numpy()
                elif isinstance(state, (list, tuple)):
                    # Try to convert list/tuple to numpy
                    # Check if it's a nested structure (list of arrays)
                    if len(state) > 0 and isinstance(state[0], (list, tuple, np.ndarray)):
                        # Nested structure - try to stack
                        state_np = np.stack([np.asarray(s, dtype=np.float32) for s in state])
                    else:
                        state_np = np.asarray(state, dtype=np.float32)
                else:
                    # Try direct conversion
                    state_np = np.asarray(state, dtype=np.float32)
                
                # Ensure contiguous
                if not state_np.flags['C_CONTIGUOUS']:
                    state_np = np.ascontiguousarray(state_np)
                state_tensor = torch.from_numpy(state_np).float().to(self.device)
            except (ValueError, TypeError) as e:
                # If all else fails, try direct tensor conversion
                try:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                except Exception:
                    raise ValueError(f"Could not convert state to tensor. State type: {type(state)}, State shape: {getattr(state, 'shape', 'N/A')}") from e
        
        state = state_tensor
        
        # Handle vectorized environments (batch of states)
        if len(state.shape) == 1:
            # Single state
            with torch.no_grad():
                action, log_pi = self.actor.sample(state.unsqueeze(0))
            return action.cpu().numpy()[0], log_pi.item()
        else:
            # Batch of states (vectorized env)
            with torch.no_grad():
                action, log_pi = self.actor.sample(state)
            return action.cpu().numpy(), log_pi.cpu().numpy()

    def exploit(self, state):
        # Convert state to tensor, handling various input types
        # First, handle tuple return from reset (obs, info)
        if isinstance(state, tuple) and len(state) == 2:
            # If it's a tuple from reset, extract the observation
            state, _ = state
        
        # Now convert state to tensor
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        elif isinstance(state, np.ndarray):
            # Direct numpy array conversion - ensure contiguous and correct dtype
            if not state.flags['C_CONTIGUOUS']:
                state = np.ascontiguousarray(state)
            state_tensor = torch.from_numpy(state).float().to(self.device)
        else:
            # Try to convert to numpy array first
            try:
                # Handle various input types
                if hasattr(state, 'cpu') and hasattr(state, 'numpy'):
                    # It's a torch tensor, convert to numpy first
                    state_np = state.cpu().numpy()
                elif isinstance(state, (list, tuple)):
                    # Try to convert list/tuple to numpy
                    # Check if it's a nested structure (list of arrays)
                    if len(state) > 0 and isinstance(state[0], (list, tuple, np.ndarray)):
                        # Nested structure - try to stack
                        state_np = np.stack([np.asarray(s, dtype=np.float32) for s in state])
                    else:
                        state_np = np.asarray(state, dtype=np.float32)
                else:
                    # Try direct conversion
                    state_np = np.asarray(state, dtype=np.float32)
                
                # Ensure contiguous
                if not state_np.flags['C_CONTIGUOUS']:
                    state_np = np.ascontiguousarray(state_np)
                state_tensor = torch.from_numpy(state_np).float().to(self.device)
            except (ValueError, TypeError) as e:
                # If all else fails, try direct tensor conversion
                try:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                except Exception:
                    raise ValueError(f"Could not convert state to tensor. State type: {type(state)}, State shape: {getattr(state, 'shape', 'N/A')}") from e
        
        state = state_tensor
        
        # Handle vectorized environments (batch of states)
        if len(state.shape) == 1:
            # Single state
            with torch.no_grad():
                action = self.actor(state.unsqueeze(0))
            return action.cpu().numpy()[0]
        else:
            # Batch of states (vectorized env)
            with torch.no_grad():
                action = self.actor(state)
            return action.cpu().numpy()

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

"""HDF5 buffer loader for GAIL training."""

import h5py
import numpy as np
import torch
from typing import Union


def flatten_dict_obs(obs: Union[dict, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Flatten dictionary observation to a 1D numpy array.
    
    Args:
        obs: Observation which can be a dict, numpy array, or torch tensor.
        
    Returns:
        Flattened observation as 1D numpy array.
    """
    if isinstance(obs, dict):
        # Flatten all values in the dict and concatenate
        flattened_parts = []
        for key in sorted(obs.keys()):  # Sort for consistency
            value = obs[key]
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            if isinstance(value, np.ndarray):
                flattened_parts.append(value.flatten())
            else:
                flattened_parts.append(np.array(value).flatten())
        return np.concatenate(flattened_parts)
    elif isinstance(obs, torch.Tensor):
        return obs.cpu().numpy().flatten()
    else:
        return np.array(obs).flatten()


class HDF5Buffer:
    """
    Buffer loader for HDF5 expert demonstrations.
    This class loads expert demonstrations from HDF5 files and converts them to the format expected by SerializedBuffer.
    """
    
    def __init__(self, path: str, 
                 device: torch.device, 
                 max_samples: int | None = None
                ):
        """Initialize HDF5 buffer loader.
        
        Args:
            path: Path to HDF5 file.
            device: Device to load data on.
            max_samples: Maximum number of samples to load (None = load all). Useful for large files.
        """
        self.device = device
        self.path = path
    
        # First, scan the file to get metadata
        with h5py.File(path, 'r') as f:
            # traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
            
            all_traj_keys = [k for k in f.keys() if k.startswith('traj_')]    # 取得所有 traj_* 鍵，並按照數字順序排序（而非字串順序）
            traj_keys = sorted(all_traj_keys, key=lambda x: int(x.split('_')[1]))[:12]  # 只取第 0~11 條示範
            print(f"    [HDF5Buffer] 只載入前 {len(traj_keys)} 條示範 (traj_0 ~ traj_{len(traj_keys)-1})")
            total_samples = 0
            for traj_key in traj_keys:
                traj = f[traj_key]
                states = traj['state']
                if len(states.shape) == 3:
                    # print(f"        [HDF5Buffer] traj_key: {traj_key}, states shape: {states.shape}")
                    T, _, _ = states.shape
                    total_samples += T
                else:
                    print(f"    [HDF5Buffer] length of traj_key: {traj_key} shape is not 3, so it will be ignored. Shape: {states.shape}")
        
        # Apply max_samples limit if specified
        if max_samples is not None and max_samples < total_samples:
            print(f"    [HDF5Buffer] Limiting samples from {total_samples} to {max_samples}")
            total_samples = max_samples

        print(f"    [HDF5Buffer] Loading {total_samples} samples into memory...")

        # load data into memory
        states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        next_states_list = []
        
        samples_loaded = 0
        with h5py.File(path, 'r') as f:

            all_traj_keys = [k for k in f.keys() if k.startswith('traj_')]    # 取得所有 traj_* 鍵，並按照數字順序排序（而非字串順序）
            traj_keys = sorted(all_traj_keys, key=lambda x: int(x.split('_')[1]))[:12]  # 只取第 0~11 條示範
            for traj_key in traj_keys:
                if max_samples is not None and samples_loaded >= max_samples:
                    break
                
                traj = f[traj_key]
                
                # Load data directly as numpy arrays (more memory efficient)
                # state shape: (T, frame_idx, obs_dim_original) where frame_idx is the number of frames (e.g., 8 for current + 7 previous)
                states_raw = traj['state'][:]  # Shape: (T, frame_idx, obs_dim_original) - 原始 state（可能包含更多維度）
                actions_raw = traj['action'][:]  # Shape: (T, action_dim_original) - 原始動作（可能包含更多維度）
                rewards = traj['reward'][:]  # Shape: (T,)
                dones = traj['done'][:]  # Shape: (T,)
                next_states_raw = traj['next_state'][:]  # Shape: (T, frame_idx, obs_dim_original)
                
                # ===== 選擇所需的 action 維度 =====
                # 根據環境配置，只使用絕對位置相關的 action：
                    # [0]: gripper_next
                    # [2-4]: ee_pos_next (原本專家資料可能是 [1-3]，但這裡選擇 [2-4])
                    # [5-8]: ee_quat_next (原本專家資料可能是 [4-7]，但這裡選擇 [5-8])
                    # 總共 8 維：索引 [0, 2, 3, 4, 5, 6, 7, 8]
                action_indices: list[int] = [0, 2, 3, 4, 5, 6, 7, 8]
                
                # 檢查 action 維度是否足夠
                max_action_idx = max(action_indices) if action_indices else 0
                if actions_raw.shape[1] < max_action_idx + 1:
                    raise ValueError(
                        f"專家示範的 action 維度 ({actions_raw.shape[1]}) 不足，"
                        f"需要至少 {max_action_idx + 1} 維來選擇索引 {action_indices}"
                    )
                
                # 選擇指定的 action 維度
                actions = actions_raw[:, action_indices]  # Shape: (T, 8)
                # print(f"        [HDF5Buffer] 從原始 action (shape: {actions_raw.shape}) 中選擇維度 {action_indices} -> 新 action (shape: {actions.shape})")
                
                # ===== 選擇所需的 state 維度 =====
                # 根據環境配置，從原始 state 中選擇特定的維度來匹配環境的 observation_space (23 維)
                state_indices: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
                
                T, frame_idx, obs_dim_original = states_raw.shape
                
                # 檢查 state 維度是否足夠
                max_state_idx = max(state_indices) if state_indices else 0
                if obs_dim_original < max_state_idx + 1:
                    raise ValueError(
                        f"專家示範的 state 維度 ({obs_dim_original}) 不足，"
                        f"需要至少 {max_state_idx + 1} 維來選擇索引 {state_indices}"
                    )
                
                states = states_raw[:, :, state_indices]  # Shape: (T, frame_idx, 23)
                next_states = next_states_raw[:, :, state_indices]  # Shape: (T, frame_idx, 23)
                # print(f"        [HDF5Buffer] 從原始 state (shape: {states_raw.shape}) 中選擇維度 {state_indices} -> 新 state (shape: {states.shape})")
                       
                # Reshape states from (T, frame_idx, obs_dim) to (T, frame_idx*obs_dim)
                states = states.reshape(T, frame_idx * len(state_indices))  # Shape: (T, frame_idx*obs_dim)
                next_states = next_states.reshape(T, frame_idx * len(state_indices))  # Shape: (T, frame_idx*obs_dim)
                # print(f"        [HDF5Buffer] 展平後的 state (shape: {states.shape})")

                # Apply max_samples limit per trajectory
                if max_samples is not None:
                    remaining = max_samples - samples_loaded
                    if states.shape[0] > remaining:
                        states = states[:remaining]
                        actions = actions[:remaining]
                        rewards = rewards[:remaining]
                        dones = dones[:remaining]
                        next_states = next_states[:remaining]
                
                states_list.append(states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                dones_list.append(dones)
                next_states_list.append(next_states)
                
                samples_loaded += states.shape[0]
                if max_samples is not None and samples_loaded >= max_samples:
                    break
        
        # Concatenate all trajectories (single operation is more efficient)
        # print(f"    [HDF5Buffer] Concatenating {len(states_list)} trajectories...")
        states_raw = np.concatenate(states_list, axis=0) if states_list else np.empty((0, 1))
        actions_raw = np.concatenate(actions_list, axis=0) if actions_list else np.empty((0, 1))
        rewards_raw = np.concatenate(rewards_list, axis=0) if rewards_list else np.empty((0,))
        dones_raw = np.concatenate(dones_list, axis=0) if dones_list else np.empty((0,))
        next_states_raw = np.concatenate(next_states_list, axis=0) if next_states_list else np.empty((0, 1))
        
        # Clear intermediate lists to free memory
        del states_list, actions_list, rewards_list, dones_list, next_states_list
        
        # Convert to torch tensors directly on device (more memory efficient)
        # print(f"    [HDF5Buffer] Converting to PyTorch tensors on {device}...")
        self.states = torch.from_numpy(states_raw).float().to(device)
        self.actions = torch.from_numpy(actions_raw).float().to(device)
        self.rewards = torch.from_numpy(rewards_raw).float().unsqueeze(1).to(device)
        self.dones = torch.from_numpy(dones_raw).float().unsqueeze(1).to(device)
        self.next_states = torch.from_numpy(next_states_raw).float().to(device)
        
        # Clear numpy arrays to free memory
        del states_raw, actions_raw, rewards_raw, dones_raw, next_states_raw
        
        self.buffer_size = self._n = self.states.size(0)
        
        # Estimate memory usage
        mem_mb = (self.states.numel() + self.actions.numel() + self.rewards.numel() + 
                    self.dones.numel() + self.next_states.numel()) * 4 / (1024 * 1024)
        print(f"    [HDF5Buffer] Loaded {self.buffer_size} samples from {len(traj_keys)} trajectories")
        print(f"    [HDF5Buffer] State shape: {self.states.shape}, Action shape: {self.actions.shape}")
        print(f"    [HDF5Buffer] State dtype: {self.states.dtype}, Action dtype: {self.actions.dtype}")
        print(f"    [HDF5Buffer] Estimated memory usage: {mem_mb:.2f} MB")
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of samples to return.
            
        Returns:
            Tuple of (states, actions, rewards, dones, next_states).
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

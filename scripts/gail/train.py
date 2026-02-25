# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
'''
python scripts/gail/train.py --task=Template-Delta-Robotic-Direct-v0 --num_envs 5 --seed 0 --algo gail --rollout_length 5000 --num_steps 1000000 --eval_interval 100000 --video --buffer /workspace/isaaclab/source/Delta_robotic/scripts/gail/export_data/out_20260126.hdf5 --max_buffer_samples 100000
'''

"""Script to train RL agent with GAIL."""

"""
Note: 
  - rollout_length: 建議使用 10000-20000，過大 (如 50000) 會消耗大量 GPU 記憶體
  - num_envs: 根據可用記憶體調整，建議從 4-10 開始
  - max_buffer_samples: 如果 HDF5 文件很大，可以限制載入的樣本數量以節省記憶體
  - buffer: 專家示範資料的 HDF5 文件路徑
"""


import argparse
import sys
import h5py
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=50000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--buffer", type=str, default=None, help="Path to expert demonstration buffer (.npz or .hdf5)")
parser.add_argument("--rollout_length", type=int, default=50000)
parser.add_argument("--num_steps", type=int, default=10**7)
parser.add_argument("--eval_interval", type=int, default=10**5)
parser.add_argument("--algo", type=str, default="gail", help="Which algo to use from ALGOS dict")
parser.add_argument("--max_buffer_samples", type=int, default=None, help="Maximum number of samples to load from buffer (None = load all). Useful for large HDF5 files to save memory.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
import math
import os
import sys
import random
import numpy as np
from datetime import datetime
import omni

# Add GAIL_for_IsaacLab to Python path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_gail_dir = os.path.join(_script_dir, "GAIL_for_IsaacLab")
_gail_dir = os.path.abspath(_gail_dir)
if _gail_dir not in sys.path:
    sys.path.insert(0, _gail_dir)

# Add Delta_robotic module to Python path
_delta_robotic_source_dir = os.path.join(_script_dir, "..", "..", "source", "Delta_robotic")
_delta_robotic_source_dir = os.path.abspath(_delta_robotic_source_dir)
if _delta_robotic_source_dir not in sys.path:
    sys.path.insert(0, _delta_robotic_source_dir)

from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer

# Import local modules using absolute path
_scripts_gail_dir = os.path.join(_script_dir)
if _scripts_gail_dir not in sys.path:
    sys.path.insert(0, _scripts_gail_dir)

from hdf5_buffer import HDF5Buffer
from obs_wrapper import FlattenDictObsWrapper
from frame_stack_wrapper import FrameStackWrapper
from vec_env_evaluator import VecEnvEvaluator

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Delta_robotic.tasks  # noqa: F401

def main():
    """Train with GAIL agent."""
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device if hasattr(args_cli, "device") and args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric if hasattr(args_cli, "use_fabric") else None,
    )

    # override configurations with non-hydra CLI arguments
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    # 設 log 目錄
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用相對路徑或當前工作目錄，而不是硬編碼絕對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    log_root_path = os.path.join(os.path.abspath(project_root), "logs", "gail", args_cli.task)
    log_dir = os.path.join(log_root_path, time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[GAIL] Logging experiment in directory: {log_dir}")
    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "env.pkl"), env_cfg)

    # 記憶體使用警告
    if args_cli.rollout_length > 20000:
        print(f"\n[WARNING] Rollout length ({args_cli.rollout_length}) 非常大！")
        print(f"  這會導致 RolloutBuffer 分配大量 GPU 記憶體。")
        print(f"  如果遇到記憶體不足，建議:")
        print(f"    1. 減少 --rollout_length (例如: 10000 或 20000)")
        print(f"    2. 減少 --num_envs (當前: {args_cli.num_envs})")

    # ===== 2. 建 Isaac Lab 環境 =====
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print(f"\n\n\n\n[GAIL] 創建任務環境: {args_cli.task}")

    # ===== 2.5. 印出環境所有資訊 =====
    def print_space_info(space, space_name: str):
        """列印空間（觀察或動作）的詳細資訊"""
        print(f"\n【{space_name}】")
        print(f"  類型: {type(space)}")
        print(f"  類型名稱: {type(space).__name__}")
        
        if hasattr(space, 'shape'):
            print(f"  形狀 (shape): {space.shape}")
            print(f"  維度數: {len(space.shape)}")
            print(f"  總元素數: {np.prod(space.shape)}")
        
        if hasattr(space, 'dtype'):
            print(f"  資料型別 (dtype): {space.dtype}")
        
        if hasattr(space, 'low'):
            print(f"  最小值範圍 (low): {space.low}")
            if hasattr(space.low, 'min'):
                print(f"    全域最小值: {space.low.min()}")
        
        if hasattr(space, 'high'):
            print(f"  最大值範圍 (high): {space.high}")
            if hasattr(space.high, 'max'):
                print(f"    全域最大值: {space.high.max()}")
        
        if hasattr(space, 'n'):
            print(f"  離散空間大小 (n): {space.n}")
        
        # 如果是 Dict 或 Tuple 空間，顯示詳細結構
        if isinstance(space, gym.spaces.Dict):
            print(f"  字典鍵: {list(space.keys())}")
            for key, sub_space in space.spaces.items():
                print(f"    '{key}': {type(sub_space).__name__}, shape={getattr(sub_space, 'shape', 'N/A')}")
        
        if isinstance(space, gym.spaces.Tuple):
            print(f"  元組長度: {len(space.spaces)}")
            for i, sub_space in enumerate(space.spaces):
                print(f"    元素 {i}: {type(sub_space).__name__}, shape={getattr(sub_space, 'shape', 'N/A')}")
    
    print("\n" + "="*80)
    print("任務環境資訊詳細報告")
    print("="*80)
    
    # 基本環境資訊
    print(f"\n【環境基本資訊】")
    print(f"  環境類型 (type): {type(env)}")
    print(f"  環境類別名稱: {type(env).__name__}")
    print(f"  環境模組: {type(env).__module__}")
    if hasattr(env, 'unwrapped'):
        print(f"  未包裝環境類型: {type(env.unwrapped)}")
        print(f"  未包裝環境類別名稱: {type(env.unwrapped).__name__}")
    
    # 使用共用函數列印觀察和動作空間資訊
    # print_space_info(env.observation_space, "觀察空間 (Observation Space)")
    # print_space_info(env.action_space, "動作空間 (Action Space)")
    
    # ===== 3. 包裝環境以展平 dict observation =====
    env = FlattenDictObsWrapper(env)
    print("\n" + "="*80)
    print(f"[GAIL] 包裝環境以展平 dict observation: {env}")
    
    # ===== 3.5. 檢查專家資料格式並添加 Frame Stacking =====
    if args_cli.buffer is None:
        raise ValueError("--buffer argument is required. Please provide path to expert demonstration buffer (.npz or .hdf5)")
    
    buffer_path = args_cli.buffer
    if os.path.isabs(buffer_path) and os.path.exists(buffer_path):
        buffer_path = os.path.abspath(buffer_path)
    
    expert_frame_idx = None
    expert_obs_dim = None
    if buffer_path.endswith('.hdf5'):
        print("\n" + "="*80)
        print("專家資料資訊詳細報告")
        print("="*80)
        print(f"專家資料路徑: {buffer_path}")
        with h5py.File(buffer_path, 'r') as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
            print(f"總共專家資料有: {len(traj_keys)} 筆")
            if traj_keys:
                print(f"以下為專家資料: {traj_keys[0]} 的資訊")
                traj = f[traj_keys[0]]
                if 'state' in traj:
                    states = traj['state']
                    if len(states.shape) == 3:
                        T, frame_idx, obs_dim = states.shape
                        expert_frame_idx = frame_idx
                        expert_obs_dim = obs_dim
                        print(f"    state shape: (T={T}, frame_idx={frame_idx}, obs_dim={obs_dim}) -> 展平後 (T, {frame_idx*obs_dim})")
                if 'action' in traj:
                    actions = traj['action']
                    if len(actions.shape) == 2:
                        T, action_dim = actions.shape
                        print(f"    action shape: (T={T}, action_dim={action_dim})")

    # ===== 4. 添加 Frame Stacking Wrapper（如果專家資料包含歷史幀）=====
    if expert_frame_idx is not None and expert_frame_idx > 1:
        print("\n" + "="*80)
        print(f"[GAIL] 添加 Frame Stacking Wrapper (num_frames={expert_frame_idx})")
        env = FrameStackWrapper(env, num_frames=expert_frame_idx)
    else:
        print(f"[GAIL] 警告: 專家資料沒有歷史幀資訊，跳過 Frame Stacking")
        print(f"[GAIL] 這可能導致專家資料和 Actor 資料格式不匹配！")
        return 0
    
    # ===== 5. 重置環境 ======================================
    print("\n" + "="*80)
    print("[GAIL] 重置環境")
    obs, info = env.reset()
    print(f"[GAIL] 重置環境後的觀察空間: {env.observation_space}")
    print(f"[GAIL] 重置環境後的動作空間: {env.action_space}")
    print(f"[GAIL] 重置環境後的觀察 shape: {obs.shape}")
    print(f"[GAIL] 重置環境後的觀察 dtype: {obs.dtype}")
    print("="*80)


    # ===== 6. 錄影 ======================================
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("\n" + "="*80)
        print("[GAIL] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ===== 7. 建立測試環境 =================================
    print("\n" + "="*80)
    print("[GAIL] 建立測試環境")
    env_test = VecEnvEvaluator(env)

    # ===== 8. 計算環境的 state_shape 和 action_shape =====
    print("\n" + "="*80)
    print("驗證 state & action shape 是否與專家資料匹配")
    print("="*80)
    def compute_space_shape(space, obs_tensor=None):
        """計算空間形狀（去除批次維度，返回單個環境的形狀）
        
        Args:
            space: gym.Space 物件
            obs_tensor: 可選的觀察張量（用於備選方案）
        
        Returns:
            tuple: 單個環境的空間形狀
        """
        space_shape = space.shape
        
        # 處理向量化環境：去除批次維度
        if len(space_shape) >= 2:
            # 向量化環境：(num_envs, dim) -> (dim,)
            return (np.prod(space_shape[1:]),)
        elif len(space_shape) == 1:
            # 單環境一維觀察：(dim,) -> 保持不變
            result_shape = space_shape
        else:
            # 空形狀或異常情況，使用備選方案
            if obs_tensor is not None and hasattr(obs_tensor, 'shape') and len(obs_tensor.shape) > 0:
                result_shape = obs_tensor.shape[1:] if len(obs_tensor.shape) > 1 else obs_tensor.shape
            else:
                try:
                    result_shape = (gym.spaces.flatdim(space),)
                except (AttributeError, TypeError):
                    result_shape = (1,)
        
        # 確保結果是有效的 tuple
        if len(result_shape) == 0:
            result_shape = (1,)
        
        return result_shape
    
    # 獲取展平後的觀察和動作空間形狀
    # 注意：由於 FlattenDictObsWrapper 已經將觀察空間轉換為 Box，所以這裡一定是 Box 類型
    state_shape = compute_space_shape(env.observation_space, obs)
    action_shape = compute_space_shape(env.action_space)
    
    print(f"[GAIL] State shape (per env): {state_shape}, Action shape (per env): {action_shape}")
    if expert_frame_idx is not None and expert_obs_dim is not None:
        expected_state_dim = expert_frame_idx * expert_obs_dim
        actual_state_dim = state_shape[0] if len(state_shape) > 0 else 1
        print(f"[GAIL] 專家資料預期 state_dim: {expected_state_dim} (frame_idx={expert_frame_idx} * obs_dim={expert_obs_dim})")
        print(f"[GAIL] Actor 實際 state_dim: {actual_state_dim}")
        if actual_state_dim != expected_state_dim:
            print(f"[GAIL] 警告: state_dim 不匹配！")
            print(f"  這可能導致訓練問題。請檢查:")
            print(f"    1. 環境的 obs_dim 是否與專家資料的 obs_dim 一致")
            print(f"    2. Frame Stacking 是否正確應用")
            if expert_obs_dim != actual_state_dim // expert_frame_idx if expert_frame_idx > 1 else actual_state_dim:
                print(f"  專家 obs_dim={expert_obs_dim}, 環境 obs_dim={actual_state_dim // expert_frame_idx if expert_frame_idx > 1 else actual_state_dim}")
    
    # ===== 9. 載入專家示範資料 =====
    print("\n" + "="*80)
    print("載入專家示範資料")
    print("="*80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.isabs(buffer_path) and os.path.exists(buffer_path):
        print(f"{buffer_path} 是絕對路徑且文件存在")
        buffer_path = os.path.abspath(buffer_path)
    else:
        print("data path is not absolute path")
        
    if buffer_path.endswith('.hdf5'):
        print(f"Loading expert demonstrations from HDF5: \n{buffer_path}")
        buffer_exp = HDF5Buffer(
            path=buffer_path,
            device=device,
            max_samples= args_cli.max_buffer_samples
        )
    else:
        raise ValueError(f"Unsupported buffer format: {buffer_path}. Supported formats: .hdf5")

    # ===== 10. 估算 RolloutBuffer 的記憶體使用量 =====
    mix_buffer = 1  # GAIL 默認 mix_buffer=1
    total_buffer_size = args_cli.rollout_length * mix_buffer
    state_size = np.prod(state_shape) if state_shape else 1
    action_size = np.prod(action_shape) if action_shape else 1
    # 每個樣本需要: states + actions + rewards + dones + log_pis + next_states
    # 每個元素 4 bytes (float32)
    buffer_mem_mb = total_buffer_size * (state_size * 2 + action_size + 3) * 4 / (1024 * 1024)
    print("\n" + "="*80)
    print(f"\n[GAIL] RolloutBuffer 記憶體估算:")
    print(f"  - Rollout length: {args_cli.rollout_length}")
    print(f"  - Mix buffer: {mix_buffer}")
    print(f"  - Total buffer size: {total_buffer_size}")
    print(f"  - State shape: {state_shape}, Action shape: {action_shape}")
    print(f"  - 估算記憶體使用: {buffer_mem_mb:.2f} MB")
    
    if buffer_mem_mb > 1000:
        print(f"\n[WARNING] RolloutBuffer 記憶體使用超過 1GB ({buffer_mem_mb:.2f} MB)")
        print(f"  建議減少 --rollout_length 參數（例如: 10000 或 20000）")
        print(f"  或者減少 --num_envs 參數以降低記憶體壓力\n")


    # ===== 11. 初始化 GAIL 演算法 =====
    print("\n" + "="*80)
    
    algo = ALGOS[args_cli.algo](
        buffer_exp=buffer_exp,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        seed=env_cfg.seed,
        rollout_length=args_cli.rollout_length,
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args_cli.num_steps,
        eval_interval=args_cli.eval_interval,
        seed=args_cli.seed,
    )

    trainer.train()

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

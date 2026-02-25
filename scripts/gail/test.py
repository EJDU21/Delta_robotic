# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test trained GAIL model."""

"""Launch Isaac Sim Simulator first."""
"""
Example usage:
python scripts/gail/test.py \
  --task=Template-Robotic-Direct-v0 \
  --checkpoint logs/gail/Template-Robotic-Direct-v0/2024-01-01_12-00-00/model/step1000000 \
  --num_episodes 10 \
  --video

Note:
  - checkpoint: 指向模型目錄（包含 actor.pth 和 discriminator.pth）
  - num_episodes: 測試回合數
  - video: 是否錄製視頻
"""


import argparse
import sys
import os
import random
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test a trained GAIL model.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint directory (contains actor.pth)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to test.")
parser.add_argument("--seed", type=int, default=None, help="Seed for testing.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during testing.")
parser.add_argument("--video_length", type=int, default=500, help="Length of recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (for testing, usually 1).")

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

from datetime import datetime
from pathlib import Path

# Add GAIL_for_IsaacLab to Python path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_gail_dir = os.path.join(_script_dir, "..", "..", "GAIL_for_IsaacLab")
_gail_dir = os.path.abspath(_gail_dir)
if _gail_dir not in sys.path:
    sys.path.insert(0, _gail_dir)

# Add Robotic module to Python path
_robotic_source_dir = os.path.join(_script_dir, "..", "..", "source", "Robotic")
_robotic_source_dir = os.path.abspath(_robotic_source_dir)
if _robotic_source_dir not in sys.path:
    sys.path.insert(0, _robotic_source_dir)

from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.network import GAILDiscrim

# Import local modules
_scripts_gail_dir = os.path.join(_script_dir)
if _scripts_gail_dir not in sys.path:
    sys.path.insert(0, _scripts_gail_dir)

from obs_wrapper import FlattenDictObsWrapper
from vec_env_evaluator import VecEnvEvaluator

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Robotic.tasks  # noqa: F401


def load_model(checkpoint_dir, state_shape, action_shape, device, algo_name="gail"):
    """載入訓練好的 GAIL 模型。
    
    Args:
        checkpoint_dir: 模型檢查點目錄路徑
        state_shape: 狀態空間形狀
        action_shape: 動作空間形狀
        device: 設備 (cuda/cpu)
        algo_name: 演算法名稱
        
    Returns:
        載入的演算法實例（只載入 actor，不需要 discriminator 和 buffer）
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # 查找 actor.pth 文件
    actor_path = checkpoint_dir / "actor.pth"
    if not actor_path.exists():
        # 嘗試查找其他可能的文件名
        possible_names = ["actor.pth", "actor.pt", "model.pth", "policy.pth"]
        actor_path = None
        for name in possible_names:
            candidate = checkpoint_dir / name
            if candidate.exists():
                actor_path = candidate
                break
        
        if actor_path is None:
            raise FileNotFoundError(
                f"找不到 actor 模型文件在: {checkpoint_dir}\n"
                f"請確認檢查點目錄包含 actor.pth 文件"
            )
    
    print(f"[測試] 載入模型從: {actor_path}")
    
    # 創建一個臨時的 dummy buffer（測試時不需要專家數據）
    # 但 GAIL 初始化需要 buffer_exp，所以我們創建一個最小的
    class DummyBuffer:
        def __init__(self):
            self.buffer_size = 1
            self._n = 1
        
        def sample(self, batch_size):
            # 返回 dummy 數據（不會被使用）
            return (
                torch.zeros((batch_size, state_shape[0]), device=device),
                torch.zeros((batch_size, action_shape[0]), device=device),
                torch.zeros((batch_size, 1), device=device),
                torch.zeros((batch_size, 1), device=device),
                torch.zeros((batch_size, state_shape[0]), device=device),
            )
    
    dummy_buffer = DummyBuffer()
    
    # 初始化 GAIL 演算法（使用默認參數）
    algo = ALGOS[algo_name](
        buffer_exp=dummy_buffer,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        seed=0,  # 測試時 seed 不重要
        rollout_length=1000,  # 測試時不需要大 buffer
    )
    
    # 載入 actor 權重
    print(f"[測試] 載入 actor 權重...")
    actor_state_dict = torch.load(actor_path, map_location=device)
    algo.actor.load_state_dict(actor_state_dict)
    algo.actor.eval()  # 設置為評估模式
    
    # 嘗試載入 discriminator（可選，測試時不需要）
    disc_path = checkpoint_dir / "discriminator.pth"
    if disc_path.exists():
        print(f"[測試] 找到 discriminator 權重，但測試時不需要（跳過）")
    
    print(f"[測試] 模型載入完成！")
    return algo


def main():
    """測試訓練好的 GAIL 模型。"""
    
    # 解析環境配置
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device if hasattr(args_cli, "device") and args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric if hasattr(args_cli, "use_fabric") else None,
    )
    
    # 設置 seed
    seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)
    env_cfg.seed = seed
    print(f"[測試] 使用 seed: {seed}")
    
    # 創建測試日誌目錄
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.join("logs", "gail", args_cli.task, "test")
    log_dir = os.path.join(log_root_path, time_str)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[測試] 測試日誌目錄: {log_dir}")
    
    # 創建環境
    print(f"[測試] 創建環境: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 轉換為單智能體環境（如果需要）
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # 包裝觀察空間（展平字典觀察）
    env = FlattenDictObsWrapper(env)
    
    # 包裝為評估環境（使用單個環境）
    env_test = VecEnvEvaluator(env)
    
    # 計算狀態和動作空間形狀
    obs_shape = env.observation_space.shape
    if len(obs_shape) >= 2:
        state_shape = (np.prod(obs_shape[1:]),)
    else:
        state_shape = obs_shape
    
    action_shape = env.action_space.shape
    if len(action_shape) >= 2:
        action_shape = (action_shape[1],)
    elif len(action_shape) == 0:
        action_shape = (1,)
    
    if len(state_shape) == 0:
        state_shape = (1,)
    if len(action_shape) == 0:
        action_shape = (1,)
    
    print(f"[測試] State shape: {state_shape}, Action shape: {action_shape}")
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[測試] 使用設備: {device}")
    
    # 載入模型
    algo = load_model(
        checkpoint_dir=args_cli.checkpoint,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        algo_name="gail"
    )
    
    # 視頻錄製設置
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        print(f"[測試] 視頻將保存到: {video_folder}")
        # 注意：這裡我們不包裝環境，因為 VecEnvEvaluator 已經處理了單環境
    
    # 測試統計
    episode_returns = []
    episode_lengths = []
    successes = []
    
    print(f"\n{'='*80}")
    print(f"開始測試 ({args_cli.num_episodes} 回合)")
    print(f"{'='*80}\n")
    
    # 運行測試回合
    for episode in range(args_cli.num_episodes):
        reset_result = env_test.reset()
        if isinstance(reset_result, tuple):
            state, info = reset_result
        else:
            state = info = reset_result
        
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # 獲取動作
            action = algo.exploit(state)
            
            # 轉換為 torch.Tensor（如果需要）
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(device)
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=device)
            
            # 執行動作
            step_result = env_test.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                state, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step return format: {step_result}")
            
            episode_return += reward
            episode_length += 1
            
            # 檢查是否超過最大步數
            if episode_length >= args_cli.video_length:
                break
        
        # 記錄統計
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        # 檢查是否成功（如果有 success 標誌）
        success = info.get('success', False) if isinstance(info, dict) else False
        successes.append(success)
        
        print(f"回合 {episode+1}/{args_cli.num_episodes}: "
              f"回報={episode_return:.2f}, "
              f"長度={episode_length}, "
              f"成功={'是' if success else '否'}")
    
    # 計算統計信息
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)
    
    mean_length = np.mean(episode_lengths)
    success_rate = np.mean(successes) if len(successes) > 0 else 0.0
    
    # 打印結果
    print(f"\n{'='*80}")
    print("測試結果 (Test Results)")
    print(f"{'='*80}")
    print(f"  回合數 (Episodes): {args_cli.num_episodes}")
    print(f"  平均回報 (Mean Return): {mean_return:.2f} ± {std_return:.2f}")
    print(f"  最小回報 (Min Return): {min_return:.2f}")
    print(f"  最大回報 (Max Return): {max_return:.2f}")
    print(f"  平均長度 (Mean Length): {mean_length:.1f}")
    if len(successes) > 0:
        print(f"  成功率 (Success Rate): {success_rate*100:.1f}% ({np.sum(successes)}/{len(successes)})")
    print(f"{'='*80}\n")
    
    # 保存結果到文件
    results_file = os.path.join(log_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write("GAIL 模型測試結果\n")
        f.write("="*80 + "\n")
        f.write(f"檢查點: {args_cli.checkpoint}\n")
        f.write(f"任務: {args_cli.task}\n")
        f.write(f"回合數: {args_cli.num_episodes}\n")
        f.write(f"Seed: {seed}\n")
        f.write("\n統計結果:\n")
        f.write(f"  平均回報: {mean_return:.2f} ± {std_return:.2f}\n")
        f.write(f"  最小回報: {min_return:.2f}\n")
        f.write(f"  最大回報: {max_return:.2f}\n")
        f.write(f"  平均長度: {mean_length:.1f}\n")
        if len(successes) > 0:
            f.write(f"  成功率: {success_rate*100:.1f}%\n")
        f.write("\n各回合詳細結果:\n")
        for i, (ret, length, succ) in enumerate(zip(episode_returns, episode_lengths, successes)):
            f.write(f"  回合 {i+1}: 回報={ret:.2f}, 長度={length}, 成功={'是' if succ else '否'}\n")
    
    print(f"[測試] 結果已保存到: {results_file}")
    
    # 關閉環境
    env.close()
    print("[測試] 測試完成！")


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()

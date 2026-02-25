# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from Delta_robotic.robots.rs_m90e7a import RS_M90E7A_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from pathlib import Path

# 這支 .py 檔案所在的位置
HERE = Path(__file__).resolve()
# 從 tasks/direct/delta_robotic/delta_robotic_env_cfg.py 往上找到專案根目錄
# delta_robotic (1) -> direct (2) -> tasks (3) -> Delta_robotic (4) -> Delta_robotic (5) -> source (6) -> Delta_robotic (專案根目錄)
REPO_ROOT = HERE.parents[6]
ASSET_DIR = REPO_ROOT / "assets"


@configclass
class DeltaRoboticEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    # - spaces definition
    action_space = 8
    # Action space: 8 dimensions (只使用絕對位置相關的 action)
    # [0]: gripper_next (歸一化的夾爪目標位置 [0, 1])
    # [1-3]: ee_pos_next (EE 目標位置 XYZ，相對於 env 原點)
    # [4-7]: ee_quat_next (EE 目標旋轉四元數 xyzw)
    observation_space = 23
    # Observation space: 23 dimensions
    # [0-2]: rel_pos (3), [3-6]: ee_quat (4)
    # [7-9]: fan_position (3), [10-13]: fan_quat (4)
    # [14]: gripper (1), [15-22]: last_action (8) - 只包含絕對位置相關的 action
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # 使用場景中的左手機器人（RS_M90E7A_Left）
    # 場景載入到 /World/envs/env_.*/Scene 後，機器人路徑為：
    # /World/envs/env_.*/Scene/RS_M90E7A_Left
    robot_cfg: ArticulationCfg = RS_M90E7A_CONFIG.replace(
        prim_path="/World/envs/env_.*/Scene/RS_M90E7A_Left",
        spawn=None,  # 不使用 spawn，因為機器人已經在場景中
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=20, env_spacing=4.0, replicate_physics=True)

    # robot configuration
    dof_names = [
        'Revolute7', 'Revolute6', 'Revolute5', 'Revolute4', 'Revolute3', 'Revolute2', 'Revolute1',  # Arm
        'Slider9', 'Slider10'  # Gripper
    ]
    ee_body_name: str = "TF_1"
    # gripper 幾何（粗估，訓練用）
    gripper_half_thickness: float = 0.01    # 手指厚度的一半（碰撞近似用）
    grasp_width_close_threshold: float = 0.005  # 視為「夾緊」的寬度（m）

    # scene USD file
    scene_usd: str = str(ASSET_DIR / "Scene" / "Scene0903.usd")

    # - reset states/conditions
    # 風扇初始位置（相對於 env 原點）
    fan_spawn_base: tuple[float, float, float] = (0.0, -0.7, 0.0)
    fan_pos_noise_xy: tuple[float, float] = (0.05, 0.05)
    fan_pos_noise_z: float = 0.0
    fan_yaw_deg_range: float = 180.0

    # - termination conditions
    fan_insert_pos_threshold: float = 0.05  # 風扇插入位置閾值（m）
    fan_insert_angle_threshold: float = 0.2  # 風扇插入角度閾值（弧度）
    fan_drop_height: float = -0.1  # 風扇掉落高度閾值（m）
    fan_too_far_distance: float = 1.0  # 風扇被推離太遠的距離閾值（m）
    joint_pos_limit_factor: float = 0.95  # 關節位置限制因子（用於判斷是否超出限制）

    # custom parameters/scales
    # - action scale
    action_scale = 1.0
    # - reward scales
    rew_scale_alive = 1.0  # 基本生存獎勵
    rew_scale_terminated = -2.0  # 終止懲罰
    rew_scale_grasp = 5.0  # 夾持獎勵（當成功夾住風扇時）
    rew_scale_insertion_progress = 10.0  # 插入進度獎勵（風扇姿態接近目標）
    rew_scale_insertion_success = 50.0  # 成功插入獎勵
    # 插入進度獎勵的衰減參數
    insertion_pos_decay = 0.1  # 位置誤差的衰減係數（越小衰減越快）
    insertion_angle_decay = 2.0  # 角度誤差的衰減係數（越小衰減越快）

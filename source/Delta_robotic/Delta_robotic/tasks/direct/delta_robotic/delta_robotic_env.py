# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_mul, quat_conjugate, convert_quat
from isaacsim.core.prims import XFormPrim

from .delta_robotic_env_cfg import DeltaRoboticEnvCfg


class DeltaRoboticEnv(DirectRLEnv):
    cfg: DeltaRoboticEnvCfg

    def __init__(self, cfg: DeltaRoboticEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 找到機器人關節索引
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.arm_dof_idx = self.dof_idx[:7]  # 前7個關節（手臂）
        
        # 找到 EE body 索引
        if self.cfg.ee_body_name not in self.robot.body_names:
            raise RuntimeError(
                f"指定的 ee_body_name='{self.cfg.ee_body_name}' 不存在。"
                f"可用的剛體名稱有：{', '.join(self.robot.body_names)}"
            )
        self.ee_body_idx = self.robot.body_names.index(self.cfg.ee_body_name)
        
        # 計算 Jacobian 索引（對於 fixed-base，需要減1）
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.ee_body_idx - 1
        else:
            self.ee_jacobi_idx = self.ee_body_idx
        
        # 初始化 Differential IK Controller（用於位置控制）
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",  # 控制位置和旋轉
            use_relative_mode=False,  # 使用絕對位置模式
            ik_method="dls",  # Damped Least Squares，對奇異點更穩定
            ik_params={"lambda_val": 0.01},  # DLS 阻尼係數
        )
        self.ik_controller = DifferentialIKController(
            cfg=ik_cfg, num_envs=self.num_envs, device=self.device
        )
        
        # 初始化張量
        self._init_tensors_once()
        
        # 初始化關節位置和速度
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # 更新機器人和風扇的 default_root_state 以反映場景中的實際位置
        # 當資產已經在場景中（spawn=None 或未指定 spawn）時，需要從場景中讀取實際位置
        # 而不是使用配置中的 init_state
        
        # 確保場景已經更新，以便讀取實際位置
        self.scene.update(dt=self.physics_dt)
        
        # ========== 1. 更新機器人的 default_root_state ==========
        if self.cfg.robot_cfg.spawn is None:
            # 讀取機器人在場景中的實際位置和旋轉（世界座標）
            robot_pos_w = self.robot.data.root_pos_w.clone()
            robot_quat_w = self.robot.data.root_quat_w.clone()
            
            # 轉換為相對於 env 原點的座標
            robot_pos_rel = robot_pos_w - self.scene.env_origins
            robot_quat_rel = robot_quat_w  # 旋轉不需要轉換
            
            # 更新 default_root_state（相對於 env 原點）
            # default_root_state 格式：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
            self.robot.data.default_root_state[:, 0:3] = robot_pos_rel
            self.robot.data.default_root_state[:, 3:7] = robot_quat_rel
            # 速度和角速度保持為零（從配置中）
        
        # ========== 2. 更新風扇的 default_root_state ==========
        # 風扇在場景中已存在（沒有指定 spawn），需要從場景中讀取實際位置
        # 讀取風扇在場景中的實際位置和旋轉（世界座標）
        fan_pos_w = self.fan.data.root_pos_w.clone()
        fan_quat_w = self.fan.data.root_quat_w.clone()
        
        # 轉換為相對於 env 原點的座標
        fan_pos_rel = fan_pos_w - self.scene.env_origins
        fan_quat_rel = fan_quat_w  # 旋轉不需要轉換
        
        # 更新 default_root_state（相對於 env 原點）
        # default_root_state 格式：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        self.fan.data.default_root_state[:, 0:3] = fan_pos_rel
        self.fan.data.default_root_state[:, 3:7] = fan_quat_rel
        # 速度和角速度保持為零（從配置中）

    def _setup_scene(self):
        # 載入場景 USD 檔案
        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.scene_usd)
        scene_cfg.func("/World/envs/env_.*/Scene", scene_cfg)

        # 載入機器人（場景中的左手機器人）
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # 載入可操作風扇物件（場景中已存在的 RigidBody）
        # 路徑：/World/envs/env_.*/Scene/Scene0903/Scene0903/tn__FANASSY_RIGHT1_nEwC
        fan_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Scene/Scene0903/Scene0903/tn__FANASSY_RIGHT1_nEwC",
        )
        self.fan = RigidObject(fan_cfg)
        
        # 使用 XFormPrim 讀取目標位置風扇（00號風扇）的姿態
        # 路徑：/World/envs/env_.*/Scene/Scene0903/Scene0903/tn__01_1_j8icW3fhh0lW6cS/tn__FANASSY_RIGHT1_nEwC_00
        # 注意：rack_fan 不是 RigidBody，所以使用 XFormPrim 來讀取姿態
        self.rack_fan_xform = XFormPrim(
            "/World/envs/env_.*/Scene/Scene0903/Scene0903/tn__01_1_j8icW3fhh0lW6cS/tn__FANASSY_RIGHT1_nEwC_00",
            reset_xform_properties=False
        )
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add rigid objects to scene
        self.scene.rigid_objects["fan"] = self.fan
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_tensors_once(self):
        """初始化用於存儲狀態的張量"""
        # 存儲上一幀的 action（用於 observation）
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # 當前 EE 位置和旋轉（相對於 env 原點）
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        
        # 目標值（從 action 解析）
        self.ee_target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.gripper_target_position_norm = torch.zeros((self.num_envs,), device=self.device)
        
        # 夾爪相關
        self.gripper_gap = torch.zeros((self.num_envs,), device=self.device)
        
        # 風扇相關
        self.fan_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fan_quat = torch.zeros((self.num_envs, 4), device=self.device)
        
        # 目標位置相關（00號風扇的位置和旋轉）
        self.rack_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.rack_target_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _compute_intermediate(self):
        """計算當前機器人狀態（EE 位置、旋轉、夾爪間距等）"""
        # 計算當前 EE 位置和旋轉（相對於 env 原點）
        ee_world_pos = self.robot.data.body_pos_w[:, self.ee_body_idx]
        self.ee_pos = ee_world_pos - self.scene.env_origins
        self.ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx]
        
        # 計算夾爪間距（兩個 slider 的距離）
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        jpos = self.robot.data.joint_pos
        self.gripper_gap = (jpos[:, idx10] - jpos[:, idx09]).abs()
        
        # 計算可操作風扇位置和旋轉（相對於 env 原點）
        fan_world_pos = self.scene.rigid_objects["fan"].data.root_pos_w
        self.fan_pos = fan_world_pos - self.scene.env_origins
        self.fan_quat = self.scene.rigid_objects["fan"].data.root_quat_w
        
        # 計算目標位置（00號風扇的位置和旋轉，相對於 env 原點）
        # 使用 XFormPrim 讀取世界座標下的姿態（支援多環境）
        rack_fan_world_pos, rack_fan_world_quat = self.rack_fan_xform.get_world_poses()
        # rack_fan_world_quat 是 (w, x, y, z) 格式，與 Isaac Lab 一致
        # 轉換為相對於 env 原點的座標
        self.rack_target_pos = rack_fan_world_pos - self.scene.env_origins
        self.rack_target_quat = rack_fan_world_quat

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """在物理步進之前處理 actions"""
        self.actions = actions.clone()
        
        # 解析 action space (8 dimensions - 只使用絕對位置相關的 action)
        # [0]: gripper_next (歸一化的夾爪目標位置 [0, 1])
        # [1-3]: ee_pos_next (EE 目標位置 XYZ，相對於 env 原點)
        # [4-7]: ee_quat_next (EE 目標旋轉四元數 xyzw)
        gripper_next = self.actions[:, 0]              # 歸一化的夾爪目標位置 [0, 1]
        ee_pos_next = self.actions[:, 1:4]              # EE 目標位置 (x, y, z)，相對於 env 原點
        ee_quat_next = self.actions[:, 4:8]             # EE 目標旋轉四元數 (x, y, z, w)
        
        # 歸一化四元數
        ee_quat_next = ee_quat_next / (torch.linalg.norm(ee_quat_next, dim=-1, keepdim=True) + 1e-8)
        
        # 保存目標值（供 observation 使用）
        self.ee_target_position = ee_pos_next.clone()
        self.ee_target_quat = ee_quat_next.clone()
        self.gripper_target_position_norm = gripper_next.clone()
        
        # 保存上一幀的 action（供 observation 使用）
        self.prev_actions = self.actions.clone()

    def _apply_action(self) -> None:
        """應用動作：使用位置控制模式，將 EE 位置/旋轉轉換為關節位置"""
        if self.actions.dim() > 2:
            self.actions = self.actions[:, 0, :]
        
        # 計算當前狀態
        self._compute_intermediate()
        
        # 獲取當前關節位置
        current_joint_pos = self.robot.data.joint_pos[:, self.arm_dof_idx]
        
        # 獲取當前 EE 位置和旋轉（世界座標）
        ee_pos_curr = self.robot.data.body_pos_w[:, self.ee_body_idx]
        ee_quat_curr = self.robot.data.body_quat_w[:, self.ee_body_idx]
        
        # 獲取目標 EE 位置和旋轉（相對於 env 原點，需要轉換為世界座標）
        ee_pos_target = self.ee_target_position + self.scene.env_origins
        ee_quat_target = self.ee_target_quat
        
        # 使用 Differential IK Controller 計算目標關節位置
        # 設定命令（pose: [x, y, z, qw, qx, qy, qz]）
        pose_command = torch.cat([ee_pos_target, ee_quat_target], dim=-1)  # [num_envs, 7]
        self.ik_controller.set_command(pose_command, ee_pos_curr, ee_quat_curr)
        
        # 獲取 Jacobian 矩陣
        try:
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_dof_idx]
            # jacobian shape: [num_envs, 6, 7]
            # 前3行是線性速度 Jacobian，後3行是角速度 Jacobian
            
            # 計算目標關節位置
            target_joint_pos = self.ik_controller.compute(
                ee_pos=ee_pos_curr,
                ee_quat=ee_quat_curr,
                jacobian=jacobian,
                joint_pos=current_joint_pos,
            )
        except Exception as e:
            # 如果 IK 計算失敗，保持當前關節位置
            print(f"[WARNING] IK computation failed: {e}, keeping current joint positions")
            target_joint_pos = current_joint_pos
        
        # 設定手臂關節位置目標
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=self.arm_dof_idx)
        
        # 處理夾爪控制
        # 將歸一化的 gripper_next [0, 1] 轉換為實際位置 [0, 0.05] m
        gripper_target_pos = self.gripper_target_position_norm * 0.05
        current_gripper_gap = self.gripper_gap
        gripper_error = gripper_target_pos - current_gripper_gap
        
        # 使用簡單的比例控制計算夾爪位置
        gripper_kp = 1.0  # 比例增益
        gripper_pos_delta = gripper_kp * gripper_error
        
        # 獲取夾爪關節索引
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        
        # 計算目標夾爪位置（兩個 slider 反向移動）
        current_slider9 = self.robot.data.joint_pos[:, idx09]
        current_slider10 = self.robot.data.joint_pos[:, idx10]
        
        target_slider9 = current_slider9 + gripper_pos_delta * 0.5
        target_slider10 = current_slider10 - gripper_pos_delta * 0.5
        
        # 限制夾爪位置範圍
        target_slider9 = torch.clamp(target_slider9, 0.0, 0.05)
        target_slider10 = torch.clamp(target_slider10, -0.05, 0.0)
        
        # 設定夾爪關節位置目標
        # set_joint_position_target 期望形狀為 (num_envs, num_joints)
        # 當 joint_ids 是單個關節時，需要 (num_envs, 1) 的形狀
        self.robot.set_joint_position_target(target_slider9.unsqueeze(-1), joint_ids=[idx09])
        self.robot.set_joint_position_target(target_slider10.unsqueeze(-1), joint_ids=[idx10])

    def _get_observations(self) -> dict:
        """計算並返回觀測值
        
        Observation space: 23 dimensions
        [0-2]: rel_pos (相對位置 EE - Fan) (3)
        [3-6]: ee_quat (EE 當前旋轉 xyzw) (4)
        [7-9]: fan_position (風扇位置) (3)
        [10-13]: fan_quat (風扇旋轉 xyzw) (4)
        [14]: gripper (Gripper 當前寬度) (1)
        [15-22]: last_action (上一步動作，8 維) (8)
        """
        # 計算當前狀態
        self._compute_intermediate()
        
        # 計算相對位置 (EE - Fan)
        rel_pos = self.ee_pos - self.fan_pos
        
        # 轉換四元數格式：Isaac Lab 使用 (w, x, y, z)，但 observation 需要 (x, y, z, w)
        ee_quat_xyzw = convert_quat(self.ee_quat, to="xyzw")
        fan_quat_xyzw = convert_quat(self.fan_quat, to="xyzw")
        
        # 組裝觀測值
        obs_list = [
            rel_pos,                              # [0-2]: 相對位置 (3)
            ee_quat_xyzw,                         # [3-6]: EE 當前旋轉 xyzw (4)
            self.fan_pos,                         # [7-9]: 風扇位置 (3)
            fan_quat_xyzw,                        # [10-13]: 風扇旋轉 xyzw (4)
            self.gripper_gap.unsqueeze(-1),       # [14]: Gripper 當前寬度 (1)
            self.prev_actions,                    # [15-22]: 上一步動作 (8)
        ]
        
        obs = torch.cat(obs_list, dim=-1)  # 維度 = 3+4+3+4+1+8 = 23
        
        # 驗證觀測維度
        if obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Observation dimension mismatch: expected {self.cfg.observation_space}, "
                f"got {obs.shape[-1]}"
            )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """計算獎勵函式
        
        獎勵組成：
        1. 基本生存獎勵（alive reward）
        2. 終止懲罰（terminated penalty）
        3. 夾持獎勵（grasp reward）：當成功夾住風扇時給予獎勵
        4. 插入進度獎勵（insertion progress reward）：風扇姿態越接近目標姿態獎勵越高
        5. 成功插入獎勵（insertion success reward）：成功插入時給予大額獎勵
        """
        self._compute_intermediate()  # 確保最新狀態可用
        
        # 初始化獎勵為零
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # ========== 1. 基本生存獎勵 ==========
        # 只要環境還在運行就給予獎勵
        rewards += self.cfg.rew_scale_alive
        
        # ========== 2. 終止懲罰 ==========
        # 檢查是否終止（成功、失敗或時間到）
        terminated, time_out = self._get_dones()
        rewards += self.cfg.rew_scale_terminated * terminated.float()
        
        # ========== 3. 夾持獎勵 ==========
        # 檢查是否成功夾持風扇（參考 PoseMonitor 的邏輯）
        is_grasping = self._check_grasping()
        rewards += self.cfg.rew_scale_grasp * is_grasping.float()
        
        # ========== 4. 插入進度獎勵 ==========
        # 計算風扇到目標位置的誤差（位置和角度）
        fan_to_target_pos_dist = torch.linalg.norm(self.fan_pos - self.rack_target_pos, dim=-1)
        # 計算角度誤差（使用四元數內積）
        quat_dot = torch.sum(self.fan_quat * self.rack_target_quat, dim=-1)
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
        fan_to_target_rot_error = 2.0 * torch.acos(quat_dot.abs())
        
        # 插入進度獎勵：誤差越小，獎勵越高
        # 使用指數衰減：exp(-decay * error)
        pos_progress = torch.exp(-self.cfg.insertion_pos_decay * fan_to_target_pos_dist)
        angle_progress = torch.exp(-self.cfg.insertion_angle_decay * fan_to_target_rot_error)
        insertion_progress = pos_progress * angle_progress
        
        rewards += self.cfg.rew_scale_insertion_progress * insertion_progress
        
        # ========== 5. 成功插入獎勵 ==========
        # 檢查是否成功插入（位置和角度都符合條件）
        success = (fan_to_target_pos_dist < self.cfg.fan_insert_pos_threshold) & \
                  (fan_to_target_rot_error < self.cfg.fan_insert_angle_threshold)
        rewards += self.cfg.rew_scale_insertion_success * success.float()
        
        return rewards
    
    def _check_grasping(self) -> torch.Tensor:
        """檢查是否成功夾持風扇（參考 PoseMonitor 的邏輯）
        
        檢查條件：
        1. 距離條件：EE 到風扇的距離在 grasp zone 內
        2. 夾爪條件：Slider9 和 Slider10 在夾持範圍內
        
        注意：這裡簡化了 PoseMonitor 的幀數確認邏輯，直接返回當前幀的狀態
        """
        # 獲取夾爪關節索引
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        
        # 獲取夾爪關節位置
        slider9 = self.robot.data.joint_pos[:, idx09]
        slider10 = self.robot.data.joint_pos[:, idx10]
        
        # 檢查夾爪是否閉合（參考 GraspDetectionConfig 的預設值）
        grip_position_min = 0.019  # 最小夾持閉合量
        grip_position_max = 0.021  # 最大夾持閉合量
        
        slider9_ok = (slider9 >= grip_position_min) & (slider9 <= grip_position_max)
        slider10_ok = (slider10 >= -grip_position_max) & (slider10 <= -grip_position_min)
        is_closed = slider9_ok & slider10_ok
        
        # 檢查距離條件（參考 GraspDetectionConfig 的預設值）
        grasp_zone_min_m = 0.01415  # 最小有效距離
        grasp_zone_max_m = 0.02415  # 最大有效距離
        
        ee_to_fan_dist = torch.linalg.norm(self.ee_pos - self.fan_pos, dim=-1)
        is_in_grasp_zone = (ee_to_fan_dist >= grasp_zone_min_m) & (ee_to_fan_dist <= grasp_zone_max_m)
        
        # 兩個條件都滿足才算夾持
        is_grasping = is_closed & is_in_grasp_zone
        
        return is_grasping

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """計算終止條件
        
        終止條件包括：
        1. 時間終止：達到最大 episode 長度
        2. 成功條件：風扇插入到 00 號風扇位置（位置和角度都符合）
        3. 失敗條件：
           - 風扇掉落（高度低於閾值）
           - 風扇被推離太遠
           - 機器人關節超出限制
        """
        self._compute_intermediate()  # 確保最新位置可用
        
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # 1. 時間終止
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 2. 成功條件：風扇插入到 00 號風扇位置
        # 計算風扇與目標位置的距離
        fan_to_target_pos_dist = torch.linalg.norm(self.fan_pos - self.rack_target_pos, dim=-1)
        # 計算風扇與目標旋轉的角度誤差
        # 使用四元數內積計算角度誤差
        quat_dot = torch.sum(self.fan_quat * self.rack_target_quat, dim=-1)
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)  # 確保在有效範圍內
        fan_to_target_rot_error = 2.0 * torch.acos(quat_dot.abs())  # 角度誤差（弧度）
        
        success = (fan_to_target_pos_dist < self.cfg.fan_insert_pos_threshold) & \
                  (fan_to_target_rot_error < self.cfg.fan_insert_angle_threshold)

        # 3. 失敗條件
        # 3.1 風扇掉落（高度低於閾值）
        fan_dropped = self.fan_pos[:, 2] < self.cfg.fan_drop_height
        
        # 3.2 風扇被推離太遠（距離初始位置太遠）
        # 使用目標位置作為參考點
        fan_too_far = torch.linalg.norm(self.fan_pos - self.rack_target_pos, dim=-1) > self.cfg.fan_too_far_distance
        
        # 3.3 機器人關節超出限制
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits
        joint_out_of_limits = torch.any(
            (self.joint_pos < soft_joint_pos_limits[:, :, 0]) | 
            (self.joint_pos > soft_joint_pos_limits[:, :, 1]),
            dim=-1
        )
        
        # 組合所有失敗條件
        failed = fan_dropped | fan_too_far | joint_out_of_limits
        
        # 總終止條件：成功、失敗或時間到
        terminated = success | failed | time_out

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定環境的狀態
        
        重置包括：
        1. 機器人狀態：重置到初始位置和關節狀態
        2. 風扇狀態：重置到隨機化的初始位置（fan_spawn_base + 隨機偏移）
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # 確保 env_ids 不是 None（類型檢查）
        assert env_ids is not None, "env_ids should not be None after initialization"
        num_envs_to_reset = len(env_ids)
        
        super()._reset_idx(env_ids)

        # ========== 1. 重置機器人狀態 ==========
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # 重置機器人根狀態
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # 重置機器人關節狀態
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        # 確保夾爪初始狀態為打開
        idx10 = self.cfg.dof_names.index("Slider10")
        idx09 = self.cfg.dof_names.index("Slider9")
        joint_pos[:, idx09] = 0.02  # 打開狀態
        joint_pos[:, idx10] = -0.02  # 打開狀態
        
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.reset()

        # ========== 2. 重置風扇狀態 ==========
        # 使用場景中的實際位置（不添加隨機偏移）
        fan_state = self.fan.data.default_root_state[env_ids].clone()
        # default_root_state 存儲的是相對於 env 原點的座標，需要轉換為世界座標
        fan_state[:, :3] += self.scene.env_origins[env_ids]
        fan_state[:, 7:] = 0.0  # 速度設為零
        
        # ===== 以下為原本的隨機偏移邏輯（已註解） =====
        # # 計算風扇初始位置（base + 隨機偏移）
        # fan_base_pos = torch.tensor(
        #     self.cfg.fan_spawn_base, device=self.device
        # ).unsqueeze(0).repeat(num_envs_to_reset, 1)
        # 
        # # 添加 XY 平面隨機偏移
        # fan_pos_noise_xy = sample_uniform(
        #     -self.cfg.fan_pos_noise_xy[0],
        #     self.cfg.fan_pos_noise_xy[1],
        #     (num_envs_to_reset, 2),
        #     self.device,
        # )
        # # 添加 Z 方向隨機偏移
        # fan_pos_noise_z = sample_uniform(
        #     -self.cfg.fan_pos_noise_z,
        #     self.cfg.fan_pos_noise_z,
        #     (num_envs_to_reset, 1),
        #     self.device,
        # )
        # fan_pos_noise = torch.cat([fan_pos_noise_xy, fan_pos_noise_z], dim=-1)
        # fan_target_pos = fan_base_pos + fan_pos_noise
        # 
        # # 添加隨機 Yaw 旋轉（繞 Z 軸）
        # fan_yaw_rad = sample_uniform(
        #     -math.radians(self.cfg.fan_yaw_deg_range / 2),
        #     math.radians(self.cfg.fan_yaw_deg_range / 2),
        #     (num_envs_to_reset,),
        #     self.device,
        # )
        # # 創建繞 Z 軸旋轉的四元數 (w, x, y, z)
        # fan_quat_w = torch.cos(fan_yaw_rad / 2)
        # fan_quat_z = torch.sin(fan_yaw_rad / 2)
        # fan_target_quat = torch.zeros((num_envs_to_reset, 4), device=self.device)
        # fan_target_quat[:, 0] = fan_quat_w  # w
        # fan_target_quat[:, 1] = 0.0  # x
        # fan_target_quat[:, 2] = 0.0  # y
        # fan_target_quat[:, 3] = fan_quat_z  # z
        # 
        # # 設定風扇狀態
        # fan_state[:, :3] = fan_target_pos + self.scene.env_origins[env_ids]
        # fan_state[:, 3:7] = fan_target_quat
        
        self.fan.write_root_state_to_sim(fan_state, env_ids)
        self.fan.reset()
        
        # 清空暫存
        self.prev_actions[env_ids] = torch.zeros_like(self.prev_actions[env_ids])
        self._compute_intermediate()  # 更新中間值


# TODO: 根據實際任務需求實作獎勵計算函式
# @torch.jit.script
# def compute_rewards(...):
#     ...
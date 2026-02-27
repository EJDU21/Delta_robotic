# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RS-M90E7A robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from pathlib import Path

# 這支 .py 檔案所在的位置
HERE = Path(__file__).resolve()
# 從 robots/rs_m90e7a.py 往上找到專案根目錄
# rs_m90e7a.py (1) -> robots (2) -> Delta_robotic (3) -> Delta_robotic (4) -> source (5) -> Delta_robotic (專案根目錄)
REPO_ROOT = HERE.parents[5]
ASSET_DIR = REPO_ROOT / "assets"

# RS-M90E7A robot configuration
RS_M90E7A_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "RS-M90E7A" / "RS-M90E7A.usd"),
        scale=(1.0, 1.0, 1.0),
    ),
    init_state={
        "pos": (0.0, 0.0, 0.0),
        "rot": (0.9914448613738104, 0.0, 0.0, -0.13052619222005157),  # quaternion (w, x, y, z)
        "lin_vel": (0.0, 0.0, 0.0),
        "ang_vel": (0.0, 0.0, 0.0),
        "joint_pos": {
            "Revolute1": 0.0,
            "Revolute2": 0.0,
            "Revolute3": 0.0,
            "Revolute4": 0.0,
            "Revolute5": 0.0,
            "Revolute6": 0.0,
            "Revolute7": 0.0,
            "Slider9": 0.0,
            "Slider10": 0.0,
        },
        "joint_vel": {".*": 0.0},
    },
    actuators={
        "arm_actuators": ImplicitActuatorCfg(
            joint_names_expr=["Revolute.*"],
            effort_limit=None,
            velocity_limit=None,
            effort_limit_sim=100000.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper_actuators": ImplicitActuatorCfg(
            joint_names_expr=["Slider.*"],
            effort_limit=None,
            velocity_limit=None,
            effort_limit_sim=7.2,
            velocity_limit_sim=0.3,
            stiffness=200.0,
            damping=40.0,
        ),
    },
    collision_group=0,
    debug_vis=False,
)

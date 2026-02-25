#!/usr/bin/env python3
"""
Pose monitor for tracking spatial relationships between robot and target objects.

This module provides a composition-based monitor class that tracks pose errors
between the robot's end-effector and target objects (fan, ground truth position).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from .array_backend import mathops as mo
from .articulation_observer import ArticulationObserver, RobotJointConfig
from .collision_observer import (
    CollisionDispatcher,
    IllegalCollisionReport,
)
from .contact_event_source import ContactEventSource
from .grasp_config import ApproachFrameConfig, GraspDetectionConfig, PosePq
from .target_object import TargetObject


__all__ = [
    "PoseError",
    "ApproachFrameConfig",
    "GraspDetectionConfig",
    "GraspDetectionStrategy",
    "PoseMonitor",
]

if TYPE_CHECKING:
    import torch
    import warp as wp

    ArrayLike = Union[np.ndarray, "torch.Tensor", "wp.indexedarray"]
    ScalarLike = Union[float, int, np.ndarray, "torch.Tensor", "wp.indexedarray"]
else:
    ArrayLike = Any
    ScalarLike = Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoseError:
    """
    Pose error between two frames.

    Attributes:
        position_error: Position difference vector (3D) from source to target.
        distance: Euclidean distance in meters (backend tensor).
        angle_error: Rotation angle error in radians (backend tensor).
    """

    position_error: Any
    distance: Any
    angle_error: Any

    def __repr__(self) -> str:
        return (
            f"PoseError(distance={self.distance}, "
            f"angle_error={self.angle_error})"
        )

    @classmethod
    def from_posepq(cls, source: PosePq, target: PosePq) -> "PoseError":
        position_error = target.p - source.p
        distance = mo.norm(position_error)

        src_q = source.q
        tgt_q = target.q
        # torch quat_diff_rad expects shape (N, 4); normalize 1D inputs.
        if hasattr(src_q, "ndim") and src_q.ndim == 1:
            src_q = src_q.reshape(1, 4)
        if hasattr(tgt_q, "ndim") and tgt_q.ndim == 1:
            tgt_q = tgt_q.reshape(1, 4)
        angle_error = mo.quat_diff_rad(tgt_q, src_q)

        return cls(
            position_error=position_error,
            distance=distance,
            angle_error=angle_error,
        )


GraspState = Literal["open", "closed", "holding"]


# ---------------------------------------------------------------------------
# GraspDetectionStrategy
# ---------------------------------------------------------------------------

class GraspDetectionStrategy:
    """
    Strategy for detecting whether the gripper is holding an object.

    This class evaluates gripper state based on instantaneous measurements,
    with optional frame-based confirmation:
    - Gripper joint positions within symmetric threshold range
    - End-effector distance to target (whether within grasp zone)
    - Consecutive frames meeting the above conditions

    Args:
        config: Grasp detection configuration. If None, uses default values.

    Example:
        >>> # Use default config
        >>> strategy = GraspDetectionStrategy()
        >>> 
        >>> # Use custom config
        >>> config = GraspDetectionConfig(
        ...     grip_position_min=0.018,
        ...     grip_position_max=0.022,
        ...     grasp_zone_max_m=0.03,
        ... )
        >>> strategy = GraspDetectionStrategy(config)
        >>> 
        >>> is_holding = strategy.evaluate(
        ...     gripper_positions=np.array([0.02, -0.02]),
        ...     ee_distance_m=0.03,
        ... )
    """

    def __init__(self, config: Optional[GraspDetectionConfig] = None) -> None:
        # Handle None to avoid mutable default argument issue
        self._config = config if config is not None else GraspDetectionConfig()

        # Internal state (for debug)
        self._state: GraspState = "open"
        self._hold_frame_count: int = 0

    # -----------------------------------------------------------------------
    # Properties (for debug)
    # -----------------------------------------------------------------------

    @property
    def config(self) -> GraspDetectionConfig:
        """The grasp detection configuration."""
        return self._config

    @property
    def state(self) -> GraspState:
        """Current grasp state (for debug purposes)."""
        return self._state

    def reset_holding_confirmation(self) -> None:
        """
        Reset the frame-based holding confirmation counter.

        After calling this, holding will remain False until the required
        number of consecutive frames meet the grasp conditions again.
        """
        self._hold_frame_count = 0
        self._state = "open"

    # -----------------------------------------------------------------------
    # Core methods
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        gripper_positions: ArrayLike,
        ee_distance_m: ScalarLike,
    ) -> bool:
        """
        Evaluate whether the gripper is currently holding an object.

        This is a frame-based check that confirms holding only after
        consecutive frames meet the instantaneous conditions.

        Grip detection logic (based on empirical testing):
        - Slider9 (left finger): ~+0.02m when holding
        - Slider10 (right finger): ~-0.02m when holding
        - Uses symmetric range for both joints:
          - Slider9:  grip_position_min <= value <= grip_position_max
          - Slider10: -grip_position_max <= value <= -grip_position_min

        Args:
            gripper_positions: Array/tensor of gripper joint positions [Slider9, Slider10].
            ee_distance_m: Distance from end-effector to target in meters.

        Returns:
            True if holding is confirmed (gripper closed, within grasp zone,
            and sustained for the configured number of frames), False otherwise.
        """
        cfg = self._config
        confirm_frames = max(1, int(cfg.hold_confirm_frames))

        def _to_bool(value: Any) -> bool:
            if hasattr(value, "item"):
                return bool(value.item())
            return bool(value)

        # Symmetric range grip detection
        slider9 = gripper_positions[0]
        slider10 = gripper_positions[1]

        slider9_ok = (slider9 >= cfg.grip_position_min) & (slider9 <= cfg.grip_position_max)
        slider10_ok = (slider10 >= -cfg.grip_position_max) & (slider10 <= -cfg.grip_position_min)
        is_closed = _to_bool(slider9_ok & slider10_ok)

        # Check if within grasp zone
        is_in_grasp_zone = _to_bool(
            (ee_distance_m >= cfg.grasp_zone_min_m)
            & (ee_distance_m <= cfg.grasp_zone_max_m)
        )

        # Frame-based confirmation
        is_candidate = is_closed and is_in_grasp_zone
        if is_candidate:
            self._hold_frame_count += 1
        else:
            self._hold_frame_count = 0
        is_confirmed = self._hold_frame_count >= confirm_frames

        # Update internal state
        if not is_closed:
            self._state = "open"
        elif is_confirmed:
            self._state = "holding"
        else:
            self._state = "closed"

        return self._state == "holding"


# ---------------------------------------------------------------------------
# PoseMonitor
# ---------------------------------------------------------------------------

class PoseMonitor:
    """
    Monitor for tracking pose errors between robot end-effector and target objects.

    This class uses the Strategy/Composition pattern to combine an ArticulationObserver
    (for robot state) with TargetObject instances (for target poses) to compute
    spatial relationships.

    Args:
        robot_observer: The ArticulationObserver for the robot.
        fan_object: Optional TargetObject for the fan.
        ground_truth_object: Optional TargetObject for the ground truth pose.

    Example:
        >>> observer = ArticulationObserver(...)
        >>> observer.initialize()
        >>> fan = TargetObject("/World/Fan")
        >>> ground_truth = TargetObject("/World/GroundTruth")
        >>> monitor = PoseMonitor(
        ...     robot_observer=observer,
        ...     fan_object=fan,
        ...     ground_truth_object=ground_truth,
        ... )
        >>> ee_to_fan = monitor.get_ee_to_fan_error()
        >>> print(f"Distance to fan: {ee_to_fan.distance}")
        >>> print(f"Angle error (rad): {ee_to_fan.angle_error}")
    """

    def __init__(
        self,
        robot_observer: ArticulationObserver,
        fan_object: Optional[TargetObject] = None,
        ground_truth_object: Optional[TargetObject] = None,
        grasp_strategy: Optional[GraspDetectionStrategy] = None,
        contact_event_source: Optional[ContactEventSource] = None,
    ) -> None:
        self.robot_observer = robot_observer
        self.fan_object = fan_object
        self.ground_truth_object = ground_truth_object
        self.grasp_strategy = grasp_strategy
        self.collision_dispatcher: Optional[CollisionDispatcher] = None
        self.contact_event_source = contact_event_source
        self.is_holding_override: Optional[bool] = None


    def get_illegal_contacts(self) -> IllegalCollisionReport:
        if self.collision_dispatcher is None:
            raise ValueError("collision_dispatcher is not set.")
        return self.collision_dispatcher.consume_illegal_report()

    def set_is_holding_override(self, value: Optional[bool]) -> None:
        """
        Override holding state for collision handling.

        Pass True/False to force holding state; pass None to restore auto mode.
        """
        self.is_holding_override = value

    def _on_contact_report(self, report) -> None:
        if self.collision_dispatcher is None:
            return
        if self.is_holding_override is not None:
            is_holding = self.is_holding_override
        else:
            is_holding = False
            if self.grasp_strategy is not None and self.fan_object is not None:
                is_holding = self.is_holding_fan()
        self.collision_dispatcher.process_contacts(report, is_holding)

    # -----------------------------------------------------------------------
    # End-effector pose access
    # -----------------------------------------------------------------------

    def get_end_effector_pose(self) -> PosePq:
        """
        Get the current end-effector pose.

        Returns:
            PosePq with:
            - p: 3D position in world frame
            - q: quaternion in wxyz format
        """
        return self.robot_observer.get_end_effector_pose()

    # -----------------------------------------------------------------------
    # Joint position access
    # -----------------------------------------------------------------------

    def get_arm_joint_positions(self) -> ArrayLike:
        """
        Get the current arm joint positions.

        Returns:
            Array/tensor of arm joint positions (7-DOF). Backend depends on
            SimulationContext (numpy ndarray or torch Tensor).
        """
        return self.robot_observer.get_arm_joint_positions()

    def get_gripper_joint_positions(self) -> ArrayLike:
        """
        Get the current gripper joint positions.

        Returns:
            Array/tensor of gripper joint positions [Slider9, Slider10]. Backend
            depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        return self.robot_observer.get_gripper_joint_positions()

    # -----------------------------------------------------------------------
    # Pose error computation
    # -----------------------------------------------------------------------

    def get_ee_to_fan_error(self) -> PoseError:
        """
        Compute pose error between end-effector and fan.

        Returns:
            PoseError containing position and rotation errors.

        Raises:
            ValueError: If fan_object is not set.
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get EE pose
        ee_pose = self.robot_observer.get_end_effector_pose()

        # Get fan pose
        fan_pos, fan_quat = self.fan_object.get_world_pose()
        fan_pose = PosePq(fan_pos, fan_quat)

        return PoseError.from_posepq(ee_pose, fan_pose)

    def get_fan_to_ground_truth_error(self) -> PoseError:
        """
        Compute pose error between fan and ground truth position.

        Returns:
            PoseError containing position and rotation errors.

        Raises:
            ValueError: If ground_truth_object is not set.
            ValueError: If fan_object is not set.
        """
        if self.ground_truth_object is None:
            raise ValueError("ground_truth_object is not set.")
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get fan pose
        fan_pos, fan_quat = self.fan_object.get_world_pose()
        fan_pose = PosePq(fan_pos, fan_quat)

        # Get ground truth pose
        gt_pos, gt_quat = self.ground_truth_object.get_world_pose()
        gt_pose = PosePq(gt_pos, gt_quat)

        return PoseError.from_posepq(fan_pose, gt_pose)

    def get_pose_error_to_target(self, target: TargetObject) -> PoseError:
        """
        Compute pose error between end-effector and an arbitrary target.

        This is a general method that can be used with any TargetObject.

        Args:
            target: The target object to compute error against.

        Returns:
            PoseError containing position and rotation errors.
        """
        # Get EE pose
        ee_pose = self.robot_observer.get_end_effector_pose()

        # Get target pose
        target_pos, target_quat = target.get_world_pose()
        target_pose = PosePq(target_pos, target_quat)

        return PoseError.from_posepq(ee_pose, target_pose)

    # -----------------------------------------------------------------------
    # Grasp detection
    # -----------------------------------------------------------------------

    def is_holding_fan(self) -> bool:
        """
        Check if the gripper is currently holding the fan.

        This method uses the configured grasp strategy to evaluate whether
        the gripper is in a holding state based on gripper positions,
        distance to the fan, and frame-based confirmation.

        Returns:
            True if holding the fan, False otherwise.

        Raises:
            ValueError: If grasp_strategy or fan_object is not set.
        """
        if self.grasp_strategy is None:
            raise ValueError("grasp_strategy is not set.")
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get gripper positions
        gripper_positions = self.robot_observer.get_gripper_joint_positions()

        # Get EE distance to fan
        ee_to_fan = self.get_ee_to_fan_error()
        ee_distance = ee_to_fan.distance

        return self.grasp_strategy.evaluate(
            gripper_positions=gripper_positions,
            ee_distance_m=ee_distance,
        )

    def reset_holding_confirmation(self) -> None:
        """
        Reset the grasp holding confirmation window.

        After calling this, holding will remain False until the required
        number of consecutive frames meet the grasp conditions again.

        Raises:
            ValueError: If grasp_strategy is not set.
        """
        if self.grasp_strategy is None:
            raise ValueError("grasp_strategy is not set.")
        self.grasp_strategy.reset_holding_confirmation()

    # -----------------------------------------------------------------------
    # Finger to handle distance
    # -----------------------------------------------------------------------

    def get_finger_to_handle_distances(self) -> Tuple[Any, Any]:
        """
        Get distances from gripper fingers to fan handles.

        This provides more precise grasp quality information than
        EE-to-fan-center distance, useful for RL reward shaping or
        grasp quality constraints.

        The handle positions are computed from the fan center using
        offsets from the fan_object's grasp_config.

        Returns:
            Tuple of (left_finger_to_left_handle_dist, right_finger_to_right_handle_dist)
            in meters (backend tensors).

        Raises:
            ValueError: If fan_object is not set.
            ValueError: If fan_object's grasp_config is not set.

        Example:
            >>> left_dist, right_dist = monitor.get_finger_to_handle_distances()
            >>> print(f"Left: {left_dist}, Right: {right_dist}")
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        # Get finger poses from robot observer
        left_finger_pose, right_finger_pose = self.robot_observer.get_finger_poses()

        # Get handle poses from fan object (uses its grasp_config)
        left_handle_pose, right_handle_pose = self.fan_object.get_handle_poses()

        # Compute distances using mathops backend
        left_dist = mo.norm(left_finger_pose.p - left_handle_pose.p)
        right_dist = mo.norm(right_finger_pose.p - right_handle_pose.p)

        return left_dist, right_dist

    def get_finger_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get the current gripper finger poses in world frame.

        This is a convenience method that delegates to the robot observer.

        Returns:
            Tuple of (left_finger_pose, right_finger_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format
        """
        return self.robot_observer.get_finger_poses()

    def get_handle_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get the virtual fan handle poses in world frame.

        This is a convenience method that delegates to the fan object.

        Returns:
            Tuple of (left_handle_pose, right_handle_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Raises:
            ValueError: If fan_object is not set.
            ValueError: If fan_object's grasp_config is not set.
        """
        if self.fan_object is None:
            raise ValueError("fan_object is not set.")

        return self.fan_object.get_handle_poses()

    # -----------------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------------

    @classmethod
    def create_default(
        cls,
        robot_prim_path: Optional[str] = None,
        fan_prim_path: Optional[str] = None,
        ground_truth_prim_path: Optional[str] = None,
        *,
        joint_config: Optional[RobotJointConfig] = None,
        grasp_config: Optional[GraspDetectionConfig] = None,
        contact_report_paths: Optional[List[str]] = None,
        cabinet_prim_path: Optional[str] = None,
    ) -> "PoseMonitor":
        """
        Factory method to create a PoseMonitor.

        This method simplifies initialization by automatically creating the
        ArticulationObserver and TargetObject instances from prim paths.

        Prim paths default to values in grasp_config when not provided.
        The robot prim path is REQUIRED and must point to an existing articulation.

        Args:
            robot_prim_path: USD prim path of the robot articulation.
                             Defaults to grasp_config.robot_prim_path if None.
            fan_prim_path: USD prim path of the fan object.
                           Defaults to grasp_config.fan_prim_path if None.
            ground_truth_prim_path: USD prim path of the ground truth object.
                                    Defaults to grasp_config.ground_truth_prim_path if None.
            joint_config: Robot joint configuration. Defaults to RobotJointConfig().
            grasp_config: Grasp detection configuration including target frame alignment.
                          If None, uses default GraspDetectionConfig (no frame transformation).
            contact_report_paths: Optional list of prim paths to enable contact reports.
                                  Defaults to [robot_prim_path, fan_prim_path].
            cabinet_prim_path: USD prim path of the cabinet object.
                               Defaults to grasp_config.cabinet_prim_path if None.

        Returns:
            Configured PoseMonitor instance (call initialize() before use).

        Raises:
            ValueError: If any of the required parameters is empty or None.

        Example:
            >>> from grasp_config import GraspDetectionConfig
            >>> # Create config for fan (front=-Y, grasp axis=+X)
            >>> config = GraspDetectionConfig()
            >>> config.approach_axis = "-y"
            >>> config.grasp_axis = "+x"
            >>> monitor = PoseMonitor.create_default(
            ...     robot_prim_path="/World/WorkSpace/RS_M90E7A_Left",
            ...     fan_prim_path="/World/WorkSpace/Scene/Fan",
            ...     ground_truth_prim_path="/World/WorkSpace/Scene/GroundTruth",
            ...     grasp_config=config,
            ... )
            >>> monitor.initialize()
            >>> error = monitor.get_ee_to_fan_error()
        """
        # Use default config if not provided
        if grasp_config is None:
            grasp_config = GraspDetectionConfig()

        def _clean_path(
            value: Optional[str], default_value: Optional[str]
        ) -> Optional[str]:
            if value is not None:
                value = str(value).strip()
            if not value:
                value = default_value
                if value is not None:
                    value = str(value).strip()
            return value if value else None

        # Resolve parameters with config defaults
        robot_prim_path = _clean_path(robot_prim_path, grasp_config.robot_prim_path)
        fan_prim_path = _clean_path(fan_prim_path, grasp_config.fan_prim_path)
        ground_truth_prim_path = _clean_path(
            ground_truth_prim_path, grasp_config.ground_truth_prim_path
        )
        cabinet_prim_path = _clean_path(
            cabinet_prim_path, grasp_config.cabinet_prim_path
        )

        grasp_config.robot_prim_path = robot_prim_path
        grasp_config.fan_prim_path = fan_prim_path
        grasp_config.ground_truth_prim_path = ground_truth_prim_path
        grasp_config.cabinet_prim_path = cabinet_prim_path

        if contact_report_paths is None:
            contact_report_paths = [robot_prim_path, fan_prim_path]

        # Create ArticulationObserver
        robot_observer = ArticulationObserver(
            prim_path=robot_prim_path,
            joint_config=joint_config,
        )

        # Create TargetObjects with grasp config (contains target_frame + handle offsets)
        fan_object = TargetObject(
            prim_path=fan_prim_path,
            grasp_config=grasp_config,
            pose_source="physx",
        )
        ground_truth_object = TargetObject(
            prim_path=ground_truth_prim_path,
            grasp_config=grasp_config,
            pose_source="xform",
        )

        # Create GraspDetectionStrategy with the config
        grasp_strategy = GraspDetectionStrategy(grasp_config)

        collision_dispatcher = CollisionDispatcher(
            whitelist_not_holding=grasp_config.collision_whitelist_not_holding,
            whitelist_holding=grasp_config.collision_whitelist_holding,
        )

        monitor = cls(
            robot_observer=robot_observer,
            fan_object=fan_object,
            ground_truth_object=ground_truth_object,
            grasp_strategy=grasp_strategy,
            contact_event_source=None,
        )
        monitor.collision_dispatcher = collision_dispatcher
        monitor.contact_event_source = ContactEventSource(
            target_paths=contact_report_paths,
            handler=monitor._on_contact_report,
        )
        return monitor

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the pose monitor.

        This initializes the underlying ArticulationObserver. Must be called
        after the simulation has started and the articulation is valid.
        Contact events are not started by default; call start_contact_events()
        to begin receiving contact reports.

        Args:
            physics_sim_view: Optional physics simulation view.
        """
        self.robot_observer.initialize(physics_sim_view)
        if self.fan_object is not None:
            self.fan_object.initialize()
        if self.ground_truth_object is not None:
            self.ground_truth_object.initialize()

    def start_contact_events(self) -> None:
        """
        Start receiving contact reports from the contact event source.

        Raises:
            ValueError: If contact_event_source is not set.
        """
        if self.contact_event_source is None:
            raise ValueError("contact_event_source is not set.")
        self.contact_event_source.start()

    def stop_contact_events(self) -> None:
        """
        Stop receiving contact reports from the contact event source.

        Raises:
            ValueError: If contact_event_source is not set.
        """
        if self.contact_event_source is None:
            raise ValueError("contact_event_source is not set.")
        self.contact_event_source.stop()



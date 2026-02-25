#!/usr/bin/env python3
"""
Configuration classes for grasp detection and target frame alignment.

This module provides configuration dataclasses for:
- PosePq: Lightweight pose representation (position + quaternion)
- ApproachFrameConfig: Defines target object coordinate frame alignment
- GraspDetectionConfig: Grasp detection thresholds and target frame settings

Example:
    >>> from grasp_config import GraspDetectionConfig, PosePq
    >>> 
    >>> # Create config for a fan (front=-Y, grasp axis=+X)
    >>> config = GraspDetectionConfig(
    ...     approach_axis="-y",
    ...     grasp_axis="+x",
    ... )
    >>> 
    >>> # Use in PoseMonitor.create_default()
    >>> monitor = PoseMonitor.create_default(..., grasp_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .array_backend import mathops as mo


__all__ = ["PosePq", "ApproachFrameConfig", "GraspDetectionConfig"]


# ---------------------------------------------------------------------------
# PosePq - Lightweight pose representation
# ---------------------------------------------------------------------------

@dataclass
class PosePq:
    """
    Lightweight pose representation using position and quaternion.

    This is a simple data class for representing 6-DOF poses without
    requiring Cortex framework dependencies.

    Attributes:
        p: 3D position vector (x, y, z), backend array/tensor.
        q: Quaternion in wxyz format (w, x, y, z), backend array/tensor.

    Example:
        >>> pose = PosePq(
        ...     p=np.array([1.0, 2.0, 3.0]),
        ...     q=np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
        ... )
        >>> T = pose.to_T()  # Get 4x4 transformation matrix
        >>> print(f"Position: {pose.p}, Orientation: {pose.q}")
    """

    p: Any  # 3D position
    q: Any  # quaternion (wxyz)
    def __post_init__(self) -> None:
        """Ensure arrays use the configured backend."""
        self.p = mo.asarray(self.p).reshape(3)
        self.q = mo.asarray(self.q).reshape(4)

    def to_T(self) -> Any:
        """
        Convert pose to 4x4 homogeneous transformation matrix.

        Returns:
            4x4 array/tensor representing the transformation matrix.
        """
        R = mo.quat_to_rot_matrix(self.q)
        T = mo.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.p
        return T


# ---------------------------------------------------------------------------
# ApproachFrameConfig
# ---------------------------------------------------------------------------

@dataclass
class ApproachFrameConfig:
    """
    Configuration for target object coordinate frame alignment.

    This defines how to transform from the object's native coordinate frame
    to a standardized "approach frame" that aligns with the end-effector's
    coordinate convention.

    The approach frame convention (matching typical EE frames):
    - X axis: approach/forward direction (EE approaches along this axis)
    - Y axis: grasp/lateral direction (gripper fingers open along this axis)
    - Z axis: up direction (derived from right-hand rule: Z = X × Y)

    Attributes:
        approach_axis: Which axis of the object corresponds to the approach
                       direction (EE's +X). Format: "+x", "-x", "+y", "-y", "+z", "-z".
                       Default is "+x" (identity, no transformation).
        grasp_axis: Which axis of the object corresponds to the grasp direction
                    (EE's +Y, gripper finger axis). Format: same as approach_axis.
                    Default is "+y" (identity, no transformation).

    Example:
        For a fan where front is -Y and left/right is X:
        >>> config = ApproachFrameConfig(approach_axis="-y", grasp_axis="+x")
    """

    approach_axis: str = "+y"
    grasp_axis: str = "-x"

    def __post_init__(self) -> None:
        """Validate axis specifications."""
        valid_axes = {"+x", "-x", "+y", "-y", "+z", "-z"}
        
        if self.approach_axis not in valid_axes:
            raise ValueError(
                f"approach_axis must be one of {valid_axes}, got '{self.approach_axis}'"
            )
        if self.grasp_axis not in valid_axes:
            raise ValueError(
                f"grasp_axis must be one of {valid_axes}, got '{self.grasp_axis}'"
            )
        
        # Check that approach and grasp are not on the same axis
        approach_base = self.approach_axis[-1]  # 'x', 'y', or 'z'
        grasp_base = self.grasp_axis[-1]
        if approach_base == grasp_base:
            raise ValueError(
                f"approach_axis and grasp_axis cannot be on the same axis: "
                f"approach={self.approach_axis}, grasp={self.grasp_axis}"
            )

    @classmethod
    def identity(cls) -> "ApproachFrameConfig":
        """
        Create an identity configuration (no transformation).

        Use this when the target object's native frame already matches
        the end-effector convention.

        Returns:
            ApproachFrameConfig with approach_axis="+x", grasp_axis="+y".
        """
        return cls(approach_axis="+x", grasp_axis="+y")


# ---------------------------------------------------------------------------
# GraspDetectionConfig
# ---------------------------------------------------------------------------

@dataclass
class GraspDetectionConfig:
    """
    Configuration for grasp detection strategy and target frame alignment.

    This class combines:
    1. Grasp detection thresholds (gripper joint position range, grasp zone distance)
    2. Target object frame alignment settings (via nested ApproachFrameConfig)
    3. Fan handle geometry for virtual handle TCP positions
    4. Project prim path defaults

    The target_frame attribute defines how to transform the target object's
    coordinate frame to align with the end-effector convention. If None,
    no transformation is applied.

    Handle offset is specified in the approach frame coordinate system:
    - X axis: approach direction (EE forward)
    - Y axis: grasp direction (gripper lateral, +Y = left handle, -Y = right handle)
    - Z axis: up direction

    Gripper joint convention (based on empirical testing):
    - Slider9 (left finger): ~+0.02m when holding
    - Slider10 (right finger): ~-0.02m when holding
    - Grip detection uses symmetric range:
      - Slider9:  grip_position_min <= value <= grip_position_max
      - Slider10: -grip_position_max <= value <= -grip_position_min

    Attributes:
        grip_position_min: Minimum joint position magnitude for holding detection.
                           Slider9 must be >= this, Slider10 must be <= -this.
                           Default 0.019m.
        grip_position_max: Maximum joint position magnitude for holding detection.
                           Slider9 must be <= this, Slider10 must be >= -this.
                           Default 0.021m.
        grasp_zone_min_m: Minimum EE-to-target distance (meters) to consider within grasp zone.
                          Default 0.0m.
        grasp_zone_max_m: Maximum EE-to-target distance (meters) to consider within grasp zone.
                          Default 0.05m.
        target_frame: Optional ApproachFrameConfig for target frame alignment.
                      If None, no coordinate transformation is applied.
        handle_y_offset: Distance from object center to each handle along Y axis (meters).
                         Left handle is at +Y, right handle is at -Y in approach frame.
                         Default 0.1m.
        handle_x_offset: Offset along approach axis X (meters).
                         Positive = in front of object center.
                         Default -0.015m.
        hold_confirm_frames: Number of consecutive frames that must satisfy the
                             instantaneous grasp conditions before reporting
                             holding. Frame-based only (no dt conversion).
                             Default 10.
        robot_prim_path: Default robot prim path for this project.
        fan_prim_path: Default fan prim path for this project.
        ground_truth_prim_path: Default ground truth prim path for this project.
        cabinet_prim_path: Default cabinet prim path for this project.
        collision_whitelist_not_holding: Collision whitelist rules when not holding.
                                         Each rule is a tuple of full path prefixes.
                                         Defaults to ignore fan <-> cabinet.
                                         Set to None or [] to disable the whitelist.
        collision_whitelist_holding: Collision whitelist rules when holding.
                                     Each rule is a tuple of full path prefixes.
                                     Defaults to ignore robot <-> fan.
                                     Set to None or [] to disable the whitelist.

    Example:
        >>> # Config for fan with custom grasp zone and handle offsets
        >>> config = GraspDetectionConfig(
        ...     approach_axis="-y",  # Fan's front is -Y
        ...     grasp_axis="+x",     # Fan's left/right is X
        ...     grasp_zone_max_m=0.03,
        ...     handle_y_offset=0.025,  # Handles 5cm apart total
        ... )
    """

    grip_position_min: float = 0.019
    grip_position_max: float = 0.021
    grasp_zone_min_m: float = 0.01415
    grasp_zone_max_m: float = 0.02415
    target_frame: Optional[ApproachFrameConfig] = field(default_factory=ApproachFrameConfig)
    handle_y_offset: float = 0.1
    handle_x_offset: float = -0.015
    hold_confirm_frames: int = 10
    robot_prim_path: Optional[str] = "/World/WorkSpace/RS_M90E7A_Left"
    fan_prim_path: Optional[str] = "/World/WorkSpace/Scene0903/Scene0903/tn__02_1_j8und3wXW0fz9/tn__FANASSY_RIGHT1_nEwC"
    ground_truth_prim_path: Optional[str] = "/World/WorkSpace/Scene0903/Scene0903/tn__01_1_j8icW3fhh0lW6cS/tn__FANASSY_RIGHT1_nEwC"
    cabinet_prim_path: Optional[str] = "/World/WorkSpace/Scene0903/Scene0903/tn__01_1_j8icW3fhh0lW6cS"
    collision_whitelist_not_holding: Optional[List[Tuple[str, ...]]] = None
    collision_whitelist_holding: Optional[List[Tuple[str, ...]]] = None

    def __post_init__(self) -> None:
        """
        Populate whitelist defaults from prim paths when not explicitly provided.

        - None means "auto": derive default rules from prim paths.
        - [] means "disable whitelist": report all collisions.
        """
        if self.collision_whitelist_not_holding is None:
            if self.cabinet_prim_path and self.fan_prim_path:
                self.collision_whitelist_not_holding = [
                    (self.cabinet_prim_path, self.fan_prim_path)
                ]
            else:
                self.collision_whitelist_not_holding = None

        if self.collision_whitelist_holding is None:
            if self.robot_prim_path and self.fan_prim_path:
                self.collision_whitelist_holding = [
                    (self.robot_prim_path, self.fan_prim_path)
                ]
            else:
                self.collision_whitelist_holding = None

    # -----------------------------------------------------------------------
    # Convenience properties for axis access
    # -----------------------------------------------------------------------

    @property
    def approach_axis(self) -> Optional[str]:
        """
        The approach axis of the target object.

        Returns None if target_frame is not set.
        """
        if self.target_frame is None:
            return None
        return self.target_frame.approach_axis

    @approach_axis.setter
    def approach_axis(self, value: str) -> None:
        """
        Set the approach axis.

        Creates a new ApproachFrameConfig if target_frame is None.
        """
        if self.target_frame is None:
            self.target_frame = ApproachFrameConfig(approach_axis=value)
        else:
            # Create new instance since dataclass is frozen-like
            self.target_frame = ApproachFrameConfig(
                approach_axis=value,
                grasp_axis=self.target_frame.grasp_axis,
            )

    @property
    def grasp_axis(self) -> Optional[str]:
        """
        The grasp axis of the target object.

        Returns None if target_frame is not set.
        """
        if self.target_frame is None:
            return None
        return self.target_frame.grasp_axis

    @grasp_axis.setter
    def grasp_axis(self, value: str) -> None:
        """
        Set the grasp axis.

        Creates a new ApproachFrameConfig if target_frame is None.
        """
        if self.target_frame is None:
            self.target_frame = ApproachFrameConfig(grasp_axis=value)
        else:
            # Create new instance since dataclass is frozen-like
            self.target_frame = ApproachFrameConfig(
                approach_axis=self.target_frame.approach_axis,
                grasp_axis=value,
            )


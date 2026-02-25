#!/usr/bin/env python3
"""Lightweight wrapper for accessing USD prim transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import re
import omni.usd
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import SingleXFormPrim
from pxr import Usd, UsdPhysics

from .array_backend import mathops as mo
from .grasp_config import PosePq

if TYPE_CHECKING:
    from .grasp_config import ApproachFrameConfig, GraspDetectionConfig


__all__ = ["TargetObject"]




# ---------------------------------------------------------------------------
# Helper functions for coordinate frame transformation
# ---------------------------------------------------------------------------

def _quat_xyzw_to_wxyz(quat: Any) -> Any:
    quat_arr = mo.asarray(quat)
    return quat_arr[..., (3, 0, 1, 2)]


def _prim_path_to_regex(path: str) -> str:
    if ".*" in path:
        escaped = re.escape(path)
        return "^" + escaped.replace("\\.\\*", ".*") + "$"
    if "*" in path:
        escaped = re.escape(path).replace("\\*", ".*")
        return "^" + escaped + "$"
    return "^" + re.escape(path) + "$"


def _resolve_template_prim_path(stage: Usd.Stage, prim_path: str) -> str:
    pattern = _prim_path_to_regex(prim_path)
    matcher = re.compile(pattern)
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if matcher.match(path):
            return path
    raise RuntimeError(f"Failed to find prim for expression: '{prim_path}'.")


def _iter_prims_under(stage: Usd.Stage, root_path: str):
    prefix = root_path.rstrip("/")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path == prefix or path.startswith(prefix + "/"):
            yield prim


# ---------------------------------------------------------------------------
# Pose sources
# ---------------------------------------------------------------------------

class PoseSource:
    def initialize(self) -> None:
        pass

    def get_world_pose(self) -> Tuple[Any, Any]:
        raise NotImplementedError


class XformPoseSource(PoseSource):
    def __init__(self, xform: SingleXFormPrim) -> None:
        self._xform = xform

    def get_world_pose(self) -> Tuple[Any, Any]:
        return self._xform.get_world_pose()


class PhysXPoseSource(PoseSource):
    def __init__(self, prim_path: str) -> None:
        self._prim_path = prim_path
        self._physics_sim_view = None
        self._root_physx_view = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        backend = "torch"
        try:
            from isaacsim.core.api.simulation_context.simulation_context import SimulationContext
        except Exception:
            SimulationContext = None
        if SimulationContext is not None:
            ctx = SimulationContext.instance()
            ctx_backend = getattr(ctx, "backend", None)
            if ctx_backend is not None:
                backend = str(ctx_backend).strip().lower()
        if backend not in ("torch", "numpy"):
            backend = "torch"

        self._physics_sim_view = physx.create_simulation_view(backend)
        self._physics_sim_view.set_subspace_roots("/")

        stage = omni.usd.get_context().get_stage()
        template_prim_path = _resolve_template_prim_path(stage, self._prim_path)

        root_prims = []
        articulation_prims = []
        for prim in _iter_prims_under(stage, template_prim_path):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                root_prims.append(prim)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_prims.append(prim)

        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self._prim_path}' for rigid objects. "
                    f"Located at: '{[p.GetPath().pathString for p in articulation_prims]}' under '{template_prim_path}'. "
                    "Disable the articulation root or use xform pose source."
                )

        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self._prim_path}'. "
                "Ensure the prim has 'USD RigidBodyAPI' applied or use xform pose source."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self._prim_path}'. "
                f"Found multiple '{[p.GetPath().pathString for p in root_prims]}' under '{template_prim_path}'."
            )

        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self._prim_path + root_prim_path[len(template_prim_path) :]
        root_prim_path_expr = root_prim_path_expr.replace(".*", "*")

        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr)
        if getattr(self._root_physx_view, "_backend", None) is None:
            raise RuntimeError(f"Failed to create rigid body at: {self._prim_path}. Please check PhysX logs.")

        self._initialized = True

    def get_world_pose(self) -> Tuple[Any, Any]:
        if not self._initialized or self._root_physx_view is None:
            raise RuntimeError("PhysX pose source is not initialized. Call initialize() first.")
        pose = self._root_physx_view.get_transforms()
        pos = pose[0, :3]
        quat = _quat_xyzw_to_wxyz(pose[0, 3:7])
        return pos, quat

def _axis_string_to_vector(axis: str) -> Any:
    """Convert axis string (e.g., "+x", "-y") to unit vector."""
    axis_vectors = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    return mo.asarray(axis_vectors[axis])


def _compute_approach_frame_rotation(approach_axis: str, grasp_axis: str) -> Any:
    """
    Compute the rotation matrix from object frame to approach frame.

    The approach frame has:
    - X axis = approach direction (from approach_axis)
    - Y axis = grasp direction (from grasp_axis)
    - Z axis = X × Y (right-hand rule)

    Args:
        approach_axis: Object axis that maps to approach frame +X.
        grasp_axis: Object axis that maps to approach frame +Y.

    Returns:
        3x3 rotation matrix R such that: orientation_approach = orientation_object @ R.T
    """
    # Get the object-frame vectors
    obj_approach = _axis_string_to_vector(approach_axis)  # Maps to +X
    obj_grasp = _axis_string_to_vector(grasp_axis)        # Maps to +Y
    
    # Compute up axis using right-hand rule: Z = X × Y
    obj_up = mo.cross(obj_approach, obj_grasp)
    obj_up = obj_up / mo.norm(obj_up)  # Normalize

    # Build rotation matrix
    # The rows of R are where each approach-frame axis comes from in object frame
    # R transforms object coords to approach coords:
    # Approach X (forward) <- obj_approach
    # Approach Y (grasp)   <- obj_grasp
    # Approach Z (up)      <- obj_up
    R = mo.vstack([obj_approach, obj_grasp, obj_up])
    return R


class TargetObject:
    """A simple wrapper around a USD prim for pose access.

    This class wraps an existing USD prim (specified by path) with a
    ``SingleXFormPrim`` and exposes methods for reading/writing the world pose
    and obtaining the 4x4 homogeneous transformation matrix. Pose retrieval
    can use either USD Xform or PhysX rigid body views.

    Optionally, a GraspDetectionConfig can be provided to configure:
    1. Approach frame transformation (via target_frame inside the config)
    2. Handle position computation for objects with graspable handles (e.g., fans)

    Args:
        prim_path: The USD prim path of an existing object in the scene.
        name: Optional friendly name. Defaults to the last segment of the path.
        grasp_config: Optional GraspDetectionConfig containing:
                      - target_frame: ApproachFrameConfig for coordinate transformation
                      - handle_y_offset, handle_x_offset: for handle position computation
        pose_source: Optional pose source selector. Use "physx" for PhysX rigid body
                     (default), or "xform" for USD Xform.
    
    Example:
        >>> from grasp_config import GraspDetectionConfig
        >>> config = GraspDetectionConfig(handle_y_offset=0.025)
        >>> fan = TargetObject(
        ...     prim_path="/World/Fan",
        ...     grasp_config=config,
        ... )
        >>> # Get handle poses
        >>> left_pose, right_pose = fan.get_handle_poses()
        >>> print(f"Left handle position: {left_pose.p}")
    """

    def __init__(
        self, 
        prim_path: str, 
        name: Optional[str] = None,
        grasp_config: Optional["GraspDetectionConfig"] = None,
        pose_source: Optional[Union[PoseSource, str]] = None,
    ) -> None:
        self._prim_path = prim_path
        self._name = name or prim_path.rsplit("/", 1)[-1]
        self._xform = SingleXFormPrim(prim_path=prim_path, name=self._name)
        self._grasp_config = grasp_config
        self._pose_source = self._resolve_pose_source(pose_source)
        
        # Pre-compute rotation matrix if approach frame is set (via grasp_config.target_frame)
        self._approach_rotation: Optional[Any] = None
        if grasp_config is not None and grasp_config.target_frame is not None:
            self._approach_rotation = _compute_approach_frame_rotation(
                grasp_config.target_frame.approach_axis,
                grasp_config.target_frame.grasp_axis,
            )

    @property
    def name(self) -> str:
        """The friendly name of this object."""
        return self._name

    @property
    def prim(self) -> Usd.Prim:
        """The underlying USD prim."""
        return self._xform.prim

    @property
    def prim_path(self) -> str:
        """The USD prim path."""
        return self._prim_path

    @property
    def grasp_config(self) -> Optional["GraspDetectionConfig"]:
        """The grasp detection configuration, if set."""
        return self._grasp_config

    @property
    def target_frame(self) -> Optional["ApproachFrameConfig"]:
        """The approach frame configuration (from grasp_config.target_frame), if set."""
        if self._grasp_config is None:
            return None
        return self._grasp_config.target_frame

    def initialize(self) -> None:
        """Initialize pose source (required for PhysX)."""
        self._pose_source.initialize()

    def get_raw_world_pose(self) -> Tuple[Any, Any]:
        """Get the object's world pose in its native frame (no transformation).

        Returns:
            A tuple (position, orientation) where position is a 3D vector
            and orientation is a quaternion in wxyz format.
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        return self._pose_source.get_world_pose()

    def get_world_pose(self) -> Tuple[Any, Any]:
        """Get the object's world pose (with approach frame transformation if set).

        If an approach_frame is configured, the returned orientation is
        transformed so that:
        - The object's approach axis aligns with +X (forward)
        - The object's grasp axis aligns with +Y (lateral)
        - The up axis (Z) is derived via right-hand rule

        Returns:
            A tuple (position, orientation) where position is a 3D vector
            and orientation is a quaternion in wxyz format.
            Backend depends on SimulationContext (numpy ndarray or torch Tensor).
        """
        position, orientation = self._pose_source.get_world_pose()
        
        if self._approach_rotation is None:
            return position, orientation
        
        # Apply approach frame transformation to orientation
        # R_world_object = rotation from object frame to world frame
        # R_approach = rotation from object frame to approach frame
        # R_world_approach = R_world_object @ R_approach^T
        R_world_object = mo.quat_to_rot_matrix(orientation)
        R_world_approach = R_world_object @ self._approach_rotation.T
        
        new_orientation = mo.rot_matrix_to_quat(R_world_approach)
        
        return position, new_orientation

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """Set the object's world pose.

        Note: This sets the raw pose without any transformation.

        Args:
            position: The 3D position (x, y, z) in world coordinates.
            orientation: The orientation quaternion in wxyz format.
        """
        self._xform.set_world_pose(position, orientation)

    def _resolve_pose_source(self, pose_source: Optional[Union[PoseSource, str]]) -> PoseSource:
        if pose_source is None:
            return PhysXPoseSource(self._prim_path)
        if isinstance(pose_source, str):
            key = pose_source.strip().lower()
            if key in ("physx", "rigidbody", "rigid_body"):
                return PhysXPoseSource(self._prim_path)
            if key in ("xform", "usd"):
                return XformPoseSource(self._xform)
            raise ValueError(f"Unknown pose_source: '{pose_source}'.")
        return pose_source

    def get_transform(self) -> Any:
        """Get the object's world pose as a 4x4 homogeneous transformation matrix.

        If an approach frame is configured, the returned matrix uses the
        transformed orientation.

        Returns:
            A 4x4 array/tensor representing the homogeneous transformation matrix.
        """
        position, orientation = self.get_world_pose()
        pose = PosePq(position, orientation)
        return pose.to_T()

    # -----------------------------------------------------------------------
    # Handle pose computation
    # -----------------------------------------------------------------------

    def get_handle_poses(self) -> Tuple[PosePq, PosePq]:
        """
        Get virtual left and right handle poses in world frame.

        The handle positions are computed from the object center using
        offsets from the grasp_config. The offsets are applied in the
        approach frame coordinate system:
        - X axis: approach direction (EE forward)
        - Y axis: grasp direction (+Y = left handle, -Y = right handle)
        - Z axis: up direction

        Both handles share the same orientation as the object (with approach
        frame transformation applied if configured).

        Returns:
            Tuple of (left_handle_pose, right_handle_pose) in world frame.
            Each is a PosePq with:
            - p: 3D position vector
            - q: quaternion in wxyz format

        Raises:
            ValueError: If grasp_config is not set.

        Example:
            >>> from grasp_config import GraspDetectionConfig
            >>> config = GraspDetectionConfig(handle_y_offset=0.025)
            >>> fan = TargetObject("/World/Fan", grasp_config=config)
            >>> left_pose, right_pose = fan.get_handle_poses()
            >>> print(f"Left handle position: {left_pose.p}")
            >>> print(f"Left handle orientation: {left_pose.q}")
        """
        if self._grasp_config is None:
            raise ValueError(
                "grasp_config is not set. Provide GraspDetectionConfig at initialization."
            )
        
        position, orientation = self.get_world_pose()
        R = mo.quat_to_rot_matrix(orientation)

        # Left handle: +Y offset in approach frame
        left_offset_local = mo.asarray([
            self._grasp_config.handle_x_offset,
            self._grasp_config.handle_y_offset,
            0.0,
        ])
        # Right handle: -Y offset in approach frame
        right_offset_local = mo.asarray([
            self._grasp_config.handle_x_offset,
            -self._grasp_config.handle_y_offset,
            0.0,
        ])

        left_handle_pos = position + R @ left_offset_local
        right_handle_pos = position + R @ right_offset_local

        # Both handles share the same orientation as the object
        left_pose = PosePq(left_handle_pos, orientation)
        right_pose = PosePq(right_handle_pos, orientation)

        return left_pose, right_pose

#!/usr/bin/env python3
"""
Array backend facade used by monitors.

This keeps numpy usage behind a minimal interface so the backend can be swapped
without changing monitor code.

Example:
    >>> from array_backend import mathops as mo
    >>> mo.set_device("cuda")
"""


import functools

import torch


__all__ = [
    "MathOps",
    "NumpyBackend",
    "TorchBackend",
    "mathops",
]


def _is_torch_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


def _is_numpy_array(value) -> bool:
    return type(value).__module__.startswith("numpy") and hasattr(value, "shape")


def _is_array_like(value) -> bool:
    return _is_torch_tensor(value) or _is_numpy_array(value)


def _iter_inputs(values):
    def _walk(sequence):
        for value in sequence:
            if value is None:
                continue
            yield value
            if _is_array_like(value):
                return True
            if isinstance(value, (list, tuple)):
                found = yield from _walk(value)
                if found:
                    return True
        return False

    yield from _walk(values)


def _dispatch(func):
    name = func.__name__

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        backend = self._select_backend_from_args(*args, **kwargs)
        return getattr(backend, name)(*args, **kwargs)

    return wrapper


def _get_simulation_context():
    try:
        from isaacsim.core.api.simulation_context.simulation_context import SimulationContext
    except Exception:
        return None
    return SimulationContext.instance()


class NumpyBackend:
    """Backend implementation that wraps numpy."""

    def __init__(self, np_module=None):
        if np_module is None:
            import numpy as np_module
        self._np = np_module
        self._dtype = None

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return getattr(self, "_dtype", None)
        if dtype is float:
            return self._np.float32
        return dtype

    def set_dtype(self, dtype):
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = self._resolve_dtype(dtype)

    def array(self, obj, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return self._np.array(obj)
        return self._np.array(obj, dtype=dtype)

    def asarray(self, obj, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return self._np.asarray(obj)
        return self._np.asarray(obj, dtype=dtype)

    def eye(self, n, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return self._np.eye(n)
        return self._np.eye(n, dtype=dtype)

    def cross(self, a, b):
        a_arr = self._np.asarray(a)
        b_arr = self._np.asarray(b)
        if a_arr.ndim == 0 or b_arr.ndim == 0:
            raise ValueError("cross expects inputs with last dimension 3; got scalar.")
        if a_arr.shape[-1] != 3 or b_arr.shape[-1] != 3:
            raise ValueError(
                f"cross expects inputs with last dimension 3; got {a_arr.shape} and {b_arr.shape}."
            )
        return self._np.cross(a_arr, b_arr, axisa=-1, axisb=-1, axisc=-1)

    def vstack(self, seq):
        return self._np.vstack(seq)

    def norm(self, a):
        return self._np.linalg.norm(a)

    def degrees(self, rad):
        return self._np.degrees(rad)

    def arccos(self, x):
        return self._np.arccos(x)

    def quat_diff_rad(self, a, b):
        np = self._np
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape
        shape = a.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)
        b = np.concatenate([b[:, :1], -b[:, 1:]], axis=-1)
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        mul = np.stack([w, x, y, z], axis=-1).reshape(shape)
        vec = mul.reshape(-1, 4)[:, 1:]
        return 2.0 * np.arcsin(np.clip(np.linalg.norm(vec, axis=-1), None, 1.0))

    def quat_to_rot_matrix(self, quat):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        return quat_to_rot_matrix(quat)

    def rot_matrix_to_quat(self, mat):
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        return rot_matrix_to_quat(mat)

    def clip(self, x, a_min, a_max):
        return self._np.clip(x, a_min, a_max)


class TorchBackend:
    """Backend implementation that wraps torch."""

    def __init__(self, *, device="cuda", dtype=None):
        self._device = device
        self._dtype = None
        self._dtype = self._resolve_dtype(dtype)

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return getattr(self, "_dtype", None)
        if dtype is float:
            return torch.float32
        return dtype

    def set_device(self, device):
        self._device = device

    def set_dtype(self, dtype):
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = self._resolve_dtype(dtype)

    def array(self, obj, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return torch.tensor(
                obj,
                device=self._device,
            )
        return torch.tensor(
            obj,
            dtype=dtype,
            device=self._device,
        )

    def asarray(self, obj, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return torch.as_tensor(
                obj,
                device=self._device,
            )
        return torch.as_tensor(
            obj,
            dtype=dtype,
            device=self._device,
        )

    def eye(self, n, dtype=None):
        dtype = self._resolve_dtype(dtype)
        if dtype is None:
            return torch.eye(
                n,
                device=self._device,
            )
        return torch.eye(
            n,
            dtype=dtype,
            device=self._device,
        )

    def cross(self, a, b):
        return torch.linalg.cross(a, b, dim=-1)

    def vstack(self, seq):
        return torch.vstack(seq)

    def norm(self, a):
        linalg = getattr(torch, "linalg", None)
        if linalg is not None and hasattr(linalg, "norm"):
            return linalg.norm(a)
        return torch.norm(a)

    def degrees(self, rad):
        if hasattr(torch, "rad2deg"):
            return torch.rad2deg(rad)
        return rad * (180.0 / torch.pi)

    def arccos(self, x):
        return torch.acos(x)

    def quat_diff_rad(self, a, b):
        from isaacsim.core.utils.torch.rotations import quat_diff_rad
        return quat_diff_rad(a, b)

    def quat_to_rot_matrix(self, quat):
        from isaacsim.core.utils.torch.rotations import quats_to_rot_matrices
        return quats_to_rot_matrices(quat)

    def rot_matrix_to_quat(self, mat):
        from isaacsim.core.utils.torch.rotations import rot_matrices_to_quats
        device = getattr(mat, "device", self._device)
        return rot_matrices_to_quats(mat, device=device)

    def clip(self, x, a_min, a_max):
        if a_min is None and a_max is None:
            return x
        if a_min is None:
            return torch.clamp_max(x, a_max)
        if a_max is None:
            return torch.clamp_min(x, a_min)
        return torch.clamp(x, a_min, a_max)


class MathOps:
    """Facade that forwards array ops to the active backend."""

    def __init__(self, backend=None, *, device="cpu", dtype=None):
        sim_context = _get_simulation_context()
        if sim_context is None:
            raise RuntimeError("SimulationContext is not initialized. Initialize it before creating MathOps.")

        self._device = self._normalize_device(device)
        self._dtype = dtype
        self._torch_backends = {}
        if backend is None:
            sim_backend = getattr(sim_context, "backend", None)
            sim_device = getattr(sim_context, "device", None)
            if sim_backend == "torch":
                selected_device = self._normalize_device(sim_device or "cpu")
                backend = TorchBackend(device=selected_device, dtype=dtype)
                self._device = selected_device
            elif sim_backend == "numpy":
                if sim_device is not None and "cuda" in str(sim_device).lower():
                    selected_device = self._normalize_device(sim_device)
                    backend = TorchBackend(device=selected_device, dtype=dtype)
                    self._device = selected_device
                else:
                    backend = NumpyBackend()
            else:
                raise ValueError(
                    f"Unsupported SimulationContext backend: {sim_backend}. "
                    "Provide a MathOps backend explicitly."
                )
        self._backend = backend
        self._numpy_backend = backend if isinstance(backend, NumpyBackend) else NumpyBackend()
        self._apply_config()
        if isinstance(self._backend, TorchBackend):
            device = getattr(self._backend, "_device", None)
            if device is not None:
                self._torch_backends[str(device)] = self._backend

    def _normalize_device(self, device):
        text = str(device).strip().lower()
        if text in ("gpu", "cuda"):
            return "cuda"
        return text or "cpu"

    def _select_backend(self, device, dtype):
        if str(device).startswith("cuda"):
            return TorchBackend(device=device, dtype=dtype)
        return NumpyBackend()

    def _get_torch_backend(self, device):
        device_key = self._normalize_device(device)
        backend = self._torch_backends.get(device_key)
        if backend is None:
            backend = TorchBackend(device=device_key, dtype=self._dtype)
            self._torch_backends[device_key] = backend
        return backend

    def _select_backend_from_args(self, *args, **kwargs):
        for value in _iter_inputs(list(args) + list(kwargs.values())):
            if _is_torch_tensor(value):
                return self._get_torch_backend(str(value.device))
            if _is_numpy_array(value):
                return self._numpy_backend
        return self._backend

    def _apply_config(self):
        if hasattr(self._backend, "set_device"):
            self._backend.set_device(self._device)
        if hasattr(self._backend, "set_dtype"):
            self._backend.set_dtype(self._dtype)
        if hasattr(self._numpy_backend, "set_dtype"):
            self._numpy_backend.set_dtype(self._dtype)

    def set_backend(self, backend):
        self._backend = backend
        if isinstance(backend, NumpyBackend):
            self._numpy_backend = backend
        self._apply_config()
        if isinstance(self._backend, TorchBackend):
            device = getattr(self._backend, "_device", None)
            if device is not None:
                self._torch_backends[str(device)] = self._backend

    def get_backend(self):
        return self._backend

    def set_device(self, device):
        self._device = self._normalize_device(device)
        if hasattr(self._backend, "set_device"):
            self._backend.set_device(self._device)

    @_dispatch
    def array(self, obj, dtype=None):
        pass

    @_dispatch
    def asarray(self, obj, dtype=None):
        pass

    @_dispatch
    def eye(self, n, dtype=None):
        pass

    @_dispatch
    def cross(self, a, b):
        pass

    @_dispatch
    def vstack(self, seq):
        pass

    @_dispatch
    def norm(self, a):
        pass

    @_dispatch
    def degrees(self, rad):
        pass

    @_dispatch
    def arccos(self, x):
        pass

    @_dispatch
    def quat_diff_rad(self, a, b):
        pass

    @_dispatch
    def quat_to_rot_matrix(self, quat):
        pass

    @_dispatch
    def rot_matrix_to_quat(self, mat):
        pass

    @_dispatch
    def clip(self, x, a_min, a_max):
        pass


_MATHOPS = None


def _get_mathops():
    """Return the process-wide MathOps instance, creating it lazily."""
    global _MATHOPS
    if _MATHOPS is None:
        _MATHOPS = MathOps()
    return _MATHOPS


class _MathOpsProxy:
    def __getattr__(self, name):
        return getattr(_get_mathops(), name)

    def __repr__(self):
        if _MATHOPS is None:
            return "<MathOpsProxy (uninitialized)>"
        return repr(_MATHOPS)


mathops = _MathOpsProxy()



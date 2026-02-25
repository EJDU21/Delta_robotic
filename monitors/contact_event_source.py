#!/usr/bin/env python3
"""
PhysX contact report source.

This module only subscribes to callbacks and forwards raw reports.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

import omni.physx
import omni.usd
from pxr import PhysxSchema


class ContactEventSource:
    def __init__(
        self,
        target_paths: Iterable[str],
        handler: Callable[[Any], None],
    ) -> None:
        self._target_paths: List[str] = list(target_paths)
        self._handler = handler
        self._subscription: Optional[Any] = None

    def set_target_paths(self, target_paths: Iterable[str]) -> None:
        self._target_paths = list(target_paths)
        if self._subscription is not None:
            self.enable_contact_reports()

    def enable_contact_reports(self) -> None:
        stage = omni.usd.get_context().get_stage()
        for path in self._target_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api = PhysxSchema.PhysxContactReportAPI(prim)
            api.CreateThresholdAttr().Set(0.0)

    def start(self) -> None:
        if self._subscription is not None:
            return
        self.enable_contact_reports()
        physx_interface = omni.physx.get_physx_interface()
        self._subscription = physx_interface.subscribe_physics_step_events(
            self._on_physics_step
        )

    def stop(self) -> None:
        if self._subscription is None:
            return
        unsubscribe = getattr(self._subscription, "unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe()
        self._subscription = None

    def _on_physics_step(self, dt: float) -> None:
        sim_interface = omni.physx.get_physx_simulation_interface()
        report = sim_interface.get_contact_report()
        if not report:
            return
        self._handler(report)

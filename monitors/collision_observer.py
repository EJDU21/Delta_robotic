#!/usr/bin/env python3
"""
Collision observation data structures.

This module hosts contact state representations that will be used by collision
dispatchers and trackers composed into PoseMonitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pxr import PhysicsSchemaTools


IllegalCollisionEvent = Dict[str, Any]


@dataclass
class IllegalCollisionReport:
    """
    Generic illegal-collision report.

    Attributes:
        occurred: 是否有至少一筆非法事件。
        events: 每一筆事件是一個 tuple(actor0, actor1, type)，例如：
            ("/World/Robot/...", "/World/Obstacle/...")
    """

    occurred: bool
    events: List[IllegalCollisionEvent]

    def __repr__(self) -> str:
        return f"IllegalCollisionReport(occurred={self.occurred}, events={self.events})"


class IllegalCollisionTracker:
    """
    Generic 版：
    - 只懂 path 與 whitelist，不懂「這是 robot / fan / 環境」。
    - 規則：
        * 先檢查 whitelist，如果任何一條 rule match -> 忽略這個 event
        * 否則視為「非法事件」，把整組 paths 存起來
    - 至於 holding / not holding、有沒有風扇，全都由外部決定
      要用幾個 tracker、給什麼 whitelist。
    """

    def __init__(self, whitelist: Optional[List[Tuple[str, ...]]] = None) -> None:
        # 每個 rule 是一個 tuple[str, ...]
        # 例如: ("/World/WorkSpace/RS_M90E7A_Left/", "/World/WorkSpace/FanRoot/")
        self._whitelist: List[Tuple[str, ...]] = whitelist or []
        self._buffer: List[IllegalCollisionEvent] = []

    # --------------- 內部 helper ----------------

    def _match_pattern(self, pattern: str, path: str) -> bool:
        """
        pattern 與 path 的比對規則：
        先用最直覺的 prefix：
            - pattern 被當作 subtree root
            - 只要 path.startswith(pattern) 就算 match

        之後你如果想改成「結尾不含斜線才是完全比對」、「支援 wildcard」，
        都只要改這一個函式就好了。
        """
        return path.startswith(pattern)

    def _tuple_matches(self, rule: Tuple[str, ...], paths: List[str]) -> bool:
        """
        判斷一條 whitelist rule 是否 cover 這個事件。

        - len(rule) == 1:
            只要 *任一* path match 這個 pattern -> 整個事件視為白名單。

        - len(rule) >= 2:
            規則是「rule 裡的 *每一個 pattern* 都必須至少被某個 path match 到」。
            也就是說：
                - (robot_root, fan_root) 這個 rule
                - 只有當事件裡同時出現了 robot 那邊的東西 *以及* fan 那邊的東西，
                  這條 rule 才算 match。
            這樣就可以表現出「robot & fan 在一起是白名單 pair」，robot+robot
            就不會被這條 rule 吃掉。
        """
        if not rule or not paths:
            return False

        # 只有一個 pattern：任一 path 符合即可
        if len(rule) == 1:
            pattern = rule[0]
            return any(self._match_pattern(pattern, p) for p in paths)

        # 多個 pattern：每個 pattern 都要在 paths 裡找到至少一個對應的 path
        for pattern in rule:
            if not any(self._match_pattern(pattern, p) for p in paths):
                return False
        return True

    # --------------- 對外介面 ----------------

    def process(self, *paths: str, event_type: str = "generic") -> None:
        """
        處理一個碰撞事件：

        - paths: 這次事件的所有 actor 路徑，通常會是兩個：
            process(actor0_path, actor1_path)
          但簽名不假設數量，以後你要塞 3 個、4 個也沒問題。

        邏輯：
            1) 如果符合任何 whitelist rule -> 直接 return（忽略）
            2) 否則 -> 視為 illegal，整個 tuple(paths) 丟進 buffer
        """
        # 濾掉 None / 空字串，保險一下
        paths = [p for p in paths if p]
        if not paths:
            return

        # 1) 先跑 whitelist
        for rule in self._whitelist:
            if self._tuple_matches(rule, paths):
                return

        # 2) 不在 whitelist -> 視為 illegal，組成 event dict 存入 buffer
        event: IllegalCollisionEvent = {
            "type": event_type,
            "actor0": paths[0] if len(paths) > 0 else None,
            "actor1": paths[1] if len(paths) > 1 else None,
            "actors": list(paths),
        }
        self._buffer.append(event)

    def consume(self) -> IllegalCollisionReport:
        """
        拿出目前累積的所有 illegal 事件並清空 buffer。
        """
        if not self._buffer:
            return IllegalCollisionReport(occurred=False, events=[])
        events = self._buffer
        self._buffer = []
        return IllegalCollisionReport(occurred=True, events=events)

    def reset(self) -> None:
        self._buffer.clear()


class CollisionDispatcher:
    """
    Whitelists are provided by the caller via config.
    高階 dispatcher：
    - 對外還是吃 whitelist rules, is_holding
    - 內部用 generic IllegalCollisionTracker 來累積「已經判定為非法」的事件
    - whitelist 與 phase 切換邏輯留在這一層
    """

    def __init__(
        self,
        whitelist_not_holding: Optional[List[Tuple[str, ...]]] = None,
        whitelist_holding: Optional[List[Tuple[str, ...]]] = None,
    ) -> None:
        # None means no whitelist (report all collisions).
        self._whitelist_by_phase: Dict[bool, List[Tuple[str, ...]]] = {
            False: list(whitelist_not_holding or []),
            True: list(whitelist_holding or []),
        }
        self._illegal_trackers: Dict[bool, IllegalCollisionTracker] = {
            phase: IllegalCollisionTracker(whitelist=rules)
            for phase, rules in self._whitelist_by_phase.items()
        }
    def process_contacts(self, contact_events, is_holding: bool) -> None:
        """
        Phase 1 (is_holding=False): report all events.
        Phase 2 (is_holding=True): ignore robot <-> fan contacts.
        """
        if not contact_events:
            return

        events = self._normalize_contact_events(contact_events)
        if not events:
            return

        tracker = self._illegal_trackers[bool(is_holding)]
        for path_a, path_b, event_type in events:
            tracker.process(path_a, path_b, event_type=event_type)

    def _normalize_contact_events(self, contact_events) -> List[Tuple[str, str, Any]]:
        if isinstance(contact_events, (list, tuple)) and contact_events:
            first = contact_events[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                normalized: List[Tuple[str, str, Any]] = []
                path_a = self._to_path(first[0])
                path_b = self._to_path(first[1])
                event_type: Any = first[2] if len(first) >= 3 else None
                if event_type is None:
                    event_type = "generic"
                if path_a and path_b:
                    normalized.append((path_a, path_b, event_type))
                    for item in contact_events[1:]:
                        if not isinstance(item, (list, tuple)) or len(item) < 2:
                            continue
                        path_a = self._to_path(item[0])
                        path_b = self._to_path(item[1])
                        event_type = item[2] if len(item) >= 3 else None
                        if event_type is None:
                            event_type = "generic"
                        if path_a and path_b:
                            normalized.append((path_a, path_b, event_type))
                    return normalized
        return self._extract_pairs_from_report(contact_events)

    def _extract_pairs_from_report(self, report) -> List[Tuple[str, str, Any]]:
        pairs: List[Tuple[str, str, Any]] = []
        try:
            buckets = list(report)
        except TypeError:
            return pairs
        for bucket in buckets:
            try:
                items = list(bucket)
            except TypeError:
                continue
            for item in items:
                pair = self._extract_pair_from_item(item)
                if pair is not None:
                    pairs.append(pair)
        return pairs

    def _extract_pair_from_item(self, item) -> Optional[Tuple[str, str, Any]]:
        if not hasattr(item, "actor0") or not hasattr(item, "actor1"):
            return None
        path_a = self._to_path(getattr(item, "actor0"))
        path_b = self._to_path(getattr(item, "actor1"))
        event_type: Any = getattr(item, "type", None)
        if event_type is None:
            event_type = "generic"
        if path_a and path_b:
            return (path_a, path_b, event_type)
        return None

    def _to_path(self, value: Any) -> Optional[str]:
        if not isinstance(value, int):
            print(f"[CollisionDispatcher] actor handle is not int: {value!r}")
            return None
        try:
            path = PhysicsSchemaTools.intToSdfPath(value)
        except Exception as exc:
            print(f"[CollisionDispatcher] intToSdfPath failed for {value!r}: {exc}")
            return None
        text = str(path)
        if not text.startswith("/"):
            print(
                f"[CollisionDispatcher] intToSdfPath returned non-path for {value!r}: {text!r}"
            )
            return None
        return text

    def consume_illegal_report(self) -> IllegalCollisionReport:
        occurred = False
        events: List[IllegalCollisionEvent] = []
        for tracker in self._illegal_trackers.values():
            report = tracker.consume()
            if report.occurred:
                occurred = True
                events.extend(report.events)
        return IllegalCollisionReport(occurred=occurred, events=events)

    def reset(self) -> None:
        for tracker in self._illegal_trackers.values():
            tracker.reset()

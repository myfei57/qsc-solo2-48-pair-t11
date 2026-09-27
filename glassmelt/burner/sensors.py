"""分区温度测点可信度看门狗。

按分区温度反馈调火力的前提是测点可信。热电偶常见三类坏法：断线/短路给出
越界值、接触不良或变送器故障导致读数跳变、信号冻死长期不变。看门狗每拍
对每个分区做三项检查（可信温区、变化率、冻死），任一不过就把该分区判为
不可信并闩锁，连续若干拍读数都正常后才恢复，避免坏点一抖就重新接管
火力。判据与原因随分配结果一起标出，现场能直接看到"这一区为什么被退回
均分"。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ..config import ZoneHeatLimits
from ..eventbus import Event, EventBus

#: 测点不可信的原因标记，随分配结果对外暴露。
REASON_MISSING = "missing"
REASON_NOT_A_NUMBER = "not_a_number"
REASON_OUT_OF_RANGE = "out_of_range"
REASON_RATE_JUMP = "rate_jump"
REASON_STUCK = "stuck"
REASON_LATCHED = "fault_latched"


@dataclass(frozen=True)
class ZoneSensorState:
    """一个分区测点当拍的可信度结论。"""

    zone: str
    trusted: bool
    temperature_c: float | None
    reason: str | None
    consecutive_faults: int
    consecutive_good: int
    age: int

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "trusted": self.trusted,
            "temperature_c": round(self.temperature_c, 3)
            if self.temperature_c is not None
            else None,
            "reason": self.reason,
            "consecutive_faults": self.consecutive_faults,
            "consecutive_good": self.consecutive_good,
            "age": self.age,
        }


class ZoneSensors:
    """逐分区温度测点的可信度看门狗。"""

    def __init__(
        self,
        clock,
        bus: EventBus,
        zones: tuple[str, ...],
        limits: ZoneHeatLimits,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._zones = tuple(zones)
        self._temperature: dict[str, float | None] = {zone: None for zone in self._zones}
        self._trusted: dict[str, bool] = {zone: False for zone in self._zones}
        self._reasons: dict[str, str | None] = {zone: None for zone in self._zones}
        self._fault_run: dict[str, int] = {zone: 0 for zone in self._zones}
        self._good_run: dict[str, int] = {zone: 0 for zone in self._zones}
        self._flat_run: dict[str, int] = {zone: 0 for zone in self._zones}
        self._age: dict[str, int] = {zone: 0 for zone in self._zones}
        self._lock = threading.RLock()

    def zones(self) -> tuple[str, ...]:
        return self._zones

    def inspect(
        self, readings: dict, setpoints_c: dict[str, float], dt: float
    ) -> dict[str, ZoneSensorState]:
        """对全部分区的当拍读数做可信度判定，返回逐分区结论。"""

        states: dict[str, ZoneSensorState] = {}
        transitions: list[tuple[str, bool, str | None]] = []
        with self._lock:
            for zone in self._zones:
                if zone not in readings:
                    raise KeyError(f"缺少分区测点读数: {zone}")
                if zone not in setpoints_c:
                    raise KeyError(f"缺少分区温度给定: {zone}")
                reason = self._check(zone, readings[zone], float(setpoints_c[zone]), float(dt))
                was_trusted = self._trusted[zone]
                if reason is None:
                    self._fault_run[zone] = 0
                    self._good_run[zone] += 1
                    now_trusted = was_trusted or (
                        self._good_run[zone] >= self._limits.fault_recover_samples
                    )
                    if not now_trusted:
                        reason = REASON_LATCHED
                else:
                    self._fault_run[zone] += 1
                    self._good_run[zone] = 0
                    now_trusted = False
                self._trusted[zone] = now_trusted
                self._reasons[zone] = reason
                self._age[zone] += 1
                states[zone] = ZoneSensorState(
                    zone=zone,
                    trusted=now_trusted,
                    temperature_c=self._temperature[zone],
                    reason=reason,
                    consecutive_faults=self._fault_run[zone],
                    consecutive_good=self._good_run[zone],
                    age=self._age[zone],
                )
                if now_trusted != was_trusted:
                    transitions.append((zone, now_trusted, reason))
        for zone, trusted, reason in transitions:
            self._bus.publish(
                Event.of(
                    "burner",
                    "burner",
                    "zone_sensor_trusted" if trusted else "zone_sensor_distrusted",
                    subject=zone,
                    severity="notice" if trusted else "warning",
                    reason=reason,
                    at=self._clock.now(),
                )
            )
        return states

    def _check(self, zone: str, raw: object, target_c: float, dt: float) -> str | None:
        """返回 None 表示本拍读数通过全部检查，否则返回坏点原因。"""

        if raw is None:
            self._temperature[zone] = None
            self._flat_run[zone] = 0
            return REASON_MISSING
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self._temperature[zone] = None
            self._flat_run[zone] = 0
            return REASON_NOT_A_NUMBER
        if not math.isfinite(value):
            self._temperature[zone] = None
            self._flat_run[zone] = 0
            return REASON_NOT_A_NUMBER
        if not self._limits.temp_plausible(value, target_c):
            self._temperature[zone] = value
            self._flat_run[zone] = 0
            return REASON_OUT_OF_RANGE
        previous = self._temperature[zone]
        if previous is None:
            self._flat_run[zone] = 0
        elif value == previous:
            self._flat_run[zone] += 1
        else:
            self._flat_run[zone] = 0
        if (
            self._trusted[zone]
            and previous is not None
            and not self._limits.rate_ok(value - previous, dt)
        ):
            self._temperature[zone] = value
            return REASON_RATE_JUMP
        if self._trusted[zone] and self._flat_run[zone] >= self._limits.stuck_samples:
            self._temperature[zone] = value
            return REASON_STUCK
        self._temperature[zone] = value
        return None

    def trusted_zones(self) -> list[str]:
        with self._lock:
            return [zone for zone in self._zones if self._trusted[zone]]

    def state_of(self, zone: str) -> ZoneSensorState:
        with self._lock:
            return ZoneSensorState(
                zone=zone,
                trusted=self._trusted[zone],
                temperature_c=self._temperature[zone],
                reason=self._reasons[zone],
                consecutive_faults=self._fault_run[zone],
                consecutive_good=self._good_run[zone],
                age=self._age[zone],
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "zones": {
                    zone: self.state_of(zone).as_dict() for zone in self._zones
                },
                "trusted_count": sum(1 for zone in self._zones if self._trusted[zone]),
                "zone_count": len(self._zones),
            }

"""按分区温度反馈分配火力。

原来"整窑一起加"的问题在于断面温差：一侧已经过热，另一侧还欠火。分配器
在总热负荷预算固定的前提下，以均分做基准，再按各区温度偏差做零和微调
（欠火区加多少、过热区就减多少），并把结果投影到每个燃烧器的火力份额
上下限内——投影用注水算法保证总和精确等于预算，**总热负荷绝不超标**。

测点不可信的分区不参与反馈：积分冻结、份额回到均分，由其余可信区消化
微调量；所有分区都不可信时整窑退回纯均分。坏点因此不可能把整窑带偏，
退回的分区与原因在结果里明确标出。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import ZoneHeatLimits
from ..eventbus import Event, EventBus
from .sensors import ZoneSensors


@dataclass(frozen=True)
class ZoneHeat:
    """一个燃烧器分区当拍的火力结算。"""

    zone: str
    heat: float
    share: float
    setpoint_c: float
    temperature_c: float | None
    error_c: float | None
    trusted: bool
    fallback: bool
    saturated: bool
    reason: str | None

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "heat": round(self.heat, 3),
            "share": round(self.share, 5),
            "setpoint_c": round(self.setpoint_c, 3),
            "temperature_c": round(self.temperature_c, 3)
            if self.temperature_c is not None
            else None,
            "error_c": round(self.error_c, 3) if self.error_c is not None else None,
            "trusted": self.trusted,
            "fallback": self.fallback,
            "saturated": self.saturated,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeatAllocation:
    """一次全窑火力分配结果。"""

    zones: tuple[ZoneHeat, ...]
    total_heat: float
    budget: float
    equal_share: float
    trusted_count: int
    fallback_count: int
    fallback_all: bool
    cross_section_spread_c: float | None
    at: float

    @property
    def budget_ok(self) -> bool:
        """分配总热负荷是否严格不超过预算。"""

        return self.total_heat <= self.budget + 1e-7

    def as_dict(self) -> dict:
        return {
            "zones": [zone.as_dict() for zone in self.zones],
            "total_heat": round(self.total_heat, 3),
            "budget": round(self.budget, 3),
            "budget_ok": self.budget_ok,
            "equal_share": round(self.equal_share, 5),
            "trusted_count": self.trusted_count,
            "fallback_count": self.fallback_count,
            "fallback_all": self.fallback_all,
            "cross_section_spread_c": round(self.cross_section_spread_c, 3)
            if self.cross_section_spread_c is not None
            else None,
            "at": round(self.at, 3),
        }


class ZoneHeatAllocator:
    """温度反馈的分区火力分配器。"""

    def __init__(
        self,
        clock,
        bus: EventBus,
        sensors: ZoneSensors,
        limits: ZoneHeatLimits,
        default_setpoint_c: float | None = None,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._sensors = sensors
        self._limits = limits
        self._default_setpoint_c = (
            float(default_setpoint_c) if default_setpoint_c is not None else None
        )
        self._zones = sensors.zones()
        self._integral: dict[str, float] = {zone: 0.0 for zone in self._zones}
        self._share: dict[str, float] = {
            zone: 1.0 / len(self._zones) for zone in self._zones
        }
        self._last: HeatAllocation | None = None
        self._history: list[dict] = []
        self._lock = threading.RLock()

    def allocate(
        self,
        readings: dict,
        total_heat: float,
        dt: float,
        setpoints_c: dict[str, float] | None = None,
    ) -> HeatAllocation:
        """按当拍测点读数把总热负荷分配到各分区。"""

        budget = float(total_heat)
        if budget < 0.0:
            raise ValueError("总热负荷不得为负")
        step_dt = max(1e-6, float(dt))
        targets = self._normalize_setpoints(setpoints_c)
        states = self._sensors.inspect(readings, targets, step_dt)

        with self._lock:
            n = len(self._zones)
            equal = 1.0 / n
            trusted = self._sensors.trusted_zones()
            fallback_all = not trusted
            errors = {
                zone: (targets[zone] - states[zone].temperature_c)
                if states[zone].trusted
                else 0.0
                for zone in self._zones
            }

            desired: dict[str, float] = {}
            increments: dict[str, float] = {zone: 0.0 for zone in self._zones}
            for zone in self._zones:
                if states[zone].trusted:
                    increment = errors[zone] * self._limits.gain_i * step_dt
                    increments[zone] = increment
                    state = self._integral[zone] + increment
                    self._integral[zone] = max(
                        -self._limits.integral_trim_cap,
                        min(self._limits.integral_trim_cap, state),
                    )
                    trim = errors[zone] * self._limits.gain_p + self._integral[zone]
                    desired[zone] = equal + trim
                else:
                    desired[zone] = equal

            if not fallback_all:
                # 可信区微调零和：仅在可信区内部重新分配，坏点退回均分后不动。
                trim_sum = sum(desired[zone] - equal for zone in trusted)
                for zone in trusted:
                    desired[zone] -= trim_sum / len(trusted)

            lowers, uppers = self._rate_bounds(step_dt, trusted)
            shares, projected = self._project(desired, lowers, uppers, set(trusted))
            for zone in self._zones:
                if states[zone].trusted:
                    self._anti_windup(zone, desired[zone], shares[zone], increments[zone])
            heats = {zone: budget * shares[zone] for zone in self._zones}

            zone_results = tuple(
                ZoneHeat(
                    zone=zone,
                    heat=heats[zone],
                    share=shares[zone],
                    setpoint_c=targets[zone],
                    temperature_c=states[zone].temperature_c,
                    error_c=errors[zone] if states[zone].trusted else None,
                    trusted=states[zone].trusted,
                    fallback=not states[zone].trusted,
                    saturated=projected[zone],
                    reason=states[zone].reason,
                )
                for zone in self._zones
            )
            trusted_temps = [
                states[zone].temperature_c
                for zone in trusted
                if states[zone].temperature_c is not None
            ]
            spread = (
                max(trusted_temps) - min(trusted_temps)
                if len(trusted_temps) >= 2
                else None
            )
            allocation = HeatAllocation(
                zones=zone_results,
                total_heat=sum(heats.values()),
                budget=budget,
                equal_share=equal,
                trusted_count=len(trusted),
                fallback_count=n - len(trusted),
                fallback_all=fallback_all,
                cross_section_spread_c=spread,
                at=self._clock.now(),
            )
            self._share = dict(shares)
            self._last = allocation
            self._history.append(allocation.as_dict())
            if len(self._history) > 200:
                del self._history[: len(self._history) - 200]
            payload = allocation.as_dict()

        self._bus.publish(
            Event.of(
                "burner",
                "burner",
                "zone_heat_allocated",
                subject="zone_heat",
                severity="warning" if fallback_all else "notice",
                total_heat=payload["total_heat"],
                budget=payload["budget"],
                budget_ok=payload["budget_ok"],
                trusted_count=len(trusted),
                fallback_count=n - len(trusted),
                fallback_all=fallback_all,
                cross_section_spread_c=payload["cross_section_spread_c"],
            )
        )
        return allocation

    def _normalize_setpoints(self, setpoints_c: dict | None) -> dict[str, float]:
        if setpoints_c is None:
            if self._default_setpoint_c is None:
                raise ValueError("未提供分区温度给定，且分配器没有默认温度给定")
            return {zone: self._default_setpoint_c for zone in self._zones}
        if not isinstance(setpoints_c, dict):
            raise TypeError("分区温度给定必须是按分区名的映射")
        missing = [zone for zone in self._zones if zone not in setpoints_c]
        if missing:
            raise KeyError(f"缺少分区温度给定: {', '.join(missing)}")
        return {zone: float(setpoints_c[zone]) for zone in self._zones}

    def _rate_bounds(
        self, dt: float, trusted: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """叠加单拍变化率限幅后的份额盒子。

        退回均分的坏点不做变化率限制：坏点必须当拍就回到均分，不能沿着
        旧份额慢慢爬，更不能用它的盒子把可信区挤偏。
        """

        step = self._limits.share_max_step * max(1.0, dt)
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}
        equal_pin = min(self._limits.share_max, max(self._limits.share_min, 1.0 / len(self._zones)))
        for zone in self._zones:
            if zone not in trusted:
                # 坏点钉死在均分：可信区顶到边界时也不允许把微调量溢到坏点上。
                lowers[zone] = equal_pin
                uppers[zone] = equal_pin
                continue
            lowers[zone] = max(self._limits.share_min, self._share[zone] - step)
            uppers[zone] = min(self._limits.share_max, self._share[zone] + step)
            if lowers[zone] > uppers[zone]:
                lowers[zone] = self._limits.share_min
                uppers[zone] = self._limits.share_max
        return lowers, uppers

    def _project(
        self,
        desired: dict[str, float],
        lowers: dict[str, float],
        uppers: dict[str, float],
        trusted: set[str],
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """把期望份额注水投影到盒子约束且总和恒为 1。

        各分区份额必须在 ``[share_min, share_max]`` 与单拍变化率限幅之内，
        全窑份额之和恒为 1（总热负荷因此精确等于预算，绝不超标）。退回
        均分的坏点盒子被钉在均分点上，投影过程不会改动它们。返回投影后的
        份额与每个可信区是否被边界顶住的标记。
        """

        n = len(self._zones)
        low_sum = sum(lowers.values())
        high_sum = sum(uppers.values())
        if low_sum > 1.0 + 1e-9 or high_sum < 1.0 - 1e-9:
            raise ValueError("分区火力份额约束无法满足总热负荷为 100%，请检查上下限配置")

        shares = {zone: min(uppers[zone], max(lowers[zone], desired[zone])) for zone in self._zones}
        for _ in range(200):
            total = sum(shares.values())
            if abs(total - 1.0) <= 1e-10:
                break
            if total < 1.0:
                movable = [
                    zone for zone in self._zones if shares[zone] < uppers[zone] - 1e-12
                ]
                if not movable:
                    break
                room = sum(uppers[zone] - shares[zone] for zone in movable)
                add = 1.0 - total
                for zone in movable:
                    shares[zone] += add * (uppers[zone] - shares[zone]) / room
            else:
                movable = [
                    zone for zone in self._zones if shares[zone] > lowers[zone] + 1e-12
                ]
                if not movable:
                    break
                room = sum(shares[zone] - lowers[zone] for zone in movable)
                drop = total - 1.0
                for zone in movable:
                    shares[zone] -= drop * (shares[zone] - lowers[zone]) / room
        saturated = {
            zone: zone in trusted
            and (
                shares[zone] <= lowers[zone] + 1e-9
                or shares[zone] >= uppers[zone] - 1e-9
            )
            for zone in self._zones
        }
        return shares, saturated

    def _anti_windup(self, zone: str, desired: float, projected: float, increment: float) -> None:
        """条件反积分：被投影顶住且微调方向仍顶边界时，撤销本拍积分增量。"""

        if abs(desired - projected) <= 1e-9 or increment == 0.0:
            return
        if (desired > projected and increment > 0.0) or (
            desired < projected and increment < 0.0
        ):
            self._integral[zone] = max(
                -self._limits.integral_trim_cap,
                min(self._limits.integral_trim_cap, self._integral[zone] - increment),
            )

    def last_allocation(self) -> HeatAllocation | None:
        with self._lock:
            return self._last

    def shares(self) -> dict:
        with self._lock:
            return dict(self._share)

    def history(self, limit: int = 5) -> list[dict]:
        with self._lock:
            return self._history[-limit:]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limits": {
                    "share_min": self._limits.share_min,
                    "share_max": self._limits.share_max,
                    "share_max_step": self._limits.share_max_step,
                    "gain_p": self._limits.gain_p,
                    "gain_i": self._limits.gain_i,
                    "integral_trim_cap": self._limits.integral_trim_cap,
                },
                "shares": {zone: round(value, 5) for zone, value in self._share.items()},
                "sensors": self._sensors.snapshot(),
                "last": self.last_allocation().as_dict() if self.last_allocation() else None,
            }

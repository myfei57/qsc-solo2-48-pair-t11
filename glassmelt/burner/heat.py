"""分区火力按温度反馈分配。

旧做法只把窑体分区对应到燃烧器，各区火力由人工给定，断面温差一大只能整窑
一起加负荷，结果一侧过热另一侧还欠。分配器在**总热负荷预算不变且不许超**
的前提下，按每区温度偏差在分区之间做零和再分配：偏冷的区多拿一点、
偏热的区少拿一点，总和保持在人工给定的总负荷上。

三条安全前提：

1. **总热负荷不许超**：总燃气流量永远不超过人工负荷（并受燃气上限
   约束）。逐区上下限投影保证各区分量之和不超过预算；若爬坡限速使本拍
   凑不满预算，允许欠发并在结果里标明 ``under_delivered``，绝不为凑满
   预算而顶破单区上限；
2. **坏点退回均分**：测点被判坏的分区钉死在均分份额上，不参与偏差
   调节；可信区数量不足以做闭环（少于两个）时整窑均分，并标注退回原因；
3. **一个坏点不带偏整窑**：偏差只在可信区之间再分配，积分项只在可信、
   未饱和的区累计，并始终重新归零。

分配份额按燃气流量（``m3_h``）结算，同时给出占总热负荷的份额，便于
控制台按百分比展示。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import BurnerLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus
from .probe import ZoneProbes

#: 可信区少于该数量时不做分区闭环，整窑退回均分。
MIN_TRUSTED_ZONES = 2


@dataclass(frozen=True)
class ZoneHeat:
    """一个分区一拍的火力结算。"""

    zone: str
    flow_m3_h: float
    fraction: float
    trusted: bool
    setpoint_c: float
    temperature_c: float
    deviation_c: float
    integral: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "flow_m3_h": round(self.flow_m3_h, 2),
            "fraction": round(self.fraction, 5),
            "trusted": self.trusted,
            "setpoint_c": round(self.setpoint_c, 2),
            "temperature_c": round(self.temperature_c, 2),
            "deviation_c": round(self.deviation_c, 2),
            "integral": round(self.integral, 5),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeatAllocation:
    """整窑一拍的火力分配结果。"""

    zones: tuple[ZoneHeat, ...]
    total_flow_m3_h: float
    demand_m3_h: float
    equal_share_m3_h: float
    mode: str
    trusted_zones: tuple[str, ...]
    bad_zones: tuple[str, ...]
    under_delivered: bool
    section_spread_c: float
    section_mean_c: float
    at: float

    def zone(self, name: str) -> ZoneHeat:
        return next(item for item in self.zones if item.zone == name)

    def as_dict(self) -> dict:
        return {
            "zones": [item.as_dict() for item in self.zones],
            "total_flow_m3_h": round(self.total_flow_m3_h, 2),
            "demand_m3_h": round(self.demand_m3_h, 2),
            "equal_share_m3_h": round(self.equal_share_m3_h, 3),
            "mode": self.mode,
            "trusted_zones": list(self.trusted_zones),
            "bad_zones": list(self.bad_zones),
            "under_delivered": self.under_delivered,
            "section_spread_c": round(self.section_spread_c, 2),
            "section_mean_c": round(self.section_mean_c, 2),
            "at": round(self.at, 3),
        }


class HeatAllocator:
    """按分区温度反馈分配火力，维持总热负荷预算。"""

    def __init__(
        self,
        bus: EventBus,
        limits: BurnerLimits,
        probes: ZoneProbes,
        reference_c: float,
    ) -> None:
        self._bus = bus
        self._limits = limits
        self._probes = probes
        zones = limits.zone_names()
        self._zones = tuple(zones)
        self._count = len(zones)
        self._setpoints: dict[str, float] = {
            zone: float(reference_c) for zone in zones
        }
        self._integral: dict[str, float] = {zone: 0.0 for zone in zones}
        self._previous: dict[str, float] | None = None
        self._allocations = 0
        self._equal_fallbacks = 0
        self._lock = threading.RLock()

    @property
    def zones(self) -> tuple[str, ...]:
        return self._zones

    def set_setpoints(self, setpoints: dict) -> None:
        """设置各分区目标温度，必须且只能覆盖全部分区。"""

        missing = [zone for zone in self._zones if zone not in setpoints]
        unknown = [zone for zone in setpoints if zone not in self._zones]
        if missing or unknown:
            raise InterlockError(
                f"分区给定值必须覆盖全部分区（缺 {missing}，多 {unknown}）",
                component="burner",
            )
        values = {zone: float(setpoints[zone]) for zone in self._zones}
        for zone, value in values.items():
            if not self._limits.zone_temp_ok(value):
                raise InterlockError(
                    f"分区 {zone} 给定值 {value} 超出可信量程", component="burner"
                )
        with self._lock:
            self._setpoints = values

    def setpoint(self, zone: str) -> float:
        with self._lock:
            return self._setpoints[zone]

    def allocate(
        self,
        readings: dict,
        total_load_fraction: float,
        dt: float,
        now: float,
    ) -> HeatAllocation:
        """喂入一拍测点读数与总负荷给定，结算各区火力。

        ``total_load_fraction`` 为人工给定的整窑负荷（0~1，相对燃气上限），
        分配器只在该预算内做分区再分配，任何情况下总流量都不超过它。
        """

        load = max(0.0, min(1.0, float(total_load_fraction)))
        step = max(0.001, float(dt))
        demand = load * self._limits.gas_max_m3_h
        equal = demand / self._count

        trusted_temps, _ = self._probes.observe(readings, now)
        trusted = tuple(zone for zone in self._zones if zone in trusted_temps)
        bad = tuple(zone for zone in self._zones if zone not in trusted_temps)

        if len(trusted) < MIN_TRUSTED_ZONES:
            mode = self._fallback_reason(trusted)
            outcome = self._equal_allocation(
                demand, equal, trusted, bad, trusted_temps, mode, now
            )
        else:
            outcome = self._closed_loop(
                demand, equal, trusted, bad, trusted_temps, step, now
            )

        with self._lock:
            self._allocations += 1
            self._previous = {item.zone: item.flow_m3_h for item in outcome.zones}
        self._publish(outcome)
        return outcome

    def force_equal_share(
        self,
        total_load_fraction: float,
        dt: float,
        now: float,
        reason: str = "manual_override",
    ) -> HeatAllocation:
        """手动强制整窑均分火力（测点检修、操作员接管时用）。"""

        self._probes.force_equal_share(reason)
        return self.allocate({}, total_load_fraction, dt, now)

    def resume_auto(self, readings: dict, total_load_fraction: float, dt: float, now: float):
        """退出手动均分，按送来的测点读数恢复分区自动闭环。"""

        self._probes.resume_auto()
        return self.allocate(readings, total_load_fraction, dt, now)

    def _fallback_reason(self, trusted: tuple[str, ...]) -> str:
        """无可信区时，区分"手动接管均分"与"测点全坏退回均分"。"""

        if self._probes.suppressed:
            return "manual_override"
        if trusted:
            return "too_few_trusted"
        reasons = set(self._probes.bad_reasons().values())
        if reasons and reasons <= {"manual_override"}:
            return "manual_override"
        return "all_probes_bad"

    def _equal_allocation(
        self,
        demand: float,
        equal: float,
        trusted: tuple[str, ...],
        bad: tuple[str, ...],
        trusted_temps: dict[str, float],
        mode: str,
        now: float,
    ) -> HeatAllocation:
        """全窑均分：坏点退回均分或可信区不足时走这里。"""

        with self._lock:
            for zone in self._zones:
                self._integral[zone] = 0.0
            self._equal_fallbacks += 1
        share = 1.0 / self._count
        zones = tuple(
            self._build_zone(
                zone,
                equal,
                share,
                zone in trusted,
                trusted_temps,
                0.0,
                mode if zone in bad else "equal_share_fallback",
            )
            for zone in self._zones
        )
        return self._finalize(
            zones, demand, demand, equal, mode, trusted, bad, now
        )

    def _closed_loop(
        self,
        demand: float,
        equal: float,
        trusted: tuple[str, ...],
        bad: tuple[str, ...],
        trusted_temps: dict[str, float],
        dt: float,
        now: float,
    ) -> HeatAllocation:
        """可信区零和再分配，坏区钉死均分。"""

        deadband = self._limits.zone_temp_deadband_c
        gain = self._limits.heat_gain
        ki = self._limits.heat_integral_ki
        cap = self._limits.heat_integral_cap_frac
        lo = self._limits.zone_heat_min_frac
        hi = self._limits.zone_heat_max_frac

        errors = {
            zone: self._deadband_error(
                self._setpoints[zone] - trusted_temps[zone], deadband
            )
            for zone in trusted
        }
        with self._lock:
            for zone in trusted:
                if errors[zone] != 0.0:
                    self._integral[zone] += ki * errors[zone] * dt
                self._integral[zone] = _clip(self._integral[zone], -cap, cap)
            self._recenter_integral(trusted)
            integrals = dict(self._integral)

        # 偏差修正先做零和（冷区想加、热区想减，份额只在它们之间转移），
        # 再裁剪到逐区上下限；裁剪产生的盈余继续转移给同向未饱和区。可信区
        # 的基线是"扣掉坏区均分之后的剩余预算"，保证中性区不被推动、
        # 全窑份额总和恒为 1。
        raw_correction = {
            zone: gain * errors[zone] + integrals[zone] for zone in trusted
        }
        corrections = self._zero_sum_corrections(raw_correction)
        bad_fraction = len(bad) / self._count
        trusted_base = (1.0 - bad_fraction) / len(trusted)
        shares = {zone: 1.0 / self._count for zone in bad}
        for zone in trusted:
            shares[zone] = _clip(
                trusted_base + corrections[zone], lo, hi
            )

        # 抗积分饱和：已被夹在上下限的区，这一拍积分不许继续往饱和方向涨。
        with self._lock:
            for zone in trusted:
                pushing_high = shares[zone] >= hi - 1e-12 and errors[zone] > 0.0
                pushing_low = shares[zone] <= lo + 1e-12 and errors[zone] < 0.0
                if pushing_high or pushing_low:
                    self._integral[zone] = integrals[zone] - ki * errors[zone] * dt

        raw_flows = {zone: shares[zone] * demand for zone in self._zones}
        flows = self._rate_limited(raw_flows, dt)
        flows = self._settle_to_budget(
            flows, raw_flows, demand, tuple(trusted), dt
        )
        total_flow = sum(flows.values())

        final_shares = {
            zone: flows[zone] / demand if demand > 0.0 else 1.0 / self._count
            for zone in self._zones
        }
        zones = tuple(
            self._build_zone(
                zone,
                flows[zone],
                final_shares[zone],
                True,
                trusted_temps,
                integrals[zone],
                "auto",
            )
            for zone in trusted
        ) + tuple(
            self._build_zone(
                zone,
                flows[zone],
                final_shares[zone],
                False,
                trusted_temps,
                0.0,
                "equal_share_bad_probe",
            )
            for zone in bad
        )
        zones = tuple(sorted(zones, key=lambda item: item.zone))
        return self._finalize(
            zones, total_flow, demand, equal, "auto", trusted, bad, now
        )

    def _recenter_integral(self, trusted: tuple[str, ...]) -> None:
        """把可信区积分整体平移到零和，防止积分总量漂走总负荷（持锁调用）。"""

        if not trusted:
            return
        mean = sum(self._integral[zone] for zone in trusted) / len(trusted)
        cap = self._limits.heat_integral_cap_frac
        for zone in trusted:
            self._integral[zone] = _clip(
                self._integral[zone] - mean, -cap, cap
            )

    def _zero_sum_corrections(self, correction: dict[str, float]) -> dict[str, float]:
        """零和偏差修正，饱和区的盈余只转给同方向、仍有空间的区。

        输入是各区相对均分的期望修正（冷区为正、热区为负）。整体零和后，
        把顶到逐区上下限的修正夹平，夹下的量按需求比例摊给**同号**且未饱和
        的区；中性区修正为 0，既不接收也不输出，因而不会被推动。找不到同号
        接收区时盈余就地丢弃（该方向需求本就到此为止），最终总量由
        :meth:`_settle_to_budget` 兜回预算。
        """

        lo = self._limits.zone_heat_min_frac
        hi = self._limits.zone_heat_max_frac
        low_cap = 1.0 / self._count - lo
        high_cap = hi - 1.0 / self._count
        zones = list(correction)
        result = dict(correction)
        mean = sum(result.values()) / max(1, len(result))
        result = {zone: value - mean for zone, value in result.items()}
        for _ in range(len(zones) + 2):
            moved = False
            for zone in zones:
                value = result[zone]
                if value > high_cap:
                    excess = value - high_cap
                    result[zone] = high_cap
                    self._hand_off(result, zone, excess, positive=True,
                                   cap=high_cap)
                    moved = True
                elif value < -low_cap:
                    excess = -low_cap - value
                    result[zone] = -low_cap
                    self._hand_off(result, zone, excess, positive=False,
                                   cap=low_cap)
                    moved = True
            if not moved:
                break
        return result

    @staticmethod
    def _hand_off(
        result: dict[str, float],
        donor: str,
        amount: float,
        *,
        positive: bool,
        cap: float,
    ) -> None:
        """把 ``donor`` 裁下的 ``amount`` 按剩余需求摊给同号未饱和区。"""

        def remaining(zone: str) -> float:
            return cap - abs(result[zone])

        candidates = [
            zone
            for zone in result
            if zone != donor
            and ((result[zone] > 0.0) if positive else (result[zone] < 0.0))
            and remaining(zone) > 1e-12
        ]
        if not candidates:
            return  # 没有同号接收区：就地丢弃，总量由结算层兜回。
        weight_total = sum(remaining(zone) for zone in candidates)
        sign = 1.0 if positive else -1.0
        for zone in candidates:
            result[zone] += sign * amount * remaining(zone) / weight_total

    def _rate_limited(self, flows: dict[str, float], dt: float) -> dict[str, float]:
        """逐区爬坡限速，避免一拍猛增；首拍从均分起步不受限。"""

        with self._lock:
            previous = dict(self._previous) if self._previous is not None else None
        if previous is None:
            return dict(flows)
        step = (
            self._limits.zone_heat_ramp_frac_per_s
            * dt
            * self._limits.gas_max_m3_h
        )
        return {
            zone: _clip(
                flows[zone],
                previous.get(zone, flows[zone]) - step,
                previous.get(zone, flows[zone]) + step,
            )
            for zone in self._zones
        }

    def _settle_to_budget(
        self,
        flows: dict[str, float],
        target_flows: dict[str, float],
        demand: float,
        trusted: tuple[str, ...],
        dt: float,
    ) -> dict[str, float]:
        """爬坡限速后把总流量结算回预算，保持逐区爬坡与单区上下限。

        热区一拍减不到目标份额时会留出一块"热区仍多发"的量。这块量只从
        冷区本拍计划增加的火力里让出（冷区最多退回到上一拍），绝不推中性区
        去补；冷区让完仍凑不满预算就保留欠发。因此总流量只可能低于或等于
        预算，绝不超过预算。
        """

        with self._lock:
            previous = (
                dict(self._previous) if self._previous is not None else {}
            )
        if not previous:
            scaled = dict(flows)
            total = sum(scaled.values())
            if total > demand + 1e-9:
                factor = demand / total
                scaled = {zone: value * factor for zone, value in scaled.items()}
            return scaled

        step = (
            self._limits.zone_heat_ramp_frac_per_s
            * dt
            * self._limits.gas_max_m3_h
        )
        result = dict(flows)
        equal_flow = demand / self._count

        # 先把总流量结算到不超过预算：逐区爬坡后热区可能仍多发，先把冷区
        # 本拍计划增加的量全部让出（冷区最多退回到上一拍）；让完仍超，
        # 或想减火的热区本身没减到位，就保留欠发，绝不推中性区、绝不超发。
        total = sum(result.values())
        if total > demand + 1e-9:
            overrun = total - demand
            cold = [
                zone
                for zone in trusted
                if target_flows[zone] > equal_flow + 1e-12
                and result[zone] > previous.get(zone, result[zone])
            ]
            give = {zone: result[zone] - previous[zone] for zone in cold}
            capacity = sum(give.values())
            take = min(overrun, capacity)
            for zone, value in give.items():
                result[zone] -= take * value / max(1e-12, capacity)

        # 若还有欠发，只让想加火（目标份额高于均分）的冷区在各自爬坡上限
        # 与单区上限内补，补不满即欠发，中性区不动。
        for _ in range(12):
            total = sum(result.values())
            if total >= demand - 1e-9:
                break
            cap_flow = self._limits.zone_heat_max_frac * demand
            room = {}
            for zone in trusted:
                if target_flows[zone] <= equal_flow + 1e-12:
                    continue
                ceiling = min(
                    previous.get(zone, result[zone]) + step,
                    cap_flow,
                    target_flows[zone],
                )
                if ceiling - result[zone] > 1e-9:
                    room[zone] = ceiling - result[zone]
            if not room:
                break
            capacity = sum(room.values())
            take = min(demand - total, capacity)
            for zone, value in room.items():
                result[zone] += take * value / capacity

        # 兜底硬保护：任何残余超发整体等比压回预算。
        total = sum(result.values())
        if total > demand + 1e-9:
            factor = demand / total
            result = {zone: value * factor for zone, value in result.items()}
        return result

    def _build_zone(
        self,
        zone: str,
        flow: float,
        fraction: float,
        trusted: bool,
        trusted_temps: dict[str, float],
        integral: float,
        reason: str,
    ) -> ZoneHeat:
        temperature = (
            trusted_temps[zone]
            if trusted
            else self._probes.reading_value(zone)
        )
        setpoint = self._setpoints[zone]
        deviation = temperature - setpoint if trusted else 0.0
        return ZoneHeat(
            zone=zone,
            flow_m3_h=flow,
            fraction=fraction,
            trusted=trusted,
            setpoint_c=setpoint,
            temperature_c=temperature,
            deviation_c=deviation,
            integral=integral,
            reason=reason,
        )

    def _deadband_error(self, error_c: float, deadband: float) -> float:
        """死带：偏差落在带内不调节，避免在目标附近来回抖。"""

        if error_c > deadband:
            return error_c - deadband
        if error_c < -deadband:
            return error_c + deadband
        return 0.0

    def _finalize(
        self,
        zones: tuple[ZoneHeat, ...],
        total_flow: float,
        demand: float,
        equal: float,
        mode: str,
        trusted: tuple[str, ...],
        bad: tuple[str, ...],
        now: float,
    ) -> HeatAllocation:
        good_temps = [
            item.temperature_c for item in zones if item.zone in trusted
        ]
        if len(good_temps) >= 2:
            spread = max(good_temps) - min(good_temps)
            mean = sum(good_temps) / len(good_temps)
        else:
            spread, mean = 0.0, 0.0
        return HeatAllocation(
            zones=zones,
            total_flow_m3_h=total_flow,
            demand_m3_h=demand,
            equal_share_m3_h=equal,
            mode=mode,
            trusted_zones=trusted,
            bad_zones=bad,
            under_delivered=total_flow < demand - 1e-6,
            section_spread_c=spread,
            section_mean_c=mean,
            at=float(now),
        )

    def _publish(self, outcome: HeatAllocation) -> None:
        self._bus.publish(
            Event.of(
                "burner",
                "burner",
                "zone_heat_allocated",
                subject="zone_heat",
                severity="notice" if outcome.mode == "auto" else "warning",
                mode=outcome.mode,
                total_flow_m3_h=round(outcome.total_flow_m3_h, 2),
                demand_m3_h=round(outcome.demand_m3_h, 2),
                under_delivered=outcome.under_delivered,
                section_spread_c=round(outcome.section_spread_c, 2),
                bad_zones=list(outcome.bad_zones),
                zones={
                    item.zone: round(item.flow_m3_h, 2) for item in outcome.zones
                },
            )
        )

    def flows(self) -> dict[str, float]:
        with self._lock:
            if self._previous is None:
                return {}
            return dict(self._previous)

    def fallback_count(self) -> int:
        with self._lock:
            return self._equal_fallbacks

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "setpoints_c": {
                    zone: round(value, 2)
                    for zone, value in self._setpoints.items()
                },
                "integrals": {
                    zone: round(value, 5)
                    for zone, value in self._integral.items()
                },
                "flows_m3_h": {
                    zone: round(value, 2)
                    for zone, value in (self._previous or {}).items()
                },
                "allocations": self._allocations,
                "equal_fallbacks": self.fallback_count(),
            }


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

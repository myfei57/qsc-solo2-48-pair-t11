"""分区热电偶可信度判定。

断面温差调节只在"每区测点可信"这一前提成立。测点坏了（断线读出 NaN、
跳出物理量程、一拍到顶的尖峰、长时间卡死不动、与其余区显著背离）还照它
调火力，会把整窑带偏。判定器对每个分区独立维护状态，判坏的分区不参与
闭环、改吃均分火力；判定带确认计数与迟滞，避免一次跳变就在自动/均分
之间来回抖动。

判据分两类：

* **硬故障**：非数值、超出量程、相对上一拍跳变超限——一拍到顶立即判坏，
  不需要等待（尖峰当场拒绝）；
* **统计判坏**：与最近的其它可信测点之差持续超过共识带。真实断面温差是
  连续渐变的（相邻区温度接近，半侧整体偏冷时两侧各自成簇），只有断线
  漂移的坏点才会离所有其它测点都很远。至少要有 3 个其余可信测点时才启用，
  防止在真实双峰断面上误判。

恢复要求连续若干拍干净，避免坏点抖动反复投退；手动接管（测点检修）期间
判定被抑制、全窑保持均分，直到操作员显式恢复自动。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ..config import BurnerLimits
from ..eventbus import Event, EventBus

#: 判坏需要连续异常的拍数，恢复需要连续干净的拍数。
FAULT_CONFIRM_TICKS = 2
CLEAR_CONFIRM_TICKS = 3

#: 卡死判定时，读数差异小于该值视为"没动"。
FREEZE_EPSILON_C = 0.5

#: 其余可信测点少于该数量时，不启用最近邻一致性判据。
CONSENSUS_MIN_PEERS = 3


@dataclass(frozen=True)
class ProbeReading:
    """一个分区一拍到的判定结论。"""

    zone: str
    temperature_c: float
    trusted: bool
    reason: str
    bad_ticks: int
    clear_ticks: int

    def as_dict(self) -> dict:
        return {
            "zone": self.zone,
            "temperature_c": round(self.temperature_c, 2),
            "trusted": self.trusted,
            "reason": self.reason,
            "bad_ticks": self.bad_ticks,
            "clear_ticks": self.clear_ticks,
        }


class ZoneProbes:
    """分区测点可信度判定器。"""

    def __init__(
        self, bus: EventBus, limits: BurnerLimits, reference_c: float
    ) -> None:
        self._bus = bus
        self._limits = limits
        zones = limits.zone_names()
        self._zones = tuple(zones)
        self._temperature: dict[str, float] = {
            zone: float(reference_c) for zone in zones
        }
        self._trusted: dict[str, bool] = {zone: True for zone in zones}
        self._bad_ticks: dict[str, int] = {zone: 0 for zone in zones}
        self._clear_ticks: dict[str, int] = {zone: 0 for zone in zones}
        self._reasons: dict[str, str] = {zone: "init" for zone in zones}
        self._last_change: dict[str, float] = {zone: 0.0 for zone in zones}
        self._seen: dict[str, int] = {zone: 0 for zone in zones}
        self._suppressed = False
        self._lock = threading.RLock()

    @property
    def zones(self) -> tuple[str, ...]:
        return self._zones

    def observe(
        self, readings: dict, now: float
    ) -> tuple[dict[str, float], dict[str, ProbeReading]]:
        """喂入一拍测点读数，返回可信温度映射与逐区判定结论。

        ``readings`` 缺区或值不可解析时，该分区按坏点处理（测点没把值
        送上来同样不能信）。
        """

        outcomes: dict[str, ProbeReading] = {}
        accepted: dict[str, float] = {}
        with self._lock:
            if self._suppressed:
                for zone in self._zones:
                    outcomes[zone] = self._suppressed_reading(zone)
                trusted_temps = {
                    zone: self._temperature[zone]
                    for zone in self._zones
                    if self._trusted[zone]
                }
                bad_zones = list(self._zones)
                view = {zone: outcomes[zone].as_dict() for zone in self._zones}
            else:
                for zone in self._zones:
                    outcome = self._check_hard_fault(zone, readings.get(zone), now)
                    if outcome is None:
                        accepted[zone] = float(readings[zone])
                    else:
                        outcomes[zone] = outcome

                nearest = self._nearest_peer_distance(accepted)
                for zone in self._zones:
                    if zone in outcomes:
                        continue
                    temperature = accepted[zone]
                    if abs(temperature - self._temperature[zone]) > FREEZE_EPSILON_C:
                        self._last_change[zone] = now
                    distance = nearest.get(zone)
                    drifting = (
                        distance is not None
                        and distance > self._limits.zone_consensus_band_c
                    )
                    outcomes[zone] = self._settle_soft(
                        zone, temperature, drifting, "isolated_from_peers", now
                    )

                trusted_temps = {
                    zone: self._temperature[zone]
                    for zone in self._zones
                    if self._trusted[zone]
                }
                bad_zones = [
                    zone for zone in self._zones if not self._trusted[zone]
                ]
                view = {zone: outcomes[zone].as_dict() for zone in self._zones}

        self._bus.publish(
            Event.of(
                "burner",
                "burner",
                "zone_probes_observed",
                subject="zone_probes",
                suppressed=self._suppressed,
                trusted_zones=sorted(trusted_temps),
                bad_zones=sorted(bad_zones),
                readings=view,
            )
        )
        return trusted_temps, outcomes

    def _suppressed_reading(self, zone: str) -> ProbeReading:
        return ProbeReading(
            zone=zone,
            temperature_c=self._temperature[zone],
            trusted=False,
            reason=self._reasons[zone],
            bad_ticks=self._bad_ticks[zone],
            clear_ticks=self._clear_ticks[zone],
        )

    def _check_hard_fault(
        self, zone: str, value: object, now: float
    ) -> ProbeReading | None:
        """检查硬故障；无故障返回 None，故障时落状态并返回判定结论。"""

        seen = self._seen[zone] + 1
        self._seen[zone] = seen
        if seen == 1:
            self._last_change[zone] = now

        if value is None:
            return self._mark_hard(zone, "missing_reading", now)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return self._mark_hard(zone, "non_numeric", now)
        temperature = float(value)
        if not math.isfinite(temperature):
            return self._mark_hard(zone, "not_finite", now)
        if not self._limits.zone_temp_ok(temperature):
            return self._mark_hard(zone, "out_of_range", now, temperature)
        if (
            seen >= 2
            and self._trusted[zone]
            and abs(temperature - self._temperature[zone])
            > self._limits.zone_temp_jump_c
        ):
            # 跳变速限只约束可信测点；坏点恢复时从不可信旧值跳回是正常的。
            return self._mark_hard(zone, "step_jump", now, temperature)
        if seen >= 2 and self._trusted[zone] and self._is_frozen(zone, temperature, now):
            return self._mark_hard(zone, "frozen", now, temperature)
        return None

    def _is_frozen(self, zone: str, temperature: float, now: float) -> bool:
        """读数在判定时长内几乎不动视为卡死（闭环温度正常必有小幅波动）。"""

        if abs(temperature - self._temperature[zone]) > FREEZE_EPSILON_C:
            self._last_change[zone] = now
            return False
        return (
            self._trusted[zone]
            and now - self._last_change[zone] > self._limits.zone_temp_freeze_s
        )

    def _mark_hard(
        self, zone: str, reason: str, now: float, temperature: float | None = None
    ) -> ProbeReading:
        """硬故障一拍到顶：立即判坏，清掉恢复计数。"""

        if temperature is not None:
            self._temperature[zone] = temperature
        was_trusted = self._trusted[zone]
        self._trusted[zone] = False
        self._reasons[zone] = reason
        self._bad_ticks[zone] = FAULT_CONFIRM_TICKS
        self._clear_ticks[zone] = 0
        outcome = self._reading(zone)
        if was_trusted:
            self._publish_fault(outcome, now)
        return outcome

    def _settle_soft(
        self,
        zone: str,
        temperature: float,
        abnormal: bool,
        reason: str,
        now: float,
    ) -> ProbeReading:
        """按确认/迟滞计数落定中位数一致性这一软判据。"""

        was_trusted = self._trusted[zone]
        self._temperature[zone] = temperature
        if abnormal:
            self._bad_ticks[zone] += 1
            self._clear_ticks[zone] = 0
            if was_trusted and self._bad_ticks[zone] >= FAULT_CONFIRM_TICKS:
                self._trusted[zone] = False
                self._reasons[zone] = reason
        else:
            self._bad_ticks[zone] = 0
            self._clear_ticks[zone] += 1
            if (
                not was_trusted
                and self._clear_ticks[zone] >= CLEAR_CONFIRM_TICKS
            ):
                self._trusted[zone] = True
                self._reasons[zone] = "recovered"
                self._last_change[zone] = now

        outcome = self._reading(zone)
        if was_trusted and not self._trusted[zone]:
            self._publish_fault(outcome, now)
        elif not was_trusted and self._trusted[zone]:
            self._bus.publish(
                Event.of(
                    "burner",
                    "burner",
                    "zone_probe_recovered",
                    subject=f"probe:{zone}",
                    severity="notice",
                    zone=zone,
                    temperature_c=round(temperature, 2),
                    at=round(float(now), 3),
                )
            )
        return outcome

    def _nearest_peer_distance(self, accepted: dict[str, float]) -> dict[str, float]:
        """每区到"其余可信测点"的最近距离（点数不足不做一致性判定）。"""

        nearest: dict[str, float] = {}
        for zone in accepted:
            peers = [
                value
                for other, value in accepted.items()
                if other != zone and self._trusted[other]
            ]
            if len(peers) >= CONSENSUS_MIN_PEERS:
                nearest[zone] = min(
                    abs(accepted[zone] - value) for value in peers
                )
        return nearest

    def _reading(self, zone: str) -> ProbeReading:
        return ProbeReading(
            zone=zone,
            temperature_c=self._temperature[zone],
            trusted=self._trusted[zone],
            reason=self._reasons[zone],
            bad_ticks=self._bad_ticks[zone],
            clear_ticks=self._clear_ticks[zone],
        )

    def _publish_fault(self, reading: ProbeReading, now: float) -> None:
        self._bus.publish(
            Event.of(
                "burner",
                "burner",
                "zone_probe_bad",
                subject=f"probe:{reading.zone}",
                severity="warning",
                zone=reading.zone,
                reason=reading.reason,
                temperature_c=round(reading.temperature_c, 2),
                at=round(float(now), 3),
            )
        )

    def trusted_temperatures(self) -> dict[str, float]:
        with self._lock:
            return {
                zone: self._temperature[zone]
                for zone in self._zones
                if self._trusted[zone]
            }

    def is_trusted(self, zone: str) -> bool:
        with self._lock:
            return self._trusted.get(zone, False)

    def reading_value(self, zone: str) -> float:
        """返回分区最近一次读数（无论可信与否；坏区仅用于展示，不参与调节）。"""

        with self._lock:
            return self._temperature[zone]

    def bad_reasons(self) -> dict[str, str]:
        """当前判坏分区及其原因，用于区分整窑回退是测点故障还是手动接管。"""

        with self._lock:
            return {
                zone: self._reasons[zone]
                for zone in self._zones
                if not self._trusted[zone]
            }

    def bad_zones(self) -> list[str]:
        with self._lock:
            return [zone for zone in self._zones if not self._trusted[zone]]

    def readings(self) -> list[dict]:
        with self._lock:
            return [self._reading(zone).as_dict() for zone in self._zones]

    def force_equal_share(self, reason: str) -> list[str]:
        """手动强制全窑均分（测点检修、操作员接管），返回原本可信的分区。

        进入抑制态：之后送来的读数不参与可信度判定，直到调用
        :meth:`resume_auto` 才恢复自动闭环。
        """

        with self._lock:
            dropped = [zone for zone in self._zones if self._trusted[zone]]
            zones = list(self._zones)
            self._suppressed = True
            for zone in self._zones:
                self._trusted[zone] = False
                self._reasons[zone] = reason
                self._bad_ticks[zone] = FAULT_CONFIRM_TICKS
                self._clear_ticks[zone] = 0
        self._bus.publish(
            Event.of(
                "burner",
                "burner",
                "zone_probes_suppressed",
                subject="zone_probes",
                severity="notice",
                reason=reason,
                zones=zones,
            )
        )
        return dropped

    def resume_auto(self) -> list[str]:
        """退出手动接管，恢复按测点可信度自动闭环，返回分区列表。"""

        with self._lock:
            was_suppressed = self._suppressed
            self._suppressed = False
            for zone in self._zones:
                self._trusted[zone] = True
                self._reasons[zone] = "auto_resumed"
                self._bad_ticks[zone] = 0
                self._clear_ticks[zone] = 0
                self._last_change[zone] = 0.0
                self._seen[zone] = 0
            zones = list(self._zones)
        if was_suppressed:
            self._bus.publish(
                Event.of(
                    "burner",
                    "burner",
                    "zone_probes_resumed",
                    subject="zone_probes",
                    severity="notice",
                    zones=zones,
                )
            )
        return zones

    @property
    def suppressed(self) -> bool:
        with self._lock:
            return self._suppressed

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "suppressed": self._suppressed,
                "readings": [
                    self._reading(zone).as_dict() for zone in self._zones
                ],
                "trusted_zones": [
                    zone for zone in self._zones if self._trusted[zone]
                ],
                "bad_zones": [
                    zone for zone in self._zones if not self._trusted[zone]
                ],
            }

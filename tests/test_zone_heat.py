"""分区火力按温度反馈分配：闭环、坏点退回均分与总负荷硬约束。"""

from __future__ import annotations

import pytest

from glassmelt.app import CycleInputs, build_platform
from glassmelt.burner.heat import MIN_TRUSTED_ZONES, HeatAllocator
from glassmelt.burner.probe import ZoneProbes
from glassmelt.config import BurnerLimits, default_config
from glassmelt.errors import InterlockError


ZONES = ("zone-1", "zone-2", "zone-3", "zone-4", "zone-5", "zone-6")
TARGET = 1425.0


def _readings(skew: dict | None = None, base: float = TARGET) -> dict:
    readings = {zone: float(base) for zone in ZONES}
    if skew:
        readings.update(skew)
    return readings


def _parts(bus=None, limits=None, reference=TARGET):
    limits = limits or default_config().burner
    probes = ZoneProbes(bus or _NullBus(), limits, reference)
    allocator = HeatAllocator(bus or _NullBus(), limits, probes, reference)
    return limits, probes, allocator


class _NullBus:
    def publish(self, event):
        return 0


# --- 基本闭环：总负荷不变，冷区多、热区少 ----------------------------------


def test_equal_temperatures_share_load_evenly():
    limits, probes, allocator = _parts()
    outcome = allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    expected = 0.8 * limits.gas_max_m3_h / 6
    assert outcome.mode == "auto"
    for zone in outcome.zones:
        assert zone.flow_m3_h == pytest.approx(expected)
        assert zone.fraction == pytest.approx(1 / 6)
        assert zone.reason == "auto"
    assert outcome.total_flow_m3_h == pytest.approx(0.8 * limits.gas_max_m3_h)
    assert outcome.under_delivered is False
    assert outcome.section_spread_c == pytest.approx(0.0)


def test_cold_zone_takes_more_and_hot_zone_less_with_zero_sum():
    limits, probes, allocator = _parts()
    # 先跑一拍均分做基线，再喂断面温差。
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    readings = _readings({"zone-1": TARGET + 40.0, "zone-6": TARGET - 40.0})
    outcome = allocator.allocate(readings, 0.8, 1.0, 1.0)
    by_zone = {z.zone: z for z in outcome.zones}
    assert by_zone["zone-1"].flow_m3_h < by_zone["zone-2"].flow_m3_h
    assert by_zone["zone-6"].flow_m3_h > by_zone["zone-5"].flow_m3_h
    # 再分配总负荷不许超：份额之和不超过 1（爬坡欠发时小于 1）。
    total_fraction = sum(z.fraction for z in outcome.zones)
    assert total_fraction <= 1.0 + 1e-9
    assert outcome.total_flow_m3_h <= 0.8 * limits.gas_max_m3_h + 1e-6

    # 调节到位（多拍后）份额之和精确回到 1。
    settled = outcome
    for tick in range(2, 12):
        settled = allocator.allocate(readings, 0.8, 1.0, float(tick))
    assert sum(z.fraction for z in settled.zones) == pytest.approx(1.0, abs=1e-6)
    assert settled.total_flow_m3_h == pytest.approx(0.8 * limits.gas_max_m3_h)


def test_total_flow_never_exceeds_operator_load_for_any_demand():
    # 每个负荷点都从均分基线独立起算，避免上一拍的爬坡历史影响本拍总量。
    for load in (0.1, 0.5, 0.9, 1.0):
        limits, _probes, allocator = _parts()
        allocator.allocate(_readings(), load, 1.0, 0.0)
        readings = _readings(
            {
                "zone-1": TARGET - 30.0,
                "zone-2": TARGET - 20.0,
                "zone-3": TARGET + 25.0,
                "zone-4": TARGET + 28.0,
                "zone-5": TARGET - 12.0,
                "zone-6": TARGET + 10.0,
            }
        )
        outcome = allocator.allocate(readings, load, 1.0, 1.0)
        # 第二拍受爬坡限速可能欠发，但任何情况下都不得超过人工负荷。
        assert outcome.total_flow_m3_h <= load * limits.gas_max_m3_h + 1e-6
        for zone in outcome.zones:
            assert zone.flow_m3_h <= load * limits.gas_max_m3_h + 1e-6
            assert zone.flow_m3_h >= 0.0


def test_deadband_prevents_action_on_small_difference():
    limits, probes, allocator = _parts()
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    small = limits.zone_temp_deadband_c
    outcome = allocator.allocate(
        _readings({"zone-1": TARGET + small, "zone-6": TARGET - small}),
        0.8,
        1.0,
        1.0,
    )
    flows = [z.flow_m3_h for z in outcome.zones]
    assert max(flows) - min(flows) == pytest.approx(0.0, abs=1e-9)


# --- 测点可信度 -------------------------------------------------------------


def test_out_of_range_probe_is_bad_immediately():
    limits, probes, allocator = _parts()
    readings = _readings({"zone-2": limits.zone_temp_max_c + 50.0})
    outcome = allocator.allocate(readings, 0.8, 1.0, 0.0)
    bad = list(outcome.bad_zones)
    assert bad == ["zone-2"]
    zone = next(z for z in outcome.zones if z.zone == "zone-2")
    assert zone.trusted is False
    assert zone.reason == "equal_share_bad_probe"
    assert zone.fraction == pytest.approx(1 / 6)
    assert probes.bad_zones() == ["zone-2"]
    assert probes.readings()[1]["reason"] == "out_of_range"


def test_step_jump_is_rejected_as_bad():
    limits, probes, allocator = _parts()
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    jump = _readings({"zone-3": TARGET + limits.zone_temp_jump_c + 5.0})
    outcome = allocator.allocate(jump, 0.8, 1.0, 1.0)
    assert list(outcome.bad_zones) == ["zone-3"]
    # 坏点不参与调节，其余可信区照样做零和，总量不破。
    assert sum(z.fraction for z in outcome.zones) == pytest.approx(1.0, abs=1e-6)


def test_missing_or_nan_reading_is_bad():
    limits, probes, allocator = _parts()
    readings = _readings()
    del readings["zone-4"]
    outcome = allocator.allocate(readings, 0.8, 1.0, 0.0)
    assert list(outcome.bad_zones) == ["zone-4"]

    outcome2 = allocator.allocate(_readings({"zone-4": float("nan")}), 0.8, 1.0, 1.0)
    assert list(outcome2.bad_zones) == ["zone-4"]


def test_one_bad_probe_does_not_bias_the_kiln():
    """坏点钉在均分上，其余区按真实偏差调节，总和仍为预算。"""

    limits, probes, allocator = _parts()
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    readings = _readings(
        {
            "zone-1": TARGET - 50.0,
            "zone-2": TARGET + 50.0,
            "zone-3": 1590.0,  # 明显坏点（超量程）
            "zone-4": TARGET - 5.0,
            "zone-5": TARGET + 5.0,
            "zone-6": TARGET,
        }
    )
    outcome = allocator.allocate(readings, 0.8, 1.0, 1.0)
    by_zone = {z.zone: z for z in outcome.zones}
    assert by_zone["zone-3"].fraction == pytest.approx(1 / 6, abs=1e-6)
    assert by_zone["zone-3"].deviation_c == 0.0
    assert by_zone["zone-1"].fraction > by_zone["zone-2"].fraction
    # 坏点不参与再分配，总量受预算约束（爬坡欠发时份额之和小于 1）。
    assert sum(z.fraction for z in outcome.zones) <= 1.0 + 1e-9
    assert outcome.total_flow_m3_h <= 0.8 * limits.gas_max_m3_h + 1e-6
    assert outcome.mode == "auto"

    # 多拍稳定后总量精确回到预算，坏区始终钉在均分。
    settled = outcome
    for tick in range(2, 12):
        settled = allocator.allocate(readings, 0.8, 1.0, float(tick))
    settled_zone = {z.zone: z for z in settled.zones}
    assert settled_zone["zone-3"].fraction == pytest.approx(1 / 6, abs=1e-6)
    assert sum(z.fraction for z in settled.zones) == pytest.approx(1.0, abs=1e-6)


def test_consensus_drift_needs_confirmation_then_recovers():
    limits, probes, allocator = _parts()
    # 一拍、两拍喂一个与其余点差一个量级的"孤立点"。
    drift = _readings({"zone-5": TARGET - 120.0})
    first = allocator.allocate(drift, 0.8, 1.0, 0.0)
    assert list(first.bad_zones) == []
    second = allocator.allocate(drift, 0.8, 1.0, 1.0)
    assert list(second.bad_zones) == ["zone-5"]

    # 恢复需要连续干净拍，避免坏点抖动。
    allocator.allocate(_readings(), 0.8, 1.0, 2.0)
    allocator.allocate(_readings(), 0.8, 1.0, 3.0)
    third = allocator.allocate(_readings(), 0.8, 1.0, 4.0)
    assert list(third.bad_zones) == []


def test_real_cross_section_skew_is_not_flagged_as_bad():
    """半侧整体偏冷是真实工况，不能因为中位数判据误伤一半测点。"""

    limits, probes, allocator = _parts()
    skew = _readings(
        {
            "zone-1": TARGET - 55.0,
            "zone-2": TARGET - 50.0,
            "zone-3": TARGET - 45.0,
            "zone-4": TARGET + 45.0,
            "zone-5": TARGET + 50.0,
            "zone-6": TARGET + 55.0,
        }
    )
    for tick in range(4):
        outcome = allocator.allocate(skew, 0.8, 1.0, float(tick))
        assert list(outcome.bad_zones) == []


# --- 回退均分 ---------------------------------------------------------------


def test_all_probes_bad_falls_back_to_equal_share():
    limits, probes, allocator = _parts()
    readings = {zone: 2000.0 for zone in ZONES}
    outcome = allocator.allocate(readings, 0.7, 1.0, 0.0)
    assert outcome.mode == "all_probes_bad"
    expected = 0.7 * limits.gas_max_m3_h / 6
    for zone in outcome.zones:
        assert zone.fraction == pytest.approx(1 / 6)
        assert zone.flow_m3_h == pytest.approx(expected)
    assert outcome.trusted_zones == ()
    assert allocator.fallback_count() == 1


def test_too_few_trusted_zones_falls_back_to_equal():
    limits, probes, allocator = _parts()
    assert MIN_TRUSTED_ZONES == 2
    readings = {
        zone: (TARGET if zone == "zone-1" else 500.0) for zone in ZONES
    }
    outcome = allocator.allocate(readings, 0.6, 1.0, 0.0)
    assert outcome.mode == "too_few_trusted"
    assert sum(z.fraction for z in outcome.zones) == pytest.approx(1.0)


def test_manual_override_forces_equal_share():
    limits, probes, allocator = _parts()
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    outcome = allocator.force_equal_share(0.8, 1.0, 1.0, "manual_override")
    assert outcome.mode == "manual_override"
    for zone in outcome.zones:
        assert zone.fraction == pytest.approx(1 / 6)
    assert allocator.fallback_count() == 1
    assert probes.suppressed is True
    # 抑制期间再送读数也不恢复自动，仍保持均分。
    held = allocator.allocate(_readings({"zone-1": TARGET - 40.0}), 0.8, 1.0, 2.0)
    assert held.mode == "manual_override"
    # 显式恢复后回到自动闭环。
    resumed = allocator.resume_auto(_readings(), 0.8, 1.0, 3.0)
    assert resumed.mode == "auto"
    assert probes.suppressed is False
    assert len(resumed.trusted_zones) == 6


def test_platform_equalize_and_resume_round_trip(platform):
    platform.start()
    held = platform.equalize_zone_heat(0.7, "测点检修")
    assert held["mode"] == "manual_override"
    assert platform.zone_probes.suppressed is True
    resumed = platform.resume_zone_heat(_readings(), total_load=0.7)
    assert resumed["mode"] == "auto"
    assert platform.zone_probes.suppressed is False


# --- 约束与爬坡 -------------------------------------------------------------


def test_ramp_limit_throttles_per_zone_change_while_total_stays_at_budget():
    limits, probes, allocator = _parts()
    allocator.allocate(_readings(), 0.8, 1.0, 0.0)
    readings = _readings(
        {"zone-1": TARGET - 40.0, "zone-2": TARGET + 40.0}
    )
    first = allocator.allocate(readings, 0.8, 1.0, 1.0)
    step = limits.zone_heat_ramp_frac_per_s * limits.gas_max_m3_h
    expected = 0.8 * limits.gas_max_m3_h / 6
    by_zone = {z.zone: z for z in first.zones}
    # 逐区爬坡：冷区一拍最多加一个限速步长、热区最多减一个步长。
    assert by_zone["zone-1"].flow_m3_h == pytest.approx(expected + step)
    assert by_zone["zone-2"].flow_m3_h == pytest.approx(expected - step)
    # 火力只在冷区与热区之间转移，中性区钉在均分；总量精确等于预算。
    for name in ("zone-3", "zone-4", "zone-5", "zone-6"):
        assert by_zone[name].flow_m3_h == pytest.approx(expected)
    assert first.under_delivered is False
    assert first.total_flow_m3_h == pytest.approx(0.8 * limits.gas_max_m3_h)

    # 多跑几拍让调节量到位：冷区明显多于热区，总量恒为预算、绝不超发。
    outcome = first
    for tick in range(2, 12):
        outcome = allocator.allocate(readings, 0.8, 1.0, float(tick))
        assert outcome.total_flow_m3_h <= 0.8 * limits.gas_max_m3_h + 1e-9
    final = {z.zone: z for z in outcome.zones}
    assert final["zone-1"].flow_m3_h > final["zone-2"].flow_m3_h
    for name in ("zone-3", "zone-4", "zone-5", "zone-6"):
        assert final[name].flow_m3_h == pytest.approx(expected)
    assert outcome.total_flow_m3_h == pytest.approx(0.8 * limits.gas_max_m3_h)


def test_setpoints_must_cover_all_zones(platform):
    with pytest.raises(InterlockError):
        platform.set_zone_setpoints({"zone-1": 1400.0})
    with pytest.raises(InterlockError):
        platform.set_zone_setpoints(
            {**{z: TARGET for z in ZONES}, "zone-9": TARGET}
        )
    result = platform.set_zone_setpoints({z: TARGET + 5.0 for z in ZONES})
    assert result["setpoints_c"]["zone-1"] == TARGET + 5.0


def test_config_rejects_infeasible_zone_limits():
    from glassmelt.config import ConfigError, config_to_mapping

    section = config_to_mapping(default_config())["limits"]["burner"]
    section["zone_heat_min_frac"] = 0.5  # 6 个区下限之和 = 3，超过总负荷
    with pytest.raises(ConfigError):
        BurnerLimits.from_mapping(section)
    section["zone_heat_min_frac"] = 0.04
    section["zone_heat_max_frac"] = 0.02
    with pytest.raises(ConfigError):
        BurnerLimits.from_mapping(section)


# --- 平台集成 ---------------------------------------------------------------


def test_platform_cycle_carries_zone_heat_and_closes_section_skew(platform):
    platform.start()
    skew = _readings(
        {"zone-1": TARGET + 45.0, "zone-6": TARGET - 45.0}
    )
    inputs = CycleInputs(
        load=0.8,
        pull_rate=0.9,
        gob_mass_g=186.0,
        dt=1.0,
        zone_temps_c=skew,
    )
    first = platform.cycle(inputs)
    assert first["heat"]["mode"] == "auto"
    assert first["heat"]["total_flow_m3_h"] <= 0.8 * platform.config.burner.gas_max_m3_h + 1e-6
    flows = {z["zone"]: z["flow_m3_h"] for z in first["heat"]["zones"]}
    assert flows["zone-6"] > flows["zone-1"]

    # 固定断面偏差下连跑若干拍，闭环把火力差建立起来，总量始终不超。
    last = first
    for _ in range(10):
        last = platform.cycle(inputs)
    assert last["heat"]["under_delivered"] is False
    assert last["heat"]["total_flow_m3_h"] == pytest.approx(
        0.8 * platform.config.burner.gas_max_m3_h
    )


def test_platform_allocate_action_flags_bad_zone(platform):
    platform.start()
    readings = _readings({"zone-3": 2200.0})
    result = platform.allocate_zone_heat(readings, total_load=0.75)
    assert result["allocation"]["bad_zones"] == ["zone-3"]
    assert result["allocation"]["mode"] == "auto"
    assert result["all_zones_trusted"] is False
    bad = next(z for z in result["allocation"]["zones"] if z["zone"] == "zone-3")
    assert bad["reason"] == "equal_share_bad_probe"
    assert bad["fraction"] == pytest.approx(1 / 6, abs=1e-4)


def test_platform_status_exposes_probes_and_heat(platform):
    status = platform.status()
    assert status["zone_probes"]["trusted_zones"] == list(ZONES)
    assert "flows_m3_h" in status["zone_heat"]
    assert status["zone_heat"]["setpoints_c"]["zone-1"] == TARGET

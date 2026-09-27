"""分区火力反馈分配：温度调平、总热负荷封顶、坏点退回均分。"""

from __future__ import annotations

import pytest

from glassmelt.burner.heat import ZoneHeatAllocator
from glassmelt.burner.sensors import (
    REASON_OUT_OF_RANGE,
    REASON_RATE_JUMP,
    REASON_STUCK,
    ZoneSensors,
)
from glassmelt.config import ZoneHeatLimits


@pytest.fixture()
def sensors(clock, platform) -> ZoneSensors:
    return ZoneSensors(clock, platform.bus, platform.zones.zones(), platform.config.zone_heat)


@pytest.fixture()
def allocator(clock, platform) -> ZoneHeatAllocator:
    watch = ZoneSensors(clock, platform.bus, platform.zones.zones(), platform.config.zone_heat)
    return ZoneHeatAllocator(
        clock,
        platform.bus,
        watch,
        platform.config.zone_heat,
        default_setpoint_c=platform.config.furnace.melt_temp_target_c,
    )


def _readings(platform, offsets: tuple[float, ...]) -> dict:
    target = platform.config.furnace.melt_temp_target_c
    return {
        zone: target + offset
        for zone, offset in zip(platform.zones.zones(), offsets)
    }


def _warmup(allocator, platform, readings, *, times=None):
    """先喂到测点全部可信，再多走一拍让反馈结算完成。"""

    times = times or platform.config.zone_heat.fault_recover_samples
    for _ in range(times + 1):
        outcome = allocator.allocate(readings, 600.0, 1.0)
    return outcome


def test_all_zones_start_at_equal_share_before_feedback(allocator, platform):
    readings = _readings(platform, (-20.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    outcome = allocator.allocate(readings, 600.0, 1.0)
    # 测点尚未经过恢复确认，整窑先按均分，不拿未确认的点调火力。
    assert outcome.fallback_all is True
    assert all(item.share == pytest.approx(1 / 6) for item in outcome.zones)
    assert all(item.heat == pytest.approx(100.0) for item in outcome.zones)


def test_cross_section_difference_is_redistributed_within_budget(allocator, platform):
    readings = _readings(platform, (-20.0, -12.0, -4.0, 4.0, 12.0, 20.0))
    outcome = _warmup(allocator, platform, readings)
    assert outcome.fallback_all is False
    assert outcome.trusted_count == 6
    shares = [item.share for item in outcome.zones]
    # 欠火区（冷，误差为正）份额高于均分，过热区低于均分，且单调。
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > shares[-1]
    assert sum(shares) == pytest.approx(1.0)
    # 总热负荷精确等于预算，绝不超总负荷。
    assert outcome.total_heat == pytest.approx(600.0)
    assert outcome.budget_ok is True
    # 断面温差被如实报出。
    assert outcome.cross_section_spread_c == pytest.approx(40.0)


def test_total_heat_never_exceeds_budget_even_when_all_zones_cold(allocator, platform):
    readings = _readings(platform, (-30.0, -25.0, -20.0, -15.0, -10.0, -5.0))
    outcome = _warmup(allocator, platform, readings)
    assert outcome.total_heat == pytest.approx(600.0)
    assert all(item.heat >= 0.0 for item in outcome.zones)
    assert sum(item.share for item in outcome.zones) == pytest.approx(1.0)


def test_untrusted_zone_falls_back_to_equal_and_is_marked(allocator, platform):
    healthy = _readings(platform, (-18.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    _warmup(allocator, platform, healthy)
    faulty = dict(healthy)
    faulty["zone-3"] = -273.0  # 断线/短路式越界值
    outcome = allocator.allocate(faulty, 600.0, 1.0)
    bad = next(item for item in outcome.zones if item.zone == "zone-3")
    assert bad.trusted is False
    assert bad.fallback is True
    assert bad.reason == REASON_OUT_OF_RANGE
    assert bad.share == pytest.approx(1 / 6)
    # 坏点的份额不被其余区借走，只在可信区内部做零和微调。
    good = [item for item in outcome.zones if item.zone != "zone-3"]
    good_adjusted = sum(item.share - 1 / 6 for item in good)
    assert good_adjusted == pytest.approx(0.0, abs=1e-9)
    assert outcome.fallback_count == 1
    assert outcome.fallback_all is False
    assert outcome.total_heat == pytest.approx(600.0)


def _settled_allocator(platform, readings):
    """构造一个已经在给定工况下完成测点确认的分配器。"""

    watch = ZoneSensors(
        platform.clock, platform.bus, platform.zones.zones(), platform.config.zone_heat
    )
    allocator = ZoneHeatAllocator(
        platform.clock,
        platform.bus,
        watch,
        platform.config.zone_heat,
        default_setpoint_c=platform.config.furnace.melt_temp_target_c,
    )
    for _ in range(platform.config.zone_heat.fault_recover_samples + 1):
        allocator.allocate(readings, 600.0, 1.0)
    return allocator


def test_one_bad_sensor_cannot_drag_the_kiln(platform):
    """坏点无论给出多离谱的值，其它区的火力方向只取决于自身温度。"""

    healthy = _readings(platform, (-18.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    for crazy_value in (-273.0, 9999.0):
        # 两条完全相同的历史并行推进：A 全健康，B 在本拍塞入坏点。
        clean_alloc = _settled_allocator(platform, healthy)
        faulty_alloc = _settled_allocator(platform, healthy)
        clean = clean_alloc.allocate(healthy, 600.0, 1.0)
        faulty = dict(healthy)
        faulty["zone-3"] = crazy_value
        outcome = faulty_alloc.allocate(faulty, 600.0, 1.0)
        bad = next(item for item in outcome.zones if item.zone == "zone-3")
        assert bad.fallback is True
        assert bad.share == pytest.approx(1 / 6)
        for item in outcome.zones:
            if item.zone != "zone-3":
                ref = next(z.share for z in clean.zones if z.zone == item.zone)
                # 仅差坏点退出后零和重新归一化的量级，火力方向与排序不变。
                assert item.share == pytest.approx(ref, abs=2e-3)
        assert outcome.total_heat == pytest.approx(600.0)


def test_all_sensors_bad_returns_to_pure_equal_split(allocator, platform):
    healthy = _readings(platform, (-18.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    _warmup(allocator, platform, healthy)
    broken = {zone: -273.0 for zone in platform.zones.zones()}
    outcome = allocator.allocate(broken, 600.0, 1.0)
    assert outcome.fallback_all is True
    assert all(item.share == pytest.approx(1 / 6) for item in outcome.zones)
    assert all(item.fallback for item in outcome.zones)
    assert outcome.total_heat == pytest.approx(600.0)


def test_fault_latches_until_readings_confirm_recovery(allocator, platform):
    healthy = _readings(platform, (-18.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    _warmup(allocator, platform, healthy)
    faulty = dict(healthy)
    faulty["zone-2"] = healthy["zone-2"] + 200.0
    allocator.allocate(faulty, 600.0, 1.0)
    # 只恢复一拍：仍然闩锁，不许立刻重新接管火力。
    first = allocator.allocate(healthy, 600.0, 1.0)
    zone2 = next(item for item in first.zones if item.zone == "zone-2")
    assert zone2.trusted is False
    # 连续恢复达到确认拍数后才接管。
    for _ in range(platform.config.zone_heat.fault_recover_samples - 1):
        outcome = allocator.allocate(healthy, 600.0, 1.0)
    zone2 = next(item for item in outcome.zones if item.zone == "zone-2")
    assert zone2.trusted is True
    assert outcome.fallback_count == 0


def test_rate_jump_and_stuck_readings_are_flagged(sensors, platform):
    target = platform.config.furnace.melt_temp_target_c
    readings = {zone: target for zone in platform.zones.zones()}
    for _ in range(platform.config.zone_heat.fault_recover_samples):
        states = sensors.inspect(readings, {z: target for z in readings}, 1.0)
    jump = dict(readings)
    jump["zone-1"] = target + 50.0
    states = sensors.inspect(jump, {z: target for z in jump}, 1.0)
    assert states["zone-1"].trusted is False
    assert states["zone-1"].reason == REASON_RATE_JUMP
    # 恢复后长期完全不变的读数判为冻死。
    for _ in range(platform.config.zone_heat.fault_recover_samples):
        states = sensors.inspect(readings, {z: target for z in readings}, 1.0)
    assert states["zone-1"].trusted is True
    stuck_state = None
    for _ in range(platform.config.zone_heat.stuck_samples + 2):
        states = sensors.inspect(readings, {z: target for z in readings}, 1.0)
        if states["zone-1"].reason == REASON_STUCK:
            stuck_state = states["zone-1"]
            break
    assert stuck_state is not None
    assert stuck_state.trusted is False


def test_missing_and_non_numeric_readings_are_rejected(allocator, platform):
    healthy = _readings(platform, (-18.0, -10.0, 0.0, 4.0, 10.0, 18.0))
    _warmup(allocator, platform, healthy)
    for bad in (None, float("nan"), "---"):
        broken = dict(healthy)
        broken["zone-4"] = bad
        outcome = allocator.allocate(broken, 600.0, 1.0)
        item = next(z for z in outcome.zones if z.zone == "zone-4")
        assert item.fallback is True
        assert item.share == pytest.approx(1 / 6)


def test_share_box_limits_are_respected(allocator, platform):
    # 极端断面温差：投影后任何一区都不能越过单区份额上下限。
    readings = _readings(platform, (-60.0, -60.0, -60.0, 60.0, 60.0, 60.0))
    outcome = _warmup(allocator, platform, readings, times=8)
    limits = platform.config.zone_heat
    for item in outcome.zones:
        assert item.share >= limits.share_min - 1e-9
        assert item.share <= limits.share_max + 1e-9
    assert sum(item.share for item in outcome.zones) == pytest.approx(1.0)
    assert outcome.total_heat == pytest.approx(600.0)


def test_rate_limit_smooths_share_movement(allocator, platform):
    limits = platform.config.zone_heat
    readings = _readings(platform, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    _warmup(allocator, platform, readings)
    extreme = _readings(platform, (-50.0, -40.0, -20.0, 20.0, 40.0, 50.0))
    outcome = allocator.allocate(extreme, 600.0, 1.0)
    for item in outcome.zones:
        assert abs(item.share - 1 / 6) <= limits.share_max_step + 1e-9


def test_infeasible_share_budget_is_rejected(clock, platform):
    base = platform.config.zone_heat
    config_zone_heat = ZoneHeatLimits(
        temp_plausible_margin_c=base.temp_plausible_margin_c,
        temp_max_rate_c_s=base.temp_max_rate_c_s,
        stuck_samples=base.stuck_samples,
        fault_recover_samples=base.fault_recover_samples,
        share_min=0.4,
        share_max=base.share_max,
        share_max_step=base.share_max_step,
        gain_p=base.gain_p,
        gain_i=base.gain_i,
        integral_trim_cap=base.integral_trim_cap,
    )
    watch = ZoneSensors(clock, platform.bus, platform.zones.zones(), config_zone_heat)
    allocator = ZoneHeatAllocator(
        clock,
        platform.bus,
        watch,
        config_zone_heat,
        default_setpoint_c=platform.config.furnace.melt_temp_target_c,
    )
    readings = _readings(platform, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        allocator.allocate(readings, 600.0, 1.0)


def test_platform_endpoint_allocates_and_reports(platform):
    readings = _readings(platform, (-20.0, -12.0, -4.0, 4.0, 12.0, 20.0))
    for _ in range(platform.config.zone_heat.fault_recover_samples):
        outcome = platform.allocate_zone_heat(readings)
    assert outcome["budget"] == platform.config.burner.gas_max_m3_h
    assert outcome["total_heat"] == pytest.approx(platform.config.burner.gas_max_m3_h)
    report = platform.zone_heat_report()
    assert report["last"]["fallback_count"] == 0
    assert set(report["sensors"]["zones"]) == set(platform.zones.zones())
    status = platform.status()
    assert status["zone_heat"]["last"]["total_heat"] == pytest.approx(
        platform.config.burner.gas_max_m3_h
    )


def test_distrust_event_is_published(platform):
    actions: list[str] = []
    platform.bus.subscribe("burner", lambda event: actions.append(event.action))
    readings = _readings(platform, (-20.0, -12.0, -4.0, 4.0, 12.0, 20.0))
    for _ in range(platform.config.zone_heat.fault_recover_samples):
        platform.allocate_zone_heat(readings)
    broken = dict(readings)
    broken["zone-1"] = -200.0
    platform.allocate_zone_heat(broken)
    assert "zone_sensor_distrusted" in actions
    assert "zone_heat_allocated" in actions

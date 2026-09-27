"""命令行入口。

五个子命令覆盖日常使用：

``serve``   启动控制台 HTTP 服务，可选地在启动后探测页面；
``demo``    在本地跑一遍完整流程（启动顺序 → 控制周期 → 投料恢复 → 点火 →
            分区大修与跳闸 → 秤标定 → 清炉 → 退火复位 → 停炉顺序），用于
            交付前的真实运行检查；
``status``  打印当前状态快照；
``config``  打印生效配置；
``reset``   清掉落盘的投料秤标定基线。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .app import CycleInputs, Platform, build_platform
from .batch.scale import BASELINE_KEY
from .clock import ManualClock, SystemClock
from .config import config_to_mapping, default_config, load_config
from .console.server import ConsoleServer, probe_pages
from .errors import GlassMeltError

DEFAULT_DEMO_CYCLES = 12


def _build(args: argparse.Namespace) -> Platform:
    config = load_config(args.config) if getattr(args, "config", None) else default_config()
    if getattr(args, "storage_root", None):
        return build_platform(config, storage_root=args.storage_root)
    return Platform(config=config)


def _dump(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _cmd_serve(args: argparse.Namespace) -> int:
    platform = _build(args)
    server = ConsoleServer(platform, host=args.host, port=args.port)
    base_url = server.start()
    payload: dict = {"listening": base_url, "pages": list(platform.config.pages)}
    if args.probe:
        payload["probe"] = probe_pages(base_url, platform.config.pages)
    _dump(payload)
    if args.duration > 0:
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:  # pragma: no cover - 交互路径
            pass
        server.stop()
        return 0
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:  # pragma: no cover - 交互路径
        server.stop()
    return 0


def _demo_zone_heat(platform: Platform) -> dict:
    """演示分区火力反馈分配：断面温差调平、坏点退回均分、测点恢复。"""

    target = platform.config.furnace.melt_temp_target_c
    zones = platform.zones.zones()
    # 断面温差：左侧欠火、右侧过热；zone-3 热电偶先给一个冻死值再恢复。
    pattern = (-18.0, -10.0, 0.0, 4.0, 12.0, 20.0)
    healthy = {zone: target + offset for zone, offset in zip(zones, pattern)}
    steps = []
    confirm = platform.config.zone_heat.fault_recover_samples + 1
    for _ in range(confirm):
        steps.append(
            platform.allocate_zone_heat(
                healthy, total_heat=platform.config.burner.gas_max_m3_h, dt=1.0
            )
        )
    faulty = dict(healthy)
    faulty["zone-3"] = -273.0
    steps.append(
        platform.allocate_zone_heat(
            faulty, total_heat=platform.config.burner.gas_max_m3_h, dt=1.0
        )
    )
    recovered_steps = 0
    outcome = steps[-1]
    for _ in range(platform.config.zone_heat.fault_recover_samples):
        outcome = platform.allocate_zone_heat(
            healthy, total_heat=platform.config.burner.gas_max_m3_h, dt=1.0
        )
        recovered_steps += 1
        if outcome["fallback_count"] == 0:
            break
    steps.append(outcome)
    return {
        "healthy": steps[confirm - 1],
        "faulty_sensor": steps[-2],
        "recovered": steps[-1],
        "recover_steps": recovered_steps,
    }


def _cmd_demo(args: argparse.Namespace) -> int:
    clock = ManualClock() if args.fast else SystemClock()
    config = load_config(args.config) if args.config else default_config()
    platform = build_platform(config, clock=clock, storage_root=args.storage_root)
    # 演示从干净基线起步：先清掉上一轮落盘的标定基线，避免继承上次的偏移。
    platform.store.remove(BASELINE_KEY)
    platform.scale.restore()

    startup = platform.start()
    cycles = [
        platform.cycle(
            CycleInputs(
                load=0.55 + 0.4 * (index / max(1, args.cycles - 1)),
                pull_rate=0.7 + 0.3 * (index / max(1, args.cycles - 1)),
                gob_mass_g=config.feeder.gob_mass_target_g,
                dt=1.0,
            )
        )
        for index in range(args.cycles)
    ]
    platform.interrupt_charging("演示用投料联锁")
    platform.level.observe(config.furnace.level_target_mm - 12.0)
    resume = platform.start_charging(path="resume")
    reignition = platform.reignite_burner(platform.furnace_temp.target_c + 25.0, "演示用失焰")
    rebuilt = platform.rebuild_zone_map(
        {zone: f"burner-{index + 7}" for index, zone in enumerate(platform.zones.zones())},
        "演示用窑体大修",
    )
    trip = platform.trip_burner("zone-3", "演示用窑温越限")
    trip_reset = platform.reset_trip()
    calibration = platform.recalibrate_scale(4.5)
    cleared = platform.clear_furnace("演示用出料")
    platform.level.observe(config.furnace.level_target_mm - 18.0)
    restored = platform.start_charging(path="resume")
    anneal_excursion = platform.evaluate_anneal(592.0)
    lehr_reset = platform.reset_lehr("演示用退火复位")
    zone_heat_demo = _demo_zone_heat(platform)
    shutdown = platform.stop()

    server = ConsoleServer(platform)
    base_url = server.start()
    probe = probe_pages(base_url, platform.config.pages)
    server.stop()

    report = {
        "startup": startup,
        "cycles_run": len(cycles),
        "last_cycle": cycles[-1] if cycles else None,
        "resume": resume,
        "reignition": reignition,
        "zone_map": rebuilt,
        "trip": trip,
        "trip_reset": trip_reset,
        "calibration": calibration,
        "cleared": cleared,
        "charged_after_clear": restored,
        "anneal_excursion": anneal_excursion,
        "lehr_reset": lehr_reset,
        "zone_heat": zone_heat_demo,
        "shutdown": shutdown,
        "probe": probe,
        "audit": platform.audit.summary(),
    }
    _dump(report)
    failures = list(startup["violations"]) + list(shutdown["violations"])
    failures += [name for name, item in probe.items() if not item["ok"]]
    failures += startup["topology_missing"]
    if failures:
        print(f"检查未通过: {failures}", file=sys.stderr)
        return 1
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    platform = _build(args)
    payload = platform.status()
    payload["topology"] = platform.topology_summary()
    payload["namespaces"] = platform.namespaces.describe()
    payload["trend"] = platform.trend.snapshot()
    _dump(payload)
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else default_config()
    _dump(config_to_mapping(config))
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    platform = _build(args)
    removed = platform.store.remove(BASELINE_KEY)
    _dump({"storage_root": platform.store.root, "scale_baseline_removed": removed})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""

    parser = argparse.ArgumentParser(prog="glassmelt", description="产线控制平台")
    parser.add_argument("--config", help="YAML 配置文件路径")
    parser.add_argument("--storage-root", help="覆盖状态存储目录")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动控制台服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--duration", type=float, default=0.0, help="运行多少秒后退出，0 表示常驻")
    serve.add_argument("--probe", action="store_true", help="启动后探测页面")
    serve.set_defaults(func=_cmd_serve)

    demo = sub.add_parser("demo", help="跑一遍完整流程")
    demo.add_argument("--cycles", type=int, default=DEFAULT_DEMO_CYCLES)
    demo.add_argument("--fast", action="store_true", help="使用可推进时钟，跳过真实等待")
    demo.set_defaults(func=_cmd_demo)

    status = sub.add_parser("status", help="打印状态快照")
    status.set_defaults(func=_cmd_status)

    config = sub.add_parser("config", help="打印生效配置")
    config.set_defaults(func=_cmd_config)

    reset = sub.add_parser("reset", help="清掉落盘的投料秤标定基线")
    reset.set_defaults(func=_cmd_reset)
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行主入口。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except GlassMeltError as error:
        print(json.dumps(error.as_record(), ensure_ascii=False), file=sys.stderr)
        return 2

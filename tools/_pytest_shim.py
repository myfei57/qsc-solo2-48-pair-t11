"""极简 pytest 兼容执行器：仅够在没有 pytest 的离线环境里跑本仓测试。"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/tmp/shim")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    conftest = load_module(ROOT / "tests" / "conftest.py", "conftest_shim")
    fixtures = {
        name: obj
        for name, obj in vars(conftest).items()
        if getattr(obj, "__pytest_fixture__", False)
    }
    passed = failed = 0
    failures = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        module = load_module(path, f"tests_{path.stem}")
        tests = [
            (name, obj)
            for name, obj in sorted(vars(module).items())
            if name.startswith("test_") and callable(obj)
        ]

        generators = []

        def call(fn, local_cache):
            params = inspect.signature(fn).parameters
            kwargs = {}
            for pname in params:
                if pname in local_cache:
                    kwargs[pname] = local_cache[pname]
                elif pname == "tmp_path":
                    local_cache[pname] = Path(tempfile.mkdtemp())
                    kwargs[pname] = local_cache[pname]
                elif pname in fixtures:
                    local_cache[pname] = call(fixtures[pname], local_cache)
                    kwargs[pname] = local_cache[pname]
                else:
                    local = getattr(module, pname, None)
                    if getattr(local, "__pytest_fixture__", False):
                        local_cache[pname] = call(local, local_cache)
                        kwargs[pname] = local_cache[pname]
                    else:
                        raise KeyError(pname)
            result = fn(**kwargs)
            if inspect.isgenerator(result):
                gen = result
                value = next(gen)
                generators.append(gen)
                return value
            return result

        for name, fn in tests:
            try:
                call(fn, {})
                passed += 1
            except Exception:
                failed += 1
                failures.append(f"{path.name}::{name}\n{traceback.format_exc()}")
            finally:
                for gen in generators:
                    with suppress(StopIteration):
                        next(gen)
                generators.clear()
    print(f"{passed} passed, {failed} failed")
    for item in failures:
        print("FAIL", item)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

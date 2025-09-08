import importlib
import sys
from pathlib import Path
import pytest


def _has_prometheus() -> bool:
    try:
        import prometheus_client  # noqa: F401
        return True
    except Exception:
        return False


def _load_scheduler_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    return importlib.import_module("orion.schedule.recursive_scheduler")


def test_scheduler_module_imports_without_prometheus() -> None:
    if _has_prometheus():
        pytest.skip("prometheus_client installed")
    mod = _load_scheduler_module()
    assert hasattr(mod, "RecursiveScheduler")
    assert hasattr(mod, "RecursionBudget")


def test_metrics_names_when_prometheus_available() -> None:
    if not _has_prometheus():
        pytest.skip("prometheus_client not installed")

    import subprocess, sys, textwrap

    code = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        import orion.schedule.recursive_scheduler  # noqa: F401
        from prometheus_client import REGISTRY, generate_latest
        text = generate_latest(REGISTRY).decode('utf-8', errors='ignore')
        print(text)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    text = result.stdout
    expected = [
        "orion_recursion_depth_sched",
        "orion_credits_remaining_sched",
        "orion_loop_detections_total_sched",
        "orion_beam_size_current_sched",
        "orion_resource_efficiency_sched",
    ]
    present = [name for name in expected if name in text]
    assert len(present) >= 2


def test_metrics_noop_behavior_without_prometheus() -> None:
    if _has_prometheus():
        pytest.skip("prometheus_client installed; skip no-op test")

    mod = _load_scheduler_module()
    scheduler = mod.RecursiveScheduler()
    best = scheduler.schedule_recursion("root", mod.example_expand_function)
    assert best is not None


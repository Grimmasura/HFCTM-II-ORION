import asyncio
import pytest

torch = pytest.importorskip("torch")
from orion_api.hfctm_safety import HFCTMII_SafetyCore, SafetyConfig


def test_safety_check_returns_metrics() -> None:
    core = HFCTMII_SafetyCore(SafetyConfig())
    result = asyncio.run(core.safety_check(torch.randn(10, 10)))
    assert "metrics" in result
    assert "lyapunov" in result["metrics"]
    assert "wavelet_energy" in result["metrics"]


def test_detect_egregore() -> None:
    core = HFCTMII_SafetyCore(SafetyConfig())
    metrics = {"lyapunov": 1.0, "wavelet_energy": 4.0}
    assert core._detect_egregore(metrics) is True
    metrics = {"lyapunov": -1.0, "wavelet_energy": 0.0}
    assert core._detect_egregore(metrics) is False

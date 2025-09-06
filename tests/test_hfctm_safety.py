import asyncio
import torch
import pytest

from orion_api.hfctm_safety import HFCTMII_SafetyCore, safety_overhead_gauge


def test_safety_check_returns_metrics() -> None:
    core = HFCTMII_SafetyCore()
    result = asyncio.run(
        core.recursive_safety_check(torch.randn(1, 64), torch.randn(2, 128))
    )
    assert "metrics" in result
    assert "mutual_info" in result["metrics"]
    assert safety_overhead_gauge._value.get() >= 0


def test_detect_egregore() -> None:
    core = HFCTMII_SafetyCore()
    metrics = {"mutual_info": 2.0, "wavelet_energy": 4.0, "lyapunov": 1.0}
    assert core._detect_egregore(metrics) is True
    metrics = {"mutual_info": 0.0, "wavelet_energy": 0.0, "lyapunov": -1.0}
    assert core._detect_egregore(metrics) is False

def test_classical_fallback() -> None:
    core = HFCTMII_SafetyCore()
    core.config.use_ironwood = False
    core.config.use_majorana1 = False
    result = asyncio.run(
        core.recursive_safety_check(torch.randn(1, 64), torch.randn(2, 128))
    )
    assert "lyapunov" in result["metrics"]
    assert "wavelet_energy" in result["metrics"]

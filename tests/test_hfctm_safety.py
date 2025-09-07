import pytest
import torch
from orion_api.hfctm_safety import HFCTMII_SafetyCore, SafetyConfig


@pytest.fixture
def safety_core():
    config = SafetyConfig(enable_quantum=False, enable_tpu=False)
    return HFCTMII_SafetyCore(config)


@pytest.mark.asyncio
async def test_safety_check(safety_core):
    """Test basic safety check functionality"""
    state = torch.randn(5, 5)
    result = await safety_core.safety_check(state)

    assert 'metrics' in result
    assert 'interventions' in result
    assert 'safe' in result
    assert isinstance(result['metrics']['lyapunov'], float)


def test_egregore_detection(safety_core):
    """Test egregore detection logic"""
    metrics = {'lyapunov': 1.0, 'wavelet_energy': 5.0}
    detected = safety_core._detect_egregore(metrics)
    assert isinstance(detected, bool)

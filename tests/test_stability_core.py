import numpy as np
import pytest

from stability_core import (
    AdaptiveDamping,
    ChiralInversion,
    E8Anchor,
    LyapunovMonitor,
    StabilityConfig,
    StabilityCore,
    WaveletScanner,
    DampingConfig,
    ChiralConfig,
    E8Config,
    LyapunovConfig,
    WaveletConfig,
)


def test_heads_callable():
    state = np.array([1.0, 2.0, 3.0, 4.0])

    lm = LyapunovMonitor(LyapunovConfig())
    state, m = lm(state)
    assert "lyapunov" in m

    damping = AdaptiveDamping(DampingConfig())
    state, m = damping(state, m["lyapunov"])
    assert "damping_factor" in m

    chiral = ChiralInversion(ChiralConfig())
    state, m = chiral(state)
    assert m["chirality"] is True

    wavelet = WaveletScanner(WaveletConfig())
    state, m = wavelet(state)
    assert "anomaly_indices" in m

    e8 = E8Anchor(E8Config(enabled=True))
    state, m = e8(state)
    assert "e8_projection" in m


def test_config_from_yaml(tmp_path):
    yaml_content = """
    damping:
      base: 0.5
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    config = StabilityConfig.from_yaml(cfg_file)
    assert config.damping.base == 0.5


def test_stability_core_step():
    core = StabilityCore(StabilityConfig(e8=E8Config(enabled=True)))
    state = np.ones(8)
    new_state, metrics = core.step(state, None)
    assert "lyapunov" in metrics and "damping_factor" in metrics
    assert "chirality" in metrics and "anomaly_indices" in metrics
    assert "e8_projection" in metrics
    assert new_state.shape == state.shape

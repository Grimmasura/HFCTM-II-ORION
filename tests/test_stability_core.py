from stability_core import (
    StabilityCore,
    load_config,
    LambdaConfig,
    DampingConfig,
    ChiralConfig,
    WaveletConfig,
    E8Config,
)
from stability_core.lambda_monitor import LambdaMonitor
from stability_core.damping import AdaptiveDamping
from stability_core.chiral import ChiralInversion
from stability_core.wavelet import WaveletScanner
from stability_core.e8_anchor import E8Anchor


def test_lambda_monitor_callable():
    monitor = LambdaMonitor(LambdaConfig())
    val = monitor.compute(0.0, 1.0)
    assert isinstance(val, float)


def test_damping_callable():
    damping = AdaptiveDamping(DampingConfig(base=0.5))
    state, factor = damping.apply(1.0, 1.0)
    assert state < 1.0 and factor < 1.0


def test_chiral_callable():
    chiral = ChiralInversion(ChiralConfig(invert=True))
    assert chiral.apply(1.0) == -1.0


def test_wavelet_callable():
    scanner = WaveletScanner(WaveletConfig(threshold=0.5))
    assert scanner.scan(1.0)


def test_e8_anchor_callable():
    anchor = E8Anchor(E8Config(enabled=True))
    assert anchor.project(9.0) == 1.0


def test_stability_core_step():
    config = load_config("stability_core/config.yaml")
    core = StabilityCore(config)
    new_state, metrics = core.step(1.0, 0.5)
    assert isinstance(new_state, float)
    assert "lambda" in metrics and "anomaly" in metrics

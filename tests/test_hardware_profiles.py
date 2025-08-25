from models.hardware_profiles import (
    load_hardware_config,
    select_profile,
    parity_check,
)


def test_load_tpu_config():
    cfg = load_hardware_config("ironwood_tpu")
    assert "fp32" in cfg.precision_modes
    assert cfg.intervals["K"] == 1024
    assert cfg.sentinel_count == 4


def test_aspen_profile():
    profile = select_profile(["gpu"])
    assert profile["profile"] == "aspen"
    assert profile["metrics"] == "reduced"
    assert profile["actions"] == ["warn", "stabilize"]


def test_parity_check():
    assert parity_check([1, 2, 3])

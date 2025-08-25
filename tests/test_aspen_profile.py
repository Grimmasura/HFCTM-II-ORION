from models.accelerator_utils import (
    get_degradation_profile,
    parity_process,
    load_config,
)


def test_aspen_profile_gpu_only():
    profile = get_degradation_profile(has_tpu=False, has_qpu=False)
    assert profile["profile"] == "aspen"
    assert profile["metrics"] == "reduced"
    assert profile["hysteresis"] == "wide"
    assert profile["actions"] == ["warn", "stabilize"]


def test_full_profile_with_tpu():
    profile = get_degradation_profile(has_tpu=True, has_qpu=False)
    assert profile["profile"] == "full"
    assert "halt" in profile["actions"]


def test_parity_host_accelerator():
    data = [1, 2, 3, 4]
    host_result = parity_process(data, use_accelerator=False)
    accel_result = parity_process(data, use_accelerator=True)
    assert host_result == accel_result


def test_ironwood_config_loads():
    config = load_config("ironwood_tpu")
    assert config["precision_modes"]
    assert set(config["intervals"]) == {"K", "M"}
    assert isinstance(config["sentinel_count"], int)
    assert isinstance(config["latency_budget_ms"], (int, float))

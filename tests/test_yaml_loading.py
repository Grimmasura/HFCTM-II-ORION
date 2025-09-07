import yaml
from models.hardware_profiles import load_hardware_config
from models.accelerator_utils import load_config


def test_load_hardware_config_uses_yaml(monkeypatch):
    def fail_json_load(*args, **kwargs):
        raise AssertionError("json.load should not be called")

    monkeypatch.setattr("models.hardware_profiles.json.load", fail_json_load)

    called = {"yaml": False, "loader": False}
    real_load = yaml.load

    def spy_load(*args, **kwargs):
        called["yaml"] = True
        if kwargs.get("Loader") is yaml.SafeLoader:
            called["loader"] = True
        return real_load(*args, **kwargs)

    monkeypatch.setattr("models.hardware_profiles.yaml.load", spy_load)

    cfg = load_hardware_config("ironwood_tpu")
    assert called["yaml"]
    assert called["loader"]
    assert cfg.precision_modes


def test_load_config_uses_yaml(monkeypatch):
    def fail_json_load(*args, **kwargs):
        raise AssertionError("json.load should not be called")

    monkeypatch.setattr("models.accelerator_utils.json.load", fail_json_load)

    called = {"yaml": False, "loader": False}
    real_load = yaml.load

    def spy_load(*args, **kwargs):
        called["yaml"] = True
        if kwargs.get("Loader") is yaml.SafeLoader:
            called["loader"] = True
        return real_load(*args, **kwargs)

    monkeypatch.setattr("models.accelerator_utils.yaml.load", spy_load)

    cfg = load_config("ironwood_tpu")
    assert called["yaml"]
    assert called["loader"]
    assert isinstance(cfg, dict)

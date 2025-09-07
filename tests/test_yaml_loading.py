import yaml
from models.hardware_profiles import load_hardware_config
from models.accelerator_utils import load_config

def test_load_hardware_config_uses_yaml(monkeypatch):
    def fail_json_load(*args, **kwargs):
        raise AssertionError("json.load should not be called")
    monkeypatch.setattr("models.hardware_profiles.json.load", fail_json_load)

    called = {"yaml": False}
    real_yaml = yaml.safe_load
    def spy_safe_load(*args, **kwargs):
        called["yaml"] = True
        return real_yaml(*args, **kwargs)
    monkeypatch.setattr("models.hardware_profiles.yaml.safe_load", spy_safe_load)

    cfg = load_hardware_config("ironwood_tpu")
    assert called["yaml"]
    assert cfg.precision_modes

def test_load_config_uses_yaml(monkeypatch):
    def fail_json_load(*args, **kwargs):
        raise AssertionError("json.load should not be called")
    monkeypatch.setattr("models.accelerator_utils.json.load", fail_json_load)

    called = {"yaml": False}
    real_yaml = yaml.safe_load
    def spy_safe_load(*args, **kwargs):
        called["yaml"] = True
        return real_yaml(*args, **kwargs)
    monkeypatch.setattr("models.accelerator_utils.yaml.safe_load", spy_safe_load)

    cfg = load_config("ironwood_tpu")
    assert called["yaml"]
    assert isinstance(cfg, dict)

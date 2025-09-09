import numpy as np

from stability_core.egregore import EgDetect, EgState, EgMitigate, EgAudit
from stability_core.egregore.detector import LOGGER, DETECTOR


def test_egregore_flow(tmp_path):
    LOGGER.path = tmp_path / "audit.log"
    np.random.seed(0)
    signal = np.concatenate([np.zeros(50), np.random.rand(50) * 100])
    phases = np.zeros(10)
    x = np.arange(100)
    y = x.copy()
    EgDetect(phases=phases, signal=signal, x=x, y=y)
    action = EgMitigate()
    assert action == "escalate"
    logs = EgAudit()
    assert logs and logs[-1].endswith(action)
    assert len(logs[-1].split()[0]) == 64
    states = EgState()
    assert all(s in {"normal", "alert"} for s in states.values())

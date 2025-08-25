import json
from pathlib import Path

from telemetry.hash_chain_logger import HashChainLogger
from telemetry.transports import FileTransport


def test_hash_chain_and_redaction(tmp_path: Path) -> None:
    log_path = tmp_path / "telemetry.log"
    logger = HashChainLogger(
        transports=[FileTransport(log_path)], redact_fields=["detector_metrics"]
    )

    r1 = logger.log(
        step=1,
        model_id="modelA",
        model_version="v1",
        detector_metrics={"acc": 0.9},
        action="step1",
    )
    r2 = logger.log(
        step=2,
        model_id="modelA",
        model_version="v1",
        detector_metrics={"acc": 0.8},
        action="step2",
    )

    assert r2.prev_hash == r1.hash_value
    assert r1.detector_metrics == {"acc": 0.9}
    assert r2.detector_metrics == {"acc": 0.8}

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "detector_metrics" not in record
        assert record["redacted_fields"] == ["detector_metrics"]


import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetry import HashChainLogger, FileTransport


def test_hash_chain_and_redaction(tmp_path):
    log_file = tmp_path / "telemetry.log"
    transport = FileTransport(log_file)
    logger = HashChainLogger(transport, redact_fields={"detector_metrics"})

    event1 = logger.log(
        model_id="m1",
        model_version="1.0",
        detector_metrics={"score": 0.5},
        action="step1",
    )
    event2 = logger.log(
        model_id="m1",
        model_version="1.0",
        detector_metrics={"score": 0.8},
        action="step2",
    )

    assert event2.prev_hash == event1.hash

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "detector_metrics" not in record

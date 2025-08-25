import json
from pathlib import Path

from telemetry.logger import HashChainLogger
from telemetry.transports import FileTransport


def test_hash_chain_and_redaction(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    logger = HashChainLogger(
        transports=[FileTransport(log_path)], redact_fields=["action"]
    )
    r1 = logger.log(
        step=1,
        model_id="modelA",
        version_id="v1",
        detector_metrics={"acc": 0.9},
        action="sensitive",
    )
    r2 = logger.log(
        step=2,
        model_id="modelA",
        version_id="v1",
        detector_metrics={"acc": 0.8},
        action="secret",
    )
    assert r2.prev_hash == r1.hash_value
    assert r1.action == "[REDACTED]"
    assert r2.action == "[REDACTED]"
    with log_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 2
    first_record = json.loads(lines[0])
    assert first_record["hash_value"] == r1.hash_value

"""Hash-chain logger producing tamper-evident telemetry records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .schema import TelemetryRecord
from .transports import TelemetryTransport


class HashChainLogger:
    """Generate telemetry records linked by cryptographic hashes.

    Each call to :meth:`log` returns a :class:`TelemetryRecord` whose
    ``hash_value`` is derived from the record contents and the previous
    record's hash. Specified fields are omitted from the payload sent to
    transports but preserved in the returned record.
    """

    def __init__(
        self,
        transports: Iterable[TelemetryTransport],
        *,
        redact_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.transports = list(transports)
        self.prev_hash: Optional[str] = None
        self.redact_fields = set(redact_fields or [])

    def log(
        self,
        *,
        step: int,
        model_id: str,
        model_version: str,
        detector_metrics: dict[str, float],
        action: Any | None,
    ) -> TelemetryRecord:
        record_dict: dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "model_version": model_version,
            "detector_metrics": detector_metrics,
            "action": action,
            "prev_hash": self.prev_hash,
        }
        serialized = json.dumps(record_dict, sort_keys=True)
        new_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        record_dict["hash_value"] = new_hash

        redacted_fields = [f for f in self.redact_fields if f in record_dict]
        record_dict["redacted_fields"] = redacted_fields
        record = TelemetryRecord(**record_dict)

        payload_dict = record_dict.copy()
        for field in redacted_fields:
            payload_dict.pop(field, None)
        payload_record = TelemetryRecord(**payload_dict)

        self.prev_hash = new_hash
        for transport in self.transports:
            transport.send(payload_record)
        return record

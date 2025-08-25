from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional

from .schema import TelemetryRecord
from .transports import TelemetryTransport


class HashChainLogger:
    """Logger producing telemetry records linked via a hash chain."""

    def __init__(
        self,
        transports: Iterable[TelemetryTransport],
        redact_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.transports = list(transports)
        self.prev_hash: Optional[str] = None
        self.redact_fields = set(redact_fields or [])

    def _apply_redaction(self, data: dict[str, Any]) -> list[str]:
        redacted = []
        for field in self.redact_fields:
            if field in data:
                data[field] = "[REDACTED]"
                redacted.append(field)
        return redacted

    def log(
        self,
        *,
        step: int,
        model_id: str,
        version_id: str,
        detector_metrics: dict[str, float],
        action: Any,
    ) -> TelemetryRecord:
        record_dict: dict[str, Any] = {
            "step": step,
            "model_id": model_id,
            "version_id": version_id,
            "detector_metrics": detector_metrics,
            "action": action,
            "prev_hash": self.prev_hash,
        }
        redacted_fields = self._apply_redaction(record_dict)
        serialized = json.dumps(record_dict, sort_keys=True)
        new_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        record_dict["hash_value"] = new_hash
        record_dict["redacted_fields"] = redacted_fields
        record = TelemetryRecord(**record_dict)
        self.prev_hash = new_hash
        for transport in self.transports:
            transport.send(record)
        return record

"""Hash-chain logger producing tamper-evident telemetry events."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from .schema import TelemetryEvent


class HashChainLogger:
    """Generate telemetry events linked by cryptographic hashes.

    Each call to :meth:`log` produces a :class:`TelemetryEvent` whose ``hash``
    is derived from the event contents and the previous event's hash. This
    creates a tamper-evident chain similar to a blockchain.

    Parameters
    ----------
    transport:
        Transport adapter responsible for delivering telemetry records.
    redact_fields:
        Iterable of field names to omit from the transmitted record. The hash
        is computed over the full event **before** redaction.
    """

    def __init__(
        self,
        transport: Any,
        *,
        redact_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.transport = transport
        self.prev_hash: Optional[str] = None
        self.step = 0
        self.redact_fields = set(redact_fields or [])

    # Internal -----------------------------------------------------------------
    def _canonical_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        canon = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                canon[key] = value.isoformat()
            else:
                canon[key] = value
        return canon

    # Public API ----------------------------------------------------------------
    def log(
        self,
        *,
        model_id: str,
        model_version: str,
        detector_metrics: Dict[str, float],
        action: str,
    ) -> TelemetryEvent:
        """Create and dispatch a new telemetry event.

        Returns the full event (before redaction).
        """

        self.step += 1
        event_dict: Dict[str, Any] = {
            "step": self.step,
            "model_id": model_id,
            "model_version": model_version,
            "detector_metrics": detector_metrics,
            "action": action,
            "timestamp": datetime.utcnow(),
            "prev_hash": self.prev_hash,
        }

        canon = self._canonical_dict(event_dict)
        serialized = json.dumps(canon, sort_keys=True)
        current_hash = hashlib.sha256(serialized.encode()).hexdigest()
        event_dict["hash"] = current_hash

        event = TelemetryEvent(**event_dict)

        # Prepare redacted payload for transport
        payload = event.model_dump()
        for field in self.redact_fields:
            payload.pop(field, None)

        payload = self._canonical_dict(payload)
        self.transport.send(payload)
        self.prev_hash = current_hash
        return event

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class TelemetryEvent(BaseModel):
    """Telemetry schema for per-step logging."""

    step: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str
    model_version: str
    detector_metrics: dict[str, float] = Field(default_factory=dict)
    action: Any = None
    prev_hash: Optional[str] = None
    hash_value: Optional[str] = None
    redacted_fields: list[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}

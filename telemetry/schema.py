"""Pydantic models for telemetry records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """Telemetry schema matching Appendix P for per-step logging."""

    step: int = Field(..., description="Incremental step number for the run")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the record was generated",
    )
    model_id: str = Field(
        ..., description="Identifier for the model generating the record"
    )
    model_version: str = Field(
        ..., description="Version of the model generating the record"
    )
    detector_metrics: dict[str, float] = Field(
        default_factory=dict, description="Metrics reported by detectors"
    )
    action: Any | None = Field(
        None, description="Action taken at this step"
    )
    prev_hash: Optional[str] = Field(
        default=None, description="Hash of the previous telemetry record"
    )
    hash_value: Optional[str] = Field(
        default=None, description="Hash of the current telemetry record"
    )
    redacted_fields: list[str] = Field(
        default_factory=list, description="Fields removed before transport"
    )

    model_config = {"protected_namespaces": ()}


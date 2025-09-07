"""Pydantic models for telemetry events."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


class TelemetryEvent(BaseModel):
    """Structured telemetry event following Appendix P schema."""

    step: int = Field(..., description="Incremental step number for the run")
    model_id: str = Field(..., description="Identifier for the model generating the event")
    model_version: str = Field(..., description="Version of the model generating the event")
    detector_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Metrics reported by detectors"
    )
    action: str = Field(..., description="Action taken at this step")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the event was generated",
    )
    prev_hash: Optional[str] = Field(
        default=None, description="Hash of the previous telemetry event"
    )
    hash: Optional[str] = Field(
        default=None, description="Hash of the current telemetry event"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Telemetry event schema matching Appendix P with tamper-evident fields."
        }
    )

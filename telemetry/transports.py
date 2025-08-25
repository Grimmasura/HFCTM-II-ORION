from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import Optional

import requests

from .schema import TelemetryEvent


class TelemetryTransport(ABC):
    """Interface for telemetry transports."""

    @abstractmethod
    def send(self, record: TelemetryEvent) -> None:
        """Send a telemetry record."""
        raise NotImplementedError


class StdoutTransport(TelemetryTransport):
    """Transport that prints records to STDOUT."""

    def send(self, record: TelemetryEvent) -> None:  # pragma: no cover - trivial
        print(record.model_dump_json())


class FileTransport(TelemetryTransport):
    """Transport that appends records to a file in JSONL format."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)

    def send(self, record: TelemetryEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json())
            f.write("\n")


class HTTPTransport(TelemetryTransport):
    """Transport that POSTs records to an HTTP endpoint."""

    def __init__(self, url: str, headers: Optional[dict[str, str]] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, record: TelemetryEvent) -> None:  # pragma: no cover - side effects
        requests.post(self.url, json=record.model_dump(), headers=self.headers)

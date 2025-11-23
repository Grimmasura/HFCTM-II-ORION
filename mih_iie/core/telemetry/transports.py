"""Transport adapters for telemetry dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional dependency
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


class TransportProtocol:
    """Simple protocol representing a telemetry transport."""

    def send(self, record: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class StdoutTransport(TransportProtocol):
    """Write telemetry records to stdout as JSON."""

    def send(self, record: Dict[str, Any]) -> None:
        print(json.dumps(record))


class FileTransport(TransportProtocol):
    """Append telemetry records to a file, one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def send(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


class HTTPTransport(TransportProtocol):
    """POST telemetry records to an HTTP endpoint."""

    def __init__(self, url: str, *, session: Optional[Any] = None) -> None:
        self.url = url
        if session is not None:
            self.session = session
        elif requests is not None:
            self.session = requests.Session()
        else:  # pragma: no cover - requests missing
            self.session = None

    def send(self, record: Dict[str, Any]) -> None:
        if self.session is not None:
            self.session.post(self.url, json=record, timeout=5)

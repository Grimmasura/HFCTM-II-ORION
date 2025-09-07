"""Transport adapters for telemetry dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - requests not installed
    requests = None


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
    """POST telemetry records to an HTTP endpoint.

    The dependency on :mod:`requests` is optional so that the package can be
    used in minimal environments.  Attempting to instantiate this transport
    without the library being available will raise a :class:`RuntimeError`.
    """

    def __init__(self, url: str, *, session: Optional["requests.Session"] = None) -> None:
        if requests is None:  # pragma: no cover - exercised when requests missing
            raise RuntimeError("requests library is required for HTTPTransport")
        self.url = url
        self.session = session or requests.Session()

    def send(self, record: Dict[str, Any]) -> None:
        assert requests is not None  # for type checkers
        self.session.post(self.url, json=record, timeout=5)

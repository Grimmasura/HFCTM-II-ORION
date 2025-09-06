"""Azure Quantum integration for synchronization checks."""

from __future__ import annotations

import os
from typing import Any


def get_sync_status() -> dict[str, Any]:
    """Return status and basic metrics from the ``majorana1`` backend.

    The function attempts to authenticate against Azure Quantum using
    subscription, resource group and workspace values supplied via environment
    variables.  If the Azure Quantum SDK or credentials are unavailable the
    error is captured and returned instead of raising an exception.
    """

    # Import is deferred so that unit tests without the SDK installed do not
    # immediately fail.
    try:
        from azure.quantum.qiskit import AzureQuantumProvider  # type: ignore
        from qiskit import QuantumCircuit  # type: ignore
    except Exception as exc:  # pragma: no cover - executed when SDK missing
        return {"status": "error", "message": f"Azure Quantum SDK not available: {exc}"}

    subscription_id = os.getenv("AZURE_QUANTUM_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_QUANTUM_RESOURCE_GROUP")
    workspace = os.getenv("AZURE_QUANTUM_WORKSPACE_NAME")
    location = os.getenv("AZURE_QUANTUM_LOCATION", "eastus")

    try:
        provider = AzureQuantumProvider(
            subscription_id=subscription_id,
            resource_group=resource_group,
            name=workspace,
            location=location,
        )

        backend = provider.get_backend("majorana1")

        # Minimal GHZ-state circuit to verify backend responsiveness
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.measure(range(3), range(3))

        job = backend.run(circuit, shots=1)

        metrics: dict[str, Any] = {}
        if hasattr(job, "metrics"):
            try:
                metrics = job.metrics()
            except Exception:  # pragma: no cover - metric retrieval optional
                metrics = {}

        status = "unknown"
        if hasattr(job, "status"):
            try:
                st = job.status()
                status = getattr(st, "name", str(st))
            except Exception:  # pragma: no cover
                status = "unknown"

        return {
            "status": status,
            "latency": metrics.get("latency"),
            "coherence_time": metrics.get("coherence_time"),
        }

    except Exception as exc:  # pragma: no cover - network/auth failures
        return {"status": "error", "message": str(exc)}


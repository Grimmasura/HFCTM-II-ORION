"""TPU-specific initialization utilities.

The functions in this module provide lightweight setup for TPU execution.  The
implementation is intentionally minimal so that unit tests can exercise the
logic without requiring an actual TPU runtime.  When :mod:`torch_xla` is
available we expose the XLA device and optionally load a sharded checkpoint.  As
an alternative path we also support JAX which is used in some of the
project's models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    import torch_xla.core.xla_model as xm  # type: ignore
    _xla_available = True
except Exception:  # pragma: no cover - torch_xla not installed
    torch = None  # type: ignore
    xm = None  # type: ignore
    _xla_available = False

try:  # pragma: no cover - optional dependency
    import jax
    from jax.experimental import pjit
    _jax_available = True
except Exception:  # pragma: no cover - jax not installed
    jax = None  # type: ignore
    pjit = None  # type: ignore
    _jax_available = False


def initialize_tpu(checkpoint: str | Path | None = None) -> Tuple[Any, Any]:
    """Initialise a TPU backend and optionally load a sharded checkpoint.

    Parameters
    ----------
    checkpoint:
        Optional path to a checkpoint file.  When provided and the respective
        backend is available, the state contained in the checkpoint is loaded
        onto the TPU device.

    Returns
    -------
    device, state
        A tuple containing the device handle and the loaded state (``None`` when
        no checkpoint was supplied).
    """

    device = None
    state = None

    if _xla_available:
        device = xm.xla_device()
        if checkpoint is not None and torch is not None:
            state = torch.load(checkpoint, map_location=device)
        return device, state

    if _jax_available:
        device = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices()[0]
        if checkpoint is not None:
            import flax.serialization as serialization
            with open(checkpoint, "rb") as fh:
                state = serialization.from_bytes(None, fh.read())
        return device, state

    raise RuntimeError("No TPU runtime is available")


__all__ = ["initialize_tpu", "pjit"]

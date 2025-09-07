"""Simplified recursive model used for tests.

The original project integrates large language models and hardware
accelerators.  For the purposes of the unit tests in this kata we
provide a lightweight deterministic implementation that mimics the
behaviour of the full system without any heavy dependencies.
"""

from __future__ import annotations

from models.stability_core import stability_core
from orion_api.config import settings


def recursive_model_live(query: str, depth: int) -> str:
    """Generate a mock recursive response.

    The input ``query`` is streamed through :mod:`stability_core` which yields
    token level telemetry.  The tokens are reassembled into a processed query
    which is then expanded recursively.  This function is deliberately
    deterministic to make unit tests reliable and to avoid requiring external
    model weights.
    """

    inference = stability_core.generate(query)
    processed_tokens = [step["token"] for step in inference]
    processed_query = " ".join(processed_tokens)
    generated_text = f"Response to {processed_query}"[: settings.max_tokens]

    if depth <= 0:
        return f"Base case: {generated_text}"

    next_depth = depth - 1
    return (
        f"Recursive Expansion of '{generated_text}' at depth {depth}\n"
        + recursive_model_live(processed_query, next_depth)
    )

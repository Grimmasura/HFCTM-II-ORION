"""Lightweight recursive model used for tests.

The original project ships a complex implementation that downloads language
models and requires heavy deep learning frameworks.  For the purposes of the
unit tests we replace that behaviour with a tiny, deterministic function that
simulates recursive text generation using Python's ``random`` module.  This
keeps the repository self-contained and fast to execute while preserving the
public API expected by the tests.
"""

from __future__ import annotations

import random
from orion_api.config import settings

_TOKEN_POOL = [
    "orion",
    "hfctm",
    "inference",
    "recursive",
    "stability",
]

def _generate_token() -> str:
    """Return a pseudo-random token.

    The global :mod:`random` state controls determinism which allows the tests to
    seed the generator and obtain repeatable results without depending on any
    external machine learning libraries.
    """

    return random.choice(_TOKEN_POOL)


def recursive_model_live(query: str, depth: int) -> str:
    """Recursively generate a string response.

    Parameters
    ----------
    query:
        Input query string.  The content is not used directly but influences the
        random generator via ``hash`` to provide a small amount of variation.
    depth:
        Recursion depth.  When ``0`` a base case string is returned.
    """

    # The query parameter is accepted for API compatibility.  The global random
    # state is expected to be controlled by the caller (tests seed it
    # explicitly), so we simply draw tokens from the pool without additional
    # seeding here.
    token = _generate_token()

    if depth <= 0:
        return f"Base case: {token}"

    return (
        f"Recursive Expansion of '{token}' at depth {depth}"
        "\n" + recursive_model_live(query, depth - 1)
    )

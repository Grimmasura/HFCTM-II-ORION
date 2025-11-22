from __future__ import annotations

from typing import Dict, Generator, List


class StabilityCore:
    """Lightweight stand-in for a Meta‑Codex stability core.

    The core tracks simple telemetry for each generation step and exposes a
    generator interface similar to the pseudocode outlined in the
    specification. Each yielded step contains token data and basic detector
    outputs which can be used for guarding responses.
    """

    def __init__(self) -> None:
        self._telemetry: List[Dict[str, object]] = []

    def generate(self, prompt: str) -> Generator[Dict[str, object], None, None]:
        """Yield generation steps for a given prompt.

        This mock implementation simply streams tokens from the prompt while
        attaching default detector outputs.
        """
        for token in prompt.split():
            step = {"token": token, "chi_Eg": 0, "lambda": 0}
            self._telemetry.append(step)
            yield step

    def snapshot(self) -> List[Dict[str, object]]:
        """Return a snapshot of recorded telemetry."""
        return list(self._telemetry)


# Instantiate a global core for shared use across the application.
stability_core = StabilityCore()

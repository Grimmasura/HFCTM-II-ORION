"""Benchmark HFCTM-II safety core on random tensors."""

import asyncio
import time

import torch

from orion_api.hfctm_safety import safety_core


async def run_benchmark() -> None:
    model_state = torch.randn(1, 64)
    latents = torch.randn(16, 128)
    start = time.time()
    result = await safety_core.recursive_safety_check(model_state, latents)
    elapsed = time.time() - start
    print("Metrics:", result["metrics"])
    print("Interventions:", result["interventions"])
    print("Elapsed seconds:", round(elapsed, 3))


if __name__ == "__main__":
    asyncio.run(run_benchmark())

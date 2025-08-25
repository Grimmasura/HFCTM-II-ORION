"""Application entrypoint compatibility wrapper.

The FastAPI ``app`` is defined in ``orion_api.main``; this module allows
``uvicorn main:app`` to continue working.
"""
from orion_api.main import app

__all__ = ["app"]

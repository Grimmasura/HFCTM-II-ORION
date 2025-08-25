"""Compatibility wrapper for the recursive model.

This module preserves the legacy import path while delegating to
``models.recursive_ai_model`` for the actual implementation.
"""
from models.recursive_ai_model import recursive_model_live, StabilityCore

__all__ = ["recursive_model_live", "StabilityCore"]

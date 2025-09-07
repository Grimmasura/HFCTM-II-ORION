"""Application configuration management using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Base configuration for the O.R.I.O.N. API.

    Values can be overridden using environment variables prefixed with
    ``ORION_`` or by providing a ``.env`` file at the project root.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    model_dir: Path = Path("models")
    recursive_model_path: Path = Path("models/recursive_live_optimization_model.zip")
    max_tokens: int = 50
    temperature: float = 1.0

    model_config = SettingsConfigDict(
        env_prefix="ORION_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


# Global settings instance used throughout the application
settings = Settings()

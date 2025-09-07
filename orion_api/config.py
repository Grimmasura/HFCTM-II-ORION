from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration using environment variables.

    Values can be overridden via environment variables with the prefix
    ``ORION_`` or by providing a ``.env`` file in the project root.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    model_dir: Path = Path("models")
    recursive_model_path: Path = Path("models/recursive_live_optimization_model.zip")
    max_tokens: int = 50
    temperature: float = 1.0

    # HFCTM-II hardware configuration
    enable_majorana1: bool = False
    enable_ironwood_tpu: bool = False
    safety_overhead_budget: float = 2.0

    azure_quantum_subscription_id: str = ""
    azure_quantum_workspace: str = ""
    quantum_shots: int = 1000

    ironwood_cluster_size: int = 256
    tpu_precision: str = "bf16"

    model_config = SettingsConfigDict(
        env_prefix="ORION_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


# Global settings instance
settings = Settings()

"""
Core configuration module for NLLB Translation Service.
Handles environment variables and application settings.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "NLLB Translation Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Configuration
    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    CUSTOM_MODEL_PATH: Optional[str] ="./models/custom-nllb/checkpoint-506"
    MODEL_CACHE_DIR: str = "./models/cache"
    USE_GPU: bool = True
    USE_HALF_PRECISION: bool = True
    MAX_LENGTH: int = 512


    # Glossary
    GLOSSARY_PATH: Optional[str] = "./glossary.json"
    APPLY_GLOSSARY_FROM_FILE: bool = True

    # Performance
    NUM_WORKERS: int = 4
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"
    TIMEOUT: int = 120
    RATE_LIMIT_PER_MINUTE: int = 60
    REQUEST_TIMEOUT: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Training
    TRAINING_OUTPUT_DIR: str = "./models/custom-nllb"
    TRAINING_BATCH_SIZE: int = 8
    TRAINING_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

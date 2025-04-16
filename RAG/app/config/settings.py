import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field



load_dotenv(dotenv_path="/Users/dhawalpanchal/Documents/practice projects/docparser/.env")



def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = "gpt-4o-mini-2024-07-18"
    embedding_model: str = "text-embedding-3-large"
    
    @classmethod
    def from_config(cls, config: Dict[str, str]):
        """Create settings from configuration dictionary."""
        return cls(
            default_model=config.get("model", "gpt-4o-mini-2024-07-18"),
            embedding_model=config.get("embedding_model", "text-embedding-3-large"),
        )


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 3072
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    
    @classmethod
    def with_config(cls, config: Dict[str, str]):
        """Create settings with specific configuration."""
        return cls(
            openai=OpenAISettings.from_config(config),
            database=DatabaseSettings(),
            vector_store=VectorStoreSettings()
        )


@lru_cache()
def get_settings(config_frozen: Optional[frozenset] = None) -> Settings:
    setup_logging()
    if config_frozen:
        config = dict(config_frozen)
        return Settings.with_config(config)
    return Settings()
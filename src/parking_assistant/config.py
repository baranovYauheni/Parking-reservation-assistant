"""Application settings loaded from environment variables and `.env`."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "var", alias="PARKING_DATA_DIR")
    knowledge_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "knowledge", alias="PARKING_KNOWLEDGE_DIR"
    )

    admin_token: SecretStr | None = Field(default=None, alias="ADMIN_TOKEN")

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_enabled: bool = Field(default=True, alias="LLM_ENABLED")

    vector_backend: Literal["weaviate", "memory"] = Field(
        default="weaviate", alias="VECTOR_BACKEND"
    )
    weaviate_mode: Literal["embedded", "local"] = Field(
        default="embedded", alias="WEAVIATE_MODE"
    )
    weaviate_host: str = Field(default="localhost", alias="WEAVIATE_HOST")
    weaviate_port: int = Field(default=8080, alias="WEAVIATE_PORT")
    weaviate_grpc_port: int = Field(default=50051, alias="WEAVIATE_GRPC_PORT")
    weaviate_collection: str = Field(
        default="ParkingKnowledge", alias="WEAVIATE_COLLECTION"
    )

    embeddings_provider: Literal["fastembed", "hashing"] = Field(
        default="fastembed", alias="EMBEDDINGS_PROVIDER"
    )
    fastembed_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="FASTEMBED_MODEL"
    )
    retrieval_k: int = Field(default=3, alias="RETRIEVAL_K")
    retrieval_min_score: float | None = Field(default=None, alias="RETRIEVAL_MIN_SCORE")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "parking.db"

    @property
    def weaviate_data_path(self) -> Path:
        return self.data_dir / "weaviate"

    @property
    def min_score(self) -> float:
        if self.retrieval_min_score is not None:
            return self.retrieval_min_score
        return 0.1 if self.embeddings_provider == "hashing" else 0.45

    @property
    def llm_available(self) -> bool:
        return self.llm_enabled and self.openai_api_key is not None

    def ensure_secrets(self) -> None:
        """Generate a process-local admin token when none is configured."""
        if self.admin_token is None:
            self.admin_token = SecretStr(secrets.token_urlsafe(24))

    def prepare_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

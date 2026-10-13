"""Application settings loaded from environment variables and `.env`."""

from __future__ import annotations

import os
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
    mcp_shared_secret: SecretStr | None = Field(default=None, alias="MCP_SHARED_SECRET")

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

    mcp_timeout_seconds: float = Field(default=30.0, alias="MCP_TIMEOUT_SECONDS")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "parking.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def export_path(self) -> Path:
        return self.data_dir / "exports" / "reservations.txt"

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

    @property
    def mcp_secret_path(self) -> Path:
        return self.data_dir / ".mcp_secret"

    def ensure_secrets(self) -> None:
        """Generate secrets that are not configured.

        The admin token is process-local. The MCP secret is persisted in the
        data directory with mode 0600 so the MCP server process reads it from
        trusted storage instead of trusting whoever spawns it.
        """
        if self.admin_token is None:
            self.admin_token = SecretStr(secrets.token_urlsafe(24))
        if self.mcp_shared_secret is None:
            self.prepare_dirs()
            path = self.mcp_secret_path
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as handle:
                    handle.write(secrets.token_urlsafe(32))
            self.mcp_shared_secret = SecretStr(path.read_text().strip())

    def load_mcp_secret(self) -> SecretStr | None:
        """Secret as seen by the MCP server: configured value or the stored file."""
        if self.mcp_shared_secret is not None:
            return self.mcp_shared_secret
        if self.mcp_secret_path.exists():
            return SecretStr(self.mcp_secret_path.read_text().strip())
        return None

    def prepare_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.export_path.parent.mkdir(parents=True, exist_ok=True)

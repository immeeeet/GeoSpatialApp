"""
Application configuration loaded from environment variables.
Uses pydantic-settings to validate and type-check all config values.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is three levels up from this file (backend/app/core/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """All application settings, read from .env file at project root."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostGIS ──────────────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "geospatial"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 86400  # 24 hours

    # ── API Keys ─────────────────────────────────────────────
    ORS_API_KEY: str = ""  # OpenRouteService — free tier: 40 req/min
    OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"

    # ── Raster Paths ─────────────────────────────────────────
    GHSL_RASTER_PATH: str = str(PROJECT_ROOT / "datasets" / "ghsl_2025.tif")
    VIIRS_RASTER_PATH: str = str(PROJECT_ROOT / "datasets" / "viirs_nightlights.tif")

    # ── ML Model ─────────────────────────────────────────────
    MODEL_PATH: str = str(
        PROJECT_ROOT / "backend" / "ml_engine" / "models" / "best_model.pkl"
    )

    # ── Server ───────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    @property
    def postgres_dsn(self) -> str:
        """Full PostgreSQL connection string for psycopg2."""
        return (
            f"host={self.POSTGRES_HOST} "
            f"port={self.POSTGRES_PORT} "
            f"dbname={self.POSTGRES_DB} "
            f"user={self.POSTGRES_USER} "
            f"password={self.POSTGRES_PASSWORD}"
        )


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — parsed once, reused everywhere."""
    return Settings()

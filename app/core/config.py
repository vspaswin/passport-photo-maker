"""Application settings (env-driven for production)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Passport Photo Maker"
    app_env: str = "development"  # development | production
    app_url: str = "http://127.0.0.1:8765"
    secret_key: str = "dev-change-me-in-production"
    host: str = "127.0.0.1"
    port: int = 8765

    # Data
    data_dir: Path = Path.home() / ".passport-photo-maker"
    job_ttl_seconds: int = 3600
    max_upload_mb: int = 25

    # Credits / freemium
    free_daily_checks: int = 20
    free_daily_converts: int = 3
    convert_credit_cost: int = 1
    # Credit packs (Stripe): price_id optional until configured
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_price_starter: Optional[str] = None  # e.g. 10 credits
    stripe_price_pro: Optional[str] = None  # e.g. 50 credits
    credit_pack_starter: int = 10
    credit_pack_pro: int = 50

    # Engine
    rembg_model: str = "u2net_human_seg"  # better for people; falls back to u2net
    use_mediapipe: bool = True

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def stripe_enabled(self) -> bool:
        return bool(self.stripe_secret_key)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()

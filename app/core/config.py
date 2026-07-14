"""Application settings (env-driven for production)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SECRET = "dev-change-me-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Passport Photo Maker"
    app_env: str = "development"  # development | production
    app_url: str = "http://127.0.0.1:8765"
    secret_key: str = DEFAULT_SECRET
    host: str = "127.0.0.1"
    port: int = 8765

    # Data
    data_dir: Path = Path.home() / ".passport-photo-maker"
    job_ttl_seconds: int = 3600
    max_upload_mb: int = 25

    # Credits / freemium (per client cookie)
    free_daily_checks: int = 20
    free_daily_converts: int = 3
    convert_credit_cost: int = 1
    # IP-level free caps (abuse control across cookie rotation)
    ip_free_daily_checks: int = 40
    ip_free_daily_converts: int = 6

    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_price_starter: Optional[str] = None
    stripe_price_pro: Optional[str] = None
    credit_pack_starter: int = 10
    credit_pack_pro: int = 50

    rembg_model: str = "u2net_human_seg"
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

    def validate_for_runtime(self) -> None:
        """Fail fast on unsafe production configuration."""
        if self.is_production:
            if not self.secret_key or self.secret_key == DEFAULT_SECRET:
                raise RuntimeError(
                    "APP_ENV=production requires a strong SECRET_KEY "
                    "(not the default dev-change-me-in-production)."
                )
            if self.app_url.startswith("http://127.") or "localhost" in self.app_url:
                # warn-level only via exception? Allow but recommend — use RuntimeError soft
                pass


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.validate_for_runtime()
    return s

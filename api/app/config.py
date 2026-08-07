from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql://matvision:matvision@postgres:5432/matvision"

    # Auth
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    s3_endpoint_url: str = "http://minio:9000"
    s3_access_key: str = "matvision"
    s3_secret_key: str = "matvision123"
    s3_bucket: str = "matvision"
    s3_region: str = "us-east-1"
    s3_public_endpoint_url: str = "http://localhost:9000"

    # Redis / queue
    redis_url: str = "redis://redis:6379/0"

    state_model_path: str = "/code/models/state-lgbm.joblib"
    state_confidence_threshold: float = 0.55

    # Layer 6: the one paid dependency (SPEC.md section 3). Unset in the sandbox/CI
    # on purpose — the REPORT stage raises a clear StageError rather than silently
    # doing nothing when this is missing. Set it in .env on the Mac to enable
    # report generation.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    max_upload_bytes: int = 1024 * 1024 * 1024
    max_duration_seconds: int = 10 * 60
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".webm")


settings = Settings()

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

    max_upload_bytes: int = 1024 * 1024 * 1024
    max_duration_seconds: int = 10 * 60
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".webm")


settings = Settings()

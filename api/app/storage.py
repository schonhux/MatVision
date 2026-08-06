import boto3
from botocore.client import Config

from app.config import settings


def _client(endpoint: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )


_internal_client = _client(settings.s3_endpoint_url)
_public_client = _client(settings.s3_public_endpoint_url)


def object_key(user_id: str, match_id: str, category: str, filename: str) -> str:
    return f"users/{user_id}/matches/{match_id}/{category}/{filename}"


def ensure_bucket() -> None:
    try:
        _internal_client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:
        _internal_client.create_bucket(Bucket=settings.s3_bucket)


def presigned_put_url(key: str, content_type: str, expires_in: int = 3600) -> str:
    return _public_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )


def presigned_get_url(key: str, expires_in: int = 3600) -> str:
    return _public_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def object_exists(key: str) -> bool:
    try:
        _internal_client.head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except Exception:
        return False


def download_to_path(key: str, local_path: str) -> None:
    _internal_client.download_file(settings.s3_bucket, key, local_path)


def upload_from_path(local_path: str, key: str, content_type: str | None = None) -> None:
    extra = {"ContentType": content_type} if content_type else {}
    _internal_client.upload_file(local_path, settings.s3_bucket, key, ExtraArgs=extra)

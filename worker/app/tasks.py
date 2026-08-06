import dramatiq
from dramatiq.brokers.redis import RedisBroker

from app.config import settings
from app.database import SessionLocal
from app.stages.runner import run_all_stages
from app.stages.clips import cut_clip_for_event

redis_broker = RedisBroker(url=settings.redis_url)
dramatiq.set_broker(redis_broker)


@dramatiq.actor(actor_name="run_pipeline", max_retries=0, time_limit=30 * 60 * 1000)
def run_pipeline(match_id: str) -> None:
    run_all_stages(match_id)


@dramatiq.actor(actor_name="cut_event_clip", max_retries=0, time_limit=5 * 60 * 1000)
def cut_event_clip(match_id: str, event_id: str) -> None:
    db = SessionLocal()
    try:
        cut_clip_for_event(match_id, event_id, db)
    finally:
        db.close()

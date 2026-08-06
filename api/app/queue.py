import dramatiq
from dramatiq.brokers.redis import RedisBroker

from app.config import settings

redis_broker = RedisBroker(url=settings.redis_url)
dramatiq.set_broker(redis_broker)


@dramatiq.actor(actor_name="run_pipeline", max_retries=0)
def _run_pipeline_stub(match_id: str) -> None:  # pragma: no cover - real body in worker
    pass


@dramatiq.actor(actor_name="cut_event_clip", max_retries=0)
def _cut_event_clip_stub(match_id: str, event_id: str) -> None:  # pragma: no cover
    pass


def enqueue_pipeline(match_id: str) -> None:
    _run_pipeline_stub.send(match_id)


def enqueue_clip_cut(match_id: str, event_id: str) -> None:
    _cut_event_clip_stub.send(match_id, event_id)

from dataclasses import dataclass
from typing import Callable, Protocol

from sqlalchemy.orm import Session

from app.models import Match


class StageFn(Protocol):
    def __call__(self, match: Match, db: Session) -> dict: ...


@dataclass
class StageResult:
    artifacts: dict


class StageError(Exception):
    """Expected pipeline failure with a user-safe message."""

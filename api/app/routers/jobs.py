from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Job
from app.schemas import JobResponse
from app.deps import get_current_user
from app.routers.matches import _get_owned_match

router = APIRouter(prefix="/matches/{match_id}/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs(
    match_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _get_owned_match(db, match_id, current_user)  # 404s if not owned
    return (
        db.query(Job)
        .filter(Job.match_id == match_id)
        .order_by(Job.created_at.asc())
        .all()
    )

"""
Celery tasks for background ingestion jobs.

Requires Redis (or another broker) and: pip install celery redis.
Set CELERY_BROKER_URL in .env (e.g. redis://localhost:6379/0).
"""

from app.core.config import get_settings
from app.ingestion.github import sync_repo

# Lazy app so config is loaded when tasks are used
_celery_app = None


def get_celery_app():
    global _celery_app
    if _celery_app is None:
        from celery import Celery
        settings = get_settings()
        _celery_app = Celery(
            "ingestion",
            broker=settings.celery_broker_url,
            backend=settings.celery_broker_url,  # optional result backend
        )
        _celery_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
        )
    return _celery_app


app = get_celery_app()


@app.task(bind=True, name="ingestion.sync_repo")
def sync_repo_task(
    self,
    repo_owner: str,
    repo_name: str,
    full_sync: bool = True,
):
    """
    Sync a single repo: fetch issues from GitHub, normalize, preprocess, upsert.

    full_sync: If True (default), sync all issues (open + closed). Reserved for
    future incremental sync when webhooks are used.
    """
   
    updated, skipped = sync_repo(repo_owner, repo_name)
    return {"updated": updated, "skipped": skipped}

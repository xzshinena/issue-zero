"""Celery tasks for background ingestion jobs.

Requires Redis (or another broker) and: pip install celery redis
Set CELERY_BROKER_URL in .env (e.g. redis://localhost:6379/0).

When Celery is NOT installed or the broker is not configured, all task
functions are still importable and fall back to synchronous execution so
the rest of the app works without changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Celery app (only created when actually needed)
# ---------------------------------------------------------------------------

_celery_app = None


def get_celery_app():
    """Return the Celery app, creating it on first call.

    Returns None when celery or redis is not installed — callers should
    fall back to synchronous / BackgroundTasks execution.
    """
    global _celery_app
    if _celery_app is not None:
        return _celery_app

    try:
        from celery import Celery  # noqa: PLC0415
    except ImportError:
        return None

    from app.core.config import get_settings  # noqa: PLC0415
    settings = get_settings()
    broker = (settings.celery_broker_url or "").strip()
    if not broker:
        return None

    _celery_app = Celery(
        "ingestion",
        broker=broker,
        backend=broker,
    )
    _celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )

    # Register tasks now that the app object exists.
    _register_tasks(_celery_app)
    return _celery_app


def _register_tasks(app):
    """Attach task definitions to a Celery app instance."""

    @app.task(
        bind=True,
        name="ingestion.sync_repo",
        autoretry_for=(Exception,),
        retry_backoff=True,
        max_retries=3,
    )
    def sync_repo_task(self, repo_owner: str, repo_name: str, index: bool = True):
        """Fetch issues from GitHub and optionally rebuild embeddings."""
        from app.ingestion.github import sync_repo  # noqa: PLC0415
        from app.ingestion.pipeline import run_index_embeddings  # noqa: PLC0415

        logger.info("sync_repo_task started: %s/%s", repo_owner, repo_name)
        updated, skipped = sync_repo(repo_owner, repo_name)
        logger.info("sync_repo_task: %d upserted, %d skipped", updated, skipped)

        emb_count = 0
        if index:
            _, emb_count = run_index_embeddings()
            logger.info("sync_repo_task: %d embeddings indexed", emb_count)

        return {"updated": updated, "skipped": skipped, "embeddings": emb_count}

    @app.task(
        bind=True,
        name="ingestion.sync_gitlab_project",
        autoretry_for=(Exception,),
        retry_backoff=True,
        max_retries=3,
    )
    def sync_gitlab_project_task(
        self, namespace: str, project_name: str, index: bool = True
    ):
        """Fetch issues from GitLab and optionally rebuild embeddings."""
        from app.ingestion.gitlab import sync_gitlab_project  # noqa: PLC0415
        from app.ingestion.pipeline import run_index_embeddings  # noqa: PLC0415

        logger.info("sync_gitlab_project_task started: %s/%s", namespace, project_name)
        updated, skipped = sync_gitlab_project(namespace, project_name)
        logger.info("sync_gitlab_project_task: %d upserted, %d skipped", updated, skipped)

        emb_count = 0
        if index:
            _, emb_count = run_index_embeddings()

        return {"updated": updated, "skipped": skipped, "embeddings": emb_count}

    return sync_repo_task, sync_gitlab_project_task


# ---------------------------------------------------------------------------
# Convenience helpers for the ingest route
# ---------------------------------------------------------------------------

def enqueue_sync(repo_owner: str, repo_name: str, index: bool = True) -> bool:
    """Enqueue a sync_repo Celery task.

    Returns True if the task was queued, False if Celery is unavailable
    (caller should fall back to BackgroundTasks).
    """
    app = get_celery_app()
    if app is None:
        return False
    try:
        app.send_task(
            "ingestion.sync_repo",
            kwargs={"repo_owner": repo_owner, "repo_name": repo_name, "index": index},
        )
        return True
    except Exception:
        logger.exception("Failed to enqueue sync_repo task")
        return False


def enqueue_sync_gitlab(
    namespace: str, project_name: str, index: bool = True
) -> bool:
    """Enqueue a sync_gitlab_project Celery task. Returns False if unavailable."""
    app = get_celery_app()
    if app is None:
        return False
    try:
        app.send_task(
            "ingestion.sync_gitlab_project",
            kwargs={"namespace": namespace, "project_name": project_name, "index": index},
        )
        return True
    except Exception:
        logger.exception("Failed to enqueue sync_gitlab_project task")
        return False

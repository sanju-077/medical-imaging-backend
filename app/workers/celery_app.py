"""
Celery application configuration for medical imaging backend.
"""
from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "medical_imaging",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'app.workers.tasks',
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
    # Task routing for different queues
    task_routes={
        'app.workers.tasks.process_inference_task': {'queue': 'inference'},
        'app.workers.tasks.process_batch_inference_task': {'queue': 'inference'},
        'app.workers.tasks.cleanup_temp_files_task': {'queue': 'cleanup'},
    },
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-files': {
            'task': 'app.workers.tasks.cleanup_temp_files_task',
            'schedule': 3600.0,  # Every hour
        },
        'update-model-metrics': {
            'task': 'app.workers.tasks.update_model_metrics_task',
            'schedule': 86400.0,  # Daily
        },
    },
)

# Start worker with: celery -A app.workers.celery_app worker --loglevel=info
# Start beat with: celery -A app.workers.celery_app beat --loglevel=info


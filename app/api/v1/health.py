"""
Health check API endpoints.
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check including database connectivity."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check database connectivity
    try:
        db.execute("SELECT 1")
        health_status["services"]["database"] = {
            "status": "healthy",
            "type": "SQLite"
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Redis connectivity (if configured)
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        health_status["services"]["redis"] = {
            "status": "healthy"
        }
    except Exception:
        health_status["services"]["redis"] = {
            "status": "unavailable",
            "note": "Redis not configured or not running"
        }
    
    # Check ML models availability
    try:
        import os
        classification_path = settings.CLASSIFICATION_MODEL_PATH
        segmentation_path = settings.SEGMENTATION_MODEL_PATH
        
        models_status = {}
        if os.path.exists(classification_path):
            models_status["classification"] = {"status": "available"}
        else:
            models_status["classification"] = {"status": "missing"}
        
        if os.path.exists(segmentation_path):
            models_status["segmentation"] = {"status": "available"}
        else:
            models_status["segmentation"] = {"status": "missing"}
        
        health_status["services"]["ml_models"] = models_status
    except Exception as e:
        health_status["services"]["ml_models"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Celery worker status
    try:
        from app.workers.celery_app import celery_app
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        if stats:
            health_status["services"]["celery"] = {
                "status": "healthy",
                "workers": len(stats)
            }
        else:
            health_status["services"]["celery"] = {
                "status": "no_workers",
                "note": "No Celery workers running"
            }
    except Exception:
        health_status["services"]["celery"] = {
            "status": "unavailable",
            "note": "Celery not configured"
        }
    
    # Set overall status based on critical services
    if health_status["services"]["database"]["status"] != "healthy":
        health_status["status"] = "unhealthy"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status


@router.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes/dockerswarm."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/live")
async def liveness_check():
    """Liveness probe for Kubernetes/dockerswarm."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def get_metrics():
    """Application metrics endpoint."""
    import psutil
    import os
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        },
        "application": {
            "pid": os.getpid(),
            "workers": 1  # Single worker for now
        }
    }


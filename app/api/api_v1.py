"""
API router configuration - combines all API endpoints.
"""
from fastapi import APIRouter
from app.api.v1 import auth, images, inference, results, health

# Create main API router
api_router = APIRouter()

# Include all API routers with version prefix
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(images.router, prefix="/images", tags=["images"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
api_router.include_router(results.router, prefix="/results", tags=["results"])
api_router.include_router(health.router, prefix="/health", tags=["health"])

# Root endpoint
@api_router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Medical Imaging Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health/",
        "status": "operational"
    }



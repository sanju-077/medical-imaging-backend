"""
Results retrieval API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.schemas.result import (
    ResultResponse, ResultListResponse, ResultStatistics,
    ExplainabilityResponse
)
from app.services.result_service import ResultService
from app.dependencies import get_db, get_current_active_user
from app.models.user import User

router = APIRouter()


@router.get("/", response_model=ResultListResponse)
async def list_results(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    inference_id: Optional[int] = Query(None),
    model_type: Optional[str] = Query(None),
    disease_type: Optional[str] = Query(None),
    confidence_min: Optional[float] = Query(None, ge=0, le=1),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List inference results with filtering options."""
    result_service = ResultService(db)
    
    results, total = result_service.get_user_results(
        user_id=current_user.id,
        page=page,
        size=size,
        inference_id=inference_id,
        model_type=model_type,
        disease_type=disease_type,
        confidence_min=confidence_min
    )
    
    return ResultListResponse(
        results=[ResultResponse.from_orm(result) for result in results],
        total=total,
        page=page,
        size=size
    )


@router.get("/{result_id}", response_model=ResultResponse)
async def get_result(
    result_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific inference result."""
    result_service = ResultService(db)
    
    result = result_service.get_result_by_id(result_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found"
        )
    
    return ResultResponse.from_orm(result)


@router.get("/{result_id}/explainability", response_model=ExplainabilityResponse)
async def get_result_explainability(
    result_id: int,
    method: str = Query("gradcam", regex="^(gradcam|integrated_gradients)$"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get explainability visualization for a result."""
    result_service = ResultService(db)
    
    result = result_service.get_result_by_id(result_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found"
        )
    
    if result.inference_request.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot generate explainability for incomplete inference"
        )
    
    # Generate explainability
    explainability = result_service.generate_explainability(
        result_id=result_id,
        method=method
    )
    
    return ExplainabilityResponse(
        result_id=result_id,
        method=method,
        heatmap_path=explainability.get("heatmap_path"),
        overlay_path=explainability.get("overlay_path"),
        explanations=explainability.get("explanations", {}),
        generated_at=explainability.get("generated_at")
    )


@router.get("/statistics/overview")
async def get_result_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get inference result statistics for the user."""
    result_service = ResultService(db)
    
    stats = result_service.get_user_statistics(
        user_id=current_user.id,
        days=days
    )
    
    return ResultStatistics(**stats)


@router.get("/statistics/disease-distribution")
async def get_disease_distribution(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get disease type distribution from results."""
    result_service = ResultService(db)
    
    distribution = result_service.get_disease_distribution(
        user_id=current_user.id,
        days=days
    )
    
    return {
        "period_days": days,
        "distribution": distribution,
        "total_results": sum(distribution.values()) if distribution else 0
    }


@router.get("/statistics/model-performance")
async def get_model_performance_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get model performance statistics."""
    result_service = ResultService(db)
    
    performance = result_service.get_model_performance_stats(
        user_id=current_user.id,
        days=days
    )
    
    return {
        "period_days": days,
        "performance": performance
    }


@router.get("/results/export")
async def export_results(
    format: str = Query("csv", regex="^(csv|json)$"),
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export results in CSV or JSON format."""
    result_service = ResultService(db)
    
    export_data = result_service.export_user_results(
        user_id=current_user.id,
        format=format,
        days=days
    )
    
    if format == "csv":
        from fastapi.responses import Response
        return Response(
            content=export_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=results_{days}days.csv"
            }
        )
    else:
        return export_data


@router.delete("/{result_id}")
async def delete_result(
    result_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an inference result."""
    result_service = ResultService(db)
    
    result = result_service.get_result_by_id(result_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found"
        )
    
    result_service.delete_result(result_id, current_user.id)
    
    return {"message": "Result deleted successfully"}


@router.post("/{result_id}/annotate")
async def annotate_result(
    result_id: int,
    annotation: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add annotation to a result."""
    result_service = ResultService(db)
    
    result = result_service.get_result_by_id(result_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found"
        )
    
    updated_result = result_service.annotate_result(
        result_id=result_id,
        user_id=current_user.id,
        annotation=annotation
    )
    
    return ResultResponse.from_orm(updated_result)


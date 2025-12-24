"""
Result schemas for inference result handling.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ResultBase(BaseModel):
    """Base result schema."""
    pass


class ResultResponse(ResultBase):
    """Schema for result response."""
    id: int
    inference_request_id: int
    disease_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    diagnosis: Optional[str] = None
    recommendation: Optional[str] = None
    
    # Result data
    result_data: Optional[Dict[str, Any]] = None
    explainability_data: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    quality_metrics: Optional[Dict[str, Any]] = None
    
    # Human review
    is_reviewed: bool
    reviewer_id: Optional[int] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, result):
        """Create response from ORM model."""
        return cls(
            id=result.id,
            inference_request_id=result.inference_request_id,
            disease_type=result.disease_type,
            confidence=result.confidence,
            diagnosis=result.diagnosis,
            recommendation=result.recommendation,
            result_data=result.result_data,
            explainability_data=result.explainability_data,
            quality_metrics=result.quality_metrics,
            is_reviewed=result.is_reviewed,
            reviewer_id=result.reviewer_id,
            review_notes=result.review_notes,
            reviewed_at=result.reviewed_at,
            created_at=result.created_at,
            updated_at=result.updated_at
        )


class ResultListResponse(BaseModel):
    """Schema for result list response."""
    results: List[ResultResponse]
    total: int
    page: int
    size: int


class ResultStatistics(BaseModel):
    """Schema for result statistics."""
    total_results: int
    average_confidence: float
    disease_distribution: Dict[str, int]
    model_usage: Dict[str, int]
    period_days: int


class ExplainabilityRequest(BaseModel):
    """Schema for explainability request."""
    method: str = Field(default="gradcam", description="Explainability method to use")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ExplainabilityResponse(BaseModel):
    """Schema for explainability response."""
    result_id: int
    method: str
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    explanations: Optional[Dict[str, Any]] = None
    generated_at: datetime


class ResultAnnotation(BaseModel):
    """Schema for result annotation."""
    annotation: Dict[str, Any]
    notes: Optional[str] = None


class ResultExport(BaseModel):
    """Schema for result export."""
    format: str = Field(default="csv", description="Export format (csv, json)")
    date_range: Optional[Dict[str, datetime]] = None
    filters: Optional[Dict[str, Any]] = None


class ResultComparison(BaseModel):
    """Schema for comparing multiple results."""
    result_ids: List[int]
    comparison_type: str = Field(default="side_by_side", description="Comparison type")
    metrics_to_compare: List[str] = Field(default_factory=lambda: ["confidence", "disease_type", "diagnosis"])


class ResultReview(BaseModel):
    """Schema for result review."""
    review_notes: str
    status: str = Field(default="reviewed", description="Review status")
    recommended_action: Optional[str] = None


class ResultFilter(BaseModel):
    """Schema for result filtering."""
    disease_type: Optional[str] = None
    confidence_min: Optional[float] = Field(None, ge=0, le=1)
    confidence_max: Optional[float] = Field(None, ge=0, le=1)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    model_type: Optional[str] = None
    is_reviewed: Optional[bool] = None


class QualityAssessment(BaseModel):
    """Schema for quality assessment."""
    image_quality_score: float = Field(..., ge=0, le=100)
    inference_confidence: float = Field(..., ge=0, le=1)
    overall_quality: str  # excellent, good, fair, poor
    recommendations: List[str]
    quality_factors: Dict[str, float]


class ResultAnalytics(BaseModel):
    """Schema for result analytics."""
    time_period: str
    total_cases: int
    positive_cases: int
    negative_cases: int
    average_confidence: float
    confidence_distribution: Dict[str, int]
    disease_trends: Dict[str, List[int]]
    model_performance: Dict[str, Dict[str, float]]


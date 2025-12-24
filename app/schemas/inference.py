"""
Inference schemas for ML inference operations.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class InferenceRequestBase(BaseModel):
    """Base inference request schema."""
    model_type: str = Field(..., description="Type of ML model to use")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional inference parameters")


class InferenceRequest(InferenceRequestBase):
    """Schema for single image inference request."""
    image_id: int = Field(..., description="ID of the image to analyze")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_models = ["tuberculosis_v1", "pneumonia_v1", "fracture_v1", "unet_v1"]
        if v not in allowed_models:
            raise ValueError(f'Model type must be one of: {allowed_models}')
        return v


class BatchInferenceRequest(InferenceRequestBase):
    """Schema for batch inference request."""
    image_ids: List[int] = Field(..., description="List of image IDs to analyze")
    
    @validator('image_ids')
    def validate_image_ids(cls, v):
        if not v:
            raise ValueError('Image IDs list cannot be empty')
        if len(v) > 50:
            raise ValueError('Batch size cannot exceed 50 images')
        return v


class InferenceResponse(BaseModel):
    """Schema for inference response."""
    request_id: int
    status: str  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class InferenceStatus(BaseModel):
    """Schema for inference status."""
    request_id: int
    status: str
    progress: int = Field(..., ge=0, le=100)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    name: str
    type: str  # classification, segmentation
    description: str
    accuracy: float
    supported_formats: List[str]
    input_size: str
    version: str = "1.0"


class InferenceResult(BaseModel):
    """Schema for inference result."""
    model_type: str
    disease_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    diagnosis: Optional[str] = None
    recommendation: Optional[str] = None
    
    # Classification specific
    predictions: Optional[List[Dict[str, Any]]] = None
    all_probabilities: Optional[Dict[str, float]] = None
    
    # Segmentation specific
    segmentation_mask: Optional[List[List[int]]] = None
    unique_classes: Optional[List[int]] = None
    class_counts: Optional[Dict[int, int]] = None
    class_percentages: Optional[Dict[int, float]] = None
    
    # Explainability
    explainability_data: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    processing_time: Optional[float] = None
    model_metadata: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None


class InferenceHistoryItem(BaseModel):
    """Schema for inference history item."""
    id: int
    image_filename: str
    model_type: str
    status: str
    created_at: datetime
    confidence: Optional[float] = None
    disease_type: Optional[str] = None


class InferenceStatistics(BaseModel):
    """Schema for inference statistics."""
    total_inferences: int
    completed_inferences: int
    failed_inferences: int
    average_processing_time: float
    most_used_model: str
    success_rate: float


class BatchProgress(BaseModel):
    """Schema for batch processing progress."""
    batch_id: str
    total_images: int
    completed_images: int
    failed_images: int
    progress_percentage: float
    estimated_time_remaining: Optional[int] = None  # in seconds


class InferenceParameter(BaseModel):
    """Schema for inference parameters."""
    name: str
    description: str
    type: str  # string, integer, float, boolean
    default_value: Any
    required: bool = False
    options: Optional[List[Any]] = None


class ModelComparison(BaseModel):
    """Schema for model comparison results."""
    image_id: int
    models_compared: List[str]
    results: Dict[str, InferenceResult]
    best_model: str
    confidence_difference: float
    consensus_analysis: Dict[str, Any]


class InferenceError(BaseModel):
    """Schema for inference errors."""
    error_type: str
    error_message: str
    error_code: Optional[int] = None
    suggestion: Optional[str] = None
    timestamp: datetime


class InferencePipelineConfig(BaseModel):
    """Schema for inference pipeline configuration."""
    preprocessing_enabled: bool = True
    enhancement_methods: List[str] = Field(default_factory=lambda: ["clahe", "noise_reduction"])
    target_size: tuple[int, int] = (224, 224)
    batch_size: int = 1
    timeout: int = 300  # seconds
    enable_explainability: bool = True
    explainability_methods: List[str] = Field(default_factory=lambda: ["gradcam", "integrated_gradients"])


class QualityMetrics(BaseModel):
    """Schema for image quality metrics."""
    brightness: float
    contrast: float
    snr_estimate: float
    dynamic_range: int
    quality_score: float = Field(..., ge=0, le=100)
    recommendations: List[str]


class InferenceValidation(BaseModel):
    """Schema for inference request validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    estimated_processing_time: Optional[float] = None


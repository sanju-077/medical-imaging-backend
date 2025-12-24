"""
Image schemas for medical image handling.
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ImageBase(BaseModel):
    """Base image schema."""
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None


class ImageCreate(ImageBase):
    """Schema for creating an image."""
    pass  # File upload is handled separately in FastAPI


class ImageUpdate(BaseModel):
    """Schema for updating image information."""
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None


class ImageResponse(ImageBase):
    """Schema for image response."""
    id: int
    filename: str
    file_size: int
    file_type: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None
    
    # DICOM fields
    dicom_patient_name: Optional[str] = None
    dicom_study_date: Optional[str] = None
    dicom_modality: Optional[str] = None
    
    # Status
    is_processed: bool
    processing_status: str
    quality_score: Optional[float] = None
    
    # Timestamps
    upload_date: datetime
    processed_date: Optional[datetime] = None
    
    # Metadata
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, image):
        """Create response from ORM model."""
        return cls(
            id=image.id,
            filename=image.filename,
            description=image.description,
            tags=image.tags.split(',') if image.tags else None,
            file_size=image.file_size,
            file_type=image.file_type,
            image_width=image.image_width,
            image_height=image.image_height,
            image_format=image.image_format,
            dicom_patient_name=image.dicom_patient_name,
            dicom_study_date=image.dicom_study_date,
            dicom_modality=image.dicom_modality,
            is_processed=image.is_processed,
            processing_status=image.processing_status,
            quality_score=image.quality_score,
            upload_date=image.upload_date,
            processed_date=image.processed_date,
            metadata=image.metadata
        )


class ImageListResponse(BaseModel):
    """Schema for image list response."""
    images: List[ImageResponse]
    total: int
    page: int
    size: int


class ImageMetadata(BaseModel):
    """Schema for image metadata."""
    dicom_patient_name: Optional[str] = None
    dicom_study_date: Optional[str] = None
    dicom_modality: Optional[str] = None
    dicom_study_uid: Optional[str] = None
    dicom_series_uid: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    additional_metadata: Optional[dict] = None


class ImageQualityAssessment(BaseModel):
    """Schema for image quality assessment."""
    overall_score: float = Field(..., ge=0, le=100)
    brightness: str  # dark, normal, bright
    contrast: float = Field(..., ge=0, le=1)
    snr_estimate: float = Field(..., ge=0)
    dynamic_range: int
    recommendations: List[str]


class ImageUploadResponse(BaseModel):
    """Schema for image upload response."""
    message: str
    image: ImageResponse
    quality_assessment: Optional[ImageQualityAssessment] = None


"""
Inference result model for storing ML inference results.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class InferenceRequest(Base):
    """Inference request model for tracking ML inference jobs."""
    __tablename__ = "inference_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Request details
    image_id = Column(Integer, ForeignKey("medical_images.id"), nullable=True)  # Nullable for batch requests
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # ML model information
    model_type = Column(String(100), nullable=False)  # e.g., "tuberculosis_v1", "pneumonia_v1"
    model_version = Column(String(50), default="1.0")
    
    # Request status and progress
    status = Column(String(50), default="pending", nullable=False)  # pending, processing, completed, failed
    progress = Column(Integer, default=0, nullable=False)  # 0-100
    error_message = Column(Text)
    
    # Batch processing
    batch_id = Column(String(100))  # For batch requests
    batch_size = Column(Integer, default=1)
    
    # Processing parameters
    parameters = Column(JSON)  # Store inference parameters as JSON
    
    # Timing
    processing_start_time = Column(DateTime(timezone=True))
    processing_end_time = Column(DateTime(timezone=True))
    processing_duration = Column(Float)  # Duration in seconds
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="inference_requests")
    image = relationship("MedicalImage", back_populates="inference_requests")
    results = relationship("InferenceResult", back_populates="inference_request", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<InferenceRequest(id={self.id}, model_type='{self.model_type}', status='{self.status}')>"
    
    @property
    def is_completed(self) -> bool:
        """Check if inference is completed."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if inference failed."""
        return self.status == "failed"
    
    @property
    def is_processing(self) -> bool:
        """Check if inference is currently processing."""
        return self.status == "processing"


class InferenceResult(Base):
    """Inference result model for storing ML inference results."""
    __tablename__ = "inference_results"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Association with inference request
    inference_request_id = Column(Integer, ForeignKey("inference_requests.id"), nullable=False)
    
    # Result data
    result_data = Column(JSON)  # Store prediction results as JSON
    
    # Classification results
    disease_type = Column(String(100))  # e.g., "tuberculosis", "pneumonia", "normal"
    confidence = Column(Float)  # Confidence score 0-1
    
    # Diagnosis and recommendation
    diagnosis = Column(Text)
    recommendation = Column(Text)
    
    # Additional metadata
    model_metadata = Column(JSON)  # Model version, input preprocessing info, etc.
    explainability_data = Column(JSON)  # Grad-CAM, Integrated Gradients results
    
    # Quality metrics
    quality_metrics = Column(JSON)  # Model accuracy, processing time, etc.
    
    # Human review
    is_reviewed = Column(Boolean, default=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"))
    review_notes = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    inference_request = relationship("InferenceRequest", back_populates="results")
    reviewer = relationship("User", foreign_keys=[reviewer_id])
    
    def __repr__(self):
        return f"<InferenceResult(id={self.id}, disease_type='{self.disease_type}', confidence={self.confidence})>"
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if result has high confidence."""
        return self.confidence >= 0.8 if self.confidence else False
    
    @property
    def is_reviewed_by_human(self) -> bool:
        """Check if result has been reviewed by human."""
        return self.is_reviewed and self.reviewer_id is not None


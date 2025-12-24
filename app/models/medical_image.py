"""
Medical image model for storing uploaded medical images.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class MedicalImage(Base):
    """Medical image model for storing uploaded images."""
    __tablename__ = "medical_images"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_type = Column(String(100), nullable=False)  # MIME type
    
    # Image metadata
    image_width = Column(Integer)
    image_height = Column(Integer)
    image_format = Column(String(50))  # jpg, png, dcm, etc.
    
    # DICOM specific fields
    dicom_patient_name = Column(String(255))
    dicom_study_date = Column(String(50))
    dicom_modality = Column(String(20))
    dicom_study_uid = Column(String(255))
    dicom_series_uid = Column(String(255))
    
    # User association
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Image details
    description = Column(Text)
    tags = Column(Text)  # JSON string of tags
    metadata = Column(Text)  # JSON string of additional metadata
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    quality_score = Column(Float)  # Image quality assessment
    
    # Timestamps
    upload_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processed_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="images")
    inference_requests = relationship("InferenceRequest", back_populates="image", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MedicalImage(id={self.id}, filename='{self.filename}', user_id={self.user_id})>"
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return round(self.file_size / (1024 * 1024), 2)
    
    @property
    def is_dicom(self) -> bool:
        """Check if image is a DICOM file."""
        return self.image_format.lower() in ['dcm', 'dicom']
    
    @property
    def dimensions(self) -> str:
        """Get image dimensions as string."""
        if self.image_width and self.image_height:
            return f"{self.image_width}x{self.image_height}"
        return "Unknown"


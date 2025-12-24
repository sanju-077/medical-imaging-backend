"""
Image service for handling medical image uploads and management.
"""
import os
import uuid
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.medical_image import MedicalImage
from app.core.config import settings
from app.core.logging import medical_logger
from app.utils.file_utils import save_upload_file, validate_file_upload


class ImageService:
    """Service for medical image management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def upload_image(
        self,
        file,
        user_id: int,
        description: Optional[str] = None
    ) -> MedicalImage:
        """Upload and save medical image."""
        # Validate file
        if not validate_file_upload(file, settings.ALLOWED_EXTENSIONS, settings.MAX_FILE_SIZE):
            raise ValueError("Invalid file format or size")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save file
        file_path = save_upload_file(file, unique_filename)
        
        # Create database record
        db_image = MedicalImage(
            filename=file.filename,
            file_path=file_path,
            file_size=file.size,
            file_type=file.content_type,
            user_id=user_id,
            description=description,
            upload_date=datetime.utcnow()
        )
        
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        
        medical_logger.log_image_upload(user_id, file.filename, file.size)
        return db_image
    
    def get_image_by_id(self, image_id: int, user_id: int) -> Optional[MedicalImage]:
        """Get image by ID for specific user."""
        return self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.id == image_id,
                MedicalImage.user_id == user_id
            )
        ).first()
    
    def get_user_images(
        self,
        user_id: int,
        page: int = 1,
        size: int = 20
    ) -> Tuple[list[MedicalImage], int]:
        """Get user's images with pagination."""
        skip = (page - 1) * size
        
        images = self.db.query(MedicalImage).filter(
            MedicalImage.user_id == user_id
        ).offset(skip).limit(size).all()
        
        total = self.db.query(MedicalImage).filter(
            MedicalImage.user_id == user_id
        ).count()
        
        return images, total
    
    def delete_image(self, image_id: int, user_id: int) -> bool:
        """Delete image."""
        image = self.get_image_by_id(image_id, user_id)
        if not image:
            return False
        
        # Delete physical file
        try:
            if os.path.exists(image.file_path):
                os.remove(image.file_path)
        except Exception as e:
            medical_logger.logger.warning(f"Failed to delete file {image.file_path}: {e}")
        
        # Delete database record
        self.db.delete(image)
        self.db.commit()
        
        return True
    
    def update_image_metadata(
        self,
        image_id: int,
        user_id: int,
        metadata: dict
    ) -> Optional[MedicalImage]:
        """Update image metadata."""
        image = self.get_image_by_id(image_id, user_id)
        if not image:
            return None
        
        # Update allowed metadata fields
        allowed_fields = ['description', 'tags', 'metadata']
        for field, value in metadata.items():
            if field in allowed_fields and hasattr(image, field):
                setattr(image, field, value)
        
        image.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(image)
        
        return image
    
    def get_images_by_date_range(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime,
        page: int = 1,
        size: int = 20
    ) -> Tuple[list[MedicalImage], int]:
        """Get images within date range."""
        skip = (page - 1) * size
        
        images = self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.user_id == user_id,
                MedicalImage.upload_date >= start_date,
                MedicalImage.upload_date <= end_date
            )
        ).offset(skip).limit(size).all()
        
        total = self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.user_id == user_id,
                MedicalImage.upload_date >= start_date,
                MedicalImage.upload_date <= end_date
            )
        ).count()
        
        return images, total
    
    def search_images(
        self,
        user_id: int,
        query: str,
        page: int = 1,
        size: int = 20
    ) -> Tuple[list[MedicalImage], int]:
        """Search images by filename or description."""
        skip = (page - 1) * size
        
        images = self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.user_id == user_id,
                MedicalImage.filename.contains(query)
            )
        ).offset(skip).limit(size).all()
        
        total = self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.user_id == user_id,
                MedicalImage.filename.contains(query)
            )
        ).count()
        
        return images, total
    
    def get_image_statistics(self, user_id: int) -> dict:
        """Get image statistics for user."""
        # Total images
        total_images = self.db.query(MedicalImage).filter(
            MedicalImage.user_id == user_id
        ).count()
        
        # Images by type
        image_types = {}
        for img_type in ['.jpg', '.jpeg', '.png', '.dcm', '.nii']:
            count = self.db.query(MedicalImage).filter(
                and_(
                    MedicalImage.user_id == user_id,
                    MedicalImage.filename.endswith(img_type)
                )
            ).count()
            if count > 0:
                image_types[img_type] = count
        
        # Storage usage
        total_size = self.db.query(MedicalImage.file_size).filter(
            MedicalImage.user_id == user_id
        ).all()
        
        storage_used = sum(size[0] for size in total_size if size[0])
        
        # Recent uploads (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_uploads = self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.user_id == user_id,
                MedicalImage.upload_date >= thirty_days_ago
            )
        ).count()
        
        return {
            "total_images": total_images,
            "images_by_type": image_types,
            "storage_used_bytes": storage_used,
            "storage_used_mb": round(storage_used / (1024 * 1024), 2),
            "recent_uploads_30_days": recent_uploads
        }


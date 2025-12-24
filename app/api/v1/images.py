"""
Image upload and management API endpoints.
"""
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
import shutil

from app.schemas.image import ImageUpload, ImageResponse, ImageListResponse
from app.services.image_service import ImageService
from app.dependencies import get_db, get_current_active_user
from app.models.user import User
from app.core.config import settings
from app.utils.file_utils import validate_file_upload, save_upload_file

router = APIRouter()


@router.post("/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    description: Optional[str] = Query(None, max_length=500),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload a medical image."""
    # Validate file
    if not validate_file_upload(file, settings.ALLOWED_EXTENSIONS, settings.MAX_FILE_SIZE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format or size"
        )
    
    image_service = ImageService(db)
    
    try:
        # Save file and create database record
        image = image_service.upload_image(
            file=file,
            user_id=current_user.id,
            description=description
        )
        
        return ImageResponse.from_orm(image)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.get("/", response_model=ImageListResponse)
async def list_images(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List user's uploaded images."""
    image_service = ImageService(db)
    
    images, total = image_service.get_user_images(
        user_id=current_user.id,
        page=page,
        size=size
    )
    
    return ImageListResponse(
        images=[ImageResponse.from_orm(img) for img in images],
        total=total,
        page=page,
        size=size
    )


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get specific image details."""
    image_service = ImageService(db)
    
    image = image_service.get_image_by_id(image_id, current_user.id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return ImageResponse.from_orm(image)


@router.delete("/{image_id}")
async def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an uploaded image."""
    image_service = ImageService(db)
    
    image = image_service.get_image_by_id(image_id, current_user.id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Delete physical file
    try:
        if os.path.exists(image.file_path):
            os.remove(image.file_path)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Failed to delete file: {e}")
    
    # Delete database record
    image_service.delete_image(image_id, current_user.id)
    
    return {"message": "Image deleted successfully"}


@router.get("/{image_id}/download")
async def download_image(
    image_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download an uploaded image."""
    from fastapi.responses import FileResponse
    
    image_service = ImageService(db)
    
    image = image_service.get_image_by_id(image_id, current_user.id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    if not os.path.exists(image.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found"
        )
    
    return FileResponse(
        path=image.file_path,
        media_type='application/octet-stream',
        filename=os.path.basename(image.file_path)
    )


@router.post("/{image_id}/metadata")
async def update_image_metadata(
    image_id: int,
    metadata: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update image metadata."""
    image_service = ImageService(db)
    
    image = image_service.get_image_by_id(image_id, current_user.id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    updated_image = image_service.update_image_metadata(image_id, current_user.id, metadata)
    
    return ImageResponse.from_orm(updated_image)


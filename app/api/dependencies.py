"""
API-specific dependencies.
"""
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies import get_db


def get_current_user_optional(
    db: Session = Depends(get_db)
) -> Optional[dict]:
    """Optional user dependency for public endpoints."""
    # This would extract user from JWT token if present
    # For now, return None to indicate anonymous access
    return None


def validate_content_type(content_type: str) -> bool:
    """Validate content type for file uploads."""
    allowed_types = [
        "image/jpeg", "image/jpg", "image/png", 
        "application/dicom", "application/nifti"
    ]
    return content_type.lower() in allowed_types


def get_pagination_params(
    page: int = 1,
    size: int = 20
) -> tuple[int, int]:
    """Get pagination parameters with validation."""
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be greater than 0"
        )
    
    if size < 1 or size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Size must be between 1 and 100"
        )
    
    return page, size


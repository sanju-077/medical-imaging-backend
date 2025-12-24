"""
FastAPI dependencies for dependency injection.
"""
from typing import Optional, Generator
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta

from app.db.session import get_db
from app.core.config import settings
from app.core.security import verify_token
from app.models.user import User
from app.core.logging import get_logger, security_logger

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer()


def get_current_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (alias for get_current_user).
    """
    return current_user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active superuser (admin).
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user


def get_current_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active doctor or admin.
    """
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor privileges required"
        )
    return current_user


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Used for endpoints that work with or without authentication.
    """
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            return get_current_user(db, credentials)
    except HTTPException:
        pass
    return None


def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session.
    """
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()


def get_request_user_id(request: Request) -> Optional[int]:
    """
    Get user ID from request (if authenticated).
    """
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            if payload:
                user_id = payload.get("sub")
                return int(user_id) if user_id else None
    except Exception:
        pass
    return None


def validate_file_upload(
    max_size: int = None,
    allowed_extensions: list = None
):
    """
    Validate file upload parameters.
    """
    if max_size is None:
        max_size = settings.MAX_FILE_SIZE
    
    if allowed_extensions is None:
        allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    def _validate(file_size: int, filename: str) -> bool:
        # Check file size
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {file_size} exceeds maximum allowed size {max_size}"
            )
        
        # Check file extension
        from pathlib import Path
        file_ext = Path(filename).suffix.lower()
        if file_ext not in [ext.lower() for ext in allowed_extensions]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension {file_ext} not allowed. Allowed: {allowed_extensions}"
            )
        
        return True
    
    return _validate


def check_rate_limit(
    user: User = Depends(get_current_user),
    requests_per_minute: int = settings.API_RATE_LIMIT
):
    """
    Check rate limit for user.
    This is a basic implementation - in production, use Redis-based rate limiting.
    """
    # TODO: Implement proper rate limiting with Redis
    # For now, just log the request
    logger.debug(f"Rate limit check for user {user.id}")


def log_api_request(
    request: Request,
    user: Optional[User] = Depends(get_optional_user)
):
    """
    Log API request for audit trail.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    if user:
        logger.info(f"API_REQUEST - User {user.id} - {request.method} {request.url.path} - {client_ip}")
        security_logger.log_data_access(
            user_id=user.id,
            resource_type="api",
            resource_id=0,
            action=f"{request.method} {request.url.path}"
        )
    else:
        logger.info(f"API_REQUEST - Anonymous - {request.method} {request.url.path} - {client_ip}")


def validate_medical_access(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Validate that user has access to medical data.
    """
    # Check if user has appropriate medical permissions
    if not current_user.is_doctor and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Medical data access requires doctor or admin privileges"
        )
    
    return current_user


def get_inference_permissions(
    model_type: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check if user has permission to use specific inference model.
    """
    # Define model permissions
    model_permissions = {
        "tuberculosis_v1": ["doctor", "admin"],
        "pneumonia_v1": ["doctor", "admin"],
        "fracture_v1": ["doctor", "admin"],
        "unet_v1": ["doctor", "admin"]
    }
    
    required_roles = model_permissions.get(model_type, ["doctor", "admin"])
    
    if current_user.role not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User role '{current_user.role}' not authorized for model '{model_type}'"
        )
    
    return current_user


def validate_batch_size(
    image_count: int,
    max_batch_size: int = settings.BATCH_SIZE
):
    """
    Validate batch processing size.
    """
    if image_count > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {image_count} exceeds maximum allowed size {max_batch_size}"
        )


def get_health_check_permissions(
    current_user: User = Depends(get_current_user)
):
    """
    Check permissions for health check endpoints.
    """
    # Health checks are available to authenticated users
    return current_user


def check_storage_quota(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check user storage quota.
    """
    # TODO: Implement storage quota checking
    # For now, allow unlimited storage
    return current_user


def validate_explainability_request(
    method: str,
    current_user: User = Depends(get_current_user)
):
    """
    Validate explainability request parameters.
    """
    allowed_methods = ["gradcam", "integrated_gradients"]
    
    if method not in allowed_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explainability method '{method}' not supported. Allowed: {allowed_methods}"
        )
    
    if not settings.EXPLAINABILITY_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability features are currently disabled"
        )
    
    return current_user


def get_export_permissions(
    format: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check permissions for data export.
    """
    allowed_formats = ["csv", "json"]
    
    if format not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export format '{format}' not supported. Allowed: {allowed_formats}"
        )
    
    # Only doctors and admins can export data
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Data export requires doctor or admin privileges"
        )
    
    return current_user


# Common dependency combinations
def get_authenticated_user_with_db(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get authenticated user with database session.
    """
    return user, db


def get_user_with_permissions(
    user: User = Depends(get_current_user),
    permissions_required: list = None
):
    """
    Get user with specific permissions.
    """
    if permissions_required:
        user_roles = {
            "doctor": ["doctor", "admin"],
            "admin": ["admin"],
            "user": ["user", "doctor", "admin"]
        }
        
        for permission in permissions_required:
            if permission not in user_roles or current_user.role not in user_roles[permission]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
    
    return user



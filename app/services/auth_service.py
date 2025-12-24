"""
Authentication service for user management.
"""
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.user import User
from app.schemas.auth import UserCreate, PasswordChange
from app.core.security import (
    get_password_hash, verify_password, create_access_token,
    generate_reset_token, verify_reset_token
)
from app.core.logging import medical_logger
from app.core.config import settings


class AuthService:
    """Service for authentication and user management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        if self.get_user_by_email(user_data.email):
            raise ValueError("Email already registered")
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        
        db_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=user_data.is_active,
            role=user_data.role or "user"
        )
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        
        medical_logger.log_user_auth(db_user.id, "register", "system")
        return db_user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            return None
        
        medical_logger.log_user_auth(user.id, "login", "system")
        return user
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user information."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def update_password(self, user: User, new_password: str):
        """Update user password."""
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        medical_logger.log_user_auth(user.id, "password_change", "system")
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        medical_logger.log_user_auth(user_id, "deactivate", "system")
        return True
    
    def activate_user(self, user_id: int) -> bool:
        """Activate user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = True
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        medical_logger.log_user_auth(user_id, "activate", "system")
        return True
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        
        medical_logger.log_user_auth(user_id, "delete", "system")
        return True
    
    def create_reset_token(self, email: str) -> Optional[str]:
        """Create password reset token."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        token = generate_reset_token(email)
        
        # Store token in user record (you might want to add a reset_token field)
        # For now, we'll use the updated_at field as a simple reset token storage
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return token
    
    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using reset token."""
        email = verify_reset_token(token)
        if not email:
            return False
        
        user = self.get_user_by_email(email)
        if not user:
            return False
        
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        medical_logger.log_user_auth(user.id, "password_reset", "system")
        return True
    
    def get_users(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get users with pagination."""
        return self.db.query(User).offset(skip).limit(limit).all()
    
    def search_users(self, query: str) -> list[User]:
        """Search users by email or name."""
        return self.db.query(User).filter(
            or_(
                User.email.contains(query),
                User.full_name.contains(query)
            )
        ).all()
    
    def update_user_role(self, user_id: int, role: str) -> bool:
        """Update user role."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.role = role
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics."""
        user = self.get_user_by_id(user_id)
        if not user:
            return {}
        
        # Get inference count
        inference_count = len(user.inference_requests) if user.inference_requests else 0
        
        # Get image count
        image_count = len(user.images) if user.images else 0
        
        return {
            "user_id": user_id,
            "total_images": image_count,
            "total_inferences": inference_count,
            "last_login": user.last_login,
            "created_at": user.created_at
        }


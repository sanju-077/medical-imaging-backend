"""
Authentication schemas for FastAPI.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: str
    phone_number: Optional[str] = None
    organization: Optional[str] = None
    license_number: Optional[str] = None
    specialization: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str
    role: Optional[str] = "user"
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    organization: Optional[str] = None
    license_number: Optional[str] = None
    specialization: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: int
    is_active: bool
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, user):
        """Create response from ORM model."""
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            phone_number=user.phone_number,
            organization=user.organization,
            license_number=user.license_number,
            specialization=user.specialization,
            is_active=user.is_active,
            role=user.role,
            created_at=user.created_at,
            last_login=user.last_login
        )


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Schema for access token response."""
    access_token: str
    token_type: str
    expires_in: int


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class PasswordReset(BaseModel):
    """Schema for password reset."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserInDB(UserResponse):
    """Schema for user in database (includes hashed password)."""
    hashed_password: str
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for user list response."""
    users: list[UserResponse]
    total: int
    page: int
    size: int


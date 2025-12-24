"""
User model for medical imaging backend.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # User status and role
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(String(50), default="user", nullable=False)  # user, admin, doctor
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Additional fields
    phone_number = Column(String(20))
    organization = Column(String(255))
    license_number = Column(String(100))  # For medical professionals
    specialization = Column(String(255))
    
    # Relationships
    images = relationship("MedicalImage", back_populates="user", cascade="all, delete-orphan")
    inference_requests = relationship("InferenceRequest", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"
    
    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.role == "admin"
    
    @property
    def is_doctor(self) -> bool:
        """Check if user is a doctor."""
        return self.role in ["doctor", "admin"]


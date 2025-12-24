"""
Audit log model for tracking system activities.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class AuditLog(Base):
    """Audit log model for tracking system activities."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system events
    user_email = Column(String(255))
    
    # Action details
    action = Column(String(100), nullable=False)  # login, logout, upload, inference, etc.
    resource_type = Column(String(100))  # image, inference_request, result, user, etc.
    resource_id = Column(Integer)  # ID of the affected resource
    
    # Action details
    description = Column(Text)
    details = Column(JSON)  # Additional action details as JSON
    ip_address = Column(String(45))  # IPv4 or IPv6 address
    user_agent = Column(Text)
    
    # Status and outcome
    status = Column(String(50), default="success")  # success, failure, error
    error_message = Column(Text)
    
    # Metadata
    session_id = Column(String(255))
    request_id = Column(String(255))  # For tracking related requests
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', user_email='{self.user_email}', status='{self.status}')>"
    
    @property
    def is_success(self) -> bool:
        """Check if the action was successful."""
        return self.status == "success"
    
    @property
    def is_failure(self) -> bool:
        """Check if the action failed."""
        return self.status in ["failure", "error"]


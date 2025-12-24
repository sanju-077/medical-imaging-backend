"""
Application configuration settings.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "sqlite:///./medical_imaging.db"
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".tiff", ".tif"]
    
    # ML Models
    CLASSIFICATION_MODEL_PATH: str = "./models/classification"
    SEGMENTATION_MODEL_PATH: str = "./models/segmentation"
    MODEL_CACHE_SIZE: int = 100
    
    # Redis/Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_FILE: str = "./logs/app.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Email (for notifications)
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    FROM_EMAIL: Optional[str] = None
    
    # External APIs
    SENTRY_DSN: Optional[str] = None
    API_RATE_LIMIT: int = 100
    
    # Healthcare Compliance
    ENCRYPTION_KEY: str = "your-encryption-key-for-sensitive-data"
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    DATA_ANONYMIZATION: bool = True
    
    # ML Model Settings
    INFERENCE_TIMEOUT: int = 300  # 5 minutes
    BATCH_SIZE: int = 10
    MAX_CONCURRENT_INFERENCES: int = 5
    
    # Explainability
    EXPLAINABILITY_ENABLED: bool = True
    HEATMAP_SAVE_PATH: str = "./explainability/heatmaps"
    GRADCAM_LAYER_NAME: str = "conv2d_3"
    
    # Monitoring
    HEALTH_CHECK_INTERVAL: int = 30
    METRICS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    
    # Development
    AUTO_RELOAD: bool = True
    SHOW_DOCS: bool = True
    DETAILED_ERRORS: bool = True
    
    # Production
    TRUST_PROXY_HEADERS: bool = False
    SECURE_COOKIES: bool = False
    SSL_VERIFY: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator('ALLOWED_HOSTS', pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                # Handle string representation of list
                import ast
                return ast.literal_eval(v)
            else:
                # Handle comma-separated string
                return [host.strip() for host in v.split(',')]
        return v
    
    @validator('ALLOWED_EXTENSIONS', pre=True)
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        """Validate secret key length."""
        if len(v) < 32:
            raise ValueError('SECRET_KEY must be at least 32 characters long')
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG
    
    @property
    def database_url_safe(self) -> str:
        """Get safe database URL for logging (hides password)."""
        if "postgresql://" in self.DATABASE_URL:
            # Hide password in PostgreSQL URL
            parts = self.DATABASE_URL.split("@")
            if len(parts) > 1:
                return f"{parts[0].split('://')[0]}://***@{parts[1]}"
        return "***hidden***" if "sqlite" not in self.DATABASE_URL else self.DATABASE_URL


# Global settings instance
settings = Settings()



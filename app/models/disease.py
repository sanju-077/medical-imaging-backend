"""
Disease model for storing medical disease information.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class Disease(Base):
    """Disease model for storing medical disease information."""
    __tablename__ = "diseases"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Disease information
    name = Column(String(255), nullable=False, unique=True)
    code = Column(String(50), unique=True, index=True)  # ICD-10 or other medical codes
    category = Column(String(100), nullable=False)  # respiratory, cardiovascular, etc.
    
    # Medical details
    description = Column(Text)
    symptoms = Column(Text)  # JSON string or comma-separated
    risk_factors = Column(Text)
    
    # Severity levels
    severity_levels = Column(Text)  # JSON string: mild, moderate, severe
    
    # AI model associations
    supported_models = Column(Text)  # JSON string of model names that can detect this disease
    
    # Statistics and metadata
    prevalence_rate = Column(Float)  # Population prevalence
    mortality_rate = Column(Float)  # Mortality rate
    treatment_options = Column(Text)
    
    # System fields
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships (if needed for future expansion)
    # results = relationship("InferenceResult", back_populates="disease")
    
    def __repr__(self):
        return f"<Disease(id={self.id}, name='{self.name}', category='{self.category}')>"
    
    @property
    def severity_list(self) -> list:
        """Get severity levels as list."""
        import json
        if self.severity_levels:
            try:
                return json.loads(self.severity_levels)
            except:
                return []
        return []
    
    @property
    def symptoms_list(self) -> list:
        """Get symptoms as list."""
        if self.symptoms:
            return [s.strip() for s in self.symptoms.split(',')]
        return []
    
    @property
    def supported_models_list(self) -> list:
        """Get supported models as list."""
        import json
        if self.supported_models:
            try:
                return json.loads(self.supported_models)
            except:
                return []
        return []


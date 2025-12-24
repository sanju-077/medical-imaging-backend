"""
Database initialization script.
"""
from sqlalchemy.orm import Session
from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.models.user import User
from app.models.disease import Disease
from app.core.security import get_password_hash
import logging

logger = logging.getLogger(__name__)


def create_db():
    """Create all database tables."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def seed_database():
    """Seed the database with initial data."""
    logger.info("Seeding database with initial data...")
    
    db = SessionLocal()
    try:
        # Create default admin user
        admin_user = User(
            email="admin@medical-imaging.com",
            full_name="System Administrator",
            hashed_password=get_password_hash("admin123"),
            role="admin",
            is_active=True,
            organization="Medical Imaging System"
        )
        
        # Check if admin already exists
        existing_admin = db.query(User).filter(User.email == admin_user.email).first()
        if not existing_admin:
            db.add(admin_user)
            db.commit()
            logger.info("Created default admin user")
        
        # Create sample diseases
        diseases_data = [
            {
                "name": "Tuberculosis",
                "code": "A15-A19",
                "category": "respiratory",
                "description": "Bacterial infection that mainly affects the lungs",
                "symptoms": "persistent cough, chest pain, weight loss, fever",
                "risk_factors": "HIV, malnutrition, diabetes, smoking",
                "severity_levels": "latent, active, severe",
                "supported_models": "tuberculosis_v1",
                "prevalence_rate": 0.134,
                "mortality_rate": 0.015
            },
            {
                "name": "Pneumonia",
                "code": "J12-J18",
                "category": "respiratory",
                "description": "Infection that inflames air sacs in one or both lungs",
                "symptoms": "cough, fever, chills, difficulty breathing",
                "risk_factors": "age, chronic disease, smoking, hospitalization",
                "severity_levels": "mild, moderate, severe",
                "supported_models": "pneumonia_v1",
                "prevalence_rate": 0.073,
                "mortality_rate": 0.05
            },
            {
                "name": "Bone Fracture",
                "code": "S02-S72",
                "category": "musculoskeletal",
                "description": "Break in the continuity of a bone",
                "symptoms": "pain, swelling, deformity, inability to move",
                "risk_factors": "trauma, osteoporosis, age, sports injury",
                "severity_levels": "simple, compound, comminuted",
                "supported_models": "fracture_v1",
                "prevalence_rate": 0.089,
                "mortality_rate": 0.008
            }
        ]
        
        for disease_data in diseases_data:
            existing_disease = db.query(Disease).filter(Disease.name == disease_data["name"]).first()
            if not existing_disease:
                disease = Disease(**disease_data)
                db.add(disease)
                db.commit()
                logger.info(f"Created disease: {disease_data['name']}")
        
        db.commit()
        logger.info("Database seeding completed successfully")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


def init_db():
    """Initialize the database."""
    logger.info("Initializing database...")
    create_db()
    seed_database()
    logger.info("Database initialization completed")


if __name__ == "__main__":
    init_db()


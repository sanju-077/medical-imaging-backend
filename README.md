# Medical Imaging Backend

A comprehensive FastAPI-based backend for AI-powered medical image analysis with support for inference, explainability, and result management.

## 🏗️ Architecture Overview

This project implements a complete medical imaging analysis system with the following components:

### API Layer
- **Authentication**: JWT-based auth with role-based access control
- **Image Management**: Secure upload, storage, and metadata handling
- **Inference**: ML model integration with batch processing support
- **Results**: Result storage, review, and export functionality
- **Health Monitoring**: Comprehensive health checks and metrics

### Business Logic Layer
- **Service Layer**: Separated business logic from API endpoints
- **Security**: HIPAA-compliant security measures and audit logging
- **File Management**: Secure file handling with validation and encryption

### ML Infrastructure
- **Model Management**: Dynamic model loading and caching
- **Inference Pipeline**: Unified pipeline for classification and segmentation
- **Explainability**: Grad-CAM and Integrated Gradients support
- **Batch Processing**: Celery-based async processing

### Data Layer
- **Database**: SQLAlchemy ORM with multiple model support
- **Storage**: File storage with backup and recovery
- **Audit**: Comprehensive audit logging for compliance

## 📁 Project Structure

```
medical-imaging-backend/
├── app/
│   ├── api/v1/                 # API endpoints
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── images.py          # Image management endpoints
│   │   ├── inference.py       # ML inference endpoints
│   │   ├── results.py         # Result management endpoints
│   │   └── health.py          # Health check endpoints
│   ├── core/                  # Core infrastructure
│   │   ├── config.py          # Configuration management
│   │   ├── security.py        # Security utilities
│   │   └── logging.py         # Logging configuration
│   ├── db/                    # Database layer
│   │   ├── base.py           # SQLAlchemy base
│   │   ├── session.py        # Database session management
│   │   └── init_db.py        # Database initialization
│   ├── models/               # Database models
│   │   ├── user.py          # User model
│   │   ├── medical_image.py  # Medical image model
│   │   ├── inference_result.py # Inference result model
│   │   ├── disease.py       # Disease information model
│   │   └── audit_log.py     # Audit logging model
│   ├── schemas/             # Pydantic schemas
│   │   ├── auth.py         # Authentication schemas
│   │   ├── image.py        # Image schemas
│   │   ├── inference.py    # Inference schemas
│   │   └── result.py       # Result schemas
│   ├── services/           # Business logic layer
│   │   ├── auth_service.py     # Authentication service
│   │   ├── image_service.py    # Image management service
│   │   ├── inference_service.py # Inference orchestration
│   │   ├── result_service.py   # Result management service
│   │   └── storage_service.py  # File storage service
│   ├── ml/                 # Machine learning infrastructure
│   │   ├── models/        # ML model implementations
│   │   ├── preprocessing/ # Image preprocessing
│   │   ├── inference/     # Inference pipeline
│   │   └── explainability/ # Explainability features
│   ├── utils/             # Utility functions
│   │   └── file_utils.py  # File handling utilities
│   ├── workers/           # Background task workers
│   │   ├── celery_app.py  # Celery configuration
│   │   └── tasks.py       # Background tasks
│   ├── api_v1.py         # API router configuration
│   ├── dependencies.py   # FastAPI dependencies
│   └── main.py           # FastAPI application
├── requirements.txt      # Python dependencies
├── .env.example         # Environment configuration template
└── COMPLETION_PLAN.md   # Project completion plan
```

## 🚀 Features

### Security & Compliance
- JWT-based authentication with refresh tokens
- Role-based access control (user, doctor, admin)
- HIPAA-compliant audit logging
- Data encryption for sensitive information
- Rate limiting and security headers

### Medical Image Processing
- Support for multiple image formats (JPEG, PNG, DICOM)
- Image quality assessment and validation
- Metadata extraction and storage
- Secure file storage with backup

### AI/ML Integration
- Model-agnostic inference pipeline
- Support for classification and segmentation models
- Batch processing with progress tracking
- Explainable AI with Grad-CAM and Integrated Gradients
- Model performance monitoring

### Data Management
- Comprehensive result storage and retrieval
- Human review workflow for AI results
- Data export in multiple formats (CSV, JSON)
- Audit trail for all data access and modifications

### System Monitoring
- Health checks for all system components
- Performance metrics and logging
- Error tracking and alerting
- Resource usage monitoring

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- Redis (for Celery workers)
- SQLite (default) or PostgreSQL

### Installation

1. **Clone and setup environment:**
   ```bash
   cd medical-imaging-backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Initialize database:**
   ```bash
   python -c "from app.db.init_db import init_db; init_db()"
   ```

4. **Start the application:**
   ```bash
   # Start FastAPI server
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   
   # Start Celery worker (in separate terminal)
   celery -A app.workers.celery_app worker --loglevel=info
   ```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 👥 User Roles

### User
- Upload medical images
- View their own results
- Request basic inference

### Doctor
- All user privileges
- Access to advanced inference models
- Review and annotate AI results
- Export data in various formats

### Admin
- All doctor privileges
- User management
- System monitoring
- Model management

## 🔒 Security Features

- **Authentication**: JWT tokens with configurable expiration
- **Authorization**: Role-based access control
- **Audit Logging**: All user actions logged for compliance
- **Data Encryption**: Sensitive data encrypted at rest
- **Rate Limiting**: API rate limiting to prevent abuse
- **CORS**: Configurable cross-origin resource sharing
- **Security Headers**: Comprehensive security headers

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/logout` - User logout

### Images
- `POST /api/v1/images/upload` - Upload medical image
- `GET /api/v1/images/` - List user images
- `GET /api/v1/images/{image_id}` - Get image details
- `DELETE /api/v1/images/{image_id}` - Delete image

### Inference
- `POST /api/v1/inference/predict` - Run inference on image
- `POST /api/v1/inference/batch` - Batch inference
- `GET /api/v1/inference/status/{request_id}` - Check inference status
- `GET /api/v1/inference/history` - Inference history

### Results
- `GET /api/v1/results/` - List results
- `GET /api/v1/results/{result_id}` - Get result details
- `PUT /api/v1/results/{result_id}/review` - Review result
- `POST /api/v1/results/export` - Export results

### Health
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed system health
- `GET /api/v1/health/metrics` - System metrics

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=sqlite:///./medical_imaging.db

# Security
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800  # 50MB

# ML Models
CLASSIFICATION_MODEL_PATH=./models/classification
SEGMENTATION_MODEL_PATH=./models/segmentation

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_auth.py
```

## 📈 Monitoring

The application includes comprehensive monitoring:

- **Health Checks**: Monitor database, Redis, ML models, and Celery workers
- **Performance Metrics**: Track inference times, success rates, and resource usage
- **Audit Logging**: Track all user actions for compliance
- **Error Tracking**: Centralized error logging and alerting

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t medical-imaging-backend .

# Run with docker-compose
docker-compose up -d
```

### Production Considerations
- Use PostgreSQL for production database
- Configure proper Redis cluster
- Set up SSL/TLS certificates
- Configure proper logging aggregation
- Set up monitoring and alerting
- Implement proper backup strategies

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions, please open an issue in the repository.

---

**Note**: This is a medical application. Ensure compliance with relevant healthcare regulations (HIPAA, GDPR, etc.) before deploying to production.
# healtcare-

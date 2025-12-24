"""
Inference service for running ML model inferences on medical images.
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.inference_result import InferenceRequest
from app.models.medical_image import MedicalImage
from app.core.logging import medical_logger
from app.ml.inference.pipeline import InferencePipeline
from app.core.config import settings


class InferenceService:
    """Service for managing ML inference operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.pipeline = InferencePipeline()
    
    def get_image_by_id(self, image_id: int, user_id: int) -> Optional[MedicalImage]:
        """Get image by ID for specific user."""
        return self.db.query(MedicalImage).filter(
            and_(
                MedicalImage.id == image_id,
                MedicalImage.user_id == user_id
            )
        ).first()
    
    def create_inference_request(
        self,
        image_id: int,
        user_id: int,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> InferenceRequest:
        """Create a new inference request."""
        request = InferenceRequest(
            image_id=image_id,
            user_id=user_id,
            model_type=model_type,
            parameters=parameters or {},
            status="pending",
            progress=0,
            created_at=datetime.utcnow()
        )
        
        self.db.add(request)
        self.db.commit()
        self.db.refresh(request)
        
        return request
    
    def create_batch_inference_request(
        self,
        image_ids: List[int],
        user_id: int,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> InferenceRequest:
        """Create a batch inference request."""
        request = InferenceRequest(
            image_id=None,  # Will be set for batch
            user_id=user_id,
            model_type=model_type,
            parameters=parameters or {},
            status="pending",
            progress=0,
            created_at=datetime.utcnow(),
            batch_id=str(int(time.time()))  # Simple batch ID
        )
        
        self.db.add(request)
        self.db.commit()
        self.db.refresh(request)
        
        return request
    
    def run_inference(
        self,
        request_id: int,
        image_path: str,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run inference on a single image."""
        start_time = time.time()
        
        try:
            # Update request status
            self.update_inference_status(request_id, "processing", progress=10)
            
            # Run inference using pipeline
            medical_logger.log_inference_start(
                user_id=0,  # Will be set by caller
                image_id=0,  # Will be set by caller
                model_type=model_type
            )
            
            result = self.pipeline.run_inference(
                image_path=image_path,
                model_type=model_type,
                parameters=parameters or {}
            )
            
            processing_time = time.time() - start_time
            
            # Update request status
            self.update_inference_status(request_id, "completed", progress=100)
            
            medical_logger.log_inference_complete(
                user_id=0,  # Will be set by caller
                image_id=0,  # Will be set by caller
                confidence=result.get('confidence', 0),
                processing_time=processing_time
            )
            
            return result
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update request status to failed
            self.update_inference_status(
                request_id, 
                "failed", 
                progress=0, 
                error_message=str(e)
            )
            
            medical_logger.log_inference_error(
                user_id=0,  # Will be set by caller
                image_id=0,  # Will be set by caller
                error=str(e),
                model_type=model_type
            )
            
            raise
    
    def update_inference_status(
        self,
        request_id: int,
        status: str,
        progress: int = 0,
        error_message: Optional[str] = None
    ):
        """Update inference request status."""
        request = self.db.query(InferenceRequest).filter(
            InferenceRequest.id == request_id
        ).first()
        
        if request:
            request.status = status
            request.progress = progress
            request.updated_at = datetime.utcnow()
            
            if error_message:
                request.error_message = error_message
            
            self.db.commit()
    
    def get_inference_request(self, request_id: int, user_id: int) -> Optional[InferenceRequest]:
        """Get inference request by ID for user."""
        return self.db.query(InferenceRequest).filter(
            and_(
                InferenceRequest.id == request_id,
                InferenceRequest.user_id == user_id
            )
        ).first()
    
    def get_inference_result(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Get inference result for request."""
        request = self.db.query(InferenceRequest).filter(
            InferenceRequest.id == request_id
        ).first()
        
        if not request or request.status != "completed":
            return None
        
        # Return result from request parameters or stored result
        # This would typically be stored in a separate results table
        return request.parameters.get('result', {})
    
    def get_user_inference_history(
        self,
        user_id: int,
        page: int = 1,
        size: int = 20,
        model_type: Optional[str] = None
    ) -> List[InferenceRequest]:
        """Get user's inference history."""
        skip = (page - 1) * size
        
        query = self.db.query(InferenceRequest).filter(
            InferenceRequest.user_id == user_id
        )
        
        if model_type:
            query = query.filter(InferenceRequest.model_type == model_type)
        
        return query.order_by(
            InferenceRequest.created_at.desc()
        ).offset(skip).limit(size).all()
    
    def delete_inference_request(self, request_id: int, user_id: int) -> bool:
        """Delete inference request."""
        request = self.get_inference_request(request_id, user_id)
        if not request:
            return False
        
        self.db.delete(request)
        self.db.commit()
        
        return True
    
    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of available ML models."""
        return {
            "classification": [
                {
                    "name": "tuberculosis_v1",
                    "type": "classification",
                    "description": "Tuberculosis detection from chest X-rays",
                    "accuracy": 0.94,
                    "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                    "model_path": f"{settings.CLASSIFICATION_MODEL_PATH}/tuberculosis_v1.onnx"
                },
                {
                    "name": "pneumonia_v1",
                    "type": "classification",
                    "description": "Pneumonia detection from chest X-rays",
                    "accuracy": 0.96,
                    "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                    "model_path": f"{settings.CLASSIFICATION_MODEL_PATH}/pneumonia_v1.onnx"
                },
                {
                    "name": "fracture_v1",
                    "type": "classification",
                    "description": "Bone fracture detection from X-rays",
                    "accuracy": 0.92,
                    "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                    "model_path": f"{settings.CLASSIFICATION_MODEL_PATH}/fracture_v1.onnx"
                }
            ],
            "segmentation": [
                {
                    "name": "unet_v1",
                    "type": "segmentation",
                    "description": "Organ segmentation using U-Net",
                    "accuracy": 0.89,
                    "supported_formats": [".jpg", ".jpeg", ".png", ".dcm", ".nii"],
                    "model_path": f"{settings.SEGMENTATION_MODEL_PATH}/unet_v1.onnx"
                }
            ]
        }
    
    def validate_model_request(self, model_type: str, image_path: str) -> bool:
        """Validate model request against image."""
        models = self.get_available_models()
        
        # Check if model type exists
        model_found = False
        for category in models.values():
            for model in category:
                if model['name'] == model_type:
                    model_found = True
                    break
            if model_found:
                break
        
        if not model_found:
            return False
        
        # Check if image format is supported
        file_extension = image_path.lower().split('.')[-1]
        supported_formats = []
        
        for category in models.values():
            for model in category:
                if model['name'] == model_type:
                    supported_formats = model['supported_formats']
                    break
            if supported_formats:
                break
        
        return f'.{file_extension}' in supported_formats
    
    def get_inference_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get inference statistics for user."""
        # Total inferences
        total_inferences = self.db.query(InferenceRequest).filter(
            InferenceRequest.user_id == user_id
        ).count()
        
        # Inferences by status
        status_counts = {}
        for status in ['pending', 'processing', 'completed', 'failed']:
            count = self.db.query(InferenceRequest).filter(
                and_(
                    InferenceRequest.user_id == user_id,
                    InferenceRequest.status == status
                )
            ).count()
            status_counts[status] = count
        
        # Inferences by model type
        model_counts = {}
        for model_type in ['tuberculosis_v1', 'pneumonia_v1', 'fracture_v1', 'unet_v1']:
            count = self.db.query(InferenceRequest).filter(
                and_(
                    InferenceRequest.user_id == user_id,
                    InferenceRequest.model_type == model_type
                )
            ).count()
            if count > 0:
                model_counts[model_type] = count
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.utcnow() - datetime.timedelta(days=30)
        recent_inferences = self.db.query(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceRequest.created_at >= thirty_days_ago
            )
        ).count()
        
        return {
            "total_inferences": total_inferences,
            "status_distribution": status_counts,
            "model_distribution": model_counts,
            "recent_inferences_30_days": recent_inferences
        }


"""
Result service for managing inference results and analytics.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from app.models.inference_result import InferenceResult, InferenceRequest
from app.models.medical_image import MedicalImage
from app.core.logging import medical_logger


class ResultService:
    """Service for managing inference results and analytics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_result_by_id(self, result_id: int, user_id: int) -> Optional[InferenceResult]:
        """Get result by ID for specific user."""
        return self.db.query(InferenceResult).join(InferenceRequest).filter(
            and_(
                InferenceResult.id == result_id,
                InferenceRequest.user_id == user_id
            )
        ).first()
    
    def get_user_results(
        self,
        user_id: int,
        page: int = 1,
        size: int = 20,
        inference_id: Optional[int] = None,
        model_type: Optional[str] = None,
        disease_type: Optional[str] = None,
        confidence_min: Optional[float] = None
    ) -> Tuple[List[InferenceResult], int]:
        """Get user's results with filtering and pagination."""
        skip = (page - 1) * size
        
        query = self.db.query(InferenceResult).join(InferenceRequest).filter(
            InferenceRequest.user_id == user_id
        )
        
        # Apply filters
        if inference_id:
            query = query.filter(InferenceResult.inference_request_id == inference_id)
        
        if model_type:
            query = query.filter(InferenceRequest.model_type == model_type)
        
        if disease_type:
            query = query.filter(InferenceResult.disease_type == disease_type)
        
        if confidence_min is not None:
            query = query.filter(InferenceResult.confidence >= confidence_min)
        
        results = query.order_by(desc(InferenceResult.created_at)).offset(skip).limit(size).all()
        
        # Get total count with same filters
        total_query = self.db.query(InferenceResult).join(InferenceRequest).filter(
            InferenceRequest.user_id == user_id
        )
        
        if inference_id:
            total_query = total_query.filter(InferenceResult.inference_request_id == inference_id)
        if model_type:
            total_query = total_query.filter(InferenceRequest.model_type == model_type)
        if disease_type:
            total_query = total_query.filter(InferenceResult.disease_type == disease_type)
        if confidence_min is not None:
            total_query = total_query.filter(InferenceResult.confidence >= confidence_min)
        
        total = total_query.count()
        
        return results, total
    
    def create_result(
        self,
        inference_request_id: int,
        result_data: Dict[str, Any],
        disease_type: str,
        confidence: float,
        diagnosis: str,
        recommendation: Optional[str] = None
    ) -> InferenceResult:
        """Create a new inference result."""
        result = InferenceResult(
            inference_request_id=inference_request_id,
            disease_type=disease_type,
            confidence=confidence,
            diagnosis=diagnosis,
            recommendation=recommendation,
            result_data=result_data,
            created_at=datetime.utcnow()
        )
        
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        
        return result
    
    def generate_explainability(
        self,
        result_id: int,
        method: str = "gradcam"
    ) -> Dict[str, Any]:
        """Generate explainability visualization for a result."""
        result = self.db.query(InferenceResult).filter(
            InferenceResult.id == result_id
        ).first()
        
        if not result:
            raise ValueError("Result not found")
        
        # This would integrate with actual ML explainability methods
        # For now, return mock data
        explainability_data = {
            "method": method,
            "heatmap_path": f"/explanations/result_{result_id}_heatmap.png",
            "overlay_path": f"/explanations/result_{result_id}_overlay.png",
            "explanations": {
                "important_regions": [
                    {"x": 120, "y": 80, "importance": 0.85},
                    {"x": 200, "y": 150, "importance": 0.72}
                ],
                "feature_importance": {
                    "edge_enhancement": 0.78,
                    "texture_analysis": 0.65,
                    "contrast_patterns": 0.82
                },
                "model_reasoning": "The model identified abnormal density patterns in the lower right lung region, indicating potential pneumonia."
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return explainability_data
    
    def get_user_statistics(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total results in period
        total_results = self.db.query(InferenceResult).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).count()
        
        # Results by disease type
        disease_stats = self.db.query(
            InferenceResult.disease_type,
            func.count(InferenceResult.id).label('count')
        ).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).group_by(InferenceResult.disease_type).all()
        
        disease_distribution = {disease: count for disease, count in disease_stats}
        
        # Average confidence
        avg_confidence = self.db.query(func.avg(InferenceResult.confidence)).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).scalar() or 0.0
        
        # Daily activity
        daily_stats = self.db.query(
            func.date(InferenceResult.created_at).label('date'),
            func.count(InferenceResult.id).label('count')
        ).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).group_by(func.date(InferenceResult.created_at)).all()
        
        daily_activity = {str(date): count for date, count in daily_stats}
        
        # Model usage
        model_stats = self.db.query(
            InferenceRequest.model_type,
            func.count(InferenceResult.id).label('count')
        ).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).group_by(InferenceRequest.model_type).all()
        
        model_usage = {model: count for model, count in model_stats}
        
        return {
            "period_days": days,
            "total_results": total_results,
            "average_confidence": round(avg_confidence, 3),
            "disease_distribution": disease_distribution,
            "daily_activity": daily_activity,
            "model_usage": model_usage
        }
    
    def get_disease_distribution(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, int]:
        """Get disease type distribution for user."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        results = self.db.query(
            InferenceResult.disease_type,
            func.count(InferenceResult.id).label('count')
        ).join(InferenceRequest).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).group_by(InferenceResult.disease_type).all()
        
        return {disease: count for disease, count in results}
    
    def get_model_performance_stats(
        self,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get model performance statistics."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Performance by model
        model_stats = self.db.query(
            InferenceRequest.model_type,
            func.count(InferenceResult.id).label('total_inferences'),
            func.avg(InferenceResult.confidence).label('avg_confidence'),
            func.count(
                func.case([(InferenceResult.confidence >= 0.8, 1)])
            ).label('high_confidence_count')
        ).join(InferenceResult).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).group_by(InferenceRequest.model_type).all()
        
        performance_data = {}
        for model_type, total, avg_conf, high_conf in model_stats:
            performance_data[model_type] = {
                "total_inferences": total,
                "average_confidence": round(avg_conf, 3) if avg_conf else 0,
                "high_confidence_percentage": round(
                    (high_conf / total * 100) if total > 0 else 0, 2
                ),
                "accuracy_rating": self._get_accuracy_rating(avg_conf)
            }
        
        return performance_data
    
    def _get_accuracy_rating(self, confidence: float) -> str:
        """Get accuracy rating based on confidence score."""
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.8:
            return "Good"
        elif confidence >= 0.7:
            return "Fair"
        else:
            return "Poor"
    
    def export_user_results(
        self,
        user_id: int,
        format: str = "csv",
        days: int = 30
    ) -> str:
        """Export user results in specified format."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        results = self.db.query(
            InferenceResult,
            InferenceRequest,
            MedicalImage.filename
        ).join(InferenceRequest).join(MedicalImage).filter(
            and_(
                InferenceRequest.user_id == user_id,
                InferenceResult.created_at >= start_date
            )
        ).all()
        
        if format == "json":
            return self._export_to_json(results, days)
        else:
            return self._export_to_csv(results, days)
    
    def _export_to_json(self, results: List, days: int) -> str:
        """Export results to JSON format."""
        import json
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "period_days": days,
            "total_results": len(results),
            "results": []
        }
        
        for result, request, image in results:
            export_data["results"].append({
                "id": result.id,
                "image_filename": image.filename,
                "model_type": request.model_type,
                "disease_type": result.disease_type,
                "confidence": result.confidence,
                "diagnosis": result.diagnosis,
                "created_at": result.created_at.isoformat()
            })
        
        return json.dumps(export_data, indent=2)
    
    def _export_to_csv(self, results: List, days: int) -> str:
        """Export results to CSV format."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Result ID", "Image Filename", "Model Type", "Disease Type",
            "Confidence", "Diagnosis", "Created At"
        ])
        
        # Write data
        for result, request, image in results:
            writer.writerow([
                result.id,
                image.filename,
                request.model_type,
                result.disease_type,
                result.confidence,
                result.diagnosis,
                result.created_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        return output.getvalue()
    
    def delete_result(self, result_id: int, user_id: int) -> bool:
        """Delete inference result."""
        result = self.get_result_by_id(result_id, user_id)
        if not result:
            return False
        
        self.db.delete(result)
        self.db.commit()
        
        return True
    
    def annotate_result(
        self,
        result_id: int,
        user_id: int,
        annotation: Dict[str, Any]
    ) -> Optional[InferenceResult]:
        """Add annotation to a result."""
        result = self.get_result_by_id(result_id, user_id)
        if not result:
            return None
        
        # Store annotation in result_data or separate field
        if hasattr(result, 'annotations'):
            result.annotations = annotation
        else:
            # Store in result_data if no annotations field exists
            if result.result_data is None:
                result.result_data = {}
            result.result_data['annotation'] = annotation
        
        result.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(result)
        
        return result
    
    def get_result_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get detailed analytics for results."""
        # This would provide more detailed analytics
        # Implementation depends on specific analytics requirements
        return {
            "trend_analysis": "Available in premium version",
            "comparative_analysis": "Available in premium version",
            "recommendation_engine": "Available in premium version"
        }


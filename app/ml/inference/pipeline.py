"""
Inference pipeline for orchestrating ML model inference.
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime

from app.ml.models.model_loader import model_manager
from app.ml.preprocessing.image_processor import ImageProcessor
from app.ml.preprocessing.validators import ImageValidator
from app.core.logging import medical_logger
from app.core.config import settings


class InferencePipeline:
    """Main inference pipeline for medical image analysis."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.image_validator = ImageValidator()
        self.model_manager = model_manager
        
        # Preload commonly used models
        self._preload_models()
    
    def _preload_models(self):
        """Preload commonly used models for faster inference."""
        common_models = ["tuberculosis_v1", "pneumonia_v1", "unet_v1"]
        for model_name in common_models:
            try:
                self.model_manager.load_model(model_name)
            except Exception as e:
                medical_logger.logger.warning(f"Failed to preload model {model_name}: {e}")
    
    def run_inference(
        self,
        image_path: str,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete inference pipeline on an image.
        
        Args:
            image_path: Path to the medical image
            model_type: Type of model to use
            parameters: Additional inference parameters
            
        Returns:
            Dict containing inference results and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input image
            validation_result = self.image_validator.validate_image_quality(image_path)
            if not validation_result["is_valid"]:
                raise ValueError(f"Image validation failed: {validation_result['errors']}")
            
            # Validate model input compatibility
            if not self.model_manager.validate_model_input(model_type, image_path):
                raise ValueError(f"Model {model_type} cannot process image {image_path}")
            
            # Preprocess image if needed
            preprocessing_params = parameters.get("preprocessing", {}) if parameters else {}
            enhancement = preprocessing_params.get("enhance", True)
            target_size = preprocessing_params.get("target_size", (224, 224))
            
            # Get model
            model = self.model_manager.get_model(model_type)
            if model is None:
                raise ValueError(f"Model {model_type} not available")
            
            # Run inference
            result = model.predict(image_path)
            
            # Add pipeline metadata
            processing_time = time.time() - start_time
            
            result.update({
                "pipeline_metadata": {
                    "model_type": model_type,
                    "processing_time": round(processing_time, 3),
                    "timestamp": datetime.utcnow().isoformat(),
                    "image_validation": validation_result,
                    "preprocessing_applied": enhancement,
                    "target_size": target_size
                }
            })
            
            medical_logger.logger.info(
                f"Inference pipeline completed - Model: {model_type}, "
                f"Time: {processing_time:.3f}s, Image: {image_path}"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            medical_logger.logger.error(
                f"Inference pipeline failed - Model: {model_type}, "
                f"Error: {str(e)}, Time: {processing_time:.3f}s, Image: {image_path}"
            )
            raise
    
    def run_multi_model_inference(
        self,
        image_path: str,
        model_types: list,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run inference using multiple models on the same image."""
        start_time = time.time()
        results = {}
        
        try:
            for model_type in model_types:
                result = self.run_inference(image_path, model_type, parameters)
                results[model_type] = result
            
            processing_time = time.time() - start_time
            
            # Aggregate results
            multi_result = {
                "multi_model_results": results,
                "aggregated_metadata": {
                    "models_used": model_types,
                    "total_processing_time": round(processing_time, 3),
                    "timestamp": datetime.utcnow().isoformat(),
                    "number_of_models": len(model_types)
                }
            }
            
            medical_logger.logger.info(
                f"Multi-model inference completed - Models: {model_types}, "
                f"Time: {processing_time:.3f}s, Image: {image_path}"
            )
            
            return multi_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            medical_logger.logger.error(
                f"Multi-model inference failed - Models: {model_types}, "
                f"Error: {str(e)}, Time: {processing_time:.3f}s, Image: {image_path}"
            )
            raise
    
    def run_sequential_inference(
        self,
        image_path: str,
        model_sequence: list,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run sequential inference where each model's output feeds into the next."""
        start_time = time.time()
        intermediate_results = []
        current_input = image_path
        
        try:
            for i, model_type in enumerate(model_sequence):
                # Run inference on current input
                result = self.run_inference(current_input, model_type, parameters)
                
                # Store intermediate result
                intermediate_results.append({
                    "step": i + 1,
                    "model_type": model_type,
                    "result": result
                })
                
                # Prepare input for next model (if needed)
                # This could involve image processing, segmentation, etc.
                current_input = self._prepare_input_for_next_model(
                    result, model_sequence[i + 1] if i + 1 < len(model_sequence) else None
                )
            
            processing_time = time.time() - start_time
            
            sequential_result = {
                "sequential_results": intermediate_results,
                "final_result": intermediate_results[-1]["result"] if intermediate_results else None,
                "sequence_metadata": {
                    "model_sequence": model_sequence,
                    "total_processing_time": round(processing_time, 3),
                    "timestamp": datetime.utcnow().isoformat(),
                    "steps_completed": len(intermediate_results)
                }
            }
            
            medical_logger.logger.info(
                f"Sequential inference completed - Sequence: {model_sequence}, "
                f"Time: {processing_time:.3f}s, Image: {image_path}"
            )
            
            return sequential_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            medical_logger.logger.error(
                f"Sequential inference failed - Sequence: {model_sequence}, "
                f"Error: {str(e)}, Time: {processing_time:.3f}s, Image: {image_path}"
            )
            raise
    
    def _prepare_input_for_next_model(self, current_result: Dict[str, Any], next_model_type: Optional[str]) -> str:
        """
        Prepare output of current model as input for next model.
        
        This is a placeholder for more sophisticated pipeline logic.
        In practice, this might involve:
        - Extracting regions of interest from segmentation masks
        - Cropping images based on classification results
        - Creating composite images from multiple models
        """
        # For now, just return the original image path
        # In a real implementation, this would modify the image based on current results
        return current_result.get("image_path", "")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and model availability."""
        return {
            "pipeline_status": "operational",
            "loaded_models": self.model_manager.get_loaded_models(),
            "available_models": list(self.model_manager.get_available_models().keys()),
            "model_statistics": self.model_manager.get_model_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def validate_pipeline_configuration(self, model_type: str, image_path: str) -> Dict[str, Any]:
        """Validate pipeline configuration before running inference."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Check if model is available
            model_metadata = self.model_manager.get_model_metadata(model_type)
            if not model_metadata:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Model {model_type} not found")
            
            # Check image validation
            image_validation = self.image_validator.validate_image_quality(image_path)
            validation_results["checks"]["image_quality"] = image_validation
            
            if not image_validation["is_valid"]:
                validation_results["is_valid"] = False
                validation_results["errors"].extend(image_validation["errors"])
            
            # Check model-image compatibility
            if model_metadata:
                image_extension = image_path.lower().split('.')[-1]
                supported_formats = model_metadata.get("supported_formats", [])
                if f'.{image_extension}' not in supported_formats:
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(
                        f"Image format .{image_extension} not supported by {model_type}"
                    )
            
            # Add warnings
            if validation_results["is_valid"]:
                if image_validation.get("quality_score", 100) < 70:
                    validation_results["warnings"].append("Image quality is suboptimal")
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Pipeline validation failed: {str(e)}")
        
        return validation_results
    
    def estimate_processing_time(self, model_type: str, image_path: str) -> float:
        """Estimate processing time for inference."""
        try:
            # Get image size for estimation
            file_size = 0
            if hasattr(image_path, '__fspath__') or os.path.exists(image_path):
                file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
            
            # Base processing times (in seconds)
            base_times = {
                "classification": 0.5,
                "segmentation": 1.5
            }
            
            # Get model type
            model_metadata = self.model_manager.get_model_metadata(model_type)
            model_category = model_metadata.get("type", "classification") if model_metadata else "classification"
            
            base_time = base_times.get(model_category, 1.0)
            
            # Adjust based on image size
            size_factor = min(file_size / (1024 * 1024), 10) / 10  # Normalize to 0-1, max at 10MB
            estimated_time = base_time * (1 + size_factor * 0.5)
            
            return round(estimated_time, 2)
            
        except Exception:
            return 2.0  # Default estimate
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        # Unload all models to free memory
        for model_name in self.model_manager.get_loaded_models():
            self.model_manager.unload_model(model_name)
        
        medical_logger.logger.info("Inference pipeline cleanup completed")


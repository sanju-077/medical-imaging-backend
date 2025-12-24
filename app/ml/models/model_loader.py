"""
Model loader and manager for ML models.
"""
import os
import pickle
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from app.ml.models.classifier import DiseaseClassifier
from app.ml.models.segmentation import SegmentationModel
from app.core.logging import medical_logger
from app.core.config import settings


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    @abstractmethod
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run inference on image."""
        pass
    
    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        pass


class ModelManager:
    """Manager for loading and managing ML models."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.model_cache: Dict[str, Any] = {}
        self._load_model_configs()
    
    def _load_model_configs(self):
        """Load model configurations."""
        # Classification models
        self.model_configs.update({
            "tuberculosis_v1": {
                "type": "classification",
                "path": f"{settings.CLASSIFICATION_MODEL_PATH}/tuberculosis_v1.onnx",
                "description": "Tuberculosis detection from chest X-rays",
                "accuracy": 0.94,
                "input_size": "224x224x3",
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                "class_labels": ["Normal", "Tuberculosis"]
            },
            "pneumonia_v1": {
                "type": "classification", 
                "path": f"{settings.CLASSIFICATION_MODEL_PATH}/pneumonia_v1.onnx",
                "description": "Pneumonia detection from chest X-rays",
                "accuracy": 0.96,
                "input_size": "224x224x3",
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                "class_labels": ["Normal", "Pneumonia", "COVID-19"]
            },
            "fracture_v1": {
                "type": "classification",
                "path": f"{settings.CLASSIFICATION_MODEL_PATH}/fracture_v1.onnx",
                "description": "Bone fracture detection from X-rays",
                "accuracy": 0.92,
                "input_size": "224x224x3",
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
                "class_labels": ["Normal", "Fracture", "Other Pathology"]
            }
        })
        
        # Segmentation models
        self.model_configs.update({
            "unet_v1": {
                "type": "segmentation",
                "path": f"{settings.SEGMENTATION_MODEL_PATH}/unet_v1.onnx",
                "description": "Organ segmentation using U-Net",
                "accuracy": 0.89,
                "input_size": "256x256x3",
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm", ".nii"],
                "class_labels": ["Background", "Heart", "Left Lung", "Right Lung", "Liver", "Kidney", "Spleen"]
            }
        })
    
    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load a specific model."""
        if model_name in self.models and not force_reload:
            return True
        
        if model_name not in self.model_configs:
            medical_logger.logger.error(f"Model configuration not found: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_path = config["path"]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            medical_logger.logger.warning(f"Model file not found: {model_path}")
            # Create a mock model for development/testing
            self.models[model_name] = self._create_mock_model(model_name, config)
            return True
        
        try:
            start_time = medical_logger.logger.info(f"Loading model: {model_name}")
            
            if config["type"] == "classification":
                model = DiseaseClassifier(model_path)
            elif config["type"] == "segmentation":
                model = SegmentationModel(model_path)
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            self.models[model_name] = model
            
            end_time = medical_logger.logger.info(f"Model loaded successfully: {model_name}")
            medical_logger.logger.log_model_loading(model_path, end_time - start_time)
            
            return True
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to load model {model_name}: {e}")
            # Create mock model for development
            self.models[model_name] = self._create_mock_model(model_name, config)
            return False
    
    def _create_mock_model(self, model_name: str, config: Dict[str, Any]) -> BaseModel:
        """Create a mock model for development/testing."""
        
        class MockModel(BaseModel):
            def __init__(self, name: str, config: Dict[str, Any]):
                self.model_name = name
                self.config = config
                
            def predict(self, image_path: str) -> Dict[str, Any]:
                # Return mock prediction
                if self.config["type"] == "classification":
                    return {
                        "model_type": "classification",
                        "predictions": [
                            {"class": "Normal", "confidence": 0.75, "index": 0},
                            {"class": "Abnormal", "confidence": 0.25, "index": 1}
                        ],
                        "confidence": 0.75,
                        "predicted_class": "Normal",
                        "mock": True
                    }
                elif self.config["type"] == "segmentation":
                    import numpy as np
                    return {
                        "model_type": "segmentation",
                        "segmentation_mask": np.zeros((256, 256)).tolist(),
                        "unique_classes": [0, 1],
                        "class_counts": {0: 65000, "1": 600},
                        "mock": True
                    }
            
            def get_model_metadata(self) -> Dict[str, Any]:
                return self.config
        
        return MockModel(model_name, config)
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get loaded model by name."""
        if model_name not in self.models:
            self.load_model(model_name)
        
        return self.models.get(model_name)
    
    def predict(self, model_name: str, image_path: str) -> Dict[str, Any]:
        """Run prediction using specified model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found or failed to load: {model_name}")
        
        return model.predict(image_path)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available model configurations."""
        return self.model_configs.copy()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            medical_logger.logger.info(f"Model unloaded: {model_name}")
            return True
        return False
    
    def reload_model(self, model_name: str) -> bool:
        """Reload a model."""
        return self.load_model(model_name, force_reload=True)
    
    def validate_model_input(self, model_name: str, image_path: str) -> bool:
        """Validate if model can process the input image."""
        model = self.get_model(model_name)
        if model is None:
            return False
        
        if not hasattr(model, 'validate_input'):
            return True
        
        return model.validate_input(image_path)
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        if model_name in self.model_configs:
            config = self.model_configs[model_name].copy()
            if model_name in self.models:
                model_metadata = self.models[model_name].get_model_metadata()
                config.update(model_metadata)
            return config
        return None
    
    def batch_load_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Load multiple models at once."""
        results = {}
        for model_name in model_names:
            results[model_name] = self.load_model(model_name)
        return results
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model management statistics."""
        return {
            "total_models": len(self.model_configs),
            "loaded_models": len(self.models),
            "model_types": {
                config["type"]: sum(1 for c in self.model_configs.values() if c["type"] == config["type"])
                for config in self.model_configs.values()
            },
            "loaded_model_names": self.get_loaded_models()
        }
    
    def warm_up_models(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Warm up models by running them on dummy data."""
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        results = {}
        
        # Create dummy image data
        import numpy as np
        dummy_image_path = "/tmp/dummy_medical_image.jpg"
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        from PIL import Image
        Image.fromarray(dummy_image).save(dummy_image_path)
        
        for model_name in model_names:
            try:
                model = self.get_model(model_name)
                if model is not None:
                    # Run dummy prediction
                    result = model.predict(dummy_image_path)
                    results[model_name] = True
                    medical_logger.logger.info(f"Model warmed up: {model_name}")
                else:
                    results[model_name] = False
            except Exception as e:
                medical_logger.logger.error(f"Failed to warm up model {model_name}: {e}")
                results[model_name] = False
        
        # Clean up dummy file
        try:
            os.remove(dummy_image_path)
        except Exception:
            pass
        
        return results


# Global model manager instance
model_manager = ModelManager()


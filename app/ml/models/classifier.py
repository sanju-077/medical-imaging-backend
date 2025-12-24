"""
Disease classification model implementation.
"""
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Tuple, List
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

from app.core.logging import medical_logger


class DiseaseClassifier:
    """Medical image disease classification model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_info = {}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and get input/output info."""
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Model metadata
            self.model_info = {
                "input_shape": self.session.get_inputs()[0].shape,
                "output_shape": self.session.get_outputs()[0].shape,
                "input_type": self.session.get_inputs()[0].type,
                "output_type": self.session.get_outputs()[0].type
            }
            
            medical_logger.logger.info(f"Disease classifier model loaded: {self.model_path}")
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to load disease classifier: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input."""
        try:
            # Load image
            if image_path.lower().endswith(('.dcm', '.dicom')):
                # Handle DICOM files
                import pydicom
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array
                
                # Normalize DICOM pixel values
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = Image.fromarray(image_array).convert('RGB')
            else:
                # Handle standard image formats
                image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_array = image_tensor.numpy()
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run inference on medical image."""
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_image})
            predictions = outputs[0]
            
            # Get probabilities
            probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
            
            # Model-specific class labels
            class_labels = self._get_class_labels()
            
            # Get top predictions
            top_predictions = self._get_top_predictions(probabilities[0], class_labels)
            
            # Prepare result
            result = {
                "model_type": "classification",
                "predictions": top_predictions,
                "confidence": float(top_predictions[0]["confidence"]),
                "predicted_class": top_predictions[0]["class"],
                "all_probabilities": {
                    class_labels[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0])
                },
                "model_info": self.model_info
            }
            
            medical_logger.logger.info(
                f"Classification completed - Image: {image_path}, "
                f"Predicted: {top_predictions[0]['class']}, "
                f"Confidence: {top_predictions[0]['confidence']:.3f}"
            )
            
            return result
            
        except Exception as e:
            medical_logger.logger.error(f"Classification failed for {image_path}: {e}")
            raise
    
    def _get_class_labels(self) -> List[str]:
        """Get class labels for the model."""
        # This would be model-specific
        if "tuberculosis" in self.model_path.lower():
            return ["Normal", "Tuberculosis"]
        elif "pneumonia" in self.model_path.lower():
            return ["Normal", "Pneumonia", "COVID-19"]
        elif "fracture" in self.model_path.lower():
            return ["Normal", "Fracture", "Other Pathology"]
        else:
            return [f"Class_{i}" for i in range(self.model_info.get("output_shape", [2])[-1])]
    
    def _get_top_predictions(self, probabilities: np.ndarray, class_labels: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top k predictions."""
        # Get indices sorted by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        
        predictions = []
        for i in range(min(top_k, len(class_labels))):
            idx = sorted_indices[i]
            predictions.append({
                "class": class_labels[idx],
                "confidence": float(probabilities[idx]),
                "index": int(idx)
            })
        
        return predictions
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": "disease_classifier",
            "model_path": self.model_path,
            "model_info": self.model_info,
            "class_labels": self._get_class_labels(),
            "input_size": "224x224x3",
            "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"],
            "description": "Medical image disease classification model"
        }
    
    def validate_input(self, image_path: str) -> bool:
        """Validate input image."""
        try:
            # Check file exists
            import os
            if not os.path.exists(image_path):
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to load and preprocess
            self.preprocess_image(image_path)
            return True
            
        except Exception:
            return False
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Run inference on multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                medical_logger.logger.error(f"Batch prediction failed for {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "success": False
                })
        
        return results


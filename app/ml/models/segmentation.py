"""
U-Net segmentation model implementation for medical image segmentation.
"""
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Tuple, List
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

from app.core.logging import medical_logger


class SegmentationModel:
    """Medical image segmentation using U-Net architecture."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_info = {}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Typical U-Net input size
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
            
            medical_logger.logger.info(f"Segmentation model loaded: {self.model_path}")
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to load segmentation model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image for segmentation model input."""
        try:
            # Load image
            if image_path.lower().endswith(('.dcm', '.dicom')):
                # Handle DICOM files
                import pydicom
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array
                
                # Store original dimensions
                original_shape = image_array.shape
                
                # Normalize DICOM pixel values
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = Image.fromarray(image_array).convert('RGB')
            else:
                # Handle standard image formats
                image = Image.open(image_path).convert('RGB')
                original_shape = image.size  # (width, height)
            
            # Store original size for later resizing
            metadata = {
                "original_size": original_shape,
                "original_mode": image.mode
            }
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_array = image_tensor.numpy()
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array, metadata
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def postprocess_segmentation(self, prediction: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess segmentation prediction."""
        try:
            # Remove batch dimension
            prediction = prediction.squeeze(0)  # (1, H, W) -> (H, W) or (C, H, W)
            
            # If output has multiple channels (C, H, W), get the argmax
            if len(prediction.shape) == 3:
                prediction = np.argmax(prediction, axis=0)
            
            # Resize back to original size
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            return prediction_resized
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to postprocess segmentation: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run segmentation inference on medical image."""
        try:
            # Preprocess image
            preprocessed_image, metadata = self.preprocess_image(image_path)
            original_size = metadata["original_size"]
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_image})
            prediction = outputs[0]
            
            # Postprocess segmentation
            segmentation_mask = self.postprocess_segmentation(prediction, original_size)
            
            # Calculate segmentation statistics
            unique_classes = np.unique(segmentation_mask)
            class_counts = {int(cls): int(np.sum(segmentation_mask == cls)) for cls in unique_classes}
            total_pixels = segmentation_mask.size
            
            # Calculate class percentages
            class_percentages = {
                int(cls): round((count / total_pixels) * 100, 2) 
                for cls, count in class_counts.items()
            }
            
            # Generate class labels
            class_labels = self._get_class_labels()
            class_names = {i: class_labels[i] if i < len(class_labels) else f"Class_{i}" 
                          for i in unique_classes}
            
            # Prepare result
            result = {
                "model_type": "segmentation",
                "segmentation_mask": segmentation_mask.tolist(),
                "unique_classes": unique_classes.tolist(),
                "class_counts": class_counts,
                "class_percentages": class_percentages,
                "class_names": class_names,
                "image_size": original_size,
                "model_info": self.model_info,
                "metadata": metadata
            }
            
            medical_logger.logger.info(
                f"Segmentation completed - Image: {image_path}, "
                f"Classes found: {len(unique_classes)}, "
                f"Shape: {original_size}"
            )
            
            return result
            
        except Exception as e:
            medical_logger.logger.error(f"Segmentation failed for {image_path}: {e}")
            raise
    
    def _get_class_labels(self) -> List[str]:
        """Get class labels for the segmentation model."""
        # Common medical segmentation classes
        return [
            "Background",
            "Heart",
            "Left Lung",
            "Right Lung",
            "Liver",
            "Kidney",
            "Spleen"
        ]
    
    def generate_overlay(self, image_path: str, segmentation_result: Dict[str, Any]) -> str:
        """Generate segmentation overlay image."""
        try:
            # Load original image
            if image_path.lower().endswith(('.dcm', '.dicom')):
                import pydicom
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = Image.fromarray(image_array).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Get segmentation mask
            mask = np.array(segmentation_result["segmentation_mask"])
            
            # Create colored overlay
            overlay = self._create_colored_overlay(image_array, mask)
            
            # Save overlay image
            overlay_path = f"/tmp/overlay_{hash(image_path)}.png"
            Image.fromarray(overlay).save(overlay_path)
            
            return overlay_path
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to generate overlay: {e}")
            raise
    
    def _create_colored_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create colored overlay from segmentation mask."""
        overlay = image.copy()
        
        # Define colors for different classes
        colors = {
            0: [0, 0, 0],       # Background - black
            1: [255, 0, 0],     # Class 1 - red
            2: [0, 255, 0],     # Class 2 - green
            3: [0, 0, 255],     # Class 3 - blue
            4: [255, 255, 0],   # Class 4 - yellow
            5: [255, 0, 255],   # Class 5 - magenta
            6: [0, 255, 255]    # Class 6 - cyan
        }
        
        # Apply colors to mask
        for class_id, color in colors.items():
            if class_id in np.unique(mask):
                mask_indices = mask == class_id
                overlay[mask_indices] = color
        
        # Blend with original image
        alpha = 0.5  # Transparency factor
        blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        return blended
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": "segmentation",
            "model_path": self.model_path,
            "model_info": self.model_info,
            "class_labels": self._get_class_labels(),
            "input_size": "256x256x3",
            "supported_formats": [".jpg", ".jpeg", ".png", ".dcm", ".nii"],
            "description": "U-Net based medical image segmentation model",
            "architecture": "U-Net",
            "output_type": "pixel-wise classification"
        }
    
    def validate_input(self, image_path: str) -> bool:
        """Validate input image."""
        try:
            # Check file exists
            import os
            if not os.path.exists(image_path):
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.nii']
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to load and preprocess
            self.preprocess_image(image_path)
            return True
            
        except Exception:
            return False
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Run segmentation on multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                medical_logger.logger.error(f"Batch segmentation failed for {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def calculate_dice_score(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate Dice coefficient for segmentation evaluation."""
        try:
            # Flatten arrays
            pred_flat = prediction.flatten()
            gt_flat = ground_truth.flatten()
            
            # Calculate intersection and union
            intersection = np.sum(pred_flat * gt_flat)
            dice = (2.0 * intersection) / (np.sum(pred_flat) + np.sum(gt_flat))
            
            return float(dice)
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to calculate Dice score: {e}")
            return 0.0


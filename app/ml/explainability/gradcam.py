"""
Grad-CAM implementation for medical image explainability.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
import cv2
from PIL import Image

from app.core.logging import medical_logger


class GradCAM:
    """Gradient-weighted Class Activation Mapping for medical images."""
    
    def __init__(self, model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: TensorFlow/Keras model
            layer_name: Name of the convolutional layer to use for Grad-CAM
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Build a model that maps the input image to the activations of the last conv layer."""
        if self.layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    self.layer_name = layer.name
                    break
        
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        return grad_model
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None,
        eps: float = 1e-8
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image as numpy array
            class_index: Index of the class to compute heatmap for (None for predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            Tuple of (heatmap, class_score)
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(np.expand_dims(image, axis=0))
            if class_index is None:
                class_index = np.argmax(predictions[0])
            
            loss = predictions[:, class_index]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of feature maps
        conv_output = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)
        
        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy(), float(predictions[0][class_index])
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        colormap: int = cv2.COLORMAP_VIRIDIS,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image
            colormap: OpenCV colormap for heatmap
            alpha: Transparency of heatmap overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is RGB
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_rgb = original_image
        else:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def generate_explanation(
        self,
        image_path: str,
        class_index: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, Any]:
        """
        Generate complete Grad-CAM explanation for an image.
        
        Args:
            image_path: Path to the image
            class_index: Target class index
            target_size: Size to resize image for processing
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path, target_size)
            
            # Compute heatmap
            heatmap, class_score = self.compute_heatmap(image, class_index)
            
            # Generate overlay
            original_image = cv2.resize(image, target_size)
            overlay = self.overlay_heatmap(heatmap, original_image)
            
            # Calculate heatmap statistics
            heatmap_stats = self._analyze_heatmap(heatmap)
            
            # Generate explanation text
            explanation = self._generate_explanation_text(heatmap, heatmap_stats, class_score)
            
            result = {
                "method": "gradcam",
                "heatmap": heatmap.tolist(),
                "overlay_image": self._image_to_base64(overlay),
                "class_score": class_score,
                "class_index": class_index,
                "heatmap_statistics": heatmap_stats,
                "explanation": explanation,
                "image_info": {
                    "original_path": image_path,
                    "processed_size": target_size,
                    "method_info": {
                        "layer_name": self.layer_name,
                        "alpha": 0.4,
                        "colormap": "viridis"
                    }
                }
            }
            
            medical_logger.logger.info(f"Grad-CAM explanation generated for {image_path}")
            return result
            
        except Exception as e:
            medical_logger.logger.error(f"Grad-CAM explanation failed for {image_path}: {e}")
            raise
    
    def _load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Load and preprocess image for Grad-CAM."""
        try:
            # Load image
            if image_path.lower().endswith(('.dcm', '.dicom')):
                import pydicom
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array
                # Normalize DICOM
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = Image.fromarray(image_array).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # Resize image
            image = image.resize(target_size)
            
            # Convert to numpy array and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to load image for Grad-CAM: {e}")
            raise
    
    def _analyze_heatmap(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """Analyze heatmap statistics."""
        stats = {
            "mean_activation": float(np.mean(heatmap)),
            "max_activation": float(np.max(heatmap)),
            "min_activation": float(np.min(heatmap)),
            "std_activation": float(np.std(heatmap)),
            "activation_range": float(np.max(heatmap) - np.min(heatmap)),
            "high_activation_percentage": float(np.sum(heatmap > 0.5) / heatmap.size * 100)
        }
        
        # Find most important regions
        threshold = np.percentile(heatmap, 90)
        important_regions = np.where(heatmap > threshold)
        
        if len(important_regions[0]) > 0:
            stats["top_region_center"] = {
                "x": int(np.mean(important_regions[1])),
                "y": int(np.mean(important_regions[0]))
            }
            stats["top_region_size"] = int(len(important_regions[0]))
        
        return stats
    
    def _generate_explanation_text(self, heatmap: np.ndarray, stats: Dict[str, Any], class_score: float) -> str:
        """Generate human-readable explanation."""
        explanation_parts = []
        
        # Overall assessment
        if stats["high_activation_percentage"] > 20:
            explanation_parts.append("The model shows strong activation patterns in the image.")
        elif stats["high_activation_percentage"] > 10:
            explanation_parts.append("The model shows moderate activation patterns.")
        else:
            explanation_parts.append("The model shows limited activation patterns.")
        
        # Confidence level
        if class_score > 0.8:
            explanation_parts.append("The prediction confidence is very high.")
        elif class_score > 0.6:
            explanation_parts.append("The prediction confidence is moderate.")
        else:
            explanation_parts.append("The prediction confidence is low.")
        
        # Activation patterns
        if stats["activation_range"] > 0.7:
            explanation_parts.append("There are distinct regions of high and low activation.")
        else:
            explanation_parts.append("The activation is relatively uniform across the image.")
        
        return " ".join(explanation_parts)
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image array to base64 string."""
        import base64
        from io import BytesIO
        
        # Convert to PIL Image
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"


class MedicalGradCAM(GradCAM):
    """Specialized Grad-CAM for medical imaging."""
    
    def __init__(self, model, layer_name: Optional[str] = None):
        super().__init__(model, layer_name)
        self.medical_terminology = {
            "chest_xray": {
                "lung_regions": ["Left lung", "Right lung", "Cardiac silhouette"],
                "pathologies": ["consolidation", "pneumothorax", "cardiomegaly", "effusion"]
            },
            "brain_mri": {
                "brain_regions": ["Frontal lobe", "Parietal lobe", "Temporal lobe", "Occipital lobe"],
                "pathologies": ["tumor", "stroke", "hemorrhage", "edema"]
            },
            "bone_xray": {
                "bone_regions": ["Cortical bone", "Trabecular bone", "Joint space"],
                "pathologies": ["fracture", "osteoporosis", "arthritis", "infection"]
            }
        }
    
    def generate_medical_explanation(
        self,
        image_path: str,
        modality: str = "chest_xray",
        class_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate medical-specific Grad-CAM explanation.
        
        Args:
            image_path: Path to the medical image
            modality: Type of medical imaging (chest_xray, brain_mri, bone_xray)
            class_index: Target class index
            
        Returns:
            Dictionary containing medical explanation
        """
        try:
            # Get basic Grad-CAM explanation
            explanation = self.generate_explanation(image_path, class_index)
            
            # Add medical-specific analysis
            medical_analysis = self._analyze_medical_regions(explanation["heatmap"], modality)
            
            # Add clinical interpretation
            clinical_interpretation = self._generate_clinical_interpretation(
                explanation, medical_analysis, modality
            )
            
            # Update explanation with medical details
            explanation.update({
                "medical_analysis": medical_analysis,
                "clinical_interpretation": clinical_interpretation,
                "modality": modality,
                "medical_context": self.medical_terminology.get(modality, {})
            })
            
            return explanation
            
        except Exception as e:
            medical_logger.logger.error(f"Medical Grad-CAM explanation failed: {e}")
            raise
    
    def _analyze_medical_regions(self, heatmap: np.ndarray, modality: str) -> Dict[str, Any]:
        """Analyze heatmap in context of medical anatomy."""
        medical_info = self.medical_terminology.get(modality, {})
        
        # Simple region analysis based on image quadrants
        height, width = heatmap.shape
        
        regions = {
            "upper_left": heatmap[:height//2, :width//2],
            "upper_right": heatmap[:height//2, width//2:],
            "lower_left": heatmap[height//2:, :width//2],
            "lower_right": heatmap[height//2:, width//2:]
        }
        
        region_analysis = {}
        for region_name, region_data in regions.items():
            region_analysis[region_name] = {
                "mean_activation": float(np.mean(region_data)),
                "max_activation": float(np.max(region_data)),
                "activation_percentage": float(np.sum(region_data > 0.3) / region_data.size * 100)
            }
        
        # Find most significant regions
        significant_regions = sorted(
            region_analysis.items(),
            key=lambda x: x[1]["mean_activation"],
            reverse=True
        )
        
        return {
            "regional_analysis": region_analysis,
            "most_significant_regions": [region[0] for region in significant_regions[:2]],
            "anatomical_relevance": self._assess_anatomical_relevance(region_analysis, modality)
        }
    
    def _assess_anatomical_relevance(self, region_analysis: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Assess anatomical relevance of activation patterns."""
        # This is a simplified assessment
        # In practice, this would use detailed anatomical atlases
        
        relevance = {
            "modality": modality,
            "key_findings": [],
            "anatomical_confidence": "moderate"
        }
        
        if modality == "chest_xray":
            # Check for bilateral vs unilateral patterns
            upper_regions = [region_analysis["upper_left"], region_analysis["upper_right"]]
            lower_regions = [region_analysis["lower_left"], region_analysis["lower_right"]]
            
            upper_activation = np.mean([r["mean_activation"] for r in upper_regions])
            lower_activation = np.mean([r["mean_activation"] for r in lower_regions])
            
            if abs(upper_activation - lower_activation) > 0.2:
                relevance["key_findings"].append("Asymmetric activation pattern detected")
            
        elif modality == "brain_mri":
            # Check for central vs peripheral patterns
            central_regions = [region_analysis["upper_left"], region_analysis["lower_right"]]
            peripheral_regions = [region_analysis["upper_right"], region_analysis["lower_left"]]
            
            central_activation = np.mean([r["mean_activation"] for r in central_regions])
            peripheral_activation = np.mean([r["mean_activation"] for r in peripheral_regions])
            
            if central_activation > peripheral_activation:
                relevance["key_findings"].append("Central pattern activation")
            else:
                relevance["key_findings"].append("Peripheral pattern activation")
        
        return relevance
    
    def _generate_clinical_interpretation(
        self,
        explanation: Dict[str, Any],
        medical_analysis: Dict[str, Any],
        modality: str
    ) -> Dict[str, Any]:
        """Generate clinical interpretation of the Grad-CAM results."""
        interpretation = {
            "modality": modality,
            "summary": "",
            "clinical_significance": [],
            "recommended_actions": [],
            "confidence_level": "moderate"
        }
        
        heatmap_stats = explanation["heatmap_statistics"]
        class_score = explanation["class_score"]
        
        # Generate summary
        if class_score > 0.8:
            interpretation["summary"] = "High-confidence AI detection with clear visualization patterns."
        elif class_score > 0.6:
            interpretation["summary"] = "Moderate-confidence AI detection with some uncertainty."
        else:
            interpretation["summary"] = "Low-confidence AI detection, manual review recommended."
        
        # Clinical significance
        if heatmap_stats["high_activation_percentage"] > 20:
            interpretation["clinical_significance"].append("Strong evidence for AI-detected pathology")
        elif heatmap_stats["high_activation_percentage"] > 10:
            interpretation["clinical_significance"].append("Moderate evidence for AI-detected findings")
        else:
            interpretation["clinical_significance"].append("Subtle or limited evidence for pathology")
        
        # Recommended actions
        if class_score < 0.6:
            interpretation["recommended_actions"].append("Manual radiological review strongly recommended")
        
        if heatmap_stats["activation_range"] > 0.7:
            interpretation["recommended_actions"].append("Focused examination of high-activation regions")
        
        return interpretation


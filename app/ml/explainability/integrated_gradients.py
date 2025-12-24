"""
Integrated Gradients implementation for medical image explainability.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
import cv2
from PIL import Image

from app.core.logging import medical_logger


class IntegratedGradients:
    """Integrated Gradients implementation for medical image interpretability."""
    
    def __init__(self, model, baseline: Optional[np.ndarray] = None):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: TensorFlow/Keras model
            baseline: Baseline image for interpolation (None for black image)
        """
        self.model = model
        self.baseline = baseline
    
    def _interpolate_images(
        self,
        baseline: np.ndarray,
        image: np.ndarray,
        alphas: np.ndarray
    ) -> np.ndarray:
        """Generate interpolated images between baseline and image."""
        alphas_x = alphas[:, np.newaxis, np.newaxis, np.newaxis]
        baseline_x = baseline[np.newaxis, :, :, :]
        input_x = image[np.newaxis, :, :, :]
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images
    
    def _compute_gradients(
        self,
        images: np.ndarray,
        target_class: int
    ) -> np.ndarray:
        """Compute gradients of the target class with respect to input images."""
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images)
            loss = predictions[:, target_class]
        
        gradients = tape.gradient(loss, images)
        return gradients
    
    def _generate_baseline(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Generate baseline image."""
        if self.baseline is not None:
            return self.baseline
        
        # Default to black image
        if len(image_shape) == 3:
            return np.zeros(image_shape, dtype=np.float32)
        else:
            return np.zeros(image_shape, dtype=np.float32)
    
    def compute_attribution(
        self,
        image: np.ndarray,
        target_class: int,
        n_steps: int = 50,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            image: Input image
            target_class: Target class index
            n_steps: Number of integration steps
            batch_size: Batch size for gradient computation
            
        Returns:
            Tuple of (attribution, prediction_score)
        """
        # Generate baseline
        baseline = self._generate_baseline(image.shape)
        
        # Generate alphas
        alphas = np.linspace(0, 1, n_steps)
        
        # Compute gradients in batches
        total_gradients = np.zeros_like(image, dtype=np.float32)
        
        # Get prediction for target class
        image_batch = np.expand_dims(image, axis=0)
        predictions = self.model(image_batch)
        prediction_score = float(predictions[0][target_class])
        
        # Process in batches
        for i in range(0, len(alphas), batch_size):
            batch_alphas = alphas[i:i+batch_size]
            
            # Generate interpolated images
            batch_images = self._interpolate_images(baseline, image, batch_alphas)
            
            # Compute gradients
            batch_gradients = self._compute_gradients(batch_images, target_class)
            
            # Accumulate gradients
            total_gradients += np.mean(batch_gradients, axis=0)
        
        # Compute integrated gradients
        attribution = (image - baseline) * total_gradients / len(alphas)
        
        return attribution, prediction_score
    
    def generate_explanation(
        self,
        image_path: str,
        target_class: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224),
        n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Generate complete Integrated Gradients explanation.
        
        Args:
            image_path: Path to the image
            target_class: Target class index (None for predicted class)
            target_size: Size to resize image for processing
            n_steps: Number of integration steps
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path, target_size)
            
            # Get prediction to determine target class
            image_batch = np.expand_dims(image, axis=0)
            predictions = self.model(image_batch)
            
            if target_class is None:
                target_class = np.argmax(predictions[0])
            
            # Compute Integrated Gradients attribution
            attribution, prediction_score = self.compute_attribution(
                image, target_class, n_steps=n_steps
            )
            
            # Generate visualization
            attribution_heatmap = self._create_attribution_heatmap(attribution)
            overlay = self._create_overlay(image, attribution_heatmap)
            
            # Analyze attribution patterns
            attribution_stats = self._analyze_attribution(attribution)
            
            # Generate explanation text
            explanation = self._generate_explanation_text(attribution_stats, prediction_score)
            
            result = {
                "method": "integrated_gradients",
                "attribution": attribution.tolist(),
                "attribution_heatmap": attribution_heatmap.tolist(),
                "overlay_image": self._image_to_base64(overlay),
                "prediction_score": prediction_score,
                "target_class": target_class,
                "attribution_statistics": attribution_stats,
                "explanation": explanation,
                "method_parameters": {
                    "n_steps": n_steps,
                    "target_size": target_size
                },
                "image_info": {
                    "original_path": image_path,
                    "processed_size": target_size
                }
            }
            
            medical_logger.logger.info(f"Integrated Gradients explanation generated for {image_path}")
            return result
            
        except Exception as e:
            medical_logger.logger.error(f"Integrated Gradients explanation failed for {image_path}: {e}")
            raise
    
    def _load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Load and preprocess image for Integrated Gradients."""
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
            medical_logger.logger.error(f"Failed to load image for Integrated Gradients: {e}")
            raise
    
    def _create_attribution_heatmap(self, attribution: np.ndarray) -> np.ndarray:
        """Create heatmap from attribution values."""
        # Normalize attribution to [0, 1]
        attribution_abs = np.abs(attribution)
        attribution_normalized = (attribution_abs - attribution_abs.min()) / (attribution_abs.max() - attribution_abs.min() + 1e-8)
        
        # Create heatmap
        heatmap = np.uint8(255 * attribution_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    def _create_overlay(self, image: np.ndarray, attribution_heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Create overlay of attribution heatmap on original image."""
        # Ensure original image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            original_rgb = (image * 255).astype(np.uint8)
        else:
            original_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap
        overlayed = cv2.addWeighted(original_rgb, 1 - alpha, attribution_heatmap, alpha, 0)
        
        return overlayed
    
    def _analyze_attribution(self, attribution: np.ndarray) -> Dict[str, Any]:
        """Analyze attribution patterns and statistics."""
        attribution_abs = np.abs(attribution)
        
        stats = {
            "mean_attribution": float(np.mean(attribution_abs)),
            "max_attribution": float(np.max(attribution_abs)),
            "min_attribution": float(np.min(attribution_abs)),
            "std_attribution": float(np.std(attribution_abs)),
            "total_attribution": float(np.sum(attribution_abs)),
            "positive_attribution": float(np.sum(attribution[attribution > 0])),
            "negative_attribution": float(np.sum(attribution[attribution < 0]))
        }
        
        # Find most important regions
        threshold = np.percentile(attribution_abs, 95)
        important_regions = np.where(attribution_abs > threshold)
        
        if len(important_regions[0]) > 0:
            stats["top_regions"] = {
                "center": {
                    "x": int(np.mean(important_regions[1])),
                    "y": int(np.mean(important_regions[0]))
                },
                "size": int(len(important_regions[0])),
                "percentage": float(len(important_regions[0]) / attribution.size * 100)
            }
        
        # Attribution distribution analysis
        attribution_flat = attribution_abs.flatten()
        percentiles = [25, 50, 75, 90, 95, 99]
        stats["percentiles"] = {
            f"p{p}": float(np.percentile(attribution_flat, p)) for p in percentiles
        }
        
        return stats
    
    def _generate_explanation_text(self, stats: Dict[str, Any], prediction_score: float) -> str:
        """Generate human-readable explanation."""
        explanation_parts = []
        
        # Prediction confidence
        if prediction_score > 0.8:
            explanation_parts.append("The model shows high confidence in its prediction.")
        elif prediction_score > 0.6:
            explanation_parts.append("The model shows moderate confidence in its prediction.")
        else:
            explanation_parts.append("The model shows low confidence in its prediction.")
        
        # Attribution patterns
        total_attr = stats["total_attribution"]
        if total_attr > np.percentile([stats["total_attribution"]], 75):
            explanation_parts.append("The image contains highly distinctive features that strongly influence the prediction.")
        elif total_attr > np.percentile([stats["total_attribution"]], 50):
            explanation_parts.append("The image contains moderately distinctive features.")
        else:
            explanation_parts.append("The image features have limited impact on the prediction.")
        
        # Attribution distribution
        if stats["positive_attribution"] > abs(stats["negative_attribution"]):
            explanation_parts.append("The positive features contribute more to the prediction than negative features.")
        else:
            explanation_parts.append("The negative features contribute more to the prediction than positive features.")
        
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


class MedicalIntegratedGradients(IntegratedGradients):
    """Specialized Integrated Gradients for medical imaging."""
    
    def __init__(self, model, baseline: Optional[np.ndarray] = None):
        super().__init__(model, baseline)
        self.medical_contexts = {
            "chest_xray": {
                "regions": {
                    "upper_lung": "Upper lung fields",
                    "lower_lung": "Lower lung fields",
                    "cardiac": "Cardiac silhouette",
                    "diaphragm": "Diaphragm and costophrenic angles"
                },
                "findings": {
                    "consolidation": "Areas of lung consolidation",
                    "effusion": "Pleural effusions",
                    "pneumothorax": "Collapsed lung areas",
                    "cardiomegaly": "Enlarged heart shadow"
                }
            },
            "brain_mri": {
                "regions": {
                    "frontal": "Frontal lobe",
                    "parietal": "Parietal lobe", 
                    "temporal": "Temporal lobe",
                    "occipital": "Occipital lobe"
                },
                "findings": {
                    "tumor": "Tumor masses",
                    "stroke": "Ischemic areas",
                    "hemorrhage": "Bleeding areas",
                    "edema": "Swelling patterns"
                }
            }
        }
    
    def generate_medical_explanation(
        self,
        image_path: str,
        modality: str = "chest_xray",
        target_class: Optional[int] = None,
        n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Generate medical-specific Integrated Gradients explanation.
        
        Args:
            image_path: Path to the medical image
            modality: Type of medical imaging
            target_class: Target class index
            n_steps: Number of integration steps
            
        Returns:
            Dictionary containing medical explanation
        """
        try:
            # Get basic Integrated Gradients explanation
            explanation = self.generate_explanation(image_path, target_class, n_steps=n_steps)
            
            # Add medical-specific analysis
            medical_analysis = self._analyze_medical_attribution(explanation["attribution"], modality)
            
            # Add clinical interpretation
            clinical_interpretation = self._generate_medical_interpretation(
                explanation, medical_analysis, modality
            )
            
            # Update explanation with medical details
            explanation.update({
                "medical_analysis": medical_analysis,
                "clinical_interpretation": clinical_interpretation,
                "modality": modality,
                "medical_context": self.medical_contexts.get(modality, {})
            })
            
            return explanation
            
        except Exception as e:
            medical_logger.logger.error(f"Medical Integrated Gradients explanation failed: {e}")
            raise
    
    def _analyze_medical_attribution(self, attribution: List[List[List[float]]], modality: str) -> Dict[str, Any]:
        """Analyze attribution in medical context."""
        attribution_array = np.array(attribution)
        
        # Simple anatomical analysis
        height, width = attribution_array.shape[:2]
        
        medical_regions = {
            "upper_third": attribution_array[:height//3, :],
            "middle_third": attribution_array[height//3:2*height//3, :],
            "lower_third": attribution_array[2*height//3:, :],
            "left_half": attribution_array[:, :width//2],
            "right_half": attribution_array[:, width//2:]
        }
        
        region_analysis = {}
        for region_name, region_data in medical_regions.items():
            region_analysis[region_name] = {
                "mean_attribution": float(np.mean(np.abs(region_data))),
                "max_attribution": float(np.max(np.abs(region_data))),
                "attribution_percentage": float(np.sum(np.abs(region_data) > 0.1) / region_data.size * 100)
            }
        
        return {
            "regional_analysis": region_analysis,
            "dominant_regions": self._find_dominant_regions(region_analysis),
            "anatomical_significance": self._assess_anatomical_significance(region_analysis, modality)
        }
    
    def _find_dominant_regions(self, region_analysis: Dict[str, Any]) -> List[str]:
        """Find regions with highest attribution."""
        sorted_regions = sorted(
            region_analysis.items(),
            key=lambda x: x[1]["mean_attribution"],
            reverse=True
        )
        return [region[0] for region in sorted_regions[:3]]
    
    def _assess_anatomical_significance(self, region_analysis: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Assess anatomical significance of attribution patterns."""
        significance = {
            "modality": modality,
            "primary_finding": "",
            "anatomical_confidence": "moderate",
            "relevant_regions": []
        }
        
        # Analyze regional patterns
        upper_regions = ["upper_third", "left_half", "right_half"]
        lower_regions = ["lower_third"]
        
        upper_activation = np.mean([region_analysis[r]["mean_attribution"] for r in upper_regions if r in region_analysis])
        lower_activation = region_analysis.get("lower_third", {}).get("mean_attribution", 0)
        
        if modality == "chest_xray":
            if upper_activation > lower_activation:
                significance["primary_finding"] = "Upper lung field pathology"
                significance["relevant_regions"] = ["Upper lung fields", "Cardiac silhouette"]
            else:
                significance["primary_finding"] = "Lower lung field pathology"
                significance["relevant_regions"] = ["Lower lung fields", "Costophrenic angles"]
        
        return significance
    
    def _generate_medical_interpretation(
        self,
        explanation: Dict[str, Any],
        medical_analysis: Dict[str, Any],
        modality: str
    ) -> Dict[str, Any]:
        """Generate clinical interpretation of Integrated Gradients results."""
        interpretation = {
            "modality": modality,
            "summary": "",
            "clinical_significance": [],
            "radiological_correlation": [],
            "confidence_assessment": "moderate"
        }
        
        attr_stats = explanation["attribution_statistics"]
        prediction_score = explanation["prediction_score"]
        
        # Generate summary
        if prediction_score > 0.8 and attr_stats["total_attribution"] > 0.5:
            interpretation["summary"] = "Strong AI indication with clear feature attribution."
        elif prediction_score > 0.6:
            interpretation["summary"] = "Moderate AI indication with moderate feature attribution."
        else:
            interpretation["summary"] = "Weak AI indication, manual review recommended."
        
        # Clinical significance
        dominant_regions = medical_analysis["dominant_regions"]
        if "upper_third" in dominant_regions:
            interpretation["clinical_significance"].append("Upper field pathology pattern detected")
        if "lower_third" in dominant_regions:
            interpretation["clinical_significance"].append("Lower field pathology pattern detected")
        if "left_half" in dominant_regions or "right_half" in dominant_regions:
            interpretation["clinical_significance"].append("Unilateral pattern detected")
        
        # Radiological correlation
        if modality == "chest_xray":
            interpretation["radiological_correlation"].append("Correlate with chest X-ray interpretation guidelines")
            interpretation["radiological_correlation"].append("Consider lateral views if indicated")
        
        return interpretation


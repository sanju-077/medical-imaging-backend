"""
Image preprocessing utilities for medical imaging.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import pydicom
from skimage import exposure, filters, restoration
from skimage.segmentation import flood_fill

from app.core.logging import medical_logger


class ImageProcessor:
    """Medical image preprocessing and enhancement."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.nii']
    
    def load_medical_image(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load medical image from various formats."""
        try:
            metadata = {"format": "unknown", "original_shape": None, "original_dtype": None}
            
            if image_path.lower().endswith(('.dcm', '.dicom')):
                # Load DICOM file
                dicom = pydicom.dcmread(image_path)
                image_array = dicom.pixel_array
                
                metadata.update({
                    "format": "dicom",
                    "patient_name": getattr(dicom, 'PatientName', 'Unknown'),
                    "study_date": getattr(dicom, 'StudyDate', 'Unknown'),
                    "modality": getattr(dicom, 'Modality', 'Unknown'),
                    "window_center": getattr(dicom, 'WindowCenter', None),
                    "window_width": getattr(dicom, 'WindowWidth', None),
                    "pixel_spacing": getattr(dicom, 'PixelSpacing', None)
                })
                
            elif image_path.lower().endswith('.nii'):
                # Load NIfTI file (requires nibabel)
                try:
                    import nibabel as nib
                    nii_img = nib.load(image_path)
                    image_array = nii_img.get_fdata()
                    image_array = np.squeeze(image_array)  # Remove singleton dimensions
                    
                    metadata.update({
                        "format": "nifti",
                        "affine": nii_img.affine.tolist(),
                        "header": str(nii_img.header)
                    })
                except ImportError:
                    raise ImportError("nibabel is required for NIfTI file support")
                    
            else:
                # Load standard image formats
                image = Image.open(image_path)
                image_array = np.array(image)
                
                metadata.update({
                    "format": "standard",
                    "mode": image.mode,
                    "size": image.size
                })
            
            metadata.update({
                "original_shape": image_array.shape,
                "original_dtype": str(image_array.dtype)
            })
            
            return image_array, metadata
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        try:
            # Convert to LAB color space for better contrast enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0]
            else:
                l_channel = image
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(l_channel)
            
            # Convert back to RGB if needed
            if len(image.shape) == 3:
                lab[:, :, 0] = enhanced
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            medical_logger.logger.debug("CLAHE enhancement applied")
            return enhanced
            
        except Exception as e:
            medical_logger.logger.error(f"CLAHE application failed: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """Apply noise reduction techniques."""
        try:
            if method == "gaussian":
                # Gaussian blur
                denoised = cv2.GaussianBlur(image, (5, 5), 0)
                
            elif method == "bilateral":
                # Bilateral filter (preserves edges)
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
                
            elif method == "non_local_means":
                # Non-local means denoising
                if len(image.shape) == 3:
                    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
                    
            elif method == "wiener":
                # Wiener filter (requires scikit-image)
                psf = np.ones((5, 5)) / 25
                denoised = restoration.wiener(image, psf, balance=0.1)
                denoised = np.clip(denoised, 0, 255).astype(np.uint8)
                
            else:
                medical_logger.logger.warning(f"Unknown noise reduction method: {method}")
                denoised = image
            
            medical_logger.logger.debug(f"Noise reduction applied: {method}")
            return denoised
            
        except Exception as e:
            medical_logger.logger.error(f"Noise reduction failed: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray, method: str = "histogram") -> np.ndarray:
        """Enhance image contrast using various methods."""
        try:
            if method == "histogram":
                # Histogram equalization
                if len(image.shape) == 3:
                    # Apply to each channel
                    enhanced = image.copy()
                    for i in range(3):
                        enhanced[:, :, i] = cv2.equalizeHist(image[:, :, i])
                else:
                    enhanced = cv2.equalizeHist(image)
                    
            elif method == "adaptive_histogram":
                # Adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
                
            elif method == "gamma_correction":
                # Gamma correction
                gamma = 1.2
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(image, table)
                
            elif method == "stretch":
                # Contrast stretching
                p2, p98 = np.percentile(image, (2, 98))
                enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))
                
            else:
                medical_logger.logger.warning(f"Unknown contrast enhancement method: {method}")
                enhanced = image
            
            medical_logger.logger.debug(f"Contrast enhancement applied: {method}")
            return enhanced
            
        except Exception as e:
            medical_logger.logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize image pixel values."""
        try:
            if method == "minmax":
                # Min-Max normalization
                min_val = image.min()
                max_val = image.max()
                normalized = (image - min_val) / (max_val - min_val)
                normalized = (normalized * 255).astype(np.uint8)
                
            elif method == "z_score":
                # Z-score normalization
                mean = image.mean()
                std = image.std()
                normalized = (image - mean) / std
                normalized = np.clip(normalized * 0.1 + 0.5, 0, 1)  # Scale to [0,1]
                normalized = (normalized * 255).astype(np.uint8)
                
            elif method == "robust":
                # Robust normalization using median and IQR
                median = np.median(image)
                iqr = np.percentile(image, 75) - np.percentile(image, 25)
                normalized = (image - median) / (1.5 * iqr)
                normalized = np.clip(normalized, -3, 3)  # Clip outliers
                normalized = (normalized + 3) / 6  # Scale to [0,1]
                normalized = (normalized * 255).astype(np.uint8)
                
            else:
                medical_logger.logger.warning(f"Unknown normalization method: {method}")
                normalized = image
            
            medical_logger.logger.debug(f"Normalization applied: {method}")
            return normalized
            
        except Exception as e:
            medical_logger.logger.error(f"Normalization failed: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = "linear") -> np.ndarray:
        """Resize image to target size."""
        try:
            if method == "linear":
                interpolation = cv2.INTER_LINEAR
            elif method == "cubic":
                interpolation = cv2.INTER_CUBIC
            elif method == "nearest":
                interpolation = cv2.INTER_NEAREST
            else:
                interpolation = cv2.INTER_LINEAR
            
            resized = cv2.resize(image, target_size, interpolation=interpolation)
            medical_logger.logger.debug(f"Image resized to {target_size}")
            return resized
            
        except Exception as e:
            medical_logger.logger.error(f"Image resizing failed: {e}")
            return image
    
    def apply_filters(self, image: np.ndarray, filter_type: str = "sharpen") -> np.ndarray:
        """Apply various image filters."""
        try:
            if filter_type == "sharpen":
                # Sharpening filter
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                filtered = cv2.filter2D(image, -1, kernel)
                
            elif filter_type == "edge_enhance":
                # Edge enhancement
                kernel = np.array([[-1,-1,-1,-1,-1],
                                 [-1, 2, 2, 2,-1],
                                 [-1, 2, 8, 2,-1],
                                 [-1, 2, 2, 2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0
                filtered = cv2.filter2D(image, -1, kernel)
                
            elif filter_type == "blur":
                # Gaussian blur
                filtered = cv2.GaussianBlur(image, (5, 5), 0)
                
            elif filter_type == "median":
                # Median filter
                filtered = cv2.medianBlur(image, 5)
                
            elif filter_type == "unsharp_mask":
                # Unsharp masking
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                filtered = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
                
            else:
                medical_logger.logger.warning(f"Unknown filter type: {filter_type}")
                filtered = image
            
            medical_logger.logger.debug(f"Filter applied: {filter_type}")
            return filtered
            
        except Exception as e:
            medical_logger.logger.error(f"Filter application failed: {e}")
            return image
    
    def window_image(self, image: np.ndarray, window_center: float, 
                    window_width: float) -> np.ndarray:
        """Apply windowing for medical images (CT/MRI)."""
        try:
            # Window parameters
            lower_bound = window_center - window_width / 2
            upper_bound = window_center + window_width / 2
            
            # Apply window
            windowed = np.clip(image, lower_bound, upper_bound)
            
            # Normalize to 0-255
            windowed = ((windowed - lower_bound) / window_width * 255).astype(np.uint8)
            
            medical_logger.logger.debug(f"Windowing applied: center={window_center}, width={window_width}")
            return windowed
            
        except Exception as e:
            medical_logger.logger.error(f"Windowing failed: {e}")
            return image
    
    def preprocess_for_inference(self, image_path: str, 
                                target_size: Tuple[int, int] = (224, 224),
                                enhancement: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline for ML inference."""
        try:
            # Load image
            image, metadata = self.load_medical_image(image_path)
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Apply enhancements if requested
            if enhancement:
                # Apply CLAHE
                image = self.apply_clahe(image)
                
                # Apply noise reduction
                image = self.reduce_noise(image, method="bilateral")
                
                # Apply contrast enhancement
                image = self.enhance_contrast(image, method="adaptive_histogram")
                
                # Apply normalization
                image = self.normalize_image(image, method="minmax")
            
            # Resize to target size
            image = self.resize_image(image, target_size)
            
            # Add preprocessing info to metadata
            metadata["preprocessing"] = {
                "target_size": target_size,
                "enhancement_applied": enhancement,
                "clahe": enhancement,
                "noise_reduction": enhancement,
                "contrast_enhancement": enhancement,
                "normalization": enhancement
            }
            
            medical_logger.logger.info(f"Preprocessing completed for {image_path}")
            return image, metadata
            
        except Exception as e:
            medical_logger.logger.error(f"Preprocessing failed for {image_path}: {e}")
            raise
    
    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality metrics."""
        try:
            metrics = {}
            
            # Calculate basic statistics
            metrics["mean"] = float(image.mean())
            metrics["std"] = float(image.std())
            metrics["min"] = float(image.min())
            metrics["max"] = float(image.max())
            metrics["dynamic_range"] = float(image.max() - image.min())
            
            # Calculate contrast (standard deviation as a proxy)
            metrics["contrast"] = float(image.std() / 255.0)  # Normalized contrast
            
            # Calculate signal-to-noise ratio estimate
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Simple SNR estimate
            signal = gray.mean()
            noise = gray.std()
            metrics["snr_estimate"] = float(signal / noise) if noise > 0 else float('inf')
            
            # Brightness assessment
            brightness = gray.mean()
            if brightness < 30:
                metrics["brightness"] = "dark"
            elif brightness > 225:
                metrics["brightness"] = "bright"
            else:
                metrics["brightness"] = "normal"
            
            # Quality score (0-100)
            quality_score = 100
            if metrics["contrast"] < 0.1:
                quality_score -= 30  # Low contrast
            if metrics["brightness"] in ["dark", "bright"]:
                quality_score -= 20  # Poor brightness
            if metrics["snr_estimate"] < 5:
                quality_score -= 20  # Low SNR
            if metrics["dynamic_range"] < 100:
                quality_score -= 20  # Limited dynamic range
            
            metrics["quality_score"] = max(0, quality_score)
            
            return metrics
            
        except Exception as e:
            medical_logger.logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}


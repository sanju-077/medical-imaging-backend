"""
Image validation utilities for medical imaging.
"""
import os
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import magic
from PIL import Image
import numpy as np

from app.core.logging import medical_logger


class ImageValidator:
    """Validator for medical images."""
    
    def __init__(self):
        self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom', '.nii']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.min_resolution = (64, 64)
        self.max_resolution = (8192, 8192)
    
    def validate_file_upload(self, file, allowed_extensions: List[str] = None, 
                           max_file_size: int = None) -> Tuple[bool, str]:
        """Validate file upload."""
        try:
            # Check if file is provided
            if not file:
                return False, "No file provided"
            
            # Check file size
            max_size = max_file_size or self.max_file_size
            if hasattr(file, 'size') and file.size > max_size:
                return False, f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            
            # Check file extension
            file_extension = os.path.splitext(file.filename)[1].lower()
            allowed_exts = allowed_extensions or self.allowed_extensions
            
            if file_extension not in allowed_exts:
                return False, f"Invalid file extension. Allowed: {', '.join(allowed_exts)}"
            
            # Check if file content is readable
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension in ['.dcm', '.dicom']:
                # Validate DICOM file
                is_valid, error = self._validate_dicom_file(file)
            elif file_extension == '.nii':
                # Validate NIfTI file
                is_valid, error = self._validate_nifti_file(file)
            else:
                # Validate standard image formats
                is_valid, error = self._validate_standard_image(file)
            
            return is_valid, error
            
        except Exception as e:
            medical_logger.logger.error(f"File validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_dicom_file(self, file) -> Tuple[bool, str]:
        """Validate DICOM file."""
        try:
            import pydicom
            
            # Try to read the file
            file_content = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            # Check magic bytes
            if len(file_content) < 132:
                return False, "File too small to be a valid DICOM file"
            
            # DICOM files have "DICM" at position 128
            if file_content[128:132] != b'DICM':
                return False, "Invalid DICOM file format"
            
            # Try to parse DICOM metadata
            dicom = pydicom.dcmread(file.file, stop_before_pixels=True)
            file.file.seek(0)  # Reset file pointer
            
            # Check required DICOM tags
            required_tags = ['PatientName', 'StudyDate', 'Modality']
            missing_tags = []
            
            for tag in required_tags:
                if not hasattr(dicom, tag.lower().replace('_', '')):
                    missing_tags.append(tag)
            
            if missing_tags:
                medical_logger.logger.warning(f"DICOM file missing tags: {missing_tags}")
            
            medical_logger.logger.info(f"DICOM validation successful: {file.filename}")
            return True, "Valid DICOM file"
            
        except ImportError:
            return False, "pydicom library required for DICOM validation"
        except Exception as e:
            return False, f"DICOM validation failed: {str(e)}"
    
    def _validate_nifti_file(self, file) -> Tuple[bool, str]:
        """Validate NIfTI file."""
        try:
            import nibabel as nib
            
            file_content = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            # Try to load NIfTI file
            nii_img = nib.load(file.file)
            file.file.seek(0)  # Reset file pointer
            
            # Check data shape
            data_shape = nii_img.shape
            if len(data_shape) < 2 or len(data_shape) > 4:
                return False, "Invalid NIfTI data shape"
            
            # Check for empty data
            if nii_img.get_fdata().size == 0:
                return False, "Empty NIfTI data"
            
            medical_logger.logger.info(f"NIfTI validation successful: {file.filename}")
            return True, "Valid NIfTI file"
            
        except ImportError:
            return False, "nibabel library required for NIfTI validation"
        except Exception as e:
            return False, f"NIfTI validation failed: {str(e)}"
    
    def _validate_standard_image(self, file) -> Tuple[bool, str]:
        """Validate standard image formats (JPEG, PNG, etc.)."""
        try:
            # Check file signature/magic bytes
            file_content = file.file.read(16)
            file.file.seek(0)  # Reset file pointer
            
            # JPEG signature
            if file_content[:2] == b'\xff\xd8':
                format_type = "JPEG"
            # PNG signature
            elif file_content[:8] == b'\x89PNG\r\n\x1a\n':
                format_type = "PNG"
            else:
                return False, "Invalid image file format"
            
            # Try to open with PIL
            image = Image.open(file.file)
            file.file.seek(0)  # Reset file pointer
            
            # Check image properties
            width, height = image.size
            
            # Check resolution
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                return False, f"Image resolution too low. Minimum: {self.min_resolution}"
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                return False, f"Image resolution too high. Maximum: {self.max_resolution}"
            
            # Check image mode
            if image.mode not in ['RGB', 'L', 'RGBA']:
                medical_logger.logger.warning(f"Unusual image mode: {image.mode}")
            
            medical_logger.logger.info(f"Standard image validation successful: {file.filename}")
            return True, f"Valid {format_type} image"
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    def validate_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive image quality validation."""
        try:
            validation_result = {
                "file_path": image_path,
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "metrics": {}
            }
            
            # Check file existence
            if not os.path.exists(image_path):
                validation_result["is_valid"] = False
                validation_result["errors"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(image_path)
            validation_result["metrics"]["file_size_bytes"] = file_size
            validation_result["metrics"]["file_size_mb"] = round(file_size / (1024*1024), 2)
            
            if file_size == 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append("File is empty")
                return validation_result
            
            if file_size > self.max_file_size:
                validation_result["is_valid"] = False
                validation_result["errors"].append("File too large")
                return validation_result
            
            # Load and analyze image
            image_extension = os.path.splitext(image_path)[1].lower()
            
            if image_extension in ['.dcm', '.dicom']:
                validation_result = self._validate_dicom_quality(image_path, validation_result)
            elif image_extension == '.nii':
                validation_result = self._validate_nifti_quality(image_path, validation_result)
            else:
                validation_result = self._validate_standard_image_quality(image_path, validation_result)
            
            medical_logger.logger.info(f"Quality validation completed for {image_path}")
            return validation_result
            
        except Exception as e:
            medical_logger.logger.error(f"Quality validation failed for {image_path}: {e}")
            return {
                "file_path": image_path,
                "is_valid": False,
                "errors": [f"Quality validation failed: {str(e)}"],
                "warnings": [],
                "metrics": {}
            }
    
    def _validate_dicom_quality(self, image_path: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DICOM image quality."""
        try:
            import pydicom
            
            dicom = pydicom.dcmread(image_path)
            
            # Check pixel data
            if not hasattr(dicom, 'pixel_array'):
                validation_result["is_valid"] = False
                validation_result["errors"].append("No pixel data found in DICOM file")
                return validation_result
            
            pixel_array = dicom.pixel_array
            height, width = pixel_array.shape
            
            # Check dimensions
            validation_result["metrics"]["width"] = width
            validation_result["metrics"]["height"] = height
            validation_result["metrics"]["dimensions"] = f"{width}x{height}"
            
            # Check bit depth
            if hasattr(dicom, 'BitsAllocated'):
                validation_result["metrics"]["bits_allocated"] = dicom.BitsAllocated
            if hasattr(dicom, 'BitsStored'):
                validation_result["metrics"]["bits_stored"] = dicom.BitsStored
            
            # Check for blank images
            if pixel_array.min() == pixel_array.max():
                validation_result["is_valid"] = False
                validation_result["errors"].append("Blank or constant image")
                return validation_result
            
            # Check dynamic range
            pixel_range = pixel_array.max() - pixel_array.min()
            validation_result["metrics"]["pixel_range"] = pixel_range
            
            if pixel_range < 10:
                validation_result["warnings"].append("Very low dynamic range")
            
            # Check DICOM tags
            required_tags = ['PatientName', 'Modality', 'StudyDate']
            for tag in required_tags:
                if not hasattr(dicom, tag.lower().replace('_', '')):
                    validation_result["warnings"].append(f"Missing DICOM tag: {tag}")
            
            return validation_result
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"DICOM quality validation failed: {str(e)}")
            return validation_result
    
    def _validate_standard_image_quality(self, image_path: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate standard image quality."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                validation_result["metrics"]["width"] = width
                validation_result["metrics"]["height"] = height
                validation_result["metrics"]["dimensions"] = f"{width}x{height}"
                validation_result["metrics"]["mode"] = img.mode
                validation_result["metrics"]["format"] = img.format
                
                # Convert to array for analysis
                img_array = np.array(img)
                
                # Check for blank images
                if img_array.min() == img_array.max():
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Blank or constant image")
                    return validation_result
                
                # Check dynamic range
                pixel_range = img_array.max() - img_array.min()
                validation_result["metrics"]["pixel_range"] = pixel_range
                
                if pixel_range < 10:
                    validation_result["warnings"].append("Very low dynamic range")
                
                # Check brightness
                brightness = img_array.mean()
                validation_result["metrics"]["brightness"] = brightness
                
                if brightness < 20:
                    validation_result["warnings"].append("Image appears very dark")
                elif brightness > 235:
                    validation_result["warnings"].append("Image appears very bright")
                
                # Check contrast
                contrast = img_array.std()
                validation_result["metrics"]["contrast"] = contrast
                
                if contrast < 10:
                    validation_result["warnings"].append("Very low contrast")
            
            return validation_result
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Standard image quality validation failed: {str(e)}")
            return validation_result
    
    def _validate_nifti_quality(self, image_path: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate NIfTI image quality."""
        try:
            import nibabel as nib
            
            nii_img = nib.load(image_path)
            data = nii_img.get_fdata()
            
            # Remove singleton dimensions
            data = np.squeeze(data)
            
            if len(data.shape) < 2:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Invalid NIfTI data dimensions")
                return validation_result
            
            height, width = data.shape[:2]
            validation_result["metrics"]["width"] = width
            validation_result["metrics"]["height"] = height
            validation_result["metrics"]["dimensions"] = f"{width}x{height}"
            validation_result["metrics"]["shape"] = data.shape
            
            # Check for blank images
            if data.min() == data.max():
                validation_result["is_valid"] = False
                validation_result["errors"].append("Blank or constant image")
                return validation_result
            
            # Check dynamic range
            pixel_range = data.max() - data.min()
            validation_result["metrics"]["pixel_range"] = pixel_range
            
            if pixel_range < 0.1:  # For float data
                validation_result["warnings"].append("Very low dynamic range")
            
            return validation_result
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"NIfTI quality validation failed: {str(e)}")
            return validation_result
    
    def check_image_integrity(self, image_path: str) -> bool:
        """Quick integrity check for image file."""
        try:
            if not os.path.exists(image_path):
                return False
            
            file_extension = os.path.splitext(image_path)[1].lower()
            
            if file_extension in ['.dcm', '.dicom']:
                import pydicom
                dicom = pydicom.dcmread(image_path, stop_before_pixels=True)
                return True
            elif file_extension == '.nii':
                import nibabel as nib
                nib.load(image_path)
                return True
            else:
                with Image.open(image_path) as img:
                    img.verify()
                return True
                
        except Exception:
            return False
    
    def generate_file_hash(self, image_path: str) -> str:
        """Generate SHA-256 hash of file for integrity verification."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            medical_logger.logger.error(f"Failed to generate file hash: {e}")
            return ""


"""
File utility functions for medical imaging backend.
"""
import os
import hashlib
import mimetypes
from typing import Optional, Tuple
from fastapi import UploadFile
import shutil
from pathlib import Path


def validate_file_upload(file: UploadFile, allowed_extensions: list, max_size: int) -> bool:
    """Validate uploaded file."""
    if not file.filename:
        return False
    
    # Check file extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in [ext.lower() for ext in allowed_extensions]:
        return False
    
    # Check file size (seek to end to get size)
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if size > max_size:
        return False
    
    return True


def save_upload_file(file: UploadFile, upload_dir: str) -> str:
    """Save uploaded file to specified directory."""
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_hash = hashlib.md5()
    file_hash.update(file.filename.encode())
    unique_filename = f"{file_hash.hexdigest()}_{file.filename}"
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path


def get_file_mime_type(file_path: str) -> str:
    """Get MIME type of file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """Calculate hash of file."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if not."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def delete_file(file_path: str) -> bool:
    """Delete file safely."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def copy_file(src: str, dst: str) -> bool:
    """Copy file from source to destination."""
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def move_file(src: str, dst: str) -> bool:
    """Move file from source to destination."""
    try:
        shutil.move(src, dst)
        return True
    except Exception:
        return False


def get_safe_filename(filename: str) -> str:
    """Get safe filename by removing/replacing unsafe characters."""
    # Replace unsafe characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return ''.join(c if c in safe_chars else '_' for c in filename)


def create_temp_directory(prefix: str = "medical_imaging_") -> str:
    """Create temporary directory."""
    import tempfile
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_temp_directory(temp_dir: str) -> bool:
    """Clean up temporary directory."""
    try:
        shutil.rmtree(temp_dir)
        return True
    except Exception:
        return False


def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return os.path.splitext(filename)[1].lower()


def is_image_file(filename: str) -> bool:
    """Check if file is an image file."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.dcm', '.dicom'}
    ext = get_file_extension(filename)
    return ext in image_extensions


def is_dicom_file(filename: str) -> bool:
    """Check if file is a DICOM file."""
    dicom_extensions = {'.dcm', '.dicom'}
    ext = get_file_extension(filename)
    return ext in dicom_extensions


def get_directory_size(directory: str) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass
    return total_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def backup_file(file_path: str, backup_dir: str) -> Optional[str]:
    """Create backup of file."""
    try:
        ensure_directory_exists(backup_dir)
        filename = os.path.basename(file_path)
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{timestamp}_{filename}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        if copy_file(file_path, backup_path):
            return backup_path
        return None
    except Exception:
        return None


def restore_file(backup_path: str, original_path: str) -> bool:
    """Restore file from backup."""
    try:
        return move_file(backup_path, original_path)
    except Exception:
        return False

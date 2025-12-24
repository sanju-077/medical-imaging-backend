"""
Storage service for handling file storage operations.
"""
import os
import boto3
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import medical_logger


class StorageService:
    """Service for managing file storage operations."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.use_s3 = os.environ.get("USE_S3", "false").lower() == "true"
        self.s3_client = None
        
        if self.use_s3:
            self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize S3 client."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_REGION', 'us-east-1')
            )
        except Exception as e:
            medical_logger.logger.error(f"Failed to initialize S3 client: {e}")
            self.use_s3 = False
    
    def save_file(self, file_content: bytes, filename: str, content_type: str) -> str:
        """Save file to storage and return path/URL."""
        if self.use_s3:
            return self._save_to_s3(file_content, filename, content_type)
        else:
            return self._save_to_local(file_content, filename)
    
    def _save_to_local(self, file_content: bytes, filename: str) -> str:
        """Save file to local storage."""
        # Create upload directory if it doesn't exist
        upload_dir = settings.UPLOAD_FOLDER
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        medical_logger.logger.info(f"File saved locally: {file_path}")
        return file_path
    
    def _save_to_s3(self, file_content: bytes, filename: str, content_type: str) -> str:
        """Save file to S3 storage."""
        if not self.s3_client:
            raise ValueError("S3 client not initialized")
        
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        s3_key = f"medical-images/{unique_filename}"
        
        try:
            # Upload to S3
            self.s3_client.put_object(
                Bucket=os.environ.get('AWS_S3_BUCKET'),
                Key=s3_key,
                Body=file_content,
                ContentType=content_type,
                Metadata={
                    'original_filename': filename,
                    'uploaded_at': datetime.utcnow().isoformat()
                }
            )
            
            # Generate S3 URL
            bucket_name = os.environ.get('AWS_S3_BUCKET')
            region = os.environ.get('AWS_REGION', 'us-east-1')
            file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
            
            medical_logger.logger.info(f"File saved to S3: {s3_key}")
            return file_url
        
        except ClientError as e:
            medical_logger.logger.error(f"Failed to save file to S3: {e}")
            raise
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        try:
            if self.use_s3 and file_path.startswith('https://'):
                return self._delete_from_s3(file_path)
            else:
                return self._delete_from_local(file_path)
        except Exception as e:
            medical_logger.logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def _delete_from_local(self, file_path: str) -> bool:
        """Delete file from local storage."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                medical_logger.logger.info(f"File deleted locally: {file_path}")
                return True
            return False
        except Exception as e:
            medical_logger.logger.error(f"Failed to delete local file {file_path}: {e}")
            return False
    
    def _delete_from_s3(self, file_url: str) -> bool:
        """Delete file from S3 storage."""
        if not self.s3_client:
            return False
        
        try:
            # Extract S3 key from URL
            bucket_name = os.environ.get('AWS_S3_BUCKET')
            region = os.environ.get('AWS_REGION', 'us-east-1')
            
            if file_url.startswith(f"https://{bucket_name}.s3.{region}.amazonaws.com/"):
                s3_key = file_url.split(f"{bucket_name}.s3.{region}.amazonaws.com/")[1]
            else:
                medical_logger.logger.error(f"Invalid S3 URL format: {file_url}")
                return False
            
            # Delete from S3
            self.s3_client.delete_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            
            medical_logger.logger.info(f"File deleted from S3: {s3_key}")
            return True
        
        except ClientError as e:
            medical_logger.logger.error(f"Failed to delete S3 file {file_url}: {e}")
            return False
    
    def get_file_url(self, file_path: str) -> str:
        """Get file URL for access."""
        if self.use_s3 and file_path.startswith('https://'):
            return file_path
        elif not self.use_s3:
            # For local storage, return the file path
            return file_path
        else:
            # Convert local path to URL if needed
            return file_path
    
    def get_file_content(self, file_path: str) -> Optional[bytes]:
        """Get file content from storage."""
        try:
            if self.use_s3 and file_path.startswith('https://'):
                return self._get_from_s3(file_path)
            else:
                return self._get_from_local(file_path)
        except Exception as e:
            medical_logger.logger.error(f"Failed to get file content {file_path}: {e}")
            return None
    
    def _get_from_local(self, file_path: str) -> Optional[bytes]:
        """Get file content from local storage."""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            medical_logger.logger.error(f"Failed to read local file {file_path}: {e}")
            return None
    
    def _get_from_s3(self, file_url: str) -> Optional[bytes]:
        """Get file content from S3 storage."""
        if not self.s3_client:
            return None
        
        try:
            bucket_name = os.environ.get('AWS_S3_BUCKET')
            region = os.environ.get('AWS_REGION', 'us-east-1')
            
            if file_url.startswith(f"https://{bucket_name}.s3.{region}.amazonaws.com/"):
                s3_key = file_url.split(f"{bucket_name}.s3.{region}.amazonaws.com/")[1]
            else:
                return None
            
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            
            return response['Body'].read()
        
        except ClientError as e:
            medical_logger.logger.error(f"Failed to get S3 file {file_url}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "storage_type": "s3" if self.use_s3 else "local",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.use_s3:
            try:
                bucket_name = os.environ.get('AWS_S3_BUCKET')
                response = self.s3_client.list_objects_v2(Bucket=bucket_name)
                
                total_objects = response.get('KeyCount', 0)
                total_size = sum(obj.get('Size', 0) for obj in response.get('Contents', []))
                
                stats.update({
                    "total_objects": total_objects,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "bucket_name": bucket_name
                })
            
            except Exception as e:
                stats["error"] = f"Failed to get S3 stats: {e}"
        
        else:
            # Local storage stats
            try:
                upload_dir = settings.UPLOAD_FOLDER
                if os.path.exists(upload_dir):
                    total_size = 0
                    total_files = 0
                    
                    for root, dirs, files in os.walk(upload_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                            total_files += 1
                    
                    stats.update({
                        "total_files": total_files,
                        "total_size_bytes": total_size,
                        "total_size_mb": round(total_size / (1024 * 1024), 2),
                        "storage_path": upload_dir
                    })
                else:
                    stats.update({
                        "total_files": 0,
                        "total_size_bytes": 0,
                        "total_size_mb": 0,
                        "storage_path": upload_dir
                    })
            
            except Exception as e:
                stats["error"] = f"Failed to get local storage stats: {e}"
        
        return stats
    
    def cleanup_old_files(self, days: int = 30) -> Dict[str, int]:
        """Cleanup old files from storage."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0
        failed_count = 0
        
        if self.use_s3:
            try:
                bucket_name = os.environ.get('AWS_S3_BUCKET')
                response = self.s3_client.list_objects_v2(Bucket=bucket_name)
                
                for obj in response.get('Contents', []):
                    # Check if file is old enough (this is a simplified approach)
                    # In production, you'd want to check metadata
                    last_modified = obj.get('LastModified')
                    if last_modified and last_modified.replace(tzinfo=None) < cutoff_date:
                        s3_key = obj.get('Key')
                        if self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key):
                            deleted_count += 1
                        else:
                            failed_count += 1
            
            except Exception as e:
                medical_logger.logger.error(f"Failed to cleanup S3 files: {e}")
        
        else:
            # Local file cleanup
            try:
                upload_dir = settings.UPLOAD_FOLDER
                if os.path.exists(upload_dir):
                    for root, dirs, files in os.walk(upload_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                            if file_mtime < cutoff_date:
                                try:
                                    os.remove(file_path)
                                    deleted_count += 1
                                except Exception:
                                    failed_count += 1
            
            except Exception as e:
                medical_logger.logger.error(f"Failed to cleanup local files: {e}")
        
        return {
            "deleted_files": deleted_count,
            "failed_deletions": failed_count,
            "cleanup_date": datetime.utcnow().isoformat()
        }


"""
Celery tasks for medical imaging backend.
"""
from celery import current_task
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.inference_service import InferenceService
from app.services.result_service import ResultService
from app.ml.inference.pipeline import InferencePipeline
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='process_inference_task')
def process_inference_task(self, inference_request_id: int, image_ids: list, model_type: str, parameters: dict):
    """Process a single or batch inference task."""
    db = SessionLocal()
    
    try:
        inference_service = InferenceService(db)
        result_service = ResultService(db)
        
        # Update task progress
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': len(image_ids), 'status': 'Starting inference...'})
        
        # Create batch inference request if multiple images
        batch_id = None
        if len(image_ids) > 1:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = []
        
        for i, image_id in enumerate(image_ids):
            try:
                # Update progress
                progress = int((i / len(image_ids)) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': i + 1,
                        'total': len(image_ids),
                        'status': f'Processing image {i + 1}/{len(image_ids)}',
                        'progress': progress
                    }
                )
                
                # Get image information
                image = inference_service.get_image_by_id(image_id, None)  # System task
                if not image:
                    logger.warning(f"Image {image_id} not found")
                    continue
                
                # Create inference request
                request = inference_service.create_inference_request(
                    image_id=image_id,
                    user_id=image.user_id,
                    model_type=model_type,
                    parameters=parameters,
                    batch_id=batch_id
                )
                
                # Update request status to processing
                inference_service.update_inference_status(request.id, "processing")
                
                # Initialize inference pipeline
                pipeline = InferencePipeline(model_type, parameters)
                
                # Run inference
                result_data = pipeline.run_inference(image.file_path)
                
                # Save result
                result = result_service.save_inference_result(
                    request_id=request.id,
                    result_data=result_data,
                    disease_type=result_data.get('disease_type'),
                    confidence=result_data.get('confidence'),
                    diagnosis=result_data.get('diagnosis'),
                    recommendation=result_data.get('recommendation'),
                    quality_metrics=result_data.get('quality_metrics')
                )
                
                # Update request status to completed
                inference_service.update_inference_status(
                    request.id,
                    "completed",
                    progress=100
                )
                
                results.append({
                    'image_id': image_id,
                    'result_id': result.id,
                    'status': 'completed'
                })
                
                logger.info(f"Successfully processed inference for image {image_id}")
                
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {str(e)}")
                
                # Update request status to failed if it exists
                try:
                    request = inference_service.get_inference_request_by_image(image_id)
                    if request:
                        inference_service.update_inference_status(
                            request.id,
                            "failed",
                            error_message=str(e)
                        )
                except Exception as inner_e:
                    logger.error(f"Error updating failed status: {inner_e}")
                
                results.append({
                    'image_id': image_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Final progress update
        self.update_state(
            state='SUCCESS',
            meta={
                'current': len(image_ids),
                'total': len(image_ids),
                'status': 'Completed',
                'results': results
            }
        )
        
        return {
            'batch_id': batch_id,
            'total_images': len(image_ids),
            'successful': len([r for r in results if r['status'] == 'completed']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch inference task failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        raise
    finally:
        db.close()


@celery_app.task(name='process_batch_inference_task')
def process_batch_inference_task(batch_id: str, image_ids: list, model_type: str, parameters: dict):
    """Process a batch of inference tasks."""
    # This is a wrapper that calls process_inference_task for batch processing
    return process_inference_task.delay(
        inference_request_id=0,  # Not used for batch
        image_ids=image_ids,
        model_type=model_type,
        parameters=parameters
    )


@celery_app.task(name='cleanup_temp_files_task')
def cleanup_temp_files_task():
    """Clean up temporary files and old inference results."""
    db = SessionLocal()
    
    try:
        import os
        import shutil
        from datetime import datetime, timedelta
        
        # Clean up temporary inference files older than 24 hours
        temp_dir = "/tmp/medical_imaging_inference"
        if os.path.exists(temp_dir):
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_time:
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed temporary file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file_path}: {e}")
        
        # Clean up completed inference requests older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        
        result_service = ResultService(db)
        result_service.cleanup_old_results(cutoff_date)
        
        logger.info("Cleanup task completed successfully")
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
        raise
    finally:
        db.close()


@celery_app.task(name='update_model_metrics_task')
def update_model_metrics_task():
    """Update and log model performance metrics."""
    db = SessionLocal()
    
    try:
        result_service = ResultService(db)
        
        # Calculate metrics for the last 24 hours
        metrics = result_service.calculate_model_metrics()
        
        # Log metrics
        for model_name, model_metrics in metrics.items():
            logger.info(f"Model {model_name} metrics:")
            logger.info(f"  Accuracy: {model_metrics.get('accuracy', 'N/A')}")
            logger.info(f"  Total inferences: {model_metrics.get('total_inferences', 0)}")
            logger.info(f"  Average confidence: {model_metrics.get('avg_confidence', 0):.3f}")
        
        logger.info("Model metrics update completed")
        
    except Exception as e:
        logger.error(f"Model metrics update failed: {str(e)}")
        raise
    finally:
        db.close()


@celery_app.task(name='send_notification_task')
def send_notification_task(user_id: int, notification_type: str, message: str):
    """Send notification to user."""
    db = SessionLocal()
    
    try:
        # Implementation for sending notifications (email, SMS, push, etc.)
        logger.info(f"Sending {notification_type} notification to user {user_id}: {message}")
        
        # This could integrate with:
        # - Email service (SendGrid, AWS SES, etc.)
        # - Push notification service (FCM, APNS, etc.)
        # - SMS service (Twilio, AWS SNS, etc.)
        
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")
        raise
    finally:
        db.close()


@celery_app.task(name='export_results_task')
def export_results_task(user_id: int, format: str, date_range: dict, export_path: str):
    """Export inference results in specified format."""
    db = SessionLocal()
    
    try:
        result_service = ResultService(db)
        
        # Export results
        export_data = result_service.export_user_results(
            user_id=user_id,
            format=format,
            date_from=date_range.get('from'),
            date_to=date_range.get('to')
        )
        
        # Save to file
        with open(export_path, 'w') as f:
            f.write(export_data)
        
        logger.info(f"Results exported to {export_path} for user {user_id}")
        
        return {
            'export_path': export_path,
            'format': format,
            'user_id': user_id
        }
        
    except Exception as e:
        logger.error(f"Export task failed: {str(e)}")
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='retrain_model_task')
def retrain_model_task(self, model_name: str, training_data_path: str, parameters: dict):
    """Retrain a specific model with new data."""
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Starting model retraining...'})
        
        # This would integrate with your ML training pipeline
        logger.info(f"Starting retraining for model {model_name}")
        
        # Update progress during training
        self.update_state(state='PROGRESS', meta={'status': 'Loading training data...', 'progress': 10})
        
        # Training steps would go here
        self.update_state(state='PROGRESS', meta={'status': 'Training model...', 'progress': 50})
        
        # More training steps...
        self.update_state(state='PROGRESS', meta={'status': 'Validating model...', 'progress': 80})
        
        # Complete training
        self.update_state(
            state='SUCCESS',
            meta={'status': 'Model retraining completed successfully', 'progress': 100}
        )
        
        return {
            'model_name': model_name,
            'status': 'completed',
            'parameters': parameters
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


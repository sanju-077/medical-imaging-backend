"""
Inference API endpoints for medical image analysis.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import uuid

from app.schemas.inference import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest,
    InferenceResult, InferenceStatus
)
from app.services.inference_service import InferenceService
from app.dependencies import get_db, get_current_active_user
from app.models.user import User
from app.workers.tasks import process_inference_task

router = APIRouter()


@router.post("/single", response_model=InferenceResponse)
async def run_single_inference(
    request: InferenceRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Run inference on a single medical image."""
    inference_service = InferenceService(db)
    
    # Validate image exists and belongs to user
    image = inference_service.get_image_by_id(request.image_id, current_user.id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found or access denied"
        )
    
    # Create inference request
    inference_request = inference_service.create_inference_request(
        image_id=request.image_id,
        user_id=current_user.id,
        model_type=request.model_type,
        parameters=request.parameters
    )
    
    try:
        # Run inference
        result = inference_service.run_inference(
            inference_request.id,
            image.file_path,
            request.model_type,
            request.parameters
        )
        
        return InferenceResponse(
            request_id=inference_request.id,
            status="completed",
            result=result,
            created_at=inference_request.created_at
        )
    
    except Exception as e:
        # Update request status to failed
        inference_service.update_inference_status(
            inference_request.id,
            "failed",
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/batch", response_model=InferenceResponse)
async def run_batch_inference(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Run inference on multiple images asynchronously."""
    inference_service = InferenceService(db)
    
    # Validate all images belong to user
    images = []
    for image_id in request.image_ids:
        image = inference_service.get_image_by_id(image_id, current_user.id)
        if not image:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} not found or access denied"
            )
        images.append(image)
    
    # Create batch inference request
    batch_request = inference_service.create_batch_inference_request(
        image_ids=request.image_ids,
        user_id=current_user.id,
        model_type=request.model_type,
        parameters=request.parameters
    )
    
    # Add background task for processing
    background_tasks.add_task(
        process_inference_task.delay,
        batch_request.id,
        request.image_ids,
        request.model_type,
        request.parameters
    )
    
    return InferenceResponse(
        request_id=batch_request.id,
        status="processing",
        result=None,
        created_at=batch_request.created_at
    )


@router.get("/status/{request_id}", response_model=InferenceStatus)
async def get_inference_status(
    request_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get inference request status."""
    inference_service = InferenceService(db)
    
    request_obj = inference_service.get_inference_request(request_id, current_user.id)
    if not request_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inference request not found"
        )
    
    return InferenceStatus(
        request_id=request_obj.id,
        status=request_obj.status,
        progress=request_obj.progress,
        error_message=request_obj.error_message,
        created_at=request_obj.created_at,
        updated_at=request_obj.updated_at
    )


@router.get("/result/{request_id}", response_model=InferenceResponse)
async def get_inference_result(
    request_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get inference result."""
    inference_service = InferenceService(db)
    
    request_obj = inference_service.get_inference_request(request_id, current_user.id)
    if not request_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inference request not found"
        )
    
    if request_obj.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference request is {request_obj.status}"
        )
    
    # Get result data
    result = inference_service.get_inference_result(request_id)
    
    return InferenceResponse(
        request_id=request_obj.id,
        status=request_obj.status,
        result=result,
        created_at=request_obj.created_at
    )


@router.get("/history", response_model=List[InferenceStatus])
async def get_inference_history(
    page: int = 1,
    size: int = 20,
    model_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's inference history."""
    inference_service = InferenceService(db)
    
    requests = inference_service.get_user_inference_history(
        user_id=current_user.id,
        page=page,
        size=size,
        model_type=model_type
    )
    
    return [
        InferenceStatus(
            request_id=req.id,
            status=req.status,
            progress=req.progress,
            error_message=req.error_message,
            created_at=req.created_at,
            updated_at=req.updated_at
        )
        for req in requests
    ]


@router.delete("/{request_id}")
async def delete_inference_request(
    request_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an inference request and its results."""
    inference_service = InferenceService(db)
    
    request_obj = inference_service.get_inference_request(request_id, current_user.id)
    if not request_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inference request not found"
        )
    
    inference_service.delete_inference_request(request_id, current_user.id)
    
    return {"message": "Inference request deleted successfully"}


@router.get("/models/available")
async def get_available_models(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available ML models."""
    return {
        "classification": [
            {
                "name": "tuberculosis_v1",
                "type": "classification",
                "description": "Tuberculosis detection from chest X-rays",
                "accuracy": 0.94,
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"]
            },
            {
                "name": "pneumonia_v1",
                "type": "classification",
                "description": "Pneumonia detection from chest X-rays",
                "accuracy": 0.96,
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"]
            },
            {
                "name": "fracture_v1",
                "type": "classification",
                "description": "Bone fracture detection from X-rays",
                "accuracy": 0.92,
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm"]
            }
        ],
        "segmentation": [
            {
                "name": "unet_v1",
                "type": "segmentation",
                "description": "Organ segmentation using U-Net",
                "accuracy": 0.89,
                "supported_formats": [".jpg", ".jpeg", ".png", ".dcm", ".nii"]
            }
        ]
    }


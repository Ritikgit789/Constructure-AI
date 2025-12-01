"""
Structured extraction routes.

Endpoints:
- POST /door-schedule - Extract door schedule
- POST /room-schedule - Extract room schedule
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from loguru import logger
import time

from app.services.extraction import extraction_service
from app.models.schemas import ExtractionRequest, ExtractionResponse

router = APIRouter()


@router.post("/door-schedule")
async def extract_door_schedule(
    filter_criteria: Optional[Dict[str, Any]] = None
):
    """
    Extract door schedule from construction documents.
    
    Returns structured JSON with door information:
    - Mark/ID
    - Location
    - Dimensions
    - Fire rating
    - Material
    - Hardware
    
    Args:
        filter_criteria: Optional filters (e.g., specific document)
        
    Returns:
        ExtractionResponse with door schedule data
    """
    logger.info("Door schedule extraction requested")
    start_time = time.time()
    
    try:
        # Extract door schedule
        door_schedule = await extraction_service.extract_door_schedule(
            filter_by=filter_criteria
        )
        
        processing_time = time.time() - start_time
        
        return ExtractionResponse(
            extraction_type="door_schedule",
            data=door_schedule.dict(),
            sources=door_schedule.sources,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting door schedule: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting door schedule: {str(e)}"
        )


@router.post("/room-schedule")
async def extract_room_schedule(
    filter_criteria: Optional[Dict[str, Any]] = None
):
    """
    Extract room schedule from construction documents.
    
    Returns structured JSON with room information:
    - Room number
    - Room name
    - Area
    - Floor finish
    - Wall finish
    - Ceiling finish
    
    Args:
        filter_criteria: Optional filters
        
    Returns:
        ExtractionResponse with room schedule data
    """
    logger.info("Room schedule extraction requested")
    start_time = time.time()
    
    try:
        # Extract room schedule
        room_schedule = await extraction_service.extract_room_schedule(
            filter_by=filter_criteria
        )
        
        processing_time = time.time() - start_time
        
        return ExtractionResponse(
            extraction_type="room_schedule",
            data=room_schedule.dict(),
            sources=room_schedule.sources,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error extracting room schedule: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting room schedule: {str(e)}"
        )


@router.post("/generic")
async def generic_extraction(request: ExtractionRequest):
    """
    Generic extraction endpoint.
    
    Supports:
    - door_schedule
    - room_schedule
    - mep_equipment
    
    Args:
        request: ExtractionRequest with type and filters
        
    Returns:
        ExtractionResponse with extracted data
    """
    logger.info(f"Generic extraction requested: {request.extraction_type}")
    
    if request.extraction_type == "door_schedule":
        return await extract_door_schedule(request.filter_criteria)
    elif request.extraction_type == "room_schedule":
        return await extract_room_schedule(request.filter_criteria)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extraction type: {request.extraction_type}"
        )

"""
Structured extraction service for construction documents.

Extracts:
- Door schedules
- Room schedules  
- MEP equipment lists
"""

from typing import Dict, Any, List
from loguru import logger
import time

from app.services.gemini_service import gemini_service
from app.services.retrieval import rag_service
from app.models.schemas import DoorSchedule, DoorEntry, RoomSchedule, RoomEntry


class ExtractionService:
    """Service for extracting structured data from construction documents"""
    
    async def extract_door_schedule(
        self,
        filter_by: Dict[str, Any] = None
    ) -> DoorSchedule:
        """
        Extract door schedule from documents.
        
        Args:
            filter_by: Optional filters (e.g., specific document)
            
        Returns:
            DoorSchedule object
        """
        logger.info("Extracting door schedule")
        start_time = time.time()
        
        try:
            # Retrieve relevant context
            text_contexts, image_contexts = await rag_service.retrieve_for_extraction(
                extraction_type="door_schedule",
                filter_by=filter_by
            )
            
            # Extract structured data with Gemini
            extraction_result = await gemini_service.extract_structured_data(
                extraction_type="door_schedule",
                contexts=text_contexts,
                image_contexts=image_contexts
            )
            
            # Parse into Pydantic model
            entries_data = extraction_result.get("entries", [])
            entries = [DoorEntry(**entry) for entry in entries_data]
            
            # Generate citations
            _, _, citations = await rag_service.retrieve_context(
                query="door schedule",
                top_k=5
            )
            
            door_schedule = DoorSchedule(
                entries=entries,
                sources=citations,
                total_doors=len(entries),
                extraction_confidence=0.85  # Simple confidence for now
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Extracted {len(entries)} doors in {elapsed:.2f}s")
            
            return door_schedule
            
        except Exception as e:
            logger.error(f"Error extracting door schedule: {e}")
            raise
    
    async def extract_room_schedule(
        self,
        filter_by: Dict[str, Any] = None
    ) -> RoomSchedule:
        """Extract room schedule from documents"""
        logger.info("Extracting room schedule")
        start_time = time.time()
        
        try:
            # Retrieve relevant context
            text_contexts, image_contexts = await rag_service.retrieve_for_extraction(
                extraction_type="room_schedule",
                filter_by=filter_by
            )
            
            # Extract structured data
            extraction_result = await gemini_service.extract_structured_data(
                extraction_type="room_schedule",
                contexts=text_contexts,
                image_contexts=image_contexts
            )
            
            # Parse into Pydantic model
            entries_data = extraction_result.get("entries", [])
            entries = [RoomEntry(**entry) for entry in entries_data]
            
            # Generate citations
            _, _, citations = await rag_service.retrieve_context(
                query="room schedule finishes",
                top_k=5
            )
            
            room_schedule = RoomSchedule(
                entries=entries,
                sources=citations,
                total_rooms=len(entries),
                extraction_confidence=0.85
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Extracted {len(entries)} rooms in {elapsed:.2f}s")
            
            return room_schedule
            
        except Exception as e:
            logger.error(f"Error extracting room schedule: {e}")
            raise


# Create singleton instance
extraction_service = ExtractionService()

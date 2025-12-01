"""
Document management routes.

Endpoints:
- POST /upload - Upload and ingest PDF
- GET /list - List all documents
- DELETE /{document_id} - Delete document
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import tempfile
from loguru import logger

from app.services.ingestion import ingestion_service
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentMetadata

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = "general"
):
    """
    Upload and ingest a PDF document.
    
    This triggers the full ingestion pipeline:
    1. Save PDF
    2. Extract text and images
    3. Generate image descriptions with Gemini Vision
    4. Create embeddings
    5. Store in FAISS
    
    Args:
        file: PDF file upload
        document_type: Type of document (drawing, specification, schedule, general)
        
    Returns:
        DocumentUploadResponse with metadata
    """
    logger.info(f"Received upload request: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 100_000_000:  # 100MB
        raise HTTPException(status_code=400, detail="File size exceeds 100MB limit")
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Ingest document
        document_id, metadata = await ingestion_service.ingest_document(
            file_path=tmp_path,
            filename=file.filename,
            document_type=document_type
        )
        
        # Clean up temp file
        os.remove(tmp_path)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            message=f"Document ingested successfully: {metadata.total_text_chunks} text chunks, {metadata.total_images} images",
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """
    List all ingested documents.
    
    Returns:
        DocumentListResponse with list of documents and metadata
    """
    try:
        documents_data = ingestion_service.list_documents()
        
        # Convert to DocumentMetadata objects
        documents = []
        for doc_data in documents_data:
            metadata = DocumentMetadata(
                filename=doc_data["filename"],
                file_size=0,  # Not stored in vector DB
                upload_time=None,  # Not stored
                total_pages=doc_data["total_pages"],
                total_text_chunks=doc_data["text_chunks"],
                total_images=doc_data["image_chunks"],
                document_type=None
            )
            documents.append(metadata)
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/stats")
async def get_vector_store_stats():
    """
    Get statistics about the vector store.
    
    Returns:
        Vector store statistics
    """
    from app.db.vector_store import vector_store
    return vector_store.get_stats()

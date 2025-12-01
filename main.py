"""
FastAPI Main Application - Project Brain Backend

Multimodal RAG system for construction documents with:
- Document ingestion (PDF with text + images)
- Gemini-powered multimodal understanding
- FAISS vector search
- Chat Q&A with citations
- Structured extraction (door/room schedules)
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path so we can import app modules
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Now we can import from app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.api.routes import documents, chat, extract, evaluate

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)

# Create FastAPI app
app = FastAPI(
    title="Project Brain API",
    description="Multimodal RAG system for construction document Q&A and extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(extract.router, prefix="/api/extract", tags=["Extraction"])
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluation"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Project Brain API - Construction Document RAG System",
        "version": "1.0.0",
        "status": "active",
        "features": [
            "Multimodal document ingestion (text + images)",
            "RAG-based Q&A with citations",
            "Structured extraction (doors, rooms)",
            "Gemini-powered vision understanding",
            "FAISS vector search"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    from app.db.vector_store import vector_store
    
    return {
        "status": "healthy",
        "vector_store": vector_store.get_stats(),
        "gemini_model": settings.GEMINI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL + " (local)"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("=" * 60)
    logger.info("PROJECT BRAIN API STARTING")
    logger.info("=" * 60)
    logger.info(f"Gemini Model: {settings.GEMINI_MODEL}")
    logger.info(f"Vector Dimension: {settings.VECTOR_DIMENSION}")
    logger.info(f"FAISS Index Path: {settings.FAISS_INDEX_PATH}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - save vector store"""
    logger.info("Shutting down - saving vector store...")
    from app.db.vector_store import vector_store
    vector_store.save()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

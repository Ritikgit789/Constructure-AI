from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Gemini API (only for chat and vision, NOT embeddings)
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # Local Embeddings (sentence-transformers)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Free local embeddings
    
    # Application
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    
    # Upload Configuration
    MAX_UPLOAD_SIZE: int = 100_000_000  # 100MB
    UPLOAD_DIR: str = "./backend/data/uploads"
    EXTRACTED_IMAGES_DIR: str = "./backend/data/extracted_images"
    
    # Vector Store (FAISS)
    FAISS_INDEX_PATH: str = "./backend/data/faiss_index"
    VECTOR_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    
    # RAG Configuration
    TOP_K_RETRIEVAL: int = 10
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_IMAGE_SIZE: int = 2048  # Max image dimension for processing
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXTRACTED_IMAGES_DIR, exist_ok=True)
os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)

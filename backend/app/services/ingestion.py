"""
Document ingestion service (FAST MODE).

Handles end-to-end document processing without image captioning
or image embeddings. Only text embeddings are generated.
"""

import os
import shutil
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple
from loguru import logger
import asyncio

from app.config import settings
from app.utils.chunking import DocumentChunker
from app.services.gemini_service import gemini_service   # only text embedding
from app.db.vector_store import vector_store
from app.models.schemas import DocumentMetadata


class IngestionService:
    """FAST document ingestion service (text-only embeddings)"""

    def __init__(self):
        self.chunker = DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            extract_images_dir=settings.EXTRACTED_IMAGES_DIR
        )
        logger.info("Initialized ingestion service (FAST MODE)")

    async def ingest_document(
        self,
        file_path: str,
        filename: str,
        document_type: str = "general"
    ) -> Tuple[str, DocumentMetadata]:

        logger.info(f"Starting ingestion for: {filename}")
        start_time = datetime.now()

        try:
            # Generate document ID
            document_id = self._generate_document_id(filename)

            # Save file
            saved_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{filename}")
            shutil.copy(file_path, saved_path)
            logger.info(f"Saved file → {saved_path}")

            # Extract text + images
            logger.info("Extracting text and images...")
            text_chunks, image_chunks = self.chunker.chunk_document(
                pdf_path=saved_path,
                document_id=document_id
            )
            logger.info(f"Chunking complete: {len(text_chunks)} text chunks, {len(image_chunks)} image chunks")

            # ------------------------------
            # FAST MODE: SKIP IMAGE DESCRIPTIONS
            # ------------------------------
            logger.info("Skipping image description generation to optimize speed")

            # ------------------------------
            # Generate embeddings (text only)
            # ------------------------------
            all_chunks = text_chunks + image_chunks
            embeddings = await self._generate_embeddings(all_chunks)

            # Store into vector DB
            logger.info("Adding chunks to FAISS vector store...")
            vector_store.add_chunks(all_chunks, embeddings)
            vector_store.save()

            # Metadata
            metadata = DocumentMetadata(
                filename=filename,
                file_size=os.path.getsize(saved_path),
                upload_time=datetime.now(),
                total_pages=max([c.get("page_number", 1) for c in all_chunks], default=1),
                total_text_chunks=len(text_chunks),
                total_images=len(image_chunks),
                document_type=document_type
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"FAST INGEST COMPLETE: {len(text_chunks)} text chunks + {len(image_chunks)} image chunks "
                f"in {elapsed:.2f}s"
            )

            return document_id, metadata

        except Exception as e:
            logger.error(f"Error ingesting {filename}: {e}")
            raise

    async def _generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """Generate embeddings for text chunks only (FAST mode)"""

        text_embeddings = []
        batch_size = 12  # slightly larger for speed

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            for chunk in batch:
                content = chunk.get("content", "")
                chunk_type = chunk.get("chunk_type", "text")

                if chunk_type == "text" and content.strip():
                    try:
                        text_embedding = await gemini_service.generate_text_embedding(content)
                        text_embeddings.append(text_embedding)
                        chunk["embedding_type"] = "text"
                    except Exception as e:
                        logger.warning(f"Embedding failed for chunk {chunk.get('chunk_id')}: {e}")
                        text_embeddings.append([0.0] * settings.VECTOR_DIMENSION)
                else:
                    # Image chunk → no embedding required
                    text_embeddings.append([0.0] * settings.VECTOR_DIMENSION)
                    chunk["embedding_type"] = "image"

            await asyncio.sleep(0.05)

        logger.info(f"Generated {len(text_embeddings)} text embeddings")
        return text_embeddings

    def _generate_document_id(self, filename: str) -> str:
        hash_value = hashlib.md5(f"{filename}_{datetime.now()}".encode()).hexdigest()[:12]
        return f"doc_{hash_value}"

    def list_documents(self) -> List[Dict[str, Any]]:
        doc_ids = list(vector_store.document_map.keys())
        results = []

        for doc_id in doc_ids:
            chunks = vector_store.get_chunks_by_document(doc_id)
            if not chunks:
                continue

            results.append({
                "document_id": doc_id,
                "filename": chunks[0].get("filename", "unknown"),
                "total_chunks": len(chunks),
                "text_chunks": sum(1 for c in chunks if c.get("chunk_type") == "text"),
                "image_chunks": sum(1 for c in chunks if c.get("chunk_type") == "image"),
            })

        return results


ingestion_service = IngestionService()

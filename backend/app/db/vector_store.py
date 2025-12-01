"""
FAISS Vector Store for multimodal RAG.

Handles:
- Text chunk embeddings
- Image description embeddings  
- Hybrid search (vector + keyword)
- Metadata filtering
- Persistence to disk
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from app.config import settings
import json


class FAISSVectorStore:
    """
    FAISS-based vector store for construction document RAG.
    
    Stores both text and image chunks with metadata for retrieval.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []  # Stores chunk metadata
        self.document_map = {}  # Maps document_id to chunks
        self.index_path = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
        self.metadata_path = os.path.join(settings.FAISS_INDEX_PATH, "metadata.pkl")
        self.docmap_path = os.path.join(settings.FAISS_INDEX_PATH, "docmap.json")
        
        logger.info(f"Initialized FAISS vector store (dimension={dimension})")
        
        # Load existing index if available
        self.load()
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: Corresponding embeddings for each chunk
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        # Create index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
            logger.info("Created new FAISS index")
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_np)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **chunk,
                "vector_id": start_idx + i
            }
            self.metadata.append(chunk_meta)
            
            # Update document map
            doc_id = chunk.get("document_id")
            if doc_id:
                if doc_id not in self.document_map:
                    self.document_map[doc_id] = []
                self.document_map[doc_id].append(start_idx + i)
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_by: Metadata filters (e.g., {"document_id": "doc123", "chunk_type": "text"})
            
        Returns:
            List of chunks with similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        # Convert to numpy and normalize
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search
        # Get more results to account for filtering
        search_k = min(top_k * 3, self.index.ntotal) if filter_by else top_k
        scores, indices = self.index.search(query_np, search_k)
        
        # Build results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            chunk_meta = self.metadata[idx].copy()
            chunk_meta["relevance_score"] = float(score)
            
            # Apply filters if specified
            if filter_by:
                if not self._matches_filter(chunk_meta, filter_by):
                    continue
            
            results.append(chunk_meta)
            
            if len(results) >= top_k:
                break
        
        logger.debug(f"FAISS search returned {len(results)} results")
        return results
    
    def keyword_search(
        self,
        keyword: str,
        top_k: int = 10,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword search in chunk content.
        
        Args:
            keyword: Keyword to search for
            top_k: Maximum results
            filter_by: Metadata filters
            
        Returns:
            List of matching chunks
        """
        keyword_lower = keyword.lower()
        results = []
        
        for chunk_meta in self.metadata:
            # Apply filters first
            if filter_by and not self._matches_filter(chunk_meta, filter_by):
                continue
            
            # Check if keyword in content
            content = chunk_meta.get("content", "")
            if keyword_lower in content.lower():
                chunk_copy = chunk_meta.copy()
                # Simple scoring based on frequency
                frequency = content.lower().count(keyword_lower)
                chunk_copy["relevance_score"] = min(frequency * 0.1, 1.0)
                results.append(chunk_copy)
        
        # Sort by score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:top_k]
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        keywords: Optional[List[str]] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,  # Weight for vector search
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            query_embedding: Query vector
            keywords: Optional keywords to boost
            top_k: Number of results
            vector_weight: Weight for vector vs keyword (0-1)
            filter_by: Metadata filters
            
        Returns:
            Ranked list of chunks
        """
        # Vector search
        vector_results = self.search(query_embedding, top_k=top_k * 2, filter_by=filter_by)
        
        # Keyword search if provided
        keyword_results = []
        if keywords:
            for keyword in keywords:
                kw_results = self.keyword_search(keyword, top_k=top_k, filter_by=filter_by)
                keyword_results.extend(kw_results)
        
        # Combine and re-rank
        chunk_scores = {}
        
        # Add vector scores
        for chunk in vector_results:
            chunk_id = chunk["chunk_id"]
            chunk_scores[chunk_id] = {
                "chunk": chunk,
                "score": chunk["relevance_score"] * vector_weight
            }
        
        # Add keyword scores
        if keyword_results:
            keyword_weight = 1.0 - vector_weight
            for chunk in keyword_results:
                chunk_id = chunk["chunk_id"]
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]["score"] += chunk["relevance_score"] * keyword_weight
                else:
                    chunk_scores[chunk_id] = {
                        "chunk": chunk,
                        "score": chunk["relevance_score"] * keyword_weight
                    }
        
        # Sort by combined score
        ranked_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Return top-k
        results = []
        for item in ranked_results[:top_k]:
            chunk = item["chunk"].copy()
            chunk["relevance_score"] = item["score"]
            results.append(chunk)
        
        logger.debug(f"Hybrid search returned {len(results)} results")
        return results
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        if document_id not in self.document_map:
            return []
        
        vector_ids = self.document_map[document_id]
        chunks = [self.metadata[vid] for vid in vector_ids if vid < len(self.metadata)]
        return chunks
    
    def delete_document(self, document_id: str) -> None:
        """
        Remove all chunks for a document.
        
        Note: FAISS doesn't support deletion, so we mark as deleted in metadata.
        For complete removal, rebuild the index.
        """
        if document_id not in self.document_map:
            logger.warning(f"Document {document_id} not found in vector store")
            return
        
        vector_ids = self.document_map[document_id]
        
        # Mark chunks as deleted
        for vid in vector_ids:
            if vid < len(self.metadata):
                self.metadata[vid]["deleted"] = True
        
        # Remove from document map
        del self.document_map[document_id]
        
        logger.info(f"Marked {len(vector_ids)} chunks as deleted for document {document_id}")
    
    def save(self) -> None:
        """Save index and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
            
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved metadata to {self.metadata_path}")
            
            with open(self.docmap_path, "w") as f:
                json.dump(self.document_map, f)
            logger.info(f"Saved document map to {self.docmap_path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def load(self) -> None:
        """Load index and metadata from disk"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded metadata ({len(self.metadata)} chunks)")
            
            if os.path.exists(self.docmap_path):
                with open(self.docmap_path, "r") as f:
                    self.document_map = json.load(f)
                logger.info(f"Loaded document map ({len(self.document_map)} documents)")
                
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    def _matches_filter(self, chunk_meta: Dict[str, Any], filter_by: Dict[str, Any]) -> bool:
        """Check if chunk matches filter criteria"""
        for key, value in filter_by.items():
            if chunk_meta.get(key) != value:
                return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_chunks": len(self.metadata),
            "total_documents": len(self.document_map),
            "dimension": self.dimension,
            "index_path": self.index_path
        }


# Create singleton instance
vector_store = FAISSVectorStore(dimension=settings.VECTOR_DIMENSION)

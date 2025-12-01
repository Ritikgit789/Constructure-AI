"""
RAG Retrieval Service - The Core of the Multimodal RAG System

Handles:
- Query processing
- Hybrid retrieval (vector + keyword)
- Context assembly (text + images)
- Re-ranking
- Multi-document support
"""

from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import time
import re

from app.config import settings
from app.services.gemini_service import gemini_service
from app.db.vector_store import vector_store
from app.models.schemas import Citation


class RAGRetrievalService:
    """
    Advanced RAG retrieval service for construction documents.
    
    This is the heart of the multimodal RAG system.
    """
    
    def __init__(self):
        self.top_k = settings.TOP_K_RETRIEVAL
        self.rerank_top_k = settings.RERANK_TOP_K
        logger.info("Initialized RAG retrieval service")
    
    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_images: bool = True,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Citation]]:
        """
        Retrieve relevant context for a query.
        
        This is the MAIN retrieval pipeline:
        1. Generate query embedding
        2. Extract keywords for hybrid search
        3. Perform hybrid search (vector + keyword)
        4. Re-rank results
        5. Separate text and image contexts
        6. Generate citations
        
        Args:
            query: User query
            top_k: Number of results (default from settings)
            include_images: Whether to include image context
            filter_by: Metadata filters
            
        Returns:
            Tuple of (text_contexts, image_contexts, citations)
        """
        start_time = time.time()
        top_k = top_k or self.rerank_top_k
        
        logger.info(f"Retrieving context for query: {query}")
        
        try:
            # Step 1: Generate query embedding
            logger.debug("Generating query embedding...")
            query_embedding = await gemini_service.generate_query_embedding(query)
            
            # Step 2: Extract keywords from query
            keywords = self._extract_keywords(query)
            logger.debug(f"Extracted keywords: {keywords}")
            
            # Step 3: Hybrid search
            logger.debug("Performing hybrid search...")
            retrieved_chunks = vector_store.hybrid_search(
                query_embedding=query_embedding,
                keywords=keywords,
                top_k=self.top_k,  # Get more for reranking
                filter_by=filter_by
            )
            
            # Step 4: Re-rank (simple scoring for now)
            reranked_chunks = self._rerank_by_query_overlap(retrieved_chunks, query)
            
            # Take top-k after reranking
            final_chunks = reranked_chunks[:top_k]
            
            # Step 5: Separate text and image contexts
            text_contexts = []
            image_contexts = []
            
            for chunk in final_chunks:
                chunk_type = chunk.get("chunk_type")
                
                if chunk_type == "text":
                    text_contexts.append(chunk.get("content", ""))
                
                elif chunk_type == "image" and include_images:
                    fallback_desc = f"[Drawing page {chunk.get('page_number')} - image available]"
                    image_contexts.append({
                        "description": chunk.get("content", fallback_desc),
                        "image_path": chunk.get("image_path"),
                        "source": chunk.get("filename", ""),
                        "page": chunk.get("page_number", 0)
                    })
            # Step 6: Generate citations
            citations = self._generate_citations(final_chunks)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Retrieved {len(text_contexts)} text chunks, "
                f"{len(image_contexts)} images in {elapsed_ms:.1f}ms"
            )
            
            return text_contexts, image_contexts, citations
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {e}")
            raise
    
    async def retrieve_for_extraction(
        self,
        extraction_type: str,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Retrieve context specifically for structured extraction.
        
        Uses tailored queries based on extraction type.
        
        Args:
            extraction_type: Type of extraction (door_schedule, room_schedule, etc.)
            filter_by: Metadata filters
            
        Returns:
            Tuple of (text_contexts, image_contexts)
        """
        # Define search queries for each extraction type
        queries = {
            "door_schedule": "door schedule doors hardware dimensions fire rating",
            "room_schedule": "room schedule finishes floor ceiling wall area",
            "mep_equipment": "MEP equipment mechanical electrical plumbing HVAC"
        }
        
        query = queries.get(extraction_type, extraction_type)
        logger.info(f"Retrieving context for {extraction_type} extraction")
        
        # Retrieve with higher top_k for extraction
        text_contexts, image_contexts, _ = await self.retrieve_context(
            query=query,
            top_k=15,  # More context for extraction
            include_images=True,
            filter_by=filter_by
        )
        
        return text_contexts, image_contexts
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query for hybrid search.
        
        Focuses on:
        - Door/room numbers (e.g., D-101, RM-205)
        - Technical terms
        - Measurements
        - Specific identifiers
        """
        keywords = []
        
        # Extract room/door codes (e.g., D-101, RM-205, R-12)
        codes = re.findall(r'\b[A-Z]{1,3}[-]?\d+\b', query, re.IGNORECASE)
        keywords.extend(codes)
        
        # Extract measurements (e.g., 900mm, 2100mm, 1 hour)
        measurements = re.findall(r'\b\d+\s*(?:mm|cm|m|ft|in|hour|hr|min)\b', query, re.IGNORECASE)
        keywords.extend(measurements)
        
        # Extract important construction terms
        construction_terms = [
            'door', 'room', 'corridor', 'lobby', 'fire rating',
            'partition', 'ceiling', 'floor', 'wall', 'finish',
            'hardware', 'accessibility', 'schedule', 'equipment',
            'MEP', 'HVAC', 'plumbing', 'electrical'
        ]
        
        query_lower = query.lower()
        for term in construction_terms:
            if term.lower() in query_lower:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def _rerank_by_query_overlap(
        self,
        chunks: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Simple re-ranking based on query term overlap.
        
        Boosts chunks that contain query terms.
        """
        query_terms = set(query.lower().split())
        
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            content_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            
            # Boost relevance score
            original_score = chunk.get("relevance_score", 0)
            chunk["relevance_score"] = original_score + (overlap * 0.2)
        
        # Re-sort by updated scores
        chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return chunks
    
    def _generate_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """
        Generate citation objects from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            
            citation = Citation(
                source=chunk.get("filename", "unknown"),
                page=chunk.get("page_number", 0),
                chunk_type=chunk.get("chunk_type", "text"),
                content_preview=preview,
                relevance_score=chunk.get("relevance_score", 0.0),
                image_url=chunk.get("image_path") if chunk.get("chunk_type") == "image" else None
            )
            citations.append(citation)
        
        return citations
    
    async def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        include_images: bool = True,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Citation]]:
        """
        Full RAG chat pipeline.
        
        Steps:
        1. Retrieve relevant context
        2. Generate response with Gemini
        3. Return response with citations
        
        Args:
            query: User question
            conversation_history: Previous messages
            include_images: Include image context
            filter_by: Document filters
            
        Returns:
            Tuple of (response, citations)
        """
        logger.info(f"RAG chat query: {query}")
        
        try:
            # Retrieve context
            text_contexts, image_contexts, citations = await self.retrieve_context(
                query=query,
                include_images=include_images,
                filter_by=filter_by
            )
            
            # Generate response
            response = await gemini_service.chat_with_context(
                query=query,
                text_contexts=text_contexts,
                image_contexts=image_contexts,
                conversation_history=conversation_history
            )
            
            logger.info(f"Generated RAG response with {len(citations)} citations")
            
            return response, citations
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            raise


# Create singleton instance
rag_service = RAGRetrievalService()

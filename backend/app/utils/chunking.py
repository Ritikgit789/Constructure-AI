"""
Advanced chunking strategies for multimodal construction documents.
Handles PDFs with text, images, drawings, and schedules.
"""

from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from loguru import logger
from datetime import datetime
import hashlib


class DocumentChunker:
    """
    Advanced document chunking for construction PDFs.
    
    Features:
    - Extracts text with page context
    - Extracts images (drawings, schedules, diagrams)
    - Maintains text-image relationships
    - Generates meaningful chunk IDs
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_image_size: int = 100,  # Min width/height in pixels
        extract_images_dir: str = "./data/extracted_images"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_image_size = min_image_size
        self.extract_images_dir = extract_images_dir
        os.makedirs(extract_images_dir, exist_ok=True)
    
    def chunk_document(
        self,
        pdf_path: str,
        document_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Chunk a PDF document into text chunks and image chunks.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique document identifier
            
        Returns:
            Tuple of (text_chunks, image_chunks)
        """
        logger.info(f"Chunking document: {pdf_path}")
        
        pdf_document = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        text_chunks = []
        image_chunks = []
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text from page
            page_text = page.get_text("text")
            
            # Create text chunks for this page
            if page_text.strip():
                page_text_chunks = self._create_text_chunks(
                    text=page_text,
                    page_num=page_num + 1,  # 1-indexed
                    document_id=document_id,
                    filename=filename
                )
                text_chunks.extend(page_text_chunks)
            
            # Extract images from page
            page_images = self._extract_page_images(
                page=page,
                page_num=page_num + 1,
                document_id=document_id,
                filename=filename
            )
            image_chunks.extend(page_images)
        
        pdf_document.close()
        
        logger.info(
            f"Chunking complete: {len(text_chunks)} text chunks, "
            f"{len(image_chunks)} image chunks"
        )
        
        return text_chunks, image_chunks
    
    def _create_text_chunks(
        self,
        text: str,
        page_num: int,
        document_id: str,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Uses recursive splitting to maintain semantic coherence.
        """
        chunks = []
        
        # Simple overlapping chunking
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence ending
                last_period = chunk_text.rfind('. ')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # At least 70% of chunk
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk_text.strip():
                chunk_id = self._generate_chunk_id(
                    document_id, page_num, chunk_index, "text"
                )
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_type": "text",
                    "content": chunk_text.strip(),
                    "chunk_index": chunk_index
                })
                
                chunk_index += 1
            
            # Move start forward with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def _extract_page_images(
        self,
        page: fitz.Page,
        page_num: int,
        document_id: str,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page.
        
        Filters out small images (likely decorative or logos).
        Saves images to disk for later vision analysis.
        """
        image_chunks = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Load image to check size
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                
                # Filter out small images
                if width < self.min_image_size or height < self.min_image_size:
                    continue
                
                # Generate unique image path
                image_filename = f"{document_id}_p{page_num}_img{img_index}.{image_ext}"
                image_path = os.path.join(self.extract_images_dir, image_filename)
                
                # Save image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                chunk_id = self._generate_chunk_id(
                    document_id, page_num, img_index, "image"
                )
                
                image_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_type": "image",
                    "content": "",  # Will be filled with vision description later
                    "image_path": image_path,
                    "image_width": width,
                    "image_height": height,
                    "image_index": img_index
                })
                
                logger.debug(f"Extracted image: {image_path} ({width}x{height})")
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                continue
        
        return image_chunks
    
    def _generate_chunk_id(
        self,
        document_id: str,
        page_num: int,
        index: int,
        chunk_type: str
    ) -> str:
        """Generate unique chunk ID"""
        content = f"{document_id}_{page_num}_{chunk_type}_{index}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{chunk_type}_{page_num}_{index}_{hash_suffix}"
    
    def get_page_context(
        self,
        chunks: List[Dict[str, Any]],
        target_chunk_id: str,
        context_window: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get surrounding chunks for additional context.
        
        Useful for expanding context during retrieval.
        """
        # Find target chunk
        target_idx = None
        for idx, chunk in enumerate(chunks):
            if chunk["chunk_id"] == target_chunk_id:
                target_idx = idx
                break
        
        if target_idx is None:
            return []
        
        # Get surrounding chunks
        start_idx = max(0, target_idx - context_window)
        end_idx = min(len(chunks), target_idx + context_window + 1)
        
        return chunks[start_idx:end_idx]

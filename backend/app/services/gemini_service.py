"""
Gemini API integration for multimodal RAG.

Handles:
- Image understanding (construction drawings, schedules) via Gemini Vision
- Chat completions with vision via Gemini
- Text embeddings via LOCAL sentence-transformers (all-MiniLM-L6-v2)
- Structured extraction via Gemini
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
from PIL import Image
import os
from loguru import logger
from sentence_transformers import SentenceTransformer
from app.config import settings

# Configure Gemini (for chat and vision only)
genai.configure(api_key=settings.GOOGLE_API_KEY)


class GeminiService:
    """Service for interacting with Gemini API and local embeddings"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        # Load local text embedding model (sentence-transformers)
        logger.info(f"Loading text embedding model: {settings.EMBEDDING_MODEL}")
        self.text_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Text embedding model loaded (dimension: {self.text_embedding_model.get_sentence_embedding_dimension()})")
        
        # Load CLIP model for image embeddings
        logger.info("Loading CLIP image embedding model: clip-ViT-B-32")
        self.image_embedding_model = SentenceTransformer('clip-ViT-B-32')
        logger.info(f"Image embedding model loaded (dimension: {self.image_embedding_model.get_sentence_embedding_dimension()})")
        
        logger.info(f"Initialized Gemini service with model: {settings.GEMINI_MODEL}")
    
    async def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using LOCAL sentence-transformers.
        
        This is FREE and runs locally - no API calls!
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (384 dimensions for all-MiniLM-L6-v2)
        """
        try:
            # Use local sentence-transformers model
            embedding = self.text_embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query using LOCAL sentence-transformers.
        
        Args:
            query: Search query
            
        Returns:
            Embedding vector (384 dimensions)
        """
        try:
            # Use same model for queries
            embedding = self.text_embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    async def generate_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate embedding for an image using CLIP (clip-ViT-B-32).
        
        This is for image-heavy pages or construction drawings.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector (512 dimensions for CLIP)
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Generate CLIP embedding
            embedding = self.image_embedding_model.encode(img, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating image embedding for {image_path}: {e}")
            raise
    
    async def describe_image(
        self,
        image_path: str,
        context: str = ""
    ) -> str:
        """
        Generate description of construction drawing/image using Gemini Vision.
        
        This is CRITICAL for RAG - creates searchable text from images.
        
        Args:
            image_path: Path to image file
            context: Additional context (e.g., "construction drawing", "door schedule")
            
        Returns:
            Detailed description of image content
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            image = Image.open(image_path)
            
            # Craft prompt for construction context
            prompt = f"""Analyze this construction document image in detail.

{context}

Provide a comprehensive description including:
1. Type of document (drawing, schedule, diagram, specification table, etc.)
2. Key elements visible (rooms, doors, equipment, dimensions, labels)
3. Any text, numbers, or codes visible
4. Measurements, scales, or dimensions shown
5. Materials, finishes, or specifications mentioned
6. Any notable details relevant to construction

Be specific and thorough - this description will be used for search and retrieval."""
            
            # Generate description
            response = self.model.generate_content([prompt, image])
            description = response.text
            
            logger.debug(f"Generated image description for {os.path.basename(image_path)}")
            return description
            
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {e}")
            return f"[Image at {os.path.basename(image_path)} - description failed]"
    
    async def chat_with_context(
        self,
        query: str,
        text_contexts: List[str],
        image_contexts: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate RAG response using retrieved text and images.
        
        This is the CORE of the multimodal RAG system.
        
        Args:
            query: User question
            text_contexts: Retrieved text chunks
            image_contexts: Retrieved images with descriptions
            conversation_history: Previous messages
            
        Returns:
            Generated response with citations
        """
        try:
            # Build context prompt
            context_parts = []
            
            # Add text context
            if text_contexts:
                context_parts.append("=== RETRIEVED TEXT CONTEXT ===")
                for idx, text in enumerate(text_contexts, 1):
                    context_parts.append(f"\n[Text Source {idx}]\n{text}\n")
            
            # Add image context
            if image_contexts:
                context_parts.append("\n=== RETRIEVED IMAGE CONTEXT ===")
                for idx, img_ctx in enumerate(image_contexts, 1):
                    desc = img_ctx.get('description', '')
                    source = img_ctx.get('source', '')
                    page = img_ctx.get('page', '')
                    context_parts.append(
                        f"\n[Image Source {idx}] (from {source}, page {page})\n{desc}\n"
                    )
            
            context_text = "\n".join(context_parts)
            
            # Build full prompt
            system_prompt = """You are a construction document AI assistant. 
Answer questions based ONLY on the provided context from construction documents.

Guidelines:
1. Provide accurate, specific answers from the context, don't answer extra, only answer accordingly to the chunks or documnwets you get.
2. Always cite sources (mention document name and page)
3. If information is in an image, mention "as shown in the drawing/schedule"
4. If you cannot find the answer in the context, say so clearly
5. For technical questions, be precise with measurements, specifications, and codes
6. When referring to visual elements, describe them clearly

Be helpful and professional."""
            
            # Build message
            prompt = f"""{system_prompt}

{context_text}

USER QUESTION: {query}

ASSISTANT RESPONSE:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in chat_with_context: {e}")
            raise
    
    async def extract_structured_data(
        self,
        extraction_type: str,
        contexts: List[str],
        image_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract structured data (door schedule, room schedule, etc.) using Gemini.
        
        Args:
            extraction_type: Type of extraction (door_schedule, room_schedule, etc.)
            contexts: Text contexts
            image_contexts: Image contexts with descriptions
            
        Returns:
            Structured JSON data
        """
        try:
            # Schema definitions
            schema_prompts = {
                "door_schedule": """Extract all door information into a JSON array with this schema:
{
  "entries": [
    {
      "mark": "door identifier/number",
      "location": "room or area",
      "width_mm": integer,
      "height_mm": integer,
      "fire_rating": "rating (e.g., '1 HR', '90 MIN')",
      "material": "door material",
      "hardware": "hardware specifications",
      "notes": "additional notes"
    }
  ]
}""",
                "room_schedule": """Extract all room information into a JSON array with this schema:
{
  "entries": [
    {
      "room_number": "room identifier",
      "room_name": "room name/function",
      "area_sqm": float,
      "floor_finish": "floor material/finish",
      "wall_finish": "wall material/finish",
      "ceiling_finish": "ceiling material/finish",
      "notes": "additional specifications"
    }
  ]
}"""
            }
            
            schema_prompt = schema_prompts.get(
                extraction_type,
                "Extract relevant information as JSON"
            )
            
            # Build extraction prompt
            context_text = "\n\n".join([
                "=== TEXT SOURCES ===",
                *[f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)],
                "\n=== IMAGE DESCRIPTIONS ===",
                *[f"[Image {i+1}]\n{img['description']}" for i, img in enumerate(image_contexts)]
            ])
            
            prompt = f"""Extract structured data from the following construction documents.

{schema_prompt}

CONTEXT:
{context_text}

Return ONLY valid JSON. Be thorough and extract all available information.
If a field is not available, use null.

JSON OUTPUT:"""
            
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Extract JSON from response (handle markdown code blocks)
            import json
            import re
            
            # Try to find JSON in code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = result_text
            
            # Parse JSON
            structured_data = json.loads(json_text)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error in structured extraction: {e}")
            raise


# Create singleton instance
gemini_service = GeminiService()

"""
Chat routes for RAG-based Q&A.

Endpoints:
- POST /query - Ask a question
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from loguru import logger
import uuid

from app.services.retrieval import rag_service
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from datetime import datetime

router = APIRouter()

# Simple in-memory conversation storage
# In production, use Redis or database
conversations: Dict[str, list] = {}


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Ask a question about construction documents using RAG.
    
    This is the MAIN chat endpoint that powers the Q&A interface.
    
    Flow:
    1. Retrieve relevant context (text + images)
    2. Generate response with Gemini
    3. Return answer with citations
    
    Args:
        request: ChatRequest with message and options
        
    Returns:
        ChatResponse with answer and citations
    """
    logger.info(f"Chat query: {request.message}")
    
    try:
        # Get or create conversation ID
        conv_id = request.conversation_id or str(uuid.uuid4())
        
        # Get conversation history
        history = conversations.get(conv_id, [])
        
        # Perform RAG
        response_text, citations = await rag_service.chat(
            query=request.message,
            conversation_history=history,
            include_images=request.include_images
        )
        
        # Create response message
        response_message = ChatMessage(
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
            citations=citations[:request.max_sources]
        )
        
        # Update conversation history
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": response_text})
        conversations[conv_id] = history
        
        # Build context debug info
        context_info = {
            "num_citations": len(citations),
            "conversation_length": len(history),
            "included_images": request.include_images
        }
        
        return ChatResponse(
            conversation_id=conv_id,
            message=response_message,
            citations=citations[:request.max_sources],
            context_used=context_info
        )
        
    except Exception as e:
        logger.error(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get conversation history.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id]
    }


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete conversation history.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Success message
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

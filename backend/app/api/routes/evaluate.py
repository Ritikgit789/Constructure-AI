"""
Evaluation routes for testing RAG quality.

Endpoints:
- POST /run - Run evaluation suite
- GET /results - Get latest results
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from loguru import logger
import json
import os
from datetime import datetime

from app.services.retrieval import rag_service
from app.models.schemas import TestQuery, EvaluationResult, EvaluationReport

router = APIRouter()

EVAL_QUERIES_PATH = "backend/evaluation/test_queries.json"
EVAL_RESULTS_PATH = "backend/evaluation/latest_results.json"


@router.post("/run", response_model=EvaluationReport)
async def run_evaluation():
    """
    Run evaluation suite on test queries.
    
    Tests:
    - Factual Q&A accuracy
    - Citation correctness
    - Extraction quality
    
    Returns:
        EvaluationReport with results
    """
    logger.info("Running evaluation suite")
    
    try:
        # Load test queries
        if not os.path.exists(EVAL_QUERIES_PATH):
            raise HTTPException(
                status_code=404,
                detail="Test queries file not found. Please create evaluation/test_queries.json"
            )
        
        with open(EVAL_QUERIES_PATH, "r") as f:
            queries_data = json.load(f)
        
        test_queries = [TestQuery(**q) for q in queries_data]
        
        # Run each query
        results = []
        correct_count = 0
        partial_count = 0
        incorrect_count = 0
        
        for test_query in test_queries:
            logger.info(f"Evaluating query: {test_query.query_id}")
            
            # Execute query
            response, citations = await rag_service.chat(
                query=test_query.query,
                include_images=True
            )
            
            # Evaluate response
            status, score, notes = _evaluate_response(
                response=response,
                citations=citations,
                expected_keywords=test_query.expected_answer_keywords,
                expected_sources=test_query.expected_sources
            )
            
            # Count status
            if status == "correct":
                correct_count += 1
            elif status == "partial":
                partial_count += 1
            else:
                incorrect_count += 1
            
            # Create result
            result = EvaluationResult(
                query_id=test_query.query_id,
                query=test_query.query,
                response=response,
                citations=citations,
                status=status,
                score=score,
                notes=notes
            )
            results.append(result)
        
        # Create report
        accuracy = correct_count / len(test_queries) if test_queries else 0.0
        
        report = EvaluationReport(
            total_queries=len(test_queries),
            correct_count=correct_count,
            partial_count=partial_count,
            incorrect_count=incorrect_count,
            accuracy=accuracy,
            results=results
        )
        
        # Save results
        os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
        with open(EVAL_RESULTS_PATH, "w") as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Evaluation complete: {accuracy:.1%} accuracy")
        
        return report
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")


def _evaluate_response(
    response: str,
    citations: List[Any],
    expected_keywords: List[str],
    expected_sources: List[str]
) -> tuple:
    """
    Evaluate a single response.
    
    Returns:
        Tuple of (status, score, notes)
    """
    response_lower = response.lower()
    
    # Check for expected keywords
    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    keyword_ratio = keywords_found / len(expected_keywords) if expected_keywords else 1.0
    
    # Check for expected sources
    citation_sources = [c.source.lower() for c in citations]
    sources_found = sum(1 for src in expected_sources if any(src.lower() in cs for cs in citation_sources))
    source_ratio = sources_found / len(expected_sources) if expected_sources else 1.0
    
    # Calculate overall score
    score = (keyword_ratio * 0.7) + (source_ratio * 0.3)
    
    # Determine status
    if score >= 0.8:
        status = "correct"
        notes = f"Strong match: {keywords_found}/{len(expected_keywords)} keywords, {sources_found}/{len(expected_sources)} sources"
    elif score >= 0.5:
        status = "partial"
        notes = f"Partial match: {keywords_found}/{len(expected_keywords)} keywords, {sources_found}/{len(expected_sources)} sources"
    else:
        status = "incorrect"
        notes = f"Weak match: {keywords_found}/{len(expected_keywords)} keywords, {sources_found}/{len(expected_sources)} sources"
    
    return status, score, notes


@router.get("/results")
async def get_latest_results():
    """
    Get latest evaluation results.
    
    Returns:
        Latest evaluation report
    """
    if not os.path.exists(EVAL_RESULTS_PATH):
        raise HTTPException(status_code=404, detail="No evaluation results found")
    
    with open(EVAL_RESULTS_PATH, "r") as f:
        results = json.load(f)
    
    return results

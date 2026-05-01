import asyncio
import json
import os
from api.routers.query import process_query
from api.routers.query import QueryRequest

async def run_lite_eval():
    print("Starting DeepDoc Lite Evaluation (Python 3.15 Optimized)")
    print("-" * 50)
    
    test_queries = [
        "What are the core components of the system?",
        "What is the SLA uptime?",
        "Tell me about the penalty clause.",
        "How is the document ingested?",
        "What models are used?"
    ]
    
    scores = []
    
    for i, q in enumerate(test_queries):
        print(f"[{i+1}/5] Querying: {q}")
        req = QueryRequest(query_string=q, user_id="eval_user")
        
        try:
            # We mock the agent call slightly to avoid heavy LLM weights if needed
            # but here we run the actual router logic
            result = await process_query(req)
            
            # Heuristics for scoring
            has_answer = len(result.answer_text) > 20
            has_citation = len(result.citations) > 0
            
            score = 0
            if has_answer: score += 50
            if has_citation: score += 50
            
            scores.append(score)
            print(f"    [OK] Success | Score: {score}% | Citation: {result.citations[0].doc_name if has_citation else 'None'}")
            
        except Exception as e:
            print(f"    [FAIL] Failed: {str(e)}")
            scores.append(0)

    avg_score = sum(scores) / len(scores)
    print("-" * 50)
    print(f"FINAL SYSTEM SCORE: {avg_score}%")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_lite_eval())

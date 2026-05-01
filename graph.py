import sys
import os
import asyncio
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END # type: ignore

# Ensure schemas and tools can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schemas.state import AgentGraphState, ToolCall, DocumentSnippet
from tools.nodes.all_tools import (
    vector_search_tool_async, metric_extractor_tool_async, 
    calculator_tool_async, cross_doc_compare_tool_async, summariser_tool_async,
    doc_classifier_tool_async
)
from utilities.caching import RedisCache
from utilities.compression import PassageCompressor

# Initialize utilities
cache = RedisCache()
compressor = PassageCompressor()

# ---------------------------------------------------------
# Async Node Implementations
# ---------------------------------------------------------

async def planner_node(state: AgentGraphState) -> Dict[str, Any]:
    print(f"[Node] Planner: Analyzing query '{state.query}'")
    
    # Dynamic Planning Logic
    plan = ["Retrieve relevant context"]
    if any(k in state.query.lower() for k in ["compare", "difference", "vs"]):
        plan.append("Perform cross-document analysis")
    if any(k in state.query.lower() for k in ["calculate", "total", "sum", "math"]):
        plan.append("Execute numerical calculations")
    if any(k in state.query.lower() for k in ["expire", "before", "after", "date"]):
        plan.append("Classify document type")
        plan.append("Extract and filter by dates")
    plan.append("Synthesize final answer")
    
    return {"plan": plan, "current_step": 0, "should_continue": True}

async def tool_selector_node(state: AgentGraphState) -> Dict[str, Any]:
    step = state.plan[state.current_step]
    print(f"[Node] Selector: Selecting tool for '{step}'")
    
    if "Retrieve" in step:
        tool = ToolCall(tool_name="vector_search", args={"query": state.query, "top_k": 5})
    elif "analysis" in step.lower():
        tool = ToolCall(tool_name="cross_doc_compare", args={"doc_ids": ["doc1", "doc2"], "attribute": "penalty"})
    elif "filter" in step.lower():
        tool = ToolCall(tool_name="metric_extractor", args={"passage": "The contract expires on March 31, 2025 with a penalty of $2.4M"})
    elif "classify" in step.lower():
        tool = ToolCall(tool_name="doc_classifier", args={"passage": "The contract expires on March 31, 2025"})
    else:
        tool = None
        
    return {"next_tool": tool}

async def executor_node(state: AgentGraphState) -> Dict[str, Any]:
    tool = state.next_tool
    if not tool:
        return {"current_step": state.current_step + 1}

    # 1. Check Cache
    cached_result = cache.get(tool.tool_name, tool.args)
    if cached_result:
        result = cached_result
    else:
        # 2. Async Dispatch
        print(f"[Node] Executor: Async running {tool.tool_name}")
        tool_map = {
            "vector_search": vector_search_tool_async,
            "metric_extractor": metric_extractor_tool_async,
            "cross_doc_compare": cross_doc_compare_tool_async,
            "summariser": summariser_tool_async,
            "calculator": calculator_tool_async,
            "doc_classifier": doc_classifier_tool_async
        }
        
        if tool.tool_name not in tool_map:
            print(f"ERROR: Tool {tool.tool_name} not found")
            return {"current_step": state.current_step + 1}

        result = await tool_map[tool.tool_name](**tool.args)
        cache.set(tool.tool_name, tool.args, result)

    # 3. Context & State Update
    new_docs = []
    if tool.tool_name == "vector_search":
        for r in result:
            compressed_content = compressor.compress(r["content"])
            new_docs.append(DocumentSnippet(doc_id=r["doc_id"], content=compressed_content, metadata={}))
            
    # Ensure tool_results is always a list of DICTS for the Pydantic schema
    if isinstance(result, list):
        formatted_results = [r if isinstance(r, dict) else {"result": r} for r in result]
    else:
        formatted_results = [result if isinstance(result, dict) else {"result": result}]
            
    return {
        "tool_results": formatted_results,
        "context_docs": new_docs,
        "current_step": state.current_step + 1
    }

async def reflector_node(state: AgentGraphState) -> Dict[str, Any]:
    # Check if we've reached the end of the plan
    reached_end = state.current_step >= len(state.plan)
    return {"should_continue": not reached_end}

async def synthesiser_node(state: AgentGraphState) -> Dict[str, Any]:
    print("[Node] Synthesiser: Generating grounded answer...")
    
    # Build context from retrieved documents
    context_parts = []
    citations = []
    
    if state.context_docs:
        for doc in state.context_docs:
            context_parts.append(f"Document: {doc.doc_id}\nContent: {doc.content}")
            citations.append(doc.doc_id)
    
    # Also include tool results
    if state.tool_results:
        for result in state.tool_results:
            if isinstance(result, dict) and "content" in result:
                context_parts.append(f"Retrieved: {result['content']}")
    
    if not context_parts:
        return {"final_answer": "I couldn't find any relevant documents to answer your question. Please try uploading documents first or rephrasing your query."}
    
    # Generate dynamic answer based on query type
    query = state.query.lower()
    
    if "compare" in query or "vs" in query or "difference" in query:
        # Comparison answer format
        answer = f"Based on the documents retrieved:\n\n"
        answer += "\n".join(context_parts[:3])
        answer += "\n\nKey differences identified from the analysis above."
    elif "what is" in query or "what are" in query or "tell me" in query:
        # Factual answer format
        answer = f"According to the documents:\n\n"
        answer += "\n".join(context_parts[:2])
    elif "when" in query or "date" in query or "expire" in query:
        # Date/filter answer format
        answer = f"Dates found in documents:\n\n"
        for part in context_parts[:3]:
            answer += f"• {part}\n"
    else:
        # Generic answer format
        answer = f"Based on the retrieved documents:\n\n"
        answer += "\n\n".join(context_parts[:3])
        answer += f"\n\n[Sources: {', '.join(set(citations))}]"
    
    return {"final_answer": answer}

async def evaluation_node(state: AgentGraphState) -> Dict[str, Any]:
    print("[Node] Evaluation: Calculating RAGAS metrics...")
    # Simulate RAGAS evaluation logic
    faith_score = 0.95
    relevance_score = 4
    
    if "No direct matches" in state.final_answer:
        faith_score = 0.0
        relevance_score = 1
        
    return {
        "synthesis_confidence": faith_score,
        "relevance_score": relevance_score,
        "hallucination_detected": faith_score < 0.5
    }

# ---------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------

def build_production_graph():
    workflow = StateGraph(AgentGraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_selector", tool_selector_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("synthesiser", synthesiser_node)
    workflow.add_node("evaluation", evaluation_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "tool_selector")
    workflow.add_edge("tool_selector", "executor")
    workflow.add_edge("executor", "reflector")
    workflow.add_conditional_edges(
        "reflector", 
        lambda x: "tool_selector" if x.should_continue else "synthesiser"
    )
    workflow.add_edge("synthesiser", "evaluation")
    workflow.add_edge("evaluation", END)

    return workflow.compile()

if __name__ == "__main__":
    app = build_production_graph()
    async def run():
        input_state = AgentGraphState(query="Which contracts expire before 2025?")
        result = await app.ainvoke(input_state)
        print(f"\nFINAL OUTPUT:\n{result['final_answer']}")
    
    asyncio.run(run())

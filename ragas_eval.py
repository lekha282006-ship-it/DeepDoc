import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
import mlflow
from typing import List, Dict

# ---------------------------------------------------------
# RAGAS Thresholds (Step 4.2)
# ---------------------------------------------------------
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevance": 0.78,
    "context_precision": 0.74,
    "context_recall": 0.76
}

def run_ragas_evaluation(samples: List[Dict]):
    """
    Step 4.2: RAGAS Automated Metrics runner.
    Takes a batch of (question, answer, contexts, ground_truth).
    """
    print(f"--- Running RAGAS Evaluation on {len(samples)} items ---")
    
    # Format dataset for RAGAS
    dataset = Dataset.from_list(samples)
    
    # Execute evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        ],
    )
    
    # Log results to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DeepDoc_RAGAS_Eval")
    
    with mlflow.start_run():
        mlflow.log_metrics(result)
        print("RAGAS Results logged to MLflow.")
        
        # Check thresholds
        breached = []
        for metric, score in result.items():
            if metric in THRESHOLDS and score < THRESHOLDS[metric]:
                breached.append(f"{metric}: {score:.2f} (Threshold: {THRESHOLDS[metric]})")
        
        if breached:
            print("\n!!! THRESHOLD BREACH DETECTED !!!")
            for b in breached:
                print(b)
            return False, result
            
    return True, result

if __name__ == "__main__":
    # Example loading from golden dataset
    sample_file = "e:\\DeepDoc\\deepdoc-intelligence\\tools\\golden_dataset\\golden_500.json"
    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            data = json.load(f)[:100] # Sample 100 as per requirement
            
        # Transform keys for RAGAS (expects: question, answer, contexts, ground_truth)
        ragas_data = []
        for item in data:
            ragas_data.append({
                "question": item["query"],
                "answer": "Mock answer", 
                "contexts": item["supporting_passages"],
                "ground_truth": item["expected_answer"]
            })
            
        success, metrics = run_ragas_evaluation(ragas_data)
        if not success:
            exit(1) # Fail CI/CD

# DeepDoc Intelligence - Workflow & Contribution Guide

This document outlines the day-to-day engineering workflows for maintaining and scaling the DeepDoc Intelligence system.

## 🔄 Daily Development Cycle

Follow this cycle for every feature or bug fix:

1.  **Start**: `git pull origin main`. Check the **MLflow Dashboard** (http://localhost:5000) for any overnight regression alerts from nightly evaluation runs.
2.  **Experiment**: Create a feature branch. Log all experimental results (hyperparameters, metrics) to MLflow with descriptive run names (e.g., `experiment-reranker-top20`).
3.  **Notebook-first**: Prototype new logic in `notebooks/`. Once a component is stable, port it to a Python module (e.g., `retrieval/`) and write a unit test in `tests/`.
4.  **PR**: Open a Pull Request. You MUST include a link to the **MLflow Comparison** between your branch and `main`. The GitHub Action `RAGAS Evaluation` must pass on the 100-item sample.
5.  **Review**: Self-review your changes. Focus on:
    -   Did this change maintain or improve RAGAS/Judge scores?
    -   Is there adequate test coverage?
6.  **Merge**: Squash merge into `main`. The full 500-item golden evaluation suite will run automatically upon merge.

## 🏷️ Data Annotation Workflow

### 1. NER Annotation (Label Studio)
-   Annotate passages using **Label Studio**.
-   Export data in **CoNLL** or **JSONL** format.
-   Run `scripts/annotation_validator.py` to verify:
    -   IOB2 consistency (no orphaned `I-` tags).
    -   Entity span overlaps.
    -   Balanced class distribution.

### 2. Dataset Versioning (W&B)
-   Version every dataset export as a **W&B Artifact**.
-   Every model training run (Layer 1) MUST reference the specific W&B artifact URI to ensure reproducibility and data lineage.

### 3. Golden Dataset Maintenance
-   Maintain the "Golden 500" items in the master **Google Sheet**.
-   Each item must have: `Question`, `Answer`, `Citations`, `Complexity_Label`, and `Reviewer_Status`.
-   Periodically sync this sheet to `tools/golden_dataset/golden_500.json`.

## 🏋️ Model Training Workflow

Follow this standard process for fine-tuning any model in Layer 1 or Layer 4:

1.  **Config**: Create a YAML file in `configs/` (e.g., `ner_v1.yaml`) defining `model_name`, `base_model`, `hyperparams`, and `mlflow_registry`.
2.  **Run**: Execute the training controller:
    ```bash
    python models/train_controller.py --config configs/ner_v1.yaml
    ```
3.  **Monitor**: Open the **W&B Dashboard** to watch training curves. Specifically, watch for **overfitting**: if validation F1 plateaus while training loss continues to fall, trigger early stopping.
4.  **Registry**: Successful runs with metrics exceeding the baseline are automatically registered in the **MLflow Model Registry**.
5.  **Eval**: Run evaluation against the held-out test set:
    ```bash
    python models/evaluate_registry.py --model registry:/ner_models/latest
    ```
6.  **Champion/Challenger**: Compare the evaluation report to the current `champion`. If the new model wins, update the `eval_report.md` and promote the model tag to `champion` in MLflow.

## 🕵️ Root-Cause Analysis (RCA) Protocol

If an evaluation run shows a drop in any metric:

1.  **Run Breakdown**: Use the breakdown script to find the worst cases:
    ```bash
    python scripts/eval_breakdown.py --metric faithfulness --top_failures 20
    ```
2.  **Inspect & Categorize**: Open the logs for the top-20 failures. Use the following decision tree:
    -   *Wrong passages retrieved?* -> **Retrieval Failure**. Fix: Update ontology, Bi-encoder, or BM25 index.
    -   *Answer not in context but model guessed?* -> **Hallucination**. Fix: Adjust SELF-RAG thresholds or prompt reflection logic.
    -   *Query misinterpreted?* -> **Parser Error**. Fix: Update Intent/Slot parser or training data.
3.  **Targeted Fix**: Apply the fix to the specific component. **Do not retrain the full model if it's a retrieval bug.**
4.  **Verify**: Rerun the evaluation suite.
5.  **Log**: Document the findings and fix in `eval_log.md`.

## 📚 Research Integration Workflow

DeepDoc Intelligence is designed to stay current with the latest NLP/IR research:

1.  **Read & Identify**: Identify the core improvement over our current baseline (e.g., *SELF-RAG: model knows when to retrieve vs. rely on parametric knowledge*).
2.  **Verify Failure Mode**: Check if this failure mode currently exists in our system by running the evaluation breakdown script.
3.  **Prototype (Notebook)**: Implement a minimal viable version in a Jupyter notebook. Measure its impact on a **50-item stratified sample** from the golden dataset.
4.  **Promote (PR)**: If the impact is **> 3% improvement** on the target metric:
    -   Implement the technique properly in the codebase.
    -   Write comprehensive unit and integration tests.
    -   Open a PR including the **before/after evaluation diff**.
5.  **Document**: Record the paper, addressed failure mode, implementation approach, and measured impact in `research_log.md`.

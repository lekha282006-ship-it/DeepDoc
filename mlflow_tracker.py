"""
MLflow Experiment Tracking
Comprehensive experiment tracking for model training and evaluation
"""
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Any, Optional
import json
import torch
from pathlib import Path


class ExperimentTracker:
    """
    MLflow experiment tracking for all model training and evaluation
    Logs metrics, parameters, artifacts, and model checkpoints
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "DeepDoc_Intelligence"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def log_training_run(
        self,
        run_name: str,
        model_type: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
        artifacts: Optional[Dict[str, str]] = None
    ):
        """
        Log a complete training run to MLflow
        """
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(config)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model type tag
            mlflow.set_tag("model_type", model_type)
            
            # Log model checkpoint
            if model is not None:
                mlflow.pytorch.log_model(model, "model")
            
            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
            
            print(f"Run '{run_name}' logged to MLflow")
    
    def log_evaluation_run(
        self,
        run_name: str,
        model_version: str,
        metrics: Dict[str, float],
        dataset_version: str = "v1.0"
    ):
        """
        Log evaluation run to MLflow
        """
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_version": model_version,
                "dataset_version": dataset_version
            })
            
            mlflow.log_metrics(metrics)
            
            mlflow.set_tag("run_type", "evaluation")
            
            print(f"Evaluation run '{run_name}' logged to MLflow")
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        stage: str = "Staging"
    ):
        """
        Register model in MLflow model registry
        """
        model_uri = f"runs:/{run_id}/model"
        
        mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Transition to specified stage
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        latest_version = client.get_latest_versions(model_name)[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=stage
        )
        
        print(f"Model '{model_name}' v{latest_version} registered and staged as '{stage}'")
    
    def compare_runs(
        self,
        run_ids: list,
        metric_name: str = "macro_f1"
    ) -> Dict[str, float]:
        """
        Compare multiple runs by a specific metric
        """
        results = {}
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            run_name = run.data.tags.get("mlflow.runName", run_id)
            metric_value = run.data.metrics.get(metric_name, 0.0)
            results[run_name] = metric_value
        
        return results
    
    def get_best_run(
        self,
        experiment_name: Optional[str] = None,
        metric_name: str = "macro_f1",
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run by a specific metric
        """
        exp_name = experiment_name or self.experiment_name
        
        runs = mlflow.search_runs(
            experiment_names=[exp_name],
            order_by=[f"metrics.{metric_name} {'DESC' if not ascending else 'ASC'}"],
            max_results=1
        )
        
        if runs:
            return {
                "run_id": runs.iloc[0]["run_id"],
                "run_name": runs.iloc[0]["tags"]["mlflow.runName"],
                "metric_value": runs.iloc[0]["metrics"][metric_name]
            }
        
        return None


class ModelCardGenerator:
    """
    Generate model cards for trained models
    """
    
    def __init__(self):
        pass
    
    def generate_model_card(
        self,
        model_name: str,
        model_type: str,
        training_data: str,
        eval_results: Dict[str, float],
        limitations: list,
        intended_use: str
    ) -> str:
        """
        Generate a model card in markdown format
        """
        card = f"""# Model Card: {model_name}

## Model Type
{model_type}

## Training Data
{training_data}

## Evaluation Results
"""
        
        for metric, value in eval_results.items():
            card += f"- {metric}: {value:.4f}\n"
        
        card += f"""
## Intended Use
{intended_use}

## Limitations
"""
        
        for limitation in limitations:
            card += f"- {limitation}\n"
        
        card += f"""
## Training Details
- Framework: PyTorch
- Training Date: {self._get_current_date()}
- Model Version: v1.0
"""
        
        return card
    
    def _get_current_date(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def save_model_card(self, model_card: str, path: str):
        """Save model card to file"""
        with open(path, 'w') as f:
            f.write(model_card)
        print(f"Model card saved to {path}")


if __name__ == "__main__":
    tracker = ExperimentTracker()
    
    # Test logging a training run
    config = {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 5
    }
    
    metrics = {
        "macro_f1": 0.92,
        "weighted_f1": 0.94,
        "accuracy": 0.93
    }
    
    tracker.log_training_run(
        run_name="classifier_v1_test",
        model_type="document_classifier",
        config=config,
        metrics=metrics
    )
    
    # Test model card generation
    card_generator = ModelCardGenerator()
    model_card = card_generator.generate_model_card(
        model_name="Document Classifier v1",
        model_type="Document Type Classification",
        training_data="800 labeled documents across 5 classes",
        eval_results=metrics,
        limitations=[
            "May not perform well on documents from unseen domains",
            "Requires pre-processing for very long documents"
        ],
        intended_use="Routing incoming documents to appropriate extraction pipelines"
    )
    
    print(f"\nModel Card:\n{model_card}")

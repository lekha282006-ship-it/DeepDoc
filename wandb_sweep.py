"""
Hyperparameter Sweep Integration
Bayesian optimization using Weights & Biases for model training
"""
import wandb
from typing import Dict, Any, List, Optional
import yaml


class SweepRunner:
    """
    Weights & Biases sweep runner for hyperparameter optimization
    Uses Bayesian search for efficient hyperparameter tuning
    """
    
    def __init__(self, project_name: str = "DeepDoc-Intelligence"):
        self.project_name = project_name
        wandb.init(project=project_name)
    
    def create_sweep_config(
        self,
        model_type: str,
        learning_rate_range: List[float] = [1e-5, 5e-5],
        batch_size_options: List[int] = [16, 32, 64],
        warmup_ratio_range: List[float] = [0.06, 0.15]
    ) -> Dict[str, Any]:
        """
        Create sweep configuration for Bayesian optimization
        """
        sweep_config = {
            "method": "bayes",  # Bayesian optimization
            "metric": {
                "name": "macro_f1",
                "goal": "maximize"
            },
            "parameters": {
                "learning_rate": {
                    "min": learning_rate_range[0],
                    "max": learning_rate_range[1]
                },
                "batch_size": {
                    "values": batch_size_options
                },
                "warmup_ratio": {
                    "min": warmup_ratio_range[0],
                    "max": warmup_ratio_range[1]
                },
                "weight_decay": {
                    "min": 0.0,
                    "max": 0.1
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 3,
                "eta": 2
            }
        }
        
        return sweep_config
    
    def launch_sweep(
        self,
        sweep_config: Dict[str, Any],
        sweep_name: str,
        count: int = 20,
        train_function: str = "models/train.py"
    ):
        """
        Launch sweep with specified configuration
        """
        # Create sweep ID
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        
        # Run sweep agent
        wandb.agent(
            sweep_id,
            function=train_function,
            count=count,
            project=self.project_name
        )
        
        print(f"Sweep '{sweep_name}' completed with {count} trials")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters to wandb"""
        wandb.config.update(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb"""
        wandb.log(metrics, step=step)
    
    def log_artifacts(self, artifact_paths: List[str]):
        """Log artifacts to wandb"""
        for path in artifact_paths:
            wandb.save(path)


class HyperparameterOptimizer:
    """
    Standalone hyperparameter optimization without W&B
    Uses grid search or random search for smaller experiments
    """
    
    def __init__(self):
        pass
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        train_function,
        metric_name: str = "macro_f1"
    ) -> Dict[str, Any]:
        """
        Grid search over parameter combinations
        """
        from itertools import product
        
        best_config = None
        best_score = 0.0
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = product(*values)
        
        total_combinations = len(list(product(*values)))
        print(f"Running grid search with {total_combinations} combinations")
        
        for i, combination in enumerate(combinations):
            config = dict(zip(keys, combination))
            
            print(f"\nTrial {i+1}/{total_combinations}: {config}")
            
            # Train with current config
            result = train_function(config)
            score = result.get(metric_name, 0.0)
            
            if score > best_score:
                best_score = score
                best_config = config
                print(f"New best score: {score:.4f}")
        
        print(f"\nGrid search complete. Best config: {best_config}, Best score: {best_score:.4f}")
        
        return {
            "best_config": best_config,
            "best_score": best_score
        }
    
    def random_search(
        self,
        param_ranges: Dict[str, tuple],
        n_trials: int = 20,
        train_function=None,
        metric_name: str = "macro_f1"
    ) -> Dict[str, Any]:
        """
        Random search over parameter ranges
        """
        import random
        
        best_config = None
        best_score = 0.0
        
        print(f"Running random search with {n_trials} trials")
        
        for i in range(n_trials):
            config = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param] = random.randint(min_val, max_val)
                else:
                    config[param] = random.uniform(min_val, max_val)
            
            print(f"\nTrial {i+1}/{n_trials}: {config}")
            
            # Train with current config
            result = train_function(config)
            score = result.get(metric_name, 0.0)
            
            if score > best_score:
                best_score = score
                best_config = config
                print(f"New best score: {score:.4f}")
        
        print(f"\nRandom search complete. Best config: {best_config}, Best score: {best_score:.4f}")
        
        return {
            "best_config": best_config,
            "best_score": best_score
        }


def mock_train_function(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Mock training function for testing
    Returns simulated metrics based on config
    """
    # Simulate training - in production, this would be actual training
    lr = config.get("learning_rate", 2e-5)
    batch_size = config.get("batch_size", 32)
    
    # Simulate that lower LR and larger batch size give better results
    simulated_f1 = 0.85 + (1e-5 / lr) * 0.05 + (batch_size / 64) * 0.03
    
    return {
        "macro_f1": min(simulated_f1, 0.95),  # Cap at 0.95
        "weighted_f1": simulated_f1 + 0.02,
        "accuracy": simulated_f1 + 0.01
    }


if __name__ == "__main__":
    # Test W&B sweep
    sweep_runner = SweepRunner()
    
    sweep_config = sweep_runner.create_sweep_config(
        model_type="classifier",
        learning_rate_range=[1e-5, 5e-5],
        batch_size_options=[16, 32, 64],
        warmup_ratio_range=[0.06, 0.15]
    )
    
    print("Sweep configuration created:")
    print(yaml.dump(sweep_config))
    
    # Test random search (without W&B)
    optimizer = HyperparameterOptimizer()
    
    param_ranges = {
        "learning_rate": (1e-5, 5e-5),
        "batch_size": (16, 64),
        "warmup_ratio": (0.06, 0.15)
    }
    
    result = optimizer.random_search(
        param_ranges=param_ranges,
        n_trials=5,
        train_function=mock_train_function
    )
    
    print(f"\nRandom search result: {result}")

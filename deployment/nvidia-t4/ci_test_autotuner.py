"""
CI/CD Pipeline Test Auto-Tuner for T4 GPU

This module provides automated parameter tuning for CI/CD pipeline tests,
dynamically adjusting test parameters based on model characteristics and
previous test results to optimize for accurate and representative testing.
"""

import os
import json
import logging
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestParameterHistory:
    """Class to track test parameter history and performance."""
    
    def __init__(self, param_name: str, window_size: int = 50):
        """
        Initialize test parameter history.
        
        Args:
            param_name: Name of the parameter
            window_size: Maximum history entries to keep
        """
        self.param_name = param_name
        self.window_size = window_size
        self.values = []
        self.test_results = []
        self.timestamps = []
        self.metadata = {}
    
    def add_result(self, 
                  value: Any, 
                  test_result: Dict[str, Any],
                  metadata: Dict[str, Any] = None):
        """
        Add test result with parameter value.
        
        Args:
            value: Parameter value used
            test_result: Test result metrics
            metadata: Additional test metadata
        """
        self.values.append(value)
        self.test_results.append(test_result)
        self.timestamps.append(datetime.now())
        
        if metadata:
            self.metadata[len(self.values) - 1] = metadata
        
        # Trim if exceeds window size
        if len(self.values) > self.window_size:
            excess = len(self.values) - self.window_size
            self.values = self.values[excess:]
            self.test_results = self.test_results[excess:]
            self.timestamps = self.timestamps[excess:]
            
            # Update metadata indices
            new_metadata = {}
            for idx, meta in self.metadata.items():
                new_idx = idx - excess
                if new_idx >= 0:
                    new_metadata[new_idx] = meta
            self.metadata = new_metadata
    
    def get_best_value(self, metric_name: str, higher_is_better: bool = True) -> Any:
        """
        Get parameter value with best test result for a metric.
        
        Args:
            metric_name: Name of the metric to optimize for
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Parameter value with best test result
        """
        if not self.test_results:
            return None
        
        best_idx = -1
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for i, result in enumerate(self.test_results):
            if metric_name in result:
                metric_value = result[metric_name]
                
                if higher_is_better:
                    if metric_value > best_value:
                        best_value = metric_value
                        best_idx = i
                else:
                    if metric_value < best_value:
                        best_value = metric_value
                        best_idx = i
        
        if best_idx >= 0:
            return self.values[best_idx]
        
        return None
    
    def get_recent_values(self, days: int = 7) -> List[Any]:
        """
        Get parameter values from recent history.
        
        Args:
            days: Look back window in days
            
        Returns:
            List of recent parameter values
        """
        if not self.timestamps:
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_indices = [i for i, ts in enumerate(self.timestamps) if ts > cutoff_time]
        
        return [self.values[i] for i in recent_indices]
    
    def get_success_rate(self, days: int = 7) -> float:
        """
        Get success rate for parameter values.
        
        Args:
            days: Look back window in days
            
        Returns:
            Success rate (0.0-1.0)
        """
        recent_indices = []
        
        if days > 0:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_indices = [i for i, ts in enumerate(self.timestamps) if ts > cutoff_time]
        else:
            recent_indices = list(range(len(self.timestamps)))
        
        if not recent_indices:
            return 0.0
        
        success_count = 0
        for i in recent_indices:
            result = self.test_results[i]
            if result.get("success", False):
                success_count += 1
        
        return success_count / len(recent_indices)
    
    def get_value_distribution(self) -> Dict[Any, int]:
        """
        Get distribution of parameter values.
        
        Returns:
            Dictionary mapping parameter values to counts
        """
        distribution = {}
        
        for value in self.values:
            str_value = str(value)
            distribution[str_value] = distribution.get(str_value, 0) + 1
        
        return distribution


class TestAutoTuner:
    """Auto-tuner for CI/CD pipeline test parameters."""
    
    def __init__(self, 
                 config_file: str = "/tmp/ci_test_autotuner.json",
                 history_window: int = 50,
                 exploration_rate: float = 0.2):
        """
        Initialize test auto-tuner.
        
        Args:
            config_file: Path to save/load tuned parameters
            history_window: Number of history entries to keep
            exploration_rate: Rate of parameter exploration (0.0-1.0)
        """
        self.config_file = config_file
        self.history_window = history_window
        self.exploration_rate = exploration_rate
        
        # Parameter history
        self.param_history = {
            "batch_sizes": TestParameterHistory("batch_sizes", window_size=history_window),
            "sequence_lengths": TestParameterHistory("sequence_lengths", window_size=history_window),
            "num_iterations": TestParameterHistory("num_iterations", window_size=history_window),
            "optimization_level": TestParameterHistory("optimization_level", window_size=history_window),
            "timeout": TestParameterHistory("timeout", window_size=history_window)
        }
        
        # Current parameters
        self.current_params = self._load_parameters()
        
        # Parameter ranges and defaults
        self.param_ranges = {
            "batch_sizes": {
                "type": "list",
                "options": [[1], [1, 4], [1, 4, 16], [1, 8, 32], [1, 16, 64], [1, 32, 128]],
                "default": [1, 8, 32]
            },
            "sequence_lengths": {
                "type": "list",
                "options": [[128], [256], [512], [1024], [128, 512], [256, 1024], [128, 512, 1024]],
                "default": [512]
            },
            "num_iterations": {
                "type": "range",
                "min": 10,
                "max": 100,
                "default": 50
            },
            "optimization_level": {
                "type": "options",
                "options": [0, 1, 2, 3, 4, 5],
                "default": 3
            },
            "timeout": {
                "type": "range",
                "min": 30,
                "max": 300,
                "default": 120
            }
        }
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load parameters from saved config or use defaults."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded test parameters from {self.config_file}")
                return params
        except Exception as e:
            logger.warning(f"Error loading parameters: {str(e)}")
        
        # Default parameters
        return {
            "batch_sizes": [1, 8, 32],
            "sequence_lengths": [512],
            "num_iterations": 50,
            "optimization_level": 3,
            "timeout": 120
        }
    
    def _save_parameters(self):
        """Save current parameters to config file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.current_params, f, indent=2)
            logger.info(f"Saved test parameters to {self.config_file}")
        except Exception as e:
            logger.warning(f"Error saving parameters: {str(e)}")
    
    def _load_history(self):
        """Load parameter history from saved config."""
        history_file = self.config_file.replace(".json", "_history.json")
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                # Restore history
                for param_name, param_history in history_data.items():
                    if param_name in self.param_history:
                        self.param_history[param_name].values = param_history.get("values", [])
                        self.param_history[param_name].test_results = param_history.get("test_results", [])
                        self.param_history[param_name].timestamps = [
                            datetime.fromisoformat(ts) for ts in param_history.get("timestamps", [])
                        ]
                        self.param_history[param_name].metadata = {
                            int(k): v for k, v in param_history.get("metadata", {}).items()
                        }
                
                logger.info(f"Loaded parameter history from {history_file}")
        except Exception as e:
            logger.warning(f"Error loading history: {str(e)}")
    
    def _save_history(self):
        """Save parameter history to config file."""
        history_file = self.config_file.replace(".json", "_history.json")
        
        try:
            history_data = {}
            
            for param_name, history in self.param_history.items():
                history_data[param_name] = {
                    "values": history.values,
                    "test_results": history.test_results,
                    "timestamps": [ts.isoformat() for ts in history.timestamps],
                    "metadata": {str(k): v for k, v in history.metadata.items()}
                }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Saved parameter history to {history_file}")
        except Exception as e:
            logger.warning(f"Error saving history: {str(e)}")
    
    def select_parameters(self, model_name: str = None) -> Dict[str, Any]:
        """
        Select test parameters, potentially exploring new values.
        
        Args:
            model_name: Optional model name for model-specific tuning
            
        Returns:
            Dictionary with selected parameters
        """
        params = {}
        
        # Determine whether to explore or exploit for each parameter
        for param_name, param_range in self.param_ranges.items():
            explore = random.random() < self.exploration_rate
            
            if explore:
                # Exploration: try new parameter values
                params[param_name] = self._explore_parameter(param_name, param_range)
                logger.info(f"Exploring parameter {param_name}: {params[param_name]}")
            else:
                # Exploitation: use current parameter or best known value
                history = self.param_history.get(param_name)
                
                if history and history.values:
                    # Use best known value for throughput
                    best_value = history.get_best_value("throughput", higher_is_better=True)
                    
                    if best_value is not None:
                        params[param_name] = best_value
                        logger.info(f"Using best known value for {param_name}: {params[param_name]}")
                    else:
                        params[param_name] = self.current_params.get(param_name, param_range["default"])
                else:
                    params[param_name] = self.current_params.get(param_name, param_range["default"])
        
        return params
    
    def _explore_parameter(self, param_name: str, param_range: Dict[str, Any]) -> Any:
        """
        Explore new parameter value within range.
        
        Args:
            param_name: Parameter name
            param_range: Parameter range configuration
            
        Returns:
            New parameter value
        """
        if param_range["type"] == "list":
            # Select random option from list of lists
            return random.choice(param_range["options"])
        elif param_range["type"] == "range":
            # Select random value within range
            return random.randint(param_range["min"], param_range["max"])
        elif param_range["type"] == "options":
            # Select random option from list
            return random.choice(param_range["options"])
        else:
            # Unknown type, use default
            return param_range["default"]
    
    def record_test_result(self, 
                          params: Dict[str, Any], 
                          test_result: Dict[str, Any],
                          model_name: str = None):
        """
        Record test result with parameters.
        
        Args:
            params: Parameters used for test
            test_result: Test result metrics
            model_name: Optional model name
        """
        metadata = {"model_name": model_name} if model_name else {}
        
        # Record result for each parameter
        for param_name, param_value in params.items():
            if param_name in self.param_history:
                self.param_history[param_name].add_result(param_value, test_result, metadata)
        
        # Save history
        self._save_history()
        
        # Update current parameters if test was successful
        if test_result.get("success", False):
            self.current_params.update(params)
            self._save_parameters()
            
            logger.info(f"Updated current parameters based on successful test")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get report on parameter optimization.
        
        Returns:
            Dictionary with optimization report
        """
        report = {
            "current_parameters": self.current_params,
            "parameter_history": {},
            "success_rates": {},
            "recommendations": []
        }
        
        # Parameter history summary
        for param_name, history in self.param_history.items():
            param_report = {
                "count": len(history.values),
                "value_distribution": history.get_value_distribution(),
                "recent_values": history.get_recent_values(days=7)
            }
            
            # Add best values for different metrics
            for metric_name, higher_is_better in [
                ("throughput", True),
                ("latency", False),
                ("memory_usage", False)
            ]:
                best_value = history.get_best_value(metric_name, higher_is_better)
                param_report[f"best_for_{metric_name}"] = best_value
            
            report["parameter_history"][param_name] = param_report
            
            # Success rates
            report["success_rates"][param_name] = {
                "overall": history.get_success_rate(days=0),
                "last_7_days": history.get_success_rate(days=7)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check if we have enough samples
        for param_name, history in self.param_history.items():
            if len(history.values) < 10:
                recommendations.append(f"Collect more test samples for parameter {param_name}")
        
        # Check for parameter patterns
        for param_name, history in self.param_history.items():
            if len(history.values) >= 10:
                success_rate = history.get_success_rate()
                
                if success_rate < 0.5:
                    recommendations.append(f"Low success rate ({success_rate:.2f}) for {param_name}, consider adjusting range")
                
                # Check if best values are at extremes of ranges
                best_value = history.get_best_value("throughput", True)
                if best_value is not None:
                    param_range = self.param_ranges.get(param_name, {})
                    
                    if param_range.get("type") == "range":
                        min_val = param_range.get("min", 0)
                        max_val = param_range.get("max", 100)
                        
                        if best_value <= min_val + (max_val - min_val) * 0.1:
                            recommendations.append(f"Best {param_name} value ({best_value}) is near minimum, consider extending range downward")
                        elif best_value >= max_val - (max_val - min_val) * 0.1:
                            recommendations.append(f"Best {param_name} value ({best_value}) is near maximum, consider extending range upward")
        
        report["recommendations"] = recommendations
        
        return report


class CITestAutoTuner:
    """Auto-tuner for CI/CD test parameters with model-specific profiles."""
    
    def __init__(self, base_dir: str = "/tmp/ci_test_autotuner"):
        """
        Initialize CI test auto-tuner.
        
        Args:
            base_dir: Base directory for configuration files
        """
        self.base_dir = base_dir
        
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Model-specific tuners
        self.tuners = {}
        
        # Default tuner
        self.default_tuner = TestAutoTuner(
            config_file=os.path.join(base_dir, "default.json")
        )
    
    def get_tuner(self, model_name: str = None) -> TestAutoTuner:
        """
        Get auto-tuner for model.
        
        Args:
            model_name: Name of the model or None for default
            
        Returns:
            TestAutoTuner instance
        """
        if not model_name:
            return self.default_tuner
        
        # Normalize model name for file paths
        normalized_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        # Create tuner if it doesn't exist
        if normalized_name not in self.tuners:
            config_file = os.path.join(self.base_dir, f"{normalized_name}.json")
            self.tuners[normalized_name] = TestAutoTuner(config_file=config_file)
        
        return self.tuners[normalized_name]
    
    def select_test_parameters(self, model_name: str = None) -> Dict[str, Any]:
        """
        Select test parameters for model.
        
        Args:
            model_name: Name of the model or None for default
            
        Returns:
            Dictionary with selected parameters
        """
        tuner = self.get_tuner(model_name)
        return tuner.select_parameters(model_name)
    
    def record_test_result(self, 
                          params: Dict[str, Any], 
                          test_result: Dict[str, Any],
                          model_name: str = None):
        """
        Record test result with parameters.
        
        Args:
            params: Parameters used for test
            test_result: Test result metrics
            model_name: Optional model name
        """
        tuner = self.get_tuner(model_name)
        tuner.record_test_result(params, test_result, model_name)
    
    def get_optimization_reports(self) -> Dict[str, Dict[str, Any]]:
        """
        Get optimization reports for all models.
        
        Returns:
            Dictionary mapping model names to optimization reports
        """
        reports = {
            "default": self.default_tuner.get_optimization_report()
        }
        
        # Add reports for model-specific tuners
        for model_name, tuner in self.tuners.items():
            reports[model_name] = tuner.get_optimization_report()
        
        return reports


# Singleton instance
_ci_test_autotuner = None

def get_autotuner(base_dir: str = "/tmp/ci_test_autotuner") -> CITestAutoTuner:
    """
    Get singleton CI test auto-tuner instance.
    
    Args:
        base_dir: Base directory for configuration files
        
    Returns:
        CITestAutoTuner instance
    """
    global _ci_test_autotuner
    
    if _ci_test_autotuner is None:
        _ci_test_autotuner = CITestAutoTuner(base_dir=base_dir)
    
    return _ci_test_autotuner


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI Test Auto-Tuner")
    parser.add_argument("--base-dir", default="/tmp/ci_test_autotuner", help="Base directory for configuration files")
    parser.add_argument("--report", action="store_true", help="Generate optimization reports")
    parser.add_argument("--model", default=None, help="Model name for model-specific tuning")
    parser.add_argument("--select", action="store_true", help="Select test parameters")
    
    args = parser.parse_args()
    
    autotuner = get_autotuner(base_dir=args.base_dir)
    
    if args.report:
        # Generate optimization reports
        reports = autotuner.get_optimization_reports()
        
        if args.model:
            # Print report for specific model
            if args.model in reports:
                print(json.dumps(reports[args.model], indent=2))
            else:
                print(f"No report available for model {args.model}")
        else:
            # Print all reports
            print(json.dumps(reports, indent=2))
    
    if args.select:
        # Select test parameters
        params = autotuner.select_test_parameters(args.model)
        print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
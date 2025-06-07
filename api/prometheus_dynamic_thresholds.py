"""
Dynamic Threshold Adapter for Prometheus Alert Rules

This module provides functionality to dynamically adjust Prometheus alert thresholds
based on observed metrics and patterns, enabling adaptive monitoring for T4 GPU workloads.
"""

import os
import re
import json
import yaml
import logging
import requests
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricHistory:
    """Class to track metric history for threshold learning."""
    
    def __init__(self, metric_name: str, window_days: int = 7):
        """
        Initialize metric history.
        
        Args:
            metric_name: Name of the Prometheus metric
            window_days: Number of days to look back for history
        """
        self.metric_name = metric_name
        self.window_days = window_days
        self.values = []
        self.timestamps = []
        
    def add_samples(self, values: List[float], timestamps: List[datetime]):
        """
        Add metric samples with timestamps.
        
        Args:
            values: List of metric values
            timestamps: List of sample timestamps
        """
        if len(values) != len(timestamps):
            logger.warning(f"Mismatch in values and timestamps length for {self.metric_name}")
            return
            
        for value, ts in zip(values, timestamps):
            self.values.append(value)
            self.timestamps.append(ts)
        
        # Trim old data
        self._trim_old_data()
    
    def _trim_old_data(self):
        """Trim data older than the window."""
        cutoff_time = datetime.now() - timedelta(days=self.window_days)
        
        # Find cutoff index
        cutoff_idx = 0
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff_time:
                cutoff_idx = i
                break
        
        # Trim data
        if cutoff_idx > 0:
            self.values = self.values[cutoff_idx:]
            self.timestamps = self.timestamps[cutoff_idx:]
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """
        Get percentile from metric history.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Percentile value or None if insufficient data
        """
        if not self.values:
            return None
            
        return float(np.percentile(self.values, percentile))
    
    def get_average(self) -> Optional[float]:
        """Get average from metric history."""
        if not self.values:
            return None
            
        return float(np.mean(self.values))
    
    def get_stddev(self) -> Optional[float]:
        """Get standard deviation from metric history."""
        if not self.values:
            return None
            
        return float(np.std(self.values))
    
    def get_anomaly_threshold(self, sensitivity: float = 3.0) -> Dict[str, float]:
        """
        Get anomaly detection thresholds using z-score.
        
        Args:
            sensitivity: Z-score sensitivity factor
            
        Returns:
            Dictionary with high and low thresholds
        """
        if not self.values or len(self.values) < 30:  # Need sufficient samples
            return {"high": None, "low": None}
            
        mean = np.mean(self.values)
        stddev = np.std(self.values)
        
        # Calculate thresholds using z-score
        high_threshold = mean + (stddev * sensitivity)
        low_threshold = mean - (stddev * sensitivity)
        
        return {"high": high_threshold, "low": low_threshold}
    
    def get_trend_based_threshold(self, 
                                  days_back: int = 1, 
                                  percentile: float = 95.0,
                                  safety_factor: float = 1.2) -> Optional[float]:
        """
        Get threshold based on recent trend.
        
        Args:
            days_back: Days to look back for recent trend
            percentile: Percentile to use
            safety_factor: Multiplier for threshold
            
        Returns:
            Threshold value or None if insufficient data
        """
        if not self.values:
            return None
            
        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_values = [v for v, ts in zip(self.values, self.timestamps) if ts >= cutoff_time]
        
        if not recent_values or len(recent_values) < 10:  # Need sufficient recent samples
            return None
            
        # Calculate threshold using percentile and safety factor
        threshold = np.percentile(recent_values, percentile) * safety_factor
        return float(threshold)


class PrometheusClient:
    """Client for interacting with Prometheus API."""
    
    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        """
        Initialize Prometheus client.
        
        Args:
            prometheus_url: URL of Prometheus server
        """
        self.prometheus_url = prometheus_url
    
    def query_range(self, 
                    query: str, 
                    start_time: datetime, 
                    end_time: datetime, 
                    step: str = "1h") -> List[Dict[str, Any]]:
        """
        Query Prometheus for a range of time.
        
        Args:
            query: Prometheus query string
            start_time: Start time for query
            end_time: End time for query
            step: Step interval
            
        Returns:
            List of result dictionaries
        """
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_time.timestamp(),
            "end": end_time.timestamp(),
            "step": step
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Prometheus query failed: {response.text}")
                return []
                
            result_data = response.json()
            if result_data["status"] != "success":
                logger.error(f"Prometheus query failed: {result_data}")
                return []
                
            return result_data["data"]["result"]
        except Exception as e:
            logger.error(f"Error querying Prometheus: {str(e)}")
            return []
    
    def get_metric_history(self, 
                         metric_name: str, 
                         days: int = 7,
                         labels: Dict[str, str] = None) -> MetricHistory:
        """
        Get history for a metric.
        
        Args:
            metric_name: Name of the metric
            days: Number of days to look back
            labels: Metric labels to filter by
            
        Returns:
            MetricHistory object
        """
        history = MetricHistory(metric_name, window_days=days)
        
        # Build query with labels if provided
        query = metric_name
        if labels:
            label_parts = [f'{k}="{v}"' for k, v in labels.items()]
            query = f"{metric_name}{{{','.join(label_parts)}}}"
        
        # Query Prometheus
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Adjust step based on time range
        hours = days * 24
        if hours <= 24:
            step = "5m"
        elif hours <= 72:
            step = "15m"
        else:
            step = "1h"
        
        results = self.query_range(query, start_time, end_time, step)
        
        # Process results
        for result in results:
            values = []
            timestamps = []
            
            for point in result["values"]:
                timestamp = datetime.fromtimestamp(point[0])
                value = float(point[1])
                
                values.append(value)
                timestamps.append(timestamp)
            
            history.add_samples(values, timestamps)
        
        return history


class AlertRuleThresholdManager:
    """Manager for dynamically updating alert rule thresholds."""
    
    def __init__(self, 
                 rules_file: str,
                 prometheus_url: str = "http://prometheus:9090",
                 config_file: str = "/tmp/dynamic_thresholds.json"):
        """
        Initialize threshold manager.
        
        Args:
            rules_file: Path to Prometheus alert rules file
            prometheus_url: URL of Prometheus server
            config_file: Path to save/load threshold configurations
        """
        self.rules_file = rules_file
        self.prometheus_client = PrometheusClient(prometheus_url)
        self.config_file = config_file
        self.thresholds_config = self._load_config()
        
        # Metric history cache
        self.metric_history = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load threshold configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        
        # Default configuration
        return {
            "metrics": {},
            "last_update": None,
            "threshold_strategies": {}
        }
    
    def _save_config(self):
        """Save threshold configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.thresholds_config, f, indent=2)
            logger.info(f"Saved threshold configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
    
    def load_rules(self) -> Dict[str, Any]:
        """
        Load alert rules from file.
        
        Returns:
            Dictionary with alert rules
        """
        try:
            with open(self.rules_file, 'r') as f:
                rules = yaml.safe_load(f)
            return rules
        except Exception as e:
            logger.error(f"Error loading rules: {str(e)}")
            return {"groups": []}
    
    def save_rules(self, rules: Dict[str, Any]):
        """
        Save alert rules to file.
        
        Args:
            rules: Dictionary with alert rules
        """
        try:
            with open(self.rules_file, 'w') as f:
                yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved alert rules to {self.rules_file}")
        except Exception as e:
            logger.error(f"Error saving rules: {str(e)}")
    
    def extract_metric_from_expr(self, expr: str) -> str:
        """
        Extract metric name from Prometheus expression.
        
        Args:
            expr: Prometheus expression
            
        Returns:
            Metric name or empty string if not found
        """
        # Simple metric extraction using regex
        match = re.match(r'^([a-zA-Z0-9_:]+)(\{|$|\s|>|<|=)', expr)
        if match:
            return match.group(1)
        return ""
    
    def analyze_rule_expressions(self) -> Dict[str, List[str]]:
        """
        Analyze alert rule expressions to extract metrics.
        
        Returns:
            Dictionary mapping alert names to metrics
        """
        rules = self.load_rules()
        alert_metrics = {}
        
        for group in rules.get("groups", []):
            for rule in group.get("rules", []):
                if "alert" in rule and "expr" in rule:
                    alert_name = rule["alert"]
                    expr = rule["expr"]
                    
                    # Extract metric name
                    metric_name = self.extract_metric_from_expr(expr)
                    if metric_name:
                        if alert_name not in alert_metrics:
                            alert_metrics[alert_name] = []
                        alert_metrics[alert_name].append(metric_name)
        
        return alert_metrics
    
    def get_metric_history(self, metric_name: str) -> MetricHistory:
        """
        Get history for a metric, using cache if available.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            MetricHistory object
        """
        if metric_name not in self.metric_history:
            # Get metric history from Prometheus
            history = self.prometheus_client.get_metric_history(metric_name)
            self.metric_history[metric_name] = history
        
        return self.metric_history[metric_name]
    
    def compute_threshold(self, 
                         metric_name: str, 
                         strategy: str = "percentile", 
                         params: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Compute threshold for a metric using specified strategy.
        
        Args:
            metric_name: Name of the metric
            strategy: Threshold strategy (percentile, anomaly, trend)
            params: Strategy parameters
            
        Returns:
            Dictionary with threshold values
        """
        history = self.get_metric_history(metric_name)
        
        if strategy == "percentile":
            # Default parameters
            p_params = {
                "high_percentile": 95.0,
                "low_percentile": 5.0,
                "safety_factor": 1.1
            }
            
            # Override with provided params
            if params:
                p_params.update(params)
            
            # Compute thresholds
            high_threshold = history.get_percentile(p_params["high_percentile"])
            if high_threshold is not None:
                high_threshold *= p_params["safety_factor"]
                
            low_threshold = history.get_percentile(p_params["low_percentile"])
            if low_threshold is not None:
                low_threshold /= p_params["safety_factor"]
            
            return {"high": high_threshold, "low": low_threshold}
            
        elif strategy == "anomaly":
            # Default parameters
            a_params = {"sensitivity": 3.0}
            
            # Override with provided params
            if params:
                a_params.update(params)
            
            # Compute thresholds
            return history.get_anomaly_threshold(a_params["sensitivity"])
            
        elif strategy == "trend":
            # Default parameters
            t_params = {
                "days_back": 1,
                "high_percentile": 95.0,
                "low_percentile": 5.0,
                "safety_factor": 1.2
            }
            
            # Override with provided params
            if params:
                t_params.update(params)
            
            # Compute thresholds
            high_threshold = history.get_trend_based_threshold(
                t_params["days_back"],
                t_params["high_percentile"],
                t_params["safety_factor"]
            )
            
            low_threshold = None
            if "low_percentile" in t_params:
                # Invert for low threshold (lower values are more extreme)
                low_values = [-v for v in history.values]
                low_history = MetricHistory(f"{metric_name}_low")
                low_history.add_samples(low_values, history.timestamps)
                
                low_threshold_neg = low_history.get_trend_based_threshold(
                    t_params["days_back"],
                    t_params["low_percentile"],
                    t_params["safety_factor"]
                )
                
                if low_threshold_neg is not None:
                    low_threshold = -low_threshold_neg
            
            return {"high": high_threshold, "low": low_threshold}
        
        # Default to None if unknown strategy
        return {"high": None, "low": None}
    
    def update_rule_thresholds(self):
        """Update alert rule thresholds based on metric history."""
        rules = self.load_rules()
        alert_metrics = self.analyze_rule_expressions()
        updated = False
        
        for group in rules.get("groups", []):
            for rule in group.get("rules", []):
                if "alert" in rule and "expr" in rule:
                    alert_name = rule["alert"]
                    expr = rule["expr"]
                    
                    # Skip if alert not in threshold config
                    if alert_name not in self.thresholds_config.get("threshold_strategies", {}):
                        continue
                    
                    # Get threshold strategy
                    strategy_config = self.thresholds_config["threshold_strategies"][alert_name]
                    strategy = strategy_config.get("strategy", "percentile")
                    params = strategy_config.get("params", {})
                    
                    # Get metrics for this alert
                    metrics = alert_metrics.get(alert_name, [])
                    if not metrics:
                        continue
                    
                    # Compute thresholds for each metric
                    for metric_name in metrics:
                        thresholds = self.compute_threshold(metric_name, strategy, params)
                        
                        # Skip if thresholds couldn't be computed
                        if thresholds["high"] is None and thresholds["low"] is None:
                            continue
                        
                        # Update expression with new thresholds
                        new_expr = self.update_expression_thresholds(expr, thresholds)
                        if new_expr != expr:
                            rule["expr"] = new_expr
                            updated = True
                            logger.info(f"Updated threshold for alert {alert_name}: {new_expr}")
        
        # Save updated rules if changed
        if updated:
            self.save_rules(rules)
            
        # Update last update timestamp
        self.thresholds_config["last_update"] = datetime.now().isoformat()
        self._save_config()
    
    def update_expression_thresholds(self, 
                                    expr: str, 
                                    thresholds: Dict[str, Optional[float]]) -> str:
        """
        Update expression with new thresholds.
        
        Args:
            expr: Prometheus expression
            thresholds: Dictionary with high and low thresholds
            
        Returns:
            Updated expression
        """
        # Replace high threshold
        if thresholds["high"] is not None:
            # Match patterns like > 80, >= 80, etc.
            high_pattern = r'(>\s*=?\s*)([0-9.]+)'
            high_match = re.search(high_pattern, expr)
            if high_match:
                operator = high_match.group(1)
                old_threshold = high_match.group(2)
                
                # Format new threshold with same precision as old
                precision = len(old_threshold.split('.')[-1]) if '.' in old_threshold else 0
                new_threshold = f"{thresholds['high']:.{precision}f}"
                
                # Replace threshold
                expr = expr.replace(f"{operator}{old_threshold}", f"{operator}{new_threshold}")
        
        # Replace low threshold
        if thresholds["low"] is not None:
            # Match patterns like < 10, <= 10, etc.
            low_pattern = r'(<\s*=?\s*)([0-9.]+)'
            low_match = re.search(low_pattern, expr)
            if low_match:
                operator = low_match.group(1)
                old_threshold = low_match.group(2)
                
                # Format new threshold with same precision as old
                precision = len(old_threshold.split('.')[-1]) if '.' in old_threshold else 0
                new_threshold = f"{thresholds['low']:.{precision}f}"
                
                # Replace threshold
                expr = expr.replace(f"{operator}{old_threshold}", f"{operator}{new_threshold}")
        
        return expr
    
    def set_threshold_strategy(self, 
                              alert_name: str, 
                              strategy: str, 
                              params: Dict[str, Any] = None):
        """
        Set threshold strategy for an alert.
        
        Args:
            alert_name: Name of the alert
            strategy: Threshold strategy (percentile, anomaly, trend)
            params: Strategy parameters
        """
        if "threshold_strategies" not in self.thresholds_config:
            self.thresholds_config["threshold_strategies"] = {}
        
        self.thresholds_config["threshold_strategies"][alert_name] = {
            "strategy": strategy,
            "params": params or {}
        }
        
        # Save config
        self._save_config()
    
    def configure_default_strategies(self):
        """Configure default threshold strategies for common alert types."""
        # Temperature alerts
        self.set_threshold_strategy(
            "T4GPUHighTemperature",
            "percentile",
            {"high_percentile": 95.0, "safety_factor": 1.05}
        )
        
        self.set_threshold_strategy(
            "T4GPUCriticalTemperature",
            "percentile",
            {"high_percentile": 99.0, "safety_factor": 1.02}
        )
        
        # Memory alerts
        self.set_threshold_strategy(
            "T4GPUMemoryNearFull",
            "anomaly",
            {"sensitivity": 2.5}
        )
        
        # Utilization alerts
        self.set_threshold_strategy(
            "T4GPUHighUtilization",
            "trend",
            {"days_back": 1, "high_percentile": 98.0, "safety_factor": 1.02}
        )
        
        self.set_threshold_strategy(
            "T4GPULowUtilization",
            "trend",
            {"days_back": 1, "low_percentile": 2.0, "safety_factor": 1.5}
        )
        
        # Power alerts
        self.set_threshold_strategy(
            "T4GPUPowerConsumptionHigh",
            "percentile",
            {"high_percentile": 95.0, "safety_factor": 1.05}
        )
        
        # Performance alerts
        self.set_threshold_strategy(
            "T4GPUThroughputDrop",
            "anomaly",
            {"sensitivity": 2.0}
        )
        
        self.set_threshold_strategy(
            "T4GPULatencyIncrease",
            "anomaly",
            {"sensitivity": 2.0}
        )
        
        logger.info("Configured default threshold strategies")
    
    def get_threshold_report(self) -> Dict[str, Any]:
        """
        Get report on threshold configuration and history.
        
        Returns:
            Dictionary with threshold report
        """
        report = {
            "last_update": self.thresholds_config.get("last_update"),
            "metrics": {},
            "alerts": {}
        }
        
        # Metrics information
        for metric_name, history in self.metric_history.items():
            if history.values:
                report["metrics"][metric_name] = {
                    "count": len(history.values),
                    "mean": float(np.mean(history.values)),
                    "p95": history.get_percentile(95.0),
                    "p99": history.get_percentile(99.0),
                    "min": float(np.min(history.values)),
                    "max": float(np.max(history.values))
                }
        
        # Alert information
        alert_metrics = self.analyze_rule_expressions()
        
        for alert_name, metrics in alert_metrics.items():
            strategy_config = self.thresholds_config.get("threshold_strategies", {}).get(alert_name, {})
            
            report["alerts"][alert_name] = {
                "metrics": metrics,
                "strategy": strategy_config.get("strategy", "none"),
                "params": strategy_config.get("params", {})
            }
            
            # Add computed thresholds if strategy is set
            if strategy_config:
                strategy = strategy_config.get("strategy")
                params = strategy_config.get("params", {})
                
                for metric_name in metrics:
                    if metric_name in self.metric_history:
                        thresholds = self.compute_threshold(metric_name, strategy, params)
                        report["alerts"][alert_name]["computed_thresholds"] = thresholds
        
        return report


def create_threshold_manager(rules_file: str, 
                           prometheus_url: str = "http://prometheus:9090",
                           config_file: str = "/tmp/dynamic_thresholds.json") -> AlertRuleThresholdManager:
    """
    Create alert rule threshold manager.
    
    Args:
        rules_file: Path to Prometheus alert rules file
        prometheus_url: URL of Prometheus server
        config_file: Path to save/load threshold configurations
        
    Returns:
        AlertRuleThresholdManager instance
    """
    manager = AlertRuleThresholdManager(
        rules_file=rules_file,
        prometheus_url=prometheus_url,
        config_file=config_file
    )
    
    # Configure default strategies
    manager.configure_default_strategies()
    
    return manager
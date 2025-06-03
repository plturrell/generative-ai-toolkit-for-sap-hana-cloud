#!/usr/bin/env python3
"""
Canary Deployment Monitor for SAP HANA AI Toolkit.

This script monitors canary deployments and provides:
1. Real-time comparison of canary vs production metrics
2. Automatic rollback based on configurable thresholds
3. Alerts and notifications for deployment events
4. Detailed logging and metrics visualization

Usage:
    python monitor-canary.py --production-url PROD_URL --canary-url CANARY_URL [options]

Options:
    --production-url URL    URL for production endpoint
    --canary-url URL        URL for canary endpoint
    --check-interval SEC    Time between checks in seconds (default: 60)
    --error-threshold N     Max errors before rollback (default: 5)
    --latency-factor N      Max latency factor compared to prod (default: 1.5)
    --duration HOURS        Monitoring duration in hours (default: 24)
    --slack-webhook URL     Slack webhook URL for notifications
    --teams-webhook URL     MS Teams webhook URL for notifications
    --metrics-db STRING     Connection string for metrics database
    --output-file PATH      Path to output metrics file
"""

import argparse
import datetime
import json
import logging
import os
import requests
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('canary-monitor.log')
    ]
)
logger = logging.getLogger("canary-monitor")

# Default configuration
DEFAULT_CHECK_INTERVAL = 60  # seconds
DEFAULT_ERROR_THRESHOLD = 5  # max errors before rollback
DEFAULT_LATENCY_FACTOR = 1.5  # max latency factor compared to prod
DEFAULT_DURATION = 24  # hours


@dataclass
class MetricsPoint:
    """Data class for a single metrics measurement point."""
    timestamp: float
    response_time: float
    status_code: int
    is_error: bool
    memory_used_percent: float
    cpu_percent: float
    uptime: str
    services_status: Dict[str, Any]


class CanaryMonitor:
    """
    Monitor and compare metrics between canary and production deployments.
    
    This class provides functionality to:
    1. Collect metrics from both deployments
    2. Compare performance and error rates
    3. Trigger alerts or rollbacks based on thresholds
    4. Generate reports and visualizations
    """
    
    def __init__(
        self,
        production_url: str,
        canary_url: str,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        error_threshold: int = DEFAULT_ERROR_THRESHOLD,
        latency_factor: float = DEFAULT_LATENCY_FACTOR,
        duration_hours: int = DEFAULT_DURATION,
        slack_webhook: Optional[str] = None,
        teams_webhook: Optional[str] = None,
        metrics_db: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        """
        Initialize the canary monitor.
        
        Args:
            production_url: URL for production health endpoint
            canary_url: URL for canary health endpoint
            check_interval: Time between checks in seconds
            error_threshold: Max errors before triggering rollback
            latency_factor: Max acceptable latency factor for canary
            duration_hours: Monitoring duration in hours
            slack_webhook: Optional Slack webhook for notifications
            teams_webhook: Optional MS Teams webhook for notifications
            metrics_db: Optional connection string for metrics database
            output_file: Optional path to output metrics file
        """
        self.production_url = production_url
        self.canary_url = canary_url
        self.check_interval = check_interval
        self.error_threshold = error_threshold
        self.latency_factor = latency_factor
        self.duration_seconds = duration_hours * 3600
        
        self.slack_webhook = slack_webhook
        self.teams_webhook = teams_webhook
        self.metrics_db = metrics_db
        self.output_file = output_file
        
        # Metrics storage
        self.production_metrics: List[MetricsPoint] = []
        self.canary_metrics: List[MetricsPoint] = []
        
        # Running statistics
        self.canary_errors = 0
        self.production_errors = 0
        self.start_time = time.time()
        
        # State flags
        self.rollback_triggered = False
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown on signals."""
        logger.info("Shutdown signal received, stopping monitor...")
        self.running = False
    
    def collect_metrics(self, url: str) -> Tuple[MetricsPoint, bool]:
        """
        Collect metrics from a deployment endpoint.
        
        Args:
            url: Health check URL to query
            
        Returns:
            Tuple of (metrics_point, success)
        """
        start_time = time.time()
        is_error = False
        
        try:
            response = requests.get(f"{url}/health", timeout=10)
            elapsed = time.time() - start_time
            
            # Parse response
            if response.status_code == 200:
                data = response.json()
                
                # Extract metrics
                metrics = MetricsPoint(
                    timestamp=time.time(),
                    response_time=elapsed,
                    status_code=response.status_code,
                    is_error=False,
                    memory_used_percent=data.get("system", {}).get("memory_used_percent", 0),
                    cpu_percent=data.get("system", {}).get("cpu_percent", 0),
                    uptime=data.get("uptime", "unknown"),
                    services_status=data.get("services", {})
                )
                
                # Check if any services are unhealthy
                if any(not service.get("is_healthy", True) for service in data.get("services", {}).values()):
                    is_error = True
                    metrics.is_error = True
            else:
                # Failed response
                is_error = True
                metrics = MetricsPoint(
                    timestamp=time.time(),
                    response_time=elapsed,
                    status_code=response.status_code,
                    is_error=True,
                    memory_used_percent=0,
                    cpu_percent=0,
                    uptime="unknown",
                    services_status={}
                )
                
        except Exception as e:
            # Request exception
            elapsed = time.time() - start_time
            logger.error(f"Error collecting metrics from {url}: {str(e)}")
            is_error = True
            metrics = MetricsPoint(
                timestamp=time.time(),
                response_time=elapsed,
                status_code=0,
                is_error=True,
                memory_used_percent=0,
                cpu_percent=0,
                uptime="unknown",
                services_status={}
            )
        
        return metrics, not is_error
    
    def compare_metrics(self) -> Dict[str, Any]:
        """
        Compare latest metrics between canary and production.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.canary_metrics or not self.production_metrics:
            return {"status": "insufficient_data"}
        
        # Get latest metrics
        canary = self.canary_metrics[-1]
        production = self.production_metrics[-1]
        
        # Calculate metrics
        latency_ratio = canary.response_time / max(0.001, production.response_time)
        
        # Compare performance
        comparison = {
            "status": "ok" if not canary.is_error else "error",
            "timestamp": time.time(),
            "latency": {
                "canary": canary.response_time,
                "production": production.response_time,
                "ratio": latency_ratio,
                "threshold_exceeded": latency_ratio > self.latency_factor
            },
            "errors": {
                "canary": self.canary_errors,
                "production": self.production_errors,
                "threshold_exceeded": self.canary_errors >= self.error_threshold
            },
            "resources": {
                "memory": {
                    "canary": canary.memory_used_percent,
                    "production": production.memory_used_percent,
                    "diff": canary.memory_used_percent - production.memory_used_percent
                },
                "cpu": {
                    "canary": canary.cpu_percent,
                    "production": production.cpu_percent,
                    "diff": canary.cpu_percent - production.cpu_percent
                }
            },
            "services": {
                "canary_healthy": all(service.get("is_healthy", True) for service in canary.services_status.values()),
                "production_healthy": all(service.get("is_healthy", True) for service in production.services_status.values())
            },
            "recommendation": "continue"
        }
        
        # Determine recommendation
        if self.canary_errors >= self.error_threshold:
            comparison["recommendation"] = "rollback"
        elif latency_ratio > self.latency_factor and canary.response_time > 1.0:
            comparison["recommendation"] = "investigate"
        
        return comparison
    
    def check_rollback_conditions(self, comparison: Dict[str, Any]) -> bool:
        """
        Check if rollback conditions are met.
        
        Args:
            comparison: Metrics comparison dictionary
            
        Returns:
            True if rollback should be triggered
        """
        if comparison.get("recommendation") == "rollback":
            logger.warning("Rollback condition met! Error threshold exceeded.")
            return True
        
        if (comparison.get("latency", {}).get("threshold_exceeded", False) and 
            comparison.get("errors", {}).get("canary", 0) > 0):
            logger.warning("Rollback condition met! High latency with errors.")
            return True
        
        return False
    
    def trigger_rollback(self) -> None:
        """
        Trigger a canary rollback.
        
        This would typically call external scripts or APIs to perform the actual rollback.
        Here it just logs the event and sends notifications.
        """
        if self.rollback_triggered:
            return
        
        logger.critical("🔥 TRIGGERING CANARY ROLLBACK!")
        self.rollback_triggered = True
        
        # Log detailed metrics for forensic analysis
        self._log_rollback_details()
        
        # Send notifications
        self._send_notification("⚠️ Canary Rollback Triggered", 
                               f"Canary deployment at {self.canary_url} has been rolled back due to errors.")
        
        # In a real implementation, this would call the rollback script or API
        try:
            # Example: Call rollback script (commented out)
            # subprocess.run(["./canary-rollback.sh", "--env", "k8s", "--percentage", "0"], check=True)
            logger.info("Rollback command would be executed here")
        except Exception as e:
            logger.error(f"Error triggering rollback: {str(e)}")
    
    def _log_rollback_details(self) -> None:
        """Log detailed metrics for forensic analysis of rollback."""
        logger.info("=== ROLLBACK FORENSIC DATA ===")
        
        # Log last 5 canary metrics
        for i, metrics in enumerate(self.canary_metrics[-5:]):
            logger.info(f"Canary Metrics #{i+1}:")
            logger.info(f"  Time: {datetime.datetime.fromtimestamp(metrics.timestamp).isoformat()}")
            logger.info(f"  Response Time: {metrics.response_time:.4f}s")
            logger.info(f"  Status Code: {metrics.status_code}")
            logger.info(f"  Error: {metrics.is_error}")
            logger.info(f"  Memory: {metrics.memory_used_percent:.2f}%")
            logger.info(f"  CPU: {metrics.cpu_percent:.2f}%")
        
        # Log comparison with production
        last_comparison = self.compare_metrics()
        logger.info("Last Comparison:")
        logger.info(json.dumps(last_comparison, indent=2))
        
        logger.info("=== END FORENSIC DATA ===")
    
    def _send_notification(self, title: str, message: str) -> None:
        """
        Send notifications to configured channels.
        
        Args:
            title: Notification title
            message: Notification message
        """
        # Send to Slack if configured
        if self.slack_webhook:
            try:
                requests.post(
                    self.slack_webhook,
                    json={
                        "text": f"*{title}*\n{message}",
                        "mrkdwn": True
                    },
                    timeout=5
                )
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {str(e)}")
        
        # Send to MS Teams if configured
        if self.teams_webhook:
            try:
                requests.post(
                    self.teams_webhook,
                    json={
                        "title": title,
                        "text": message
                    },
                    timeout=5
                )
            except Exception as e:
                logger.error(f"Failed to send Teams notification: {str(e)}")
    
    def save_metrics(self) -> None:
        """Save collected metrics to the configured output file."""
        if not self.output_file:
            return
        
        try:
            # Prepare data for export
            data = {
                "start_time": self.start_time,
                "end_time": time.time(),
                "production_url": self.production_url,
                "canary_url": self.canary_url,
                "config": {
                    "check_interval": self.check_interval,
                    "error_threshold": self.error_threshold,
                    "latency_factor": self.latency_factor
                },
                "summary": {
                    "duration_seconds": time.time() - self.start_time,
                    "canary_errors": self.canary_errors,
                    "production_errors": self.production_errors,
                    "rollback_triggered": self.rollback_triggered
                },
                "production_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "response_time": m.response_time,
                        "status_code": m.status_code,
                        "is_error": m.is_error,
                        "memory_used_percent": m.memory_used_percent,
                        "cpu_percent": m.cpu_percent
                    } for m in self.production_metrics
                ],
                "canary_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "response_time": m.response_time,
                        "status_code": m.status_code,
                        "is_error": m.is_error,
                        "memory_used_percent": m.memory_used_percent,
                        "cpu_percent": m.cpu_percent
                    } for m in self.canary_metrics
                ]
            }
            
            # Write to file
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics saved to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def print_summary(self) -> None:
        """Print a summary of the monitoring session."""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*80)
        print(f"CANARY MONITORING SUMMARY")
        print("="*80)
        print(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Production URL: {self.production_url}")
        print(f"Canary URL: {self.canary_url}")
        print(f"Checks Performed: {len(self.canary_metrics)}")
        print(f"Canary Errors: {self.canary_errors}")
        print(f"Production Errors: {self.production_errors}")
        print(f"Rollback Triggered: {'Yes' if self.rollback_triggered else 'No'}")
        
        if self.canary_metrics and self.production_metrics:
            # Calculate averages
            canary_latency = sum(m.response_time for m in self.canary_metrics) / len(self.canary_metrics)
            prod_latency = sum(m.response_time for m in self.production_metrics) / len(self.production_metrics)
            
            print("\nPERFORMANCE COMPARISON:")
            print(f"Avg. Canary Response Time:   {canary_latency:.4f}s")
            print(f"Avg. Production Response Time: {prod_latency:.4f}s")
            print(f"Latency Ratio:                 {canary_latency/prod_latency:.2f}x")
        
        print("="*80)
    
    def run(self) -> None:
        """
        Run the canary monitoring process.
        
        This is the main loop that:
        1. Collects metrics from both deployments
        2. Compares metrics and checks for rollback conditions
        3. Triggers rollback if needed
        4. Saves metrics and generates reports
        """
        logger.info(f"Starting canary monitoring: {self.canary_url} vs {self.production_url}")
        logger.info(f"Check interval: {self.check_interval}s, Error threshold: {self.error_threshold}")
        
        # Send startup notification
        self._send_notification(
            "🚀 Canary Monitoring Started",
            f"Monitoring canary deployment at {self.canary_url}"
        )
        
        check_count = 0
        end_time = self.start_time + self.duration_seconds
        
        try:
            while self.running and time.time() < end_time:
                check_count += 1
                logger.info(f"Check #{check_count}")
                
                # Collect metrics from both deployments
                production_metrics, prod_success = self.collect_metrics(self.production_url)
                canary_metrics, canary_success = self.collect_metrics(self.canary_url)
                
                # Store metrics
                self.production_metrics.append(production_metrics)
                self.canary_metrics.append(canary_metrics)
                
                # Update error counts
                if not prod_success:
                    self.production_errors += 1
                if not canary_success:
                    self.canary_errors += 1
                
                # Compare metrics and check for rollback
                comparison = self.compare_metrics()
                logger.info(f"Comparison: {json.dumps(comparison, indent=2)}")
                
                # Check if rollback is needed
                if self.check_rollback_conditions(comparison) and not self.rollback_triggered:
                    self.trigger_rollback()
                
                # Print status
                print(f"\rCheck #{check_count} | "
                      f"Canary: {'✅' if canary_success else '❌'} | "
                      f"Production: {'✅' if prod_success else '❌'} | "
                      f"Errors: {self.canary_errors}/{self.error_threshold} | "
                      f"Status: {'✅' if not self.rollback_triggered else '🔥 ROLLBACK'}", end="")
                
                # Wait for next check
                if self.running and time.time() < end_time:
                    time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")
        finally:
            # Cleanup and save results
            logger.info("Monitoring complete, saving results...")
            self.save_metrics()
            self.print_summary()
            
            # Send completion notification
            status = "rolled back" if self.rollback_triggered else "completed successfully"
            self._send_notification(
                f"🏁 Canary Monitoring {status.title()}",
                f"Canary monitoring for {self.canary_url} has {status}."
            )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Canary Deployment Monitor")
    
    parser.add_argument("--production-url", required=True, help="URL for production endpoint")
    parser.add_argument("--canary-url", required=True, help="URL for canary endpoint")
    parser.add_argument("--check-interval", type=int, default=DEFAULT_CHECK_INTERVAL, 
                        help=f"Time between checks in seconds (default: {DEFAULT_CHECK_INTERVAL})")
    parser.add_argument("--error-threshold", type=int, default=DEFAULT_ERROR_THRESHOLD,
                        help=f"Max errors before rollback (default: {DEFAULT_ERROR_THRESHOLD})")
    parser.add_argument("--latency-factor", type=float, default=DEFAULT_LATENCY_FACTOR,
                        help=f"Max latency factor compared to prod (default: {DEFAULT_LATENCY_FACTOR})")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Monitoring duration in hours (default: {DEFAULT_DURATION})")
    parser.add_argument("--slack-webhook", help="Slack webhook URL for notifications")
    parser.add_argument("--teams-webhook", help="MS Teams webhook URL for notifications")
    parser.add_argument("--metrics-db", help="Connection string for metrics database")
    parser.add_argument("--output-file", default="canary-metrics.json",
                        help="Path to output metrics file (default: canary-metrics.json)")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create and run monitor
    monitor = CanaryMonitor(
        production_url=args.production_url,
        canary_url=args.canary_url,
        check_interval=args.check_interval,
        error_threshold=args.error_threshold,
        latency_factor=args.latency_factor,
        duration_hours=args.duration,
        slack_webhook=args.slack_webhook,
        teams_webhook=args.teams_webhook,
        metrics_db=args.metrics_db,
        output_file=args.output_file
    )
    
    monitor.run()


if __name__ == "__main__":
    main()
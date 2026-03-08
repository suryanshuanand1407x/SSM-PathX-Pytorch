"""
Metric Aggregator
=================
Centralized collection and batching of metrics before logging.
"""

from typing import Dict, Any, Optional
from .logger_factory import BaseLogger


class MetricAggregator:
    """
    Aggregates metrics before flushing to logger.

    Allows collecting multiple metrics from different sources,
    then logging them all together with a single step number.
    """

    def __init__(self, logger: BaseLogger):
        """
        Args:
            logger: Logger to flush metrics to
        """
        self.logger = logger
        self.metrics_buffer = {}

    def add_metric(self, name: str, value: Any):
        """
        Add a metric to the buffer.

        Args:
            name: Metric name (e.g., 'train/loss', 'stability/spectral_radius')
            value: Metric value (float, int, etc.)
        """
        self.metrics_buffer[name] = value

    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Add multiple metrics at once.

        Args:
            metrics: Dictionary of metrics to add
        """
        self.metrics_buffer.update(metrics)

    def flush(self, step: Optional[int] = None):
        """
        Flush all buffered metrics to logger and clear buffer.

        Args:
            step: Current training step
        """
        if len(self.metrics_buffer) > 0:
            self.logger.log(self.metrics_buffer, step=step)
            self.metrics_buffer = {}

    def clear(self):
        """Clear the metrics buffer without logging."""
        self.metrics_buffer = {}

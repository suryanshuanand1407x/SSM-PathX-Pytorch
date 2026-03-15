"""
Logger Factory
==============
Creates loggers with W&B integration.
"""

import logging
from typing import Dict, Any, Optional


class BaseLogger:
    """Base class for all loggers."""

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        raise NotImplementedError

    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        raise NotImplementedError

    def finish(self):
        """Finish logging (cleanup)."""
        pass


class WandBLogger(BaseLogger):
    """Weights & Biases logger."""

    def __init__(self, project: str, config: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            config: Configuration dictionary
            name: Optional run name
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                config=config,
                name=name,
                resume='allow',  # Allow resuming runs
            )
            logging.info(f"✓ W&B initialized: {self.run.url}")
        except ImportError:
            logging.error("wandb not installed! Install with: pip install wandb")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize W&B: {e}")
            raise

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        self.wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]):
        """Update W&B config."""
        self.wandb.config.update(config)

    def finish(self):
        """Finish W&B run."""
        self.wandb.finish()


class JSONLogger(BaseLogger):
    """Fallback JSON logger (if W&B not available)."""

    def __init__(self, log_dir: str):
        """
        Initialize JSON logger.

        Args:
            log_dir: Directory to save JSON logs
        """
        import os
        import json

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, 'metrics.jsonl')
        self.config_file = os.path.join(log_dir, 'config.json')

        logging.info(f"✓ JSON logger initialized: {self.log_file}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to JSON file."""
        import json

        log_entry = {'step': step, **metrics}

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_config(self, config: Dict[str, Any]):
        """Save config to JSON file."""
        import json

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)


class LoggerFactory:
    """Factory for creating loggers."""

    @staticmethod
    def create_logger(
        backend: str = 'wandb',
        project: str = 'tustin-mamba',
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        log_dir: str = './logs',
    ) -> BaseLogger:
        """
        Create a logger based on specified backend.

        Args:
            backend: 'wandb' or 'json'
            project: W&B project name
            config: Configuration dictionary
            name: Optional run name
            log_dir: Directory for JSON logs (fallback)

        Returns:
            Logger instance
        """
        if config is None:
            config = {}

        if backend == 'wandb':
            try:
                return WandBLogger(project=project, config=config, name=name)
            except Exception as e:
                logging.warning(f"W&B initialization failed: {e}")
                logging.warning("Falling back to JSON logger")
                return JSONLogger(log_dir=log_dir)
        elif backend == 'json':
            return JSONLogger(log_dir=log_dir)
        else:
            raise ValueError(f"Unknown backend: {backend}")

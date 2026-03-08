"""
Checkpoint Manager
==================
Handles saving, loading, and cleanup of model checkpoints with time-based intervals.
"""

import os
import time
import torch
import glob
from typing import Optional, Dict, Any
import logging


class CheckpointManager:
    """
    Manages model checkpoints with time-based saving (e.g., every 2 hours).

    Features:
    - Time-based checkpointing (instead of step-based)
    - Auto-resume from latest checkpoint
    - Automatic cleanup of old checkpoints
    - Full state preservation (model, optimizer, scaler, RNG states)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval_hours: float = 2.0,
        keep_last_n: int = 5,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval_hours: Save checkpoint every N hours
            keep_last_n: Keep only the last N checkpoints (0 = keep all)
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval_seconds = checkpoint_interval_hours * 3600
        self.keep_last_n = keep_last_n
        self.last_checkpoint_time = None

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        logging.info(f"CheckpointManager initialized:")
        logging.info(f"  Directory: {checkpoint_dir}")
        logging.info(f"  Interval: {checkpoint_interval_hours} hours")
        logging.info(f"  Keep last: {keep_last_n} checkpoints")

    def should_save_checkpoint(self) -> bool:
        """
        Check if it's time to save a checkpoint based on time interval.

        Returns:
            True if checkpoint should be saved, False otherwise
        """
        if self.last_checkpoint_time is None:
            return True

        elapsed = time.time() - self.last_checkpoint_time
        return elapsed >= self.checkpoint_interval_seconds

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        dataset_version: Dict[str, str],
        wall_clock_time: float,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            scaler: Gradient scaler (or None)
            metrics: Current metrics (loss, accuracy, etc.)
            config: Full training configuration
            dataset_version: Dataset hash/version info
            wall_clock_time: Total elapsed time (seconds)

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint filename with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_step{step}_{timestamp}.pt'
        )

        # Get model state dict and strip _orig_mod. prefix if present (from torch.compile)
        # This makes checkpoints portable between compiled and non-compiled models
        state_dict = model.state_dict()
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Build checkpoint state
        checkpoint = {
            'step': step,
            'timestamp': timestamp,
            'wall_clock_time': wall_clock_time,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'metrics': metrics,
            'config': config,
            'dataset_version': dataset_version,
            'random_state': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update last checkpoint time
        self.last_checkpoint_time = time.time()

        logging.info(f"✓ Saved checkpoint: {checkpoint_path}")
        logging.info(f"  Step: {step}, Wall-clock: {wall_clock_time/3600:.2f}h")

        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint state dictionary, or None if no checkpoint exists
        """
        # Find all checkpoints
        checkpoint_files = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_*.pt')
        )

        if len(checkpoint_files) == 0:
            logging.info("No existing checkpoints found. Starting from scratch.")
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]

        logging.info(f"Loading checkpoint: {latest_checkpoint}")

        # Load checkpoint
        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            logging.info(f"✓ Loaded checkpoint from step {checkpoint['step']}")
            logging.info(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
            logging.info(f"  Wall-clock: {checkpoint.get('wall_clock_time', 0)/3600:.2f}h")

            # Set last checkpoint time to avoid immediate re-save
            self.last_checkpoint_time = time.time()

            return checkpoint

        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self):
        """
        Delete old checkpoints, keeping only the last N.
        """
        if self.keep_last_n <= 0:
            return  # Keep all checkpoints

        # Find all checkpoints
        checkpoint_files = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_*.pt')
        )

        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)

        # Delete oldest checkpoints beyond keep_last_n
        num_to_delete = len(checkpoint_files) - self.keep_last_n
        if num_to_delete > 0:
            for checkpoint_file in checkpoint_files[:num_to_delete]:
                try:
                    os.remove(checkpoint_file)
                    logging.info(f"Deleted old checkpoint: {os.path.basename(checkpoint_file)}")
                except Exception as e:
                    logging.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")

    def save_best_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        dataset_version: Dict[str, str],
    ) -> str:
        """
        Save a separate 'best' checkpoint (not subject to cleanup).

        Args:
            Same as save_checkpoint

        Returns:
            Path to saved best checkpoint
        """
        best_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            'best_checkpoint.pt'
        )

        # Strip _orig_mod. prefix if present (from torch.compile)
        state_dict = model.state_dict()
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        checkpoint = {
            'step': step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'metrics': metrics,
            'config': config,
            'dataset_version': dataset_version,
        }

        torch.save(checkpoint, best_checkpoint_path)
        logging.info(f"✓ Saved BEST checkpoint: {best_checkpoint_path}")
        logging.info(f"  Step: {step}, Val loss: {metrics.get('val_loss', 'N/A')}")

        return best_checkpoint_path

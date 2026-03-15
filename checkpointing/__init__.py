"""
Checkpointing Module
====================
Time-based checkpointing with auto-resume and data versioning.
"""

from .checkpoint_manager import CheckpointManager
from .data_versioning import compute_dataset_hash, compute_dataset_version

__all__ = ['CheckpointManager', 'compute_dataset_hash', 'compute_dataset_version']

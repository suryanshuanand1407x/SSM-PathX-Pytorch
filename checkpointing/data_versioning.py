"""
Data Versioning
===============
Compute dataset hashes to track data versions and avoid "zombie" experiments.
"""

import hashlib
import os
from typing import Dict


def compute_dataset_hash(file_path: str, chunk_size_mb: int = 10) -> str:
    """
    Compute MD5 hash of dataset file.

    To avoid reading entire large files, we hash only the first N MB.

    Args:
        file_path: Path to dataset file
        chunk_size_mb: Number of MB to hash (default: 10)

    Returns:
        MD5 hash string
    """
    if not os.path.exists(file_path):
        return "FILE_NOT_FOUND"

    hasher = hashlib.md5()
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size_bytes)
            hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return f"ERROR_{str(e)[:20]}"


def compute_dataset_version(data_paths: Dict[str, str]) -> Dict[str, str]:
    """
    Compute version info for all datasets.

    Args:
        data_paths: Dictionary mapping dataset names to file paths
                   e.g., {'train': 'path/to/train.pkl', 'val': 'path/to/val.pkl'}

    Returns:
        Dictionary with hash for each dataset
    """
    import time

    dataset_version = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for name, path in data_paths.items():
        dataset_version[f'{name}_hash'] = compute_dataset_hash(path)

    return dataset_version


def validate_dataset_version(
    checkpoint_version: Dict[str, str],
    current_version: Dict[str, str],
) -> bool:
    """
    Check if dataset versions match between checkpoint and current data.

    Args:
        checkpoint_version: Dataset version from checkpoint
        current_version: Current dataset version

    Returns:
        True if versions match, False otherwise
    """
    # Check all hash keys
    for key in checkpoint_version:
        if key.endswith('_hash'):
            checkpoint_hash = checkpoint_version[key]
            current_hash = current_version.get(key, None)

            if checkpoint_hash != current_hash:
                return False

    return True

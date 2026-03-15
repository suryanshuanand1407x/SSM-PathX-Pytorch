"""
Performance Metrics
===================
Tracks computational performance: VRAM, throughput, timing.
"""

import torch
import time
from typing import Dict, Optional


class PerformanceMonitor:
    """
    Monitors computational performance metrics.

    Tracks:
    - GPU VRAM usage (current, peak, allocated)
    - Throughput (tokens/sec, samples/sec, steps/sec)
    - Wall-clock time
    """

    @staticmethod
    def get_gpu_memory_usage(device: torch.device) -> Dict[str, float]:
        """
        Get current and peak GPU memory usage.

        Args:
            device: The device (cuda/mps/cpu)

        Returns:
            Dictionary with memory stats in MB
        """
        if device.type == 'cuda':
            return {
                'vram_current_mb': torch.cuda.memory_allocated(device) / 1e6,
                'vram_peak_mb': torch.cuda.max_memory_allocated(device) / 1e6,
                'vram_reserved_mb': torch.cuda.memory_reserved(device) / 1e6,
            }
        elif device.type == 'mps':
            # MPS doesn't expose memory stats directly
            # Return placeholder values
            return {
                'vram_current_mb': 0.0,
                'vram_peak_mb': 0.0,
                'vram_reserved_mb': 0.0,
            }
        else:
            # CPU
            return {
                'vram_current_mb': 0.0,
                'vram_peak_mb': 0.0,
                'vram_reserved_mb': 0.0,
            }

    @staticmethod
    def reset_peak_memory(device: torch.device):
        """Reset peak memory counter."""
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def compute_throughput(
        iter_num: int,
        start_time: float,
        batch_size: int,
        seq_len: int,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Compute throughput metrics.

        Args:
            iter_num: Current iteration number
            start_time: Training start time (from time.time())
            batch_size: Batch size per step
            seq_len: Sequence length
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Dictionary with throughput metrics
        """
        elapsed = time.time() - start_time

        if elapsed < 1e-6:
            return {
                'tokens_per_sec': 0.0,
                'samples_per_sec': 0.0,
                'steps_per_sec': 0.0,
            }

        # Total samples processed (accounting for gradient accumulation)
        total_samples = (iter_num + 1) * batch_size * gradient_accumulation_steps
        total_tokens = total_samples * seq_len
        total_steps = iter_num + 1

        return {
            'tokens_per_sec': total_tokens / elapsed,
            'samples_per_sec': total_samples / elapsed,
            'steps_per_sec': total_steps / elapsed,
        }

    @staticmethod
    def compute_timing_stats(
        step_times: list,
        window_size: int = 100,
    ) -> Dict[str, float]:
        """
        Compute timing statistics.

        Args:
            step_times: List of step durations (in seconds)
            window_size: Window for rolling average

        Returns:
            Dictionary with timing stats
        """
        if len(step_times) == 0:
            return {
                'ms_per_step_mean': 0.0,
                'ms_per_step_std': 0.0,
                'ms_per_step_min': 0.0,
                'ms_per_step_max': 0.0,
            }

        # Get recent window
        recent = step_times[-window_size:]
        recent_ms = [t * 1000 for t in recent]  # Convert to ms

        import numpy as np
        return {
            'ms_per_step_mean': float(np.mean(recent_ms)),
            'ms_per_step_std': float(np.std(recent_ms)),
            'ms_per_step_min': float(np.min(recent_ms)),
            'ms_per_step_max': float(np.max(recent_ms)),
        }

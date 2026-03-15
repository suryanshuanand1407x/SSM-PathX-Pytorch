"""
Gradient Diagnostics
====================
Tracks gradient norms and detects vanishing/exploding gradients.
"""

import torch
from typing import Dict, List


class GradientDiagnostics:
    """
    Monitors gradient flow through the model.

    Tracks:
    - Frobenius norm of gradients for key parameters (Δ, A)
    - Gradient statistics (mean, std, max)
    - Vanishing/exploding gradient detection
    """

    @staticmethod
    def compute_gradient_norms(
        model: torch.nn.Module,
        param_patterns: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compute Frobenius norm of gradients for specified parameters.

        Args:
            model: The model to analyze
            param_patterns: List of parameter name patterns to track.
                          Default: ['A_log', 'dt_proj', 'delta_norm']

        Returns:
            Dictionary mapping parameter names to gradient norms
        """
        if param_patterns is None:
            param_patterns = ['A_log', 'dt_proj', 'delta_norm', 'D']

        grad_norms = {}

        for name, param in model.named_parameters():
            # Check if this parameter matches any pattern
            if any(pattern in name for pattern in param_patterns):
                if param.grad is not None:
                    grad_norm = param.grad.norm(p=2).item()  # Frobenius norm
                    grad_norms[name] = grad_norm
                else:
                    grad_norms[name] = 0.0

        return grad_norms

    @staticmethod
    def compute_gradient_statistics(
        model: torch.nn.Module,
    ) -> Dict[str, float]:
        """
        Compute global gradient statistics across all parameters.

        Returns:
            Dictionary with mean, std, max, min gradient norms
        """
        all_grads = []

        for param in model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.flatten())

        if len(all_grads) == 0:
            return {
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'grad_max': 0.0,
                'grad_min': 0.0,
            }

        all_grads = torch.cat(all_grads)

        return {
            'grad_mean': all_grads.mean().item(),
            'grad_std': all_grads.std().item(),
            'grad_max': all_grads.abs().max().item(),
            'grad_min': all_grads.abs().min().item(),
        }

    @staticmethod
    def detect_gradient_issues(
        grad_norms: Dict[str, float],
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 100.0,
    ) -> Dict[str, bool]:
        """
        Detect vanishing or exploding gradients.

        Args:
            grad_norms: Dictionary of gradient norms
            vanishing_threshold: Threshold for vanishing gradients
            exploding_threshold: Threshold for exploding gradients

        Returns:
            Dictionary with boolean flags for issues
        """
        has_vanishing = any(v < vanishing_threshold for v in grad_norms.values())
        has_exploding = any(v > exploding_threshold for v in grad_norms.values())

        return {
            'has_vanishing_gradients': has_vanishing,
            'has_exploding_gradients': has_exploding,
        }

    @staticmethod
    def analyze_all(
        model: torch.nn.Module,
        param_patterns: List[str] = None,
    ) -> Dict[str, any]:
        """
        Compute all gradient diagnostics at once.

        Args:
            model: The model to analyze
            param_patterns: Parameter patterns to track

        Returns:
            Comprehensive gradient diagnostics
        """
        grad_norms = GradientDiagnostics.compute_gradient_norms(model, param_patterns)
        grad_stats = GradientDiagnostics.compute_gradient_statistics(model)
        grad_issues = GradientDiagnostics.detect_gradient_issues(grad_norms)

        return {
            'norms': grad_norms,
            'statistics': grad_stats,
            'issues': grad_issues,
        }

"""
Research Metrics Module
=======================
Comprehensive metric collection for Tustin Mamba paper.
"""

from .stability_metrics import StabilityAnalyzer
from .discretization_error import DiscretizationErrorAnalyzer
from .gradient_diagnostics import GradientDiagnostics
from .performance_metrics import PerformanceMonitor

__all__ = [
    'StabilityAnalyzer',
    'DiscretizationErrorAnalyzer',
    'GradientDiagnostics',
    'PerformanceMonitor',
]

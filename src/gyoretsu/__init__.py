"""Gyoretsu: differentiable discrete-event simulation in JAX.

Phase 0 (prototype) surface: a differentiable single-server FIFO queue, the smoothing
operators that make discrete choices differentiable, gradient-estimator baselines, and
closed-form queueing results used for validation.
"""

from gyoretsu import dist, grad, smooth, stats, theory
from gyoretsu.dist import Deterministic, Exponential, Gamma, LogNormal
from gyoretsu.engine import QueueSample, kiefer_wolfowitz, lindley, simulate_gg1, simulate_ggc
from gyoretsu.priority import PrioritySample, priority2, simulate_priority2

__version__ = "0.0.1"

__all__ = [
    "Deterministic",
    "Exponential",
    "Gamma",
    "LogNormal",
    "PrioritySample",
    "QueueSample",
    "dist",
    "grad",
    "kiefer_wolfowitz",
    "lindley",
    "priority2",
    "simulate_gg1",
    "simulate_ggc",
    "simulate_priority2",
    "smooth",
    "stats",
    "theory",
]

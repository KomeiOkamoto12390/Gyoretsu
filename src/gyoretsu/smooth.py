"""Smooth surrogates for the discrete choices inside a simulation.

Every operator takes a temperature ``tau``:

* ``tau == 0`` (a Python number) gives the exact, hard operation.
* ``tau > 0`` gives a smooth relaxation that converges to the hard one as ``tau -> 0``.
  ``tau`` may be a traced JAX value, so annealing schedules do not force recompilation.
* ``straight_through=True`` keeps the forward value exact and uses the smooth
  relaxation only for the backward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike


def is_exact(tau: ArrayLike) -> bool:
    """True when ``tau`` is a concrete Python zero, i.e. exact (unsmoothed) mode."""
    if isinstance(tau, (int, float)):
        if tau < 0:
            raise ValueError(f"temperature tau must be >= 0, got {tau}")
        return tau == 0
    return False


def straight_through(hard: Array, soft: Array) -> Array:
    """Forward value of ``hard``, gradient of ``soft``."""
    return soft + jax.lax.stop_gradient(hard - soft)


def smooth_relu(x: ArrayLike, tau: ArrayLike = 0.0, *, straight_through: bool = False) -> Array:
    """``max(x, 0)``, relaxed to ``tau * softplus(x / tau)``.

    The relaxation is an upper bound on ``max(x, 0)`` with gap at most ``tau * log(2)``.
    """
    x = jnp.asarray(x)
    hard = jnp.maximum(x, 0.0)
    if is_exact(tau):
        return hard
    soft = tau * jax.nn.softplus(x / tau)
    return _ste(hard, soft) if straight_through else soft


def smooth_max(
    a: ArrayLike, b: ArrayLike, tau: ArrayLike = 0.0, *, straight_through: bool = False
) -> Array:
    """``max(a, b)`` with the same relaxation as :func:`smooth_relu`."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    return b + smooth_relu(a - b, tau, straight_through=straight_through)


def soft_min(
    x: ArrayLike, tau: ArrayLike = 0.0, axis: int = -1, *, straight_through: bool = False
) -> Array:
    """``min(x, axis)``, relaxed to ``-tau * logsumexp(-x / tau)``.

    Used for next-event selection. The relaxation is a lower bound on the minimum with
    gap at most ``tau * log(n)``.
    """
    x = jnp.asarray(x)
    hard = jnp.min(x, axis=axis)
    if is_exact(tau):
        return hard
    soft = -tau * logsumexp(-x / tau, axis=axis)
    return _ste(hard, soft) if straight_through else soft


def soft_argmin_weights(x: ArrayLike, tau: ArrayLike = 0.0, axis: int = -1) -> Array:
    """Weights over candidates: one-hot argmin when exact, ``softmax(-x / tau)`` otherwise."""
    x = jnp.asarray(x)
    if is_exact(tau):
        n = x.shape[axis]
        return jax.nn.one_hot(jnp.argmin(x, axis=axis), n, axis=axis, dtype=x.dtype)
    return jax.nn.softmax(-x / tau, axis=axis)


_ste = straight_through

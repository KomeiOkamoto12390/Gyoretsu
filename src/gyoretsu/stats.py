"""Output analysis across independent replications."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtri
from jax.typing import ArrayLike


def mean_ci(x: ArrayLike, level: float = 0.95, axis: int = 0) -> tuple[Array, Array]:
    """Mean and normal-approximation confidence half-width across replications."""
    x = jnp.asarray(x)
    n = x.shape[axis]
    if n < 2:
        raise ValueError("need at least two replications for a confidence interval")
    z = ndtri(0.5 + level / 2.0)
    se = jnp.std(x, axis=axis, ddof=1) / jnp.sqrt(n)
    return jnp.mean(x, axis=axis), z * se

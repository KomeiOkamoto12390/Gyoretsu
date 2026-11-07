"""Closed-form queueing results used as ground truth in validation.

All functions take the mean interarrival time ``a`` and mean service time ``s`` (so the
utilization is ``rho = s / a`` for one server, ``s / (c a)`` for c servers) and work on
Python floats or JAX arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _check_stable(a: ArrayLike, s: ArrayLike) -> None:
    if isinstance(a, (int, float)) and isinstance(s, (int, float)) and s >= a:
        raise ValueError(f"unstable queue: utilization s/a = {s / a:.3f} >= 1")


def mg1_wait(a: ArrayLike, s: ArrayLike, scv: ArrayLike) -> Array:
    """Pollaczek-Khinchine steady-state mean wait in queue for M/G/1.

    ``Wq = lambda * E[S^2] / (2 (1 - rho)) = s^2 (1 + scv) / (2 (a - s))``.
    """
    _check_stable(a, s)
    a, s, scv = jnp.asarray(a), jnp.asarray(s), jnp.asarray(scv)
    return s**2 * (1.0 + scv) / (2.0 * (a - s))


def mm1_wait(a: ArrayLike, s: ArrayLike) -> Array:
    """Steady-state mean wait in queue for M/M/1: ``Wq = s^2 / (a - s)``."""
    return mg1_wait(a, s, 1.0)


def mm1_wait_grad(a: ArrayLike, s: ArrayLike) -> Array:
    """``[dWq/da, dWq/ds]`` for M/M/1, written out by hand."""
    _check_stable(a, s)
    a, s = jnp.asarray(a), jnp.asarray(s)
    d = (a - s) ** 2
    return jnp.stack([-(s**2) / d, s * (2.0 * a - s) / d])


def erlang_c(servers: int, offered_load: ArrayLike) -> Array:
    """Probability an arriving customer waits in M/M/c (Erlang C).

    ``offered_load`` is ``A = s / a``. Computed via the numerically stable Erlang B
    recursion ``B_k = A B_{k-1} / (k + A B_{k-1})``, then ``C = c B / (c - A (1 - B))``.
    """
    if not isinstance(servers, int) or servers < 1:
        raise ValueError(f"servers must be a positive Python int, got {servers!r}")
    load = jnp.asarray(offered_load)
    b = jnp.ones_like(load)
    for k in range(1, servers + 1):
        b = load * b / (k + load * b)
    return servers * b / (servers - load * (1.0 - b))


def mmc_wait(a: ArrayLike, s: ArrayLike, servers: int) -> Array:
    """Steady-state mean wait in queue for M/M/c: ``Wq = C(c, A) * s / (c - A)``."""
    if isinstance(a, (int, float)) and isinstance(s, (int, float)) and s >= servers * a:
        raise ValueError(f"unstable queue: utilization s/(c a) = {s / (servers * a):.3f} >= 1")
    a, s = jnp.asarray(a), jnp.asarray(s)
    load = s / a
    return erlang_c(servers, load) * s / (servers - load)

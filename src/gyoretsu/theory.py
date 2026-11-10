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


def priority_waits(
    a: tuple[ArrayLike, ...], s: tuple[ArrayLike, ...], scv: tuple[ArrayLike, ...] | None = None
) -> Array:
    """Cobham's formula: mean wait per class in an M/G/1 non-preemptive priority queue.

    Class 0 has the highest priority. ``a[k]``, ``s[k]`` and ``scv[k]`` are class ``k``'s
    mean interarrival time, mean service time and squared coefficient of variation of
    service (default 1, exponential). With ``sigma_k = rho_0 + ... + rho_k`` and residual
    work ``W0 = sum_i lambda_i E[S_i^2] / 2``:

        W_k = W0 / ((1 - sigma_{k-1}) (1 - sigma_k)).
    """
    a = [jnp.asarray(x) for x in a]
    s = [jnp.asarray(x) for x in s]
    scv = [jnp.asarray(1.0)] * len(s) if scv is None else [jnp.asarray(x) for x in scv]
    rho = [si / ai for ai, si in zip(a, s, strict=True)]
    w0 = sum(si**2 * (1.0 + ci) / (2.0 * ai) for ai, si, ci in zip(a, s, scv, strict=True))
    out, sigma_prev = [], 0.0
    for r in rho:
        sigma = sigma_prev + r
        out.append(w0 / ((1.0 - sigma_prev) * (1.0 - sigma)))
        sigma_prev = sigma
    return jnp.stack(out)

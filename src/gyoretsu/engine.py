"""Vectorized event engine.

Phase 0 implements the single-server FIFO queue through the Lindley recursion

    W[0] = 0,    W[n+1] = max(0, W[n] + S[n] - A[n+1])

where ``W`` is the wait in queue, ``S`` the service time and ``A[n]`` the gap between
arrivals ``n-1`` and ``n``. Each replication is one ``lax.scan``; replications are
batched with ``vmap``. In exact mode, reverse-mode autodiff through this recursion
reproduces the classic IPA estimator.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from gyoretsu.dist import Distribution
from gyoretsu.smooth import smooth_relu

# Event steps fused per scan iteration. Sequential event loops are launch-latency bound on
# GPU; unrolling 32 steps made 10k replications ~60x faster on an RTX 4050 (Nov 2026) and
# leaves CPU speed unchanged. Larger values mainly increase compile time.
DEFAULT_UNROLL = 32


class QueueSample(NamedTuple):
    """Batched sample paths, each array shaped ``(replications, n_customers)``."""

    interarrivals: Array
    services: Array
    waits: Array

    @property
    def sojourns(self) -> Array:
        """Time in system: wait in queue plus service."""
        return self.waits + self.services

    @property
    def arrival_times(self) -> Array:
        """Arrival epochs, with the first customer arriving at time 0."""
        return jnp.cumsum(self.interarrivals.at[..., 0].set(0.0), axis=-1)

    def mean_wait(self, warmup: int = 0) -> Array:
        """Mean wait per replication, discarding the first ``warmup`` customers."""
        return self.waits[..., warmup:].mean(axis=-1)

    def mean_sojourn(self, warmup: int = 0) -> Array:
        """Mean time in system per replication, discarding the first ``warmup`` customers."""
        return self.sojourns[..., warmup:].mean(axis=-1)


def lindley(
    interarrivals: Array,
    services: Array,
    tau: ArrayLike = 0.0,
    *,
    straight_through: bool = False,
    unroll: int = DEFAULT_UNROLL,
) -> Array:
    """Waits in queue for one replication of a FIFO single-server queue.

    Args:
        interarrivals: shape ``(n,)``; entry ``k`` is the gap before arrival ``k``.
            Entry 0 is ignored (the first customer arrives to an empty system).
        services: shape ``(n,)`` service times.
        tau: smoothing temperature; 0 is exact.
        straight_through: exact forward pass with smoothed gradients.
        unroll: event steps fused per scan iteration (performance only).
    """
    w0 = jnp.zeros((), dtype=services.dtype)

    def step(w: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        s, a_next = x
        w_next = smooth_relu(w + s - a_next, tau, straight_through=straight_through)
        return w_next, w_next

    _, ws = jax.lax.scan(step, w0, (services[:-1], interarrivals[1:]), unroll=unroll)
    return jnp.concatenate([w0[None], ws])


def simulate_gg1(
    key: Array,
    interarrival: Distribution,
    service: Distribution,
    *,
    n_customers: int,
    replications: int = 1,
    tau: ArrayLike = 0.0,
    straight_through: bool = False,
    unroll: int = DEFAULT_UNROLL,
) -> QueueSample:
    """Simulate ``replications`` independent runs of a G/G/1 FIFO queue.

    Reusing ``key`` across parameter values gives common random numbers, so differences
    and finite-difference gradients have low variance.
    """
    if n_customers < 2:
        raise ValueError("n_customers must be at least 2")
    k_arr, k_svc = jax.random.split(key)
    size = (replications, n_customers)
    a = interarrival.sample(k_arr, size)
    s = service.sample(k_svc, size)
    sim = partial(lindley, tau=tau, straight_through=straight_through, unroll=unroll)
    w = jax.vmap(sim)(a, s)
    return QueueSample(interarrivals=a, services=s, waits=w)

"""Vectorized event engine.

Single server (G/G/1): the Lindley recursion

    W[0] = 0,    W[n+1] = max(0, W[n] + S[n] - A[n+1])

where ``W`` is the wait in queue, ``S`` the service time and ``A[n]`` the gap between
arrivals ``n-1`` and ``n``.

Multiple identical servers (G/G/c, FIFO): the Kiefer-Wolfowitz recursion on the sorted
vector ``V`` of remaining workload per server,

    W[n] = V[n][0],    V[n+1] = sort(max(0, V[n] + S[n] e_0 - A[n+1])).

Customer ``n`` joins the least-loaded server. Because servers are identical, ties in the
sort are harmless (permuting equal entries leaves ``V`` unchanged), so the sample path is
continuous in the parameters and exact autodiff gives pathwise (IPA) gradients without
any smoothing of the server choice.

Each replication is one ``lax.scan``; replications are batched with ``vmap``.
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


def kiefer_wolfowitz(
    interarrivals: Array,
    services: Array,
    servers: int,
    tau: ArrayLike = 0.0,
    *,
    straight_through: bool = False,
    unroll: int = DEFAULT_UNROLL,
) -> Array:
    """Waits in queue for one replication of a FIFO queue with ``servers`` identical servers.

    Same conventions as :func:`lindley`, which it reduces to when ``servers == 1``.
    ``servers`` must be a Python int. Cost per customer is a sort of ``servers`` values.
    """
    if not isinstance(servers, int) or servers < 1:
        raise ValueError(f"servers must be a positive Python int, got {servers!r}")
    v0 = jnp.zeros((servers,), dtype=services.dtype)
    # Gap to the *next* arrival; the last customer's is unused.
    gaps = jnp.concatenate([interarrivals[1:], jnp.zeros((1,), interarrivals.dtype)])

    def step(v: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        s, a_next = x
        w = v[0]
        v = jnp.sort(v.at[0].add(s))
        v = smooth_relu(v - a_next, tau, straight_through=straight_through)
        return v, w

    _, ws = jax.lax.scan(step, v0, (services, gaps), unroll=unroll)
    return ws


def simulate_ggc(
    key: Array,
    interarrival: Distribution,
    service: Distribution,
    *,
    servers: int,
    n_customers: int,
    replications: int = 1,
    tau: ArrayLike = 0.0,
    straight_through: bool = False,
    unroll: int = DEFAULT_UNROLL,
) -> QueueSample:
    """Simulate ``replications`` independent runs of a G/G/c FIFO queue.

    Reusing ``key`` across parameter values gives common random numbers, so differences
    and finite-difference gradients have low variance. ``servers == 1`` uses the Lindley
    recursion, which is cheaper.
    """
    if n_customers < 2:
        raise ValueError("n_customers must be at least 2")
    k_arr, k_svc = jax.random.split(key)
    size = (replications, n_customers)
    a = interarrival.sample(k_arr, size)
    s = service.sample(k_svc, size)
    opts = dict(tau=tau, straight_through=straight_through, unroll=unroll)
    if servers == 1:
        sim = partial(lindley, **opts)
    else:
        sim = partial(kiefer_wolfowitz, servers=servers, **opts)
    w = jax.vmap(sim)(a, s)
    return QueueSample(interarrivals=a, services=s, waits=w)


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
    """Simulate ``replications`` independent runs of a G/G/1 FIFO queue."""
    return simulate_ggc(
        key,
        interarrival,
        service,
        servers=1,
        n_customers=n_customers,
        replications=replications,
        tau=tau,
        straight_through=straight_through,
        unroll=unroll,
    )

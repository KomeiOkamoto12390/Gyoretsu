"""Two-class non-preemptive priority queue with one server.

Class 0 has priority over class 1; within a class, service is FIFO. When the server frees
up at time ``t``, it serves the class-0 head-of-line customer if it has arrived (``T0 <=
t``); otherwise the class-1 head if it has arrived; otherwise whichever arrives first.
Both cases reduce to one test on the decision margin

    d = max(t, T1) - T0,    serve class 0  iff  d >= 0.

This is the first model where small parameter changes swap the order of events: when a
class-0 arrival crosses the moment the server frees up, a waiting class-1 customer is
served before or after it, and both customers' waits jump by a full service time. Exact
autodiff (``tau == 0``) cannot see these jumps. With ``tau > 0`` the forward path stays
exact and the backward pass uses ``sigmoid(d / tau)`` as the choice weight (a
straight-through estimator), which adds a gradient term for paths within ~``tau`` of a
swap.

Each replication runs ``n_customers`` service steps against ``n_customers`` generated
arrivals per class. Each step serves exactly one customer, so no class can run out.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from gyoretsu.dist import Distribution
from gyoretsu.engine import resolve_unroll
from gyoretsu.smooth import is_exact
from gyoretsu.smooth import straight_through as _ste


class PrioritySample(NamedTuple):
    """Batched results, arrays shaped ``(replications, steps)``.

    ``served0`` is True where the step served class 0. ``wait0`` and ``wait1`` hold the
    wait of the customer served at that step in its class's column and 0 in the other.
    Their forward values are exact; with ``tau > 0`` their gradients are smoothed.
    """

    served0: Array
    wait0: Array
    wait1: Array

    def mean_waits(self, warmup: int = 0) -> Array:
        """Per-replication mean wait of each class, shape ``(replications, 2)``."""
        m0 = self.served0[..., warmup:].astype(self.wait0.dtype)
        m1 = 1.0 - m0
        w0 = self.wait0[..., warmup:].sum(-1) / m0.sum(-1)
        w1 = self.wait1[..., warmup:].sum(-1) / m1.sum(-1)
        return jnp.stack([w0, w1], axis=-1)


def priority2(
    arrivals0: Array,
    arrivals1: Array,
    services0: Array,
    services1: Array,
    tau: ArrayLike = 0.0,
    *,
    unroll: int | None = None,
) -> tuple[Array, Array, Array]:
    """One replication. Arrival *times* (sorted) and services per class, each shape ``(n,)``.

    Returns ``(served0, wait0, wait1)``, each shape ``(n,)`` (one entry per service step).
    """
    exact = is_exact(tau)

    def step(carry, _):
        t, i0, i1 = carry
        t0, t1 = arrivals0[i0], arrivals1[i1]
        d = jnp.maximum(t, t1) - t0
        pick0 = d >= 0
        start0, start1 = jnp.maximum(t, t0), jnp.maximum(t, t1)
        w0, w1 = start0 - t0, start1 - t1
        end0, end1 = start0 + services0[i0], start1 + services1[i1]
        t_next = jnp.where(pick0, end0, end1)
        c0 = jnp.where(pick0, w0, 0.0)
        c1 = jnp.where(pick0, 0.0, w1)
        if not exact:
            p = jax.nn.sigmoid(d / tau)
            t_next = _ste(t_next, p * end0 + (1.0 - p) * end1)
            c0 = _ste(c0, p * w0)
            c1 = _ste(c1, (1.0 - p) * w1)
        carry = (t_next, i0 + pick0, i1 + (1 - pick0))
        return carry, (pick0, c0, c1)

    zero = jnp.zeros((), arrivals0.dtype)
    idx = jnp.zeros((), jnp.int32)
    _, out = jax.lax.scan(
        step, (zero, idx, idx), None, length=arrivals0.shape[0], unroll=resolve_unroll(unroll)
    )
    return out


def simulate_priority2(
    key: Array,
    interarrivals: tuple[Distribution, Distribution],
    services: tuple[Distribution, Distribution],
    *,
    n_customers: int,
    replications: int = 1,
    tau: ArrayLike = 0.0,
    unroll: int | None = None,
) -> PrioritySample:
    """Simulate ``replications`` runs of the two-class priority queue.

    ``interarrivals[k]`` and ``services[k]`` describe class ``k`` (class 0 has priority).
    Reusing ``key`` gives common random numbers across parameter values.
    """
    keys = jax.random.split(key, 4)
    size = (replications, n_customers)
    arr0 = jnp.cumsum(interarrivals[0].sample(keys[0], size), axis=-1)
    arr1 = jnp.cumsum(interarrivals[1].sample(keys[1], size), axis=-1)
    svc0 = services[0].sample(keys[2], size)
    svc1 = services[1].sample(keys[3], size)
    run = jax.vmap(lambda a0, a1, s0, s1: priority2(a0, a1, s0, s1, tau, unroll=unroll))
    served0, w0, w1 = run(arr0, arr1, svc0, svc1)
    return PrioritySample(served0=served0, wait0=w0, wait1=w1)

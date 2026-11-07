"""Gradient-estimator baselines used to validate pathwise (autodiff) gradients."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def central_difference(
    f: Callable[[Array], ArrayLike], theta: ArrayLike, rel_step: float = 1e-3
) -> Array:
    """Central finite-difference gradient of a scalar function.

    ``f`` should reuse a fixed PRNG key (common random numbers); otherwise the estimate
    is dominated by simulation noise. Costs ``2 * theta.size`` evaluations of ``f``.
    """
    theta = jnp.asarray(theta, dtype=jnp.result_type(float))
    flat = theta.ravel()
    out = []
    for i in range(flat.size):
        h = rel_step * jnp.maximum(1.0, jnp.abs(flat[i]))
        e = jnp.zeros_like(flat).at[i].set(h)
        up = f((flat + e).reshape(theta.shape))
        down = f((flat - e).reshape(theta.shape))
        out.append((jnp.asarray(up) - jnp.asarray(down)) / (2.0 * h))
    return jnp.stack(out).reshape(theta.shape)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> Array:
    a, b = jnp.ravel(jnp.asarray(a)), jnp.ravel(jnp.asarray(b))
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def lindley_ipa(
    interarrivals: Array,
    services: Array,
    d_interarrivals: Array,
    d_services: Array,
) -> tuple[Array, Array]:
    """Hand-written infinitesimal perturbation analysis for the Lindley recursion.

    Propagates ``dW[n+1] = 1{W[n] + S[n] - A[n+1] > 0} * (dW[n] + dS[n] - dA[n+1])``
    for one replication, given the derivatives of the inputs with respect to a single
    parameter (for a scale parameter ``m``, ``dX/dm = X / m``).

    Returns ``(waits, d_waits)``. Kept independent of autodiff so it can serve as a check.
    """
    zero = jnp.zeros((), dtype=services.dtype)

    def step(carry, x):
        w, dw = carry
        s, a, ds, da = x
        z = w + s - a
        busy = z > 0
        w_next = jnp.where(busy, z, 0.0)
        dw_next = jnp.where(busy, dw + ds - da, 0.0)
        return (w_next, dw_next), (w_next, dw_next)

    xs = (services[:-1], interarrivals[1:], d_services[:-1], d_interarrivals[1:])
    _, (ws, dws) = jax.lax.scan(step, (zero, zero), xs)
    return jnp.concatenate([zero[None], ws]), jnp.concatenate([zero[None], dws])

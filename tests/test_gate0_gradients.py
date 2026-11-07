"""Gate 0 (prototype, Nov-Dec 2026): pathwise gradients of a single-server queue.

Pass criterion from the proposal: pathwise gradients agree with finite differences
(cosine similarity >= 0.95). We additionally check them against the closed-form M/M/1
derivative and against a hand-written IPA estimator, and show smoothing bias -> 0.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from gyoretsu import Exponential, simulate_gg1
from gyoretsu.grad import central_difference, cosine_similarity, lindley_ipa
from gyoretsu.stats import mean_ci
from gyoretsu.theory import mm1_wait_grad

N, WARMUP, B = 20_000, 2_000, 64
THETA = jnp.array([1.0, 0.5])  # [mean interarrival, mean service], rho = 0.5
KEY = jax.random.PRNGKey(2026)


@partial(jax.jit, static_argnames=("n", "reps", "tau", "straight_through"))
def mean_wait(theta, key, n=N, reps=B, tau=0.0, straight_through=False):
    sample = simulate_gg1(
        key,
        Exponential(theta[0]),
        Exponential(theta[1]),
        n_customers=n,
        replications=reps,
        tau=tau,
        straight_through=straight_through,
    )
    return sample.mean_wait(WARMUP).mean()


grad_mean_wait = jax.jit(
    jax.grad(mean_wait), static_argnames=("n", "reps", "tau", "straight_through")
)


def test_pathwise_gradient_matches_closed_form():
    keys = jax.random.split(KEY, B)
    per_rep = jax.vmap(lambda k: grad_mean_wait(THETA, k, reps=1))(keys)  # (B, 2)
    mean, half = mean_ci(per_rep, level=0.999)
    truth = mm1_wait_grad(1.0, 0.5)  # [-1, 3]
    # CI half-width plus 2% allowance for finite-horizon bias.
    assert jnp.all(jnp.abs(mean - truth) <= half + 0.02 * jnp.abs(truth)), (mean, half)


def test_pathwise_gradient_matches_finite_difference():
    g = grad_mean_wait(THETA, KEY)
    fd = central_difference(lambda t: mean_wait(t, KEY), THETA)
    assert float(cosine_similarity(g, fd)) >= 0.95  # proposal's success criterion
    assert jnp.allclose(g, fd, rtol=2e-2), (g, fd)


def test_autodiff_reproduces_hand_written_ipa():
    n, reps = 2_000, 4
    sample = simulate_gg1(
        KEY, Exponential(THETA[0]), Exponential(THETA[1]), n_customers=n, replications=reps
    )
    # Derivative w.r.t. the service mean: dS/ds = S / s, interarrivals unaffected.
    _, dw = jax.vmap(lindley_ipa)(
        sample.interarrivals,
        sample.services,
        jnp.zeros_like(sample.interarrivals),
        sample.services / THETA[1],
    )
    ipa = dw.mean()
    auto = jax.grad(
        lambda t: simulate_gg1(
            KEY, Exponential(t[0]), Exponential(t[1]), n_customers=n, replications=reps
        ).waits.mean()
    )(THETA)[1]
    assert float(auto) == pytest.approx(float(ipa), rel=1e-4)


def test_smoothing_bias_vanishes_as_tau_decreases():
    n, reps = 5_000, 16
    exact = float(mean_wait(THETA, KEY, n=n, reps=reps))
    taus = [0.1, 0.03, 0.01, 0.003]
    vals = [float(mean_wait(THETA, KEY, n=n, reps=reps, tau=t)) for t in taus]
    # Softplus is a pathwise upper bound and increasing in tau, so bias is monotone.
    assert all(v >= exact - 1e-5 for v in vals)
    assert all(a >= b - 1e-5 for a, b in zip(vals, vals[1:], strict=False))
    assert vals[-1] - exact < 0.01 * exact

    g_exact = grad_mean_wait(THETA, KEY, n=n, reps=reps)
    g_soft = grad_mean_wait(THETA, KEY, n=n, reps=reps, tau=1e-3)
    assert float(jnp.linalg.norm(g_soft - g_exact) / jnp.linalg.norm(g_exact)) < 0.05


def test_straight_through_keeps_forward_exact():
    n, reps = 5_000, 16
    exact = mean_wait(THETA, KEY, n=n, reps=reps)
    st = mean_wait(THETA, KEY, n=n, reps=reps, tau=0.05, straight_through=True)
    assert float(st) == pytest.approx(float(exact), rel=1e-5)
    g_st = grad_mean_wait(THETA, KEY, n=n, reps=reps, tau=0.05, straight_through=True)
    assert float(cosine_similarity(g_st, grad_mean_wait(THETA, KEY, n=n, reps=reps))) > 0.95

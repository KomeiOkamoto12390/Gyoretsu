"""Two-class non-preemptive priority queue: forward correctness and the gradient finding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyoretsu import Exponential, priority2, simulate_priority2
from gyoretsu.stats import mean_ci
from gyoretsu.theory import priority_waits

THETA = jnp.array([1 / 0.3, 2.5, 1.0, 1.0])  # [a0, a1, s0, s1]; rho0 = 0.3, rho1 = 0.4
KEY = jax.random.PRNGKey(5)


def _waits(th, key, n, reps, tau=0.0):
    s = simulate_priority2(
        key,
        (Exponential(th[0]), Exponential(th[1])),
        (Exponential(th[2]), Exponential(th[3])),
        n_customers=n,
        replications=reps,
        tau=tau,
    )
    return s.mean_waits(n // 10)


def test_cobham_known_values():
    w = priority_waits((1 / 0.3, 2.5), (1.0, 1.0))
    assert jnp.allclose(w, jnp.array([1.0, 10 / 3]), rtol=1e-5)
    # One class reduces to M/M/1: Wq = s^2 / (a - s).
    assert float(priority_waits((1.0,), (0.5,))[0]) == pytest.approx(0.5)


def _simpy_priority_waits(arr0, arr1, svc0, svc1):
    simpy = pytest.importorskip("simpy")
    env = simpy.Environment()
    server = simpy.PriorityResource(env, capacity=1)
    waits = [np.zeros(len(arr0)), np.zeros(len(arr1))]

    def customer(cls, i, service):
        arrived = env.now
        with server.request(priority=cls) as req:  # lower number = higher priority
            yield req
            waits[cls][i] = env.now - arrived
            yield env.timeout(service)

    def source(cls, times, services):
        last = 0.0
        for i, (t, s) in enumerate(zip(times, services, strict=True)):
            yield env.timeout(t - last)
            last = t
            env.process(customer(cls, i, s))

    env.process(source(0, arr0, svc0))
    env.process(source(1, arr1, svc1))
    env.run()
    return waits


def test_waits_match_simpy_priority_resource():
    k = jax.random.split(KEY, 4)
    n = 800
    arr0 = jnp.cumsum(Exponential(1 / 0.35).sample(k[0], (n,)))
    arr1 = jnp.cumsum(Exponential(1 / 0.5).sample(k[1], (n,)))
    svc0 = Exponential(1.0).sample(k[2], (n,))
    svc1 = Exponential(1.0).sample(k[3], (n,))  # rho = 0.85: frequent priority decisions
    served0, w0, w1 = (np.asarray(x) for x in priority2(arr0, arr1, svc0, svc1))
    ref0, ref1 = _simpy_priority_waits(*(np.asarray(x, float) for x in (arr0, arr1, svc0, svc1)))
    ours0, ours1 = w0[served0], w1[~served0]  # in service order = FIFO index order
    assert len(ours0) > 100 and len(ours1) > 100
    np.testing.assert_allclose(ours0, ref0[: len(ours0)], atol=5e-4)
    np.testing.assert_allclose(ours1, ref1[: len(ours1)], atol=5e-4)


def test_steady_state_waits_match_cobham():
    per_rep = _waits(THETA, KEY, 10_000, 64)  # (64, 2)
    mean, half = mean_ci(per_rep, level=0.999)
    truth = priority_waits(THETA[:2], THETA[2:])
    assert jnp.all(jnp.abs(mean - truth) <= half), (mean, half, truth)


def test_tau_does_not_change_forward_path():
    exact = _waits(THETA, KEY, 2_000, 4)
    smooth = _waits(THETA, KEY, 2_000, 4, tau=0.1)
    assert jnp.allclose(exact, smooth, rtol=1e-6)


def test_finite_differences_recover_cobham_gradient():
    """CRN finite differences are a valid gradient reference for this model."""
    f = jax.jit(lambda th: _waits(th, KEY, 5_000, 256).mean(0))
    truth = jax.jacrev(lambda t: priority_waits(t[:2], t[2:]))(THETA)
    fd = []
    for i in range(4):
        h = 0.02 * float(THETA[i])
        e = jnp.zeros(4).at[i].set(h)
        fd.append((f(THETA + e) - f(THETA - e)) / (2 * h))
    fd = jnp.stack(fd, axis=1)
    assert float(jnp.linalg.norm(fd - truth) / jnp.linalg.norm(truth)) < 0.05


def test_exact_pathwise_gradient_diverges_with_horizon():
    """Documents the Gate 0 finding: freezing the service order makes arrival-rate gradients
    grow linearly with the horizon instead of converging. If this test starts failing, the
    formulation changed and the gradient study in benchmarks/ must be re-run."""
    grad = jax.jit(
        jax.vmap(
            lambda k, n: jax.jacrev(lambda t: _waits(t, k, n, 1)[0])(THETA)[0, 0], in_axes=(0, None)
        ),
        static_argnums=1,
    )
    keys = jax.random.split(KEY, 64)
    g_short = float(grad(keys, 1_000).mean())
    g_long = float(grad(keys, 4_000).mean())
    assert g_long / g_short > 3.0  # ~4x: linear growth, true value is -0.257

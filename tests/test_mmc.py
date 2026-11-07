"""Multi-server FIFO queue (G/G/c): forward results and gradients."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyoretsu import Exponential, kiefer_wolfowitz, lindley, simulate_ggc
from gyoretsu.grad import central_difference, cosine_similarity
from gyoretsu.stats import mean_ci
from gyoretsu.theory import erlang_c, mm1_wait, mmc_wait

N, WARMUP, B = 20_000, 2_000, 64
RHO = 0.7
KEY = jax.random.PRNGKey(31)


def theta_for(c):
    return jnp.array([1.0, RHO * c])  # [mean interarrival, mean service]


@partial(jax.jit, static_argnames=("c", "n", "reps", "tau", "straight_through"))
def mean_wait(theta, key, c, n=N, reps=B, tau=0.0, straight_through=False):
    sample = simulate_ggc(
        key,
        Exponential(theta[0]),
        Exponential(theta[1]),
        servers=c,
        n_customers=n,
        replications=reps,
        tau=tau,
        straight_through=straight_through,
    )
    return sample.mean_wait(WARMUP).mean()


grad_mean_wait = jax.jit(
    jax.grad(mean_wait), static_argnames=("c", "n", "reps", "tau", "straight_through")
)


# --------------------------------------------------------------------------- theory


def test_erlang_c_known_values():
    assert float(erlang_c(1, 0.5)) == pytest.approx(0.5)  # M/M/1: P(wait) = rho
    assert float(erlang_c(2, 1.0)) == pytest.approx(1 / 3)
    assert float(mmc_wait(1.0, 1.0, 2)) == pytest.approx(1 / 3)
    assert float(mmc_wait(1.0, 0.5, 1)) == pytest.approx(float(mm1_wait(1.0, 0.5)))


def test_erlang_c_stable_for_many_servers():
    c = erlang_c(200, 180.0)  # naive A^c / c! overflows float32 here
    assert jnp.isfinite(c) and 0.0 < float(c) < 1.0


# --------------------------------------------------------------------------- forward


def test_one_server_reduces_to_lindley():
    k1, k2 = jax.random.split(KEY)
    inter = Exponential(1.0).sample(k1, (2_000,))
    serv = Exponential(0.8).sample(k2, (2_000,))
    assert jnp.allclose(kiefer_wolfowitz(inter, serv, 1), lindley(inter, serv), atol=1e-5)


def _simpy_waits(inter, serv, c):
    simpy = pytest.importorskip("simpy")
    env = simpy.Environment()
    pool = simpy.Resource(env, capacity=c)
    waits = np.zeros(len(serv))

    def customer(i):
        arrived = env.now
        with pool.request() as req:
            yield req
            waits[i] = env.now - arrived
            yield env.timeout(serv[i])

    def source():
        for i in range(len(inter)):
            if i > 0:
                yield env.timeout(inter[i])
            env.process(customer(i))

    env.process(source())
    env.run()
    return waits


@pytest.mark.parametrize("c", [2, 3, 5])
def test_waits_match_simpy(c):
    k1, k2 = jax.random.split(jax.random.PRNGKey(c))
    inter = Exponential(1.0).sample(k1, (1_000,))
    serv = Exponential(0.9 * c).sample(k2, (1_000,))  # rho = 0.9: plenty of queueing
    ref = _simpy_waits(np.asarray(inter, float), np.asarray(serv, float), c)
    assert ref.mean() > 0.5
    np.testing.assert_allclose(np.asarray(kiefer_wolfowitz(inter, serv, c)), ref, atol=2e-4)


@pytest.mark.parametrize("c", [2, 4, 8])
def test_steady_state_wait_matches_erlang_c(c):
    sample = simulate_ggc(
        KEY, Exponential(1.0), Exponential(RHO * c), servers=c, n_customers=N, replications=B
    )
    mean, half = mean_ci(sample.mean_wait(WARMUP), level=0.999)
    truth = float(mmc_wait(1.0, RHO * c, c))
    assert abs(float(mean) - truth) <= float(half), (float(mean), float(half), truth)


# --------------------------------------------------------------------------- gradients


@pytest.mark.parametrize("c", [2, 4])
def test_pathwise_gradient_matches_closed_form(c):
    theta = theta_for(c)
    keys = jax.random.split(KEY, B)
    per_rep = jax.vmap(lambda k: grad_mean_wait(theta, k, c, reps=1))(keys)
    mean, half = mean_ci(per_rep, level=0.999)
    truth = jax.grad(lambda t: mmc_wait(t[0], t[1], c))(theta)
    # CI half-width plus 2% allowance for finite-horizon bias.
    assert jnp.all(jnp.abs(mean - truth) <= half + 0.02 * jnp.abs(truth)), (mean, half, truth)


@pytest.mark.parametrize("c", [2, 4])
def test_pathwise_gradient_matches_finite_difference(c):
    theta = theta_for(c)
    g = grad_mean_wait(theta, KEY, c)
    fd = central_difference(lambda t: mean_wait(t, KEY, c), theta)
    assert float(cosine_similarity(g, fd)) >= 0.95  # proposal's success criterion
    assert jnp.allclose(g, fd, rtol=2e-2), (g, fd)


def test_smoothing_converges_to_exact():
    c, n, reps = 3, 5_000, 16
    theta = theta_for(c)
    exact = float(mean_wait(theta, KEY, c, n=n, reps=reps))
    errs = [
        abs(float(mean_wait(theta, KEY, c, n=n, reps=reps, tau=t)) - exact)
        for t in (0.1, 0.01, 0.001)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.01 * exact

    g_exact = grad_mean_wait(theta, KEY, c, n=n, reps=reps)
    g_soft = grad_mean_wait(theta, KEY, c, n=n, reps=reps, tau=1e-3)
    assert float(jnp.linalg.norm(g_soft - g_exact) / jnp.linalg.norm(g_exact)) < 0.05


def test_straight_through_keeps_forward_exact():
    c, n, reps = 3, 5_000, 16
    theta = theta_for(c)
    exact = mean_wait(theta, KEY, c, n=n, reps=reps)
    st = mean_wait(theta, KEY, c, n=n, reps=reps, tau=0.05, straight_through=True)
    assert float(st) == pytest.approx(float(exact), rel=1e-5)


def test_invalid_servers_rejected():
    x = jnp.ones(10)
    with pytest.raises(ValueError):
        kiefer_wolfowitz(x, x, 0)
    with pytest.raises(ValueError):
        kiefer_wolfowitz(x, x, 2.0)

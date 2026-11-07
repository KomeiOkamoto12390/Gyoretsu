"""Forward correctness: steady-state results vs queueing theory, and path-level cross-checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gyoretsu import Deterministic, Exponential, Gamma, LogNormal, lindley, simulate_gg1
from gyoretsu.stats import mean_ci
from gyoretsu.theory import mg1_wait

N, WARMUP, B = 20_000, 2_000, 64
A, S = 1.0, 0.5  # rho = 0.5


@pytest.mark.parametrize(
    "service",
    [Exponential(S), Deterministic(S), LogNormal(S, 0.25), Gamma(S, shape=4.0)],
    ids=["M/M/1", "M/D/1", "M/LN/1", "M/Gamma/1"],
)
def test_mg1_steady_state_wait(service):
    sample = simulate_gg1(
        jax.random.PRNGKey(1), Exponential(A), service, n_customers=N, replications=B
    )
    mean, half = mean_ci(sample.mean_wait(WARMUP), level=0.999)
    truth = float(mg1_wait(A, S, service.scv))
    assert abs(float(mean) - truth) <= float(half), (float(mean), float(half), truth)


def _reference_waits(inter, serv):
    """Textbook departure-time recursion in plain Python."""
    t, last_departure, waits = 0.0, 0.0, []
    for i, (a, s) in enumerate(zip(inter, serv, strict=True)):
        t = 0.0 if i == 0 else t + a
        start = max(t, last_departure)
        waits.append(start - t)
        last_departure = start + s
    return np.array(waits)


def _simpy_waits(inter, serv):
    simpy = pytest.importorskip("simpy")
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    waits = np.zeros(len(serv))

    def customer(i):
        arrived = env.now
        with server.request() as req:
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


@pytest.fixture(scope="module")
def small_path():
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    inter = Exponential(1.0).sample(k1, (500,))
    serv = Exponential(0.9).sample(k2, (500,))
    return inter, serv, lindley(inter, serv)


def test_lindley_matches_departure_recursion(small_path):
    inter, serv, waits = small_path
    ref = _reference_waits(np.asarray(inter, float), np.asarray(serv, float))
    np.testing.assert_allclose(np.asarray(waits), ref, rtol=1e-4, atol=1e-4)


def test_lindley_matches_simpy(small_path):
    inter, serv, waits = small_path
    ref = _simpy_waits(np.asarray(inter, float), np.asarray(serv, float))
    np.testing.assert_allclose(np.asarray(waits), ref, rtol=1e-4, atol=1e-4)


def test_same_key_gives_common_random_numbers():
    key = jax.random.PRNGKey(3)
    lo = simulate_gg1(key, Exponential(1.0), Exponential(0.5), n_customers=100)
    hi = simulate_gg1(key, Exponential(1.0), Exponential(0.6), n_customers=100)
    assert jnp.allclose(hi.services, lo.services * 1.2)
    assert jnp.all(hi.waits >= lo.waits - 1e-6)  # longer services never shorten waits


def test_unroll_does_not_change_results():
    """Unrolling is a pure performance knob, including when n is not a multiple of it."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(11))
    inter = Exponential(1.0).sample(k1, (1_001,))
    serv = Exponential(0.8).sample(k2, (1_001,))
    ref = lindley(inter, serv, unroll=1)
    for u in (8, 32, 64):
        assert jnp.allclose(lindley(inter, serv, unroll=u), ref, rtol=1e-6, atol=1e-6)

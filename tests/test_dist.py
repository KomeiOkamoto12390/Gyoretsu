import jax
import jax.numpy as jnp
import pytest

from gyoretsu.dist import Deterministic, Exponential, Gamma, LogNormal

KEY = jax.random.PRNGKey(0)
SIZE = (200_000,)

CASES = [
    (Exponential(2.0), 2.0, 1.0),
    (Deterministic(2.0), 2.0, 0.0),
    (LogNormal(2.0, 1.0), 2.0, 0.25),
    (Gamma(2.0, shape=4.0), 2.0, 0.25),
]


@pytest.mark.parametrize(("d", "mean", "scv"), CASES)
def test_moments(d, mean, scv):
    x = d.sample(KEY, SIZE)
    assert jnp.all(x >= 0)
    assert float(x.mean()) == pytest.approx(mean, rel=0.02)
    assert float(x.var() / x.mean() ** 2) == pytest.approx(scv, rel=0.05, abs=1e-6)
    assert float(d.scv) == pytest.approx(scv)


def test_noise_is_parameter_free():
    """Same key, different parameters -> same base noise (common random numbers)."""
    assert jnp.array_equal(Exponential(1.0).noise(KEY, (10,)), Exponential(5.0).noise(KEY, (10,)))
    assert jnp.array_equal(
        LogNormal(1.0, 0.5).noise(KEY, (10,)), LogNormal(3.0, 2.0).noise(KEY, (10,))
    )


@pytest.mark.parametrize(
    "make", [Exponential, Deterministic, lambda m: LogNormal(m, 1.0), lambda m: Gamma(m, 4.0)]
)
def test_reparameterized_mean_gradient(make):
    """d E[X] / d mean = 1 for every family, estimated pathwise."""
    g = jax.grad(lambda m: make(m).sample(KEY, SIZE).mean())(2.0)
    assert float(g) == pytest.approx(1.0, rel=0.02)

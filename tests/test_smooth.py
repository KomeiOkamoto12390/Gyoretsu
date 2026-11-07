import jax
import jax.numpy as jnp
import pytest

from gyoretsu.smooth import smooth_max, smooth_relu, soft_argmin_weights, soft_min

X = jnp.linspace(-3.0, 3.0, 61)


def test_exact_mode_is_hard():
    assert jnp.array_equal(smooth_relu(X, 0.0), jnp.maximum(X, 0.0))
    assert jnp.array_equal(smooth_max(X, 1.0, 0), jnp.maximum(X, 1.0))
    assert float(soft_min(X, 0.0)) == float(X.min())


@pytest.mark.parametrize("tau", [1.0, 0.1, 0.01])
def test_relu_relaxation_is_tight_upper_bound(tau):
    gap = smooth_relu(X, tau) - jnp.maximum(X, 0.0)
    assert jnp.all(gap >= -1e-6)
    assert jnp.all(gap <= tau * jnp.log(2.0) + 1e-6)


@pytest.mark.parametrize("tau", [1.0, 0.1, 0.01])
def test_soft_min_is_tight_lower_bound(tau):
    v = jnp.array([3.0, 1.0, 2.0, 1.5])
    s = float(soft_min(v, tau))
    assert s <= 1.0 + 1e-6
    assert s >= 1.0 - tau * float(jnp.log(4.0)) - 1e-6


def test_straight_through_forward_exact_backward_smooth():
    tau = 0.5
    st = smooth_relu(X, tau, straight_through=True)
    assert jnp.allclose(st, jnp.maximum(X, 0.0), atol=1e-6)
    g = jax.vmap(jax.grad(lambda x: smooth_relu(x, tau, straight_through=True)))(X)
    assert jnp.allclose(g, jax.nn.sigmoid(X / tau), atol=1e-6)


def test_tau_can_be_traced():
    f = jax.jit(lambda x, tau: smooth_relu(x, tau))
    assert jnp.allclose(f(X, 0.1), smooth_relu(X, 0.1))


def test_soft_argmin_weights():
    v = jnp.array([3.0, 1.0, 2.0])
    assert jnp.array_equal(soft_argmin_weights(v, 0.0), jnp.array([0.0, 1.0, 0.0]))
    w = soft_argmin_weights(v, 0.5)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert int(jnp.argmax(w)) == 1


def test_negative_tau_rejected():
    with pytest.raises(ValueError):
        smooth_relu(X, -0.1)

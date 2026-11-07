import jax
import jax.numpy as jnp
import pytest

from gyoretsu.theory import mg1_wait, mm1_wait, mm1_wait_grad


def test_mm1_known_value():
    # rho = 0.5: Wq = rho * s / (1 - rho) = 0.5
    assert float(mm1_wait(1.0, 0.5)) == pytest.approx(0.5)


def test_mg1_special_cases():
    assert float(mg1_wait(1.0, 0.5, 1.0)) == pytest.approx(float(mm1_wait(1.0, 0.5)))
    assert float(mg1_wait(1.0, 0.5, 0.0)) == pytest.approx(0.25)  # M/D/1 is half of M/M/1


def test_mm1_hand_gradient_matches_autodiff():
    theta = jnp.array([1.0, 0.5])
    auto = jax.grad(lambda t: mm1_wait(t[0], t[1]))(theta)
    hand = mm1_wait_grad(1.0, 0.5)
    assert jnp.allclose(auto, hand, rtol=1e-6)
    assert jnp.allclose(hand, jnp.array([-1.0, 3.0]))


def test_unstable_queue_raises():
    with pytest.raises(ValueError):
        mm1_wait(1.0, 1.0)

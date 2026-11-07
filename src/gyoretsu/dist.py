"""Reparameterized distributions.

Each distribution splits sampling into two steps:

1. ``noise(key, size)`` draws parameter-free base noise.
2. ``transform(noise)`` maps that noise to samples as a differentiable function of the
   distribution's parameters.

Because the noise never depends on parameters, two runs with the same PRNG key share
common random numbers, and every sample path is a deterministic, differentiable function
of the parameters (the reparameterization trick).

Parameters marked "static" below must be plain Python numbers; they shape the noise and
are not differentiable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

Size = tuple[int, ...]


class Distribution(ABC):
    """A nonnegative duration distribution with a reparameterized sampler."""

    @abstractmethod
    def noise(self, key: Array, size: Size) -> Array:
        """Parameter-free base noise of the given shape."""

    @abstractmethod
    def transform(self, noise: Array) -> Array:
        """Map base noise to samples; differentiable in the distribution's parameters."""

    @property
    @abstractmethod
    def scv(self) -> ArrayLike:
        """Squared coefficient of variation, Var[X] / E[X]^2."""

    def sample(self, key: Array, size: Size) -> Array:
        return self.transform(self.noise(key, size))


@dataclass(frozen=True)
class Exponential(Distribution):
    """Exponential with the given mean (differentiable)."""

    mean: ArrayLike

    def noise(self, key: Array, size: Size) -> Array:
        return jax.random.exponential(key, size)

    def transform(self, noise: Array) -> Array:
        return jnp.asarray(self.mean) * noise

    @property
    def scv(self) -> ArrayLike:
        return 1.0


@dataclass(frozen=True)
class Deterministic(Distribution):
    """A constant duration (differentiable)."""

    value: ArrayLike

    @property
    def mean(self) -> ArrayLike:
        return self.value

    def noise(self, key: Array, size: Size) -> Array:
        del key
        return jnp.zeros(size)

    def transform(self, noise: Array) -> Array:
        return noise + jnp.asarray(self.value)

    @property
    def scv(self) -> ArrayLike:
        return 0.0


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Lognormal parameterized by its own mean and standard deviation (both differentiable)."""

    mean: ArrayLike
    sd: ArrayLike

    def noise(self, key: Array, size: Size) -> Array:
        return jax.random.normal(key, size)

    def transform(self, noise: Array) -> Array:
        mean, sd = jnp.asarray(self.mean), jnp.asarray(self.sd)
        sigma2 = jnp.log1p((sd / mean) ** 2)
        mu = jnp.log(mean) - 0.5 * sigma2
        return jnp.exp(mu + jnp.sqrt(sigma2) * noise)

    @property
    def scv(self) -> ArrayLike:
        return (jnp.asarray(self.sd) / jnp.asarray(self.mean)) ** 2


@dataclass(frozen=True)
class Gamma(Distribution):
    """Gamma with differentiable ``mean`` and static ``shape`` (scv = 1 / shape)."""

    mean: ArrayLike
    shape: float

    def noise(self, key: Array, size: Size) -> Array:
        # Unit-mean gamma noise; depends only on the static shape.
        return jax.random.gamma(key, self.shape, size) / self.shape

    def transform(self, noise: Array) -> Array:
        return jnp.asarray(self.mean) * noise

    @property
    def scv(self) -> ArrayLike:
        return 1.0 / self.shape

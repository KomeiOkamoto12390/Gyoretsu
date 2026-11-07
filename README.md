# Gyoretsu

**Gyoretsu** (行列, "queue") is an open-source Python library that makes discrete-event
simulation differentiable. You write a queueing or process model once and, from a single
batch of runs, get the performance estimate, its gradient with respect to every input
parameter, and thousands of parallel replications on CPU or GPU. Built on JAX.

> Status: **Phase 0 prototype** (Nov–Dec 2026). The API is unstable. See [ROADMAP.md](ROADMAP.md).

## What works today

- A differentiable G/G/1 FIFO queue (Lindley recursion in `lax.scan`, batched with `vmap`)
- Reparameterized distributions (Exponential, Deterministic, LogNormal, Gamma) with common
  random numbers by default
- Smoothing operators with temperature `tau` (soft ReLU/max, soft-min, straight-through)
- Baselines: central finite differences and a hand-written IPA estimator
- Closed-form M/M/1 and M/G/1 results for validation

```python
import jax, jax.numpy as jnp
from gyoretsu import Exponential, simulate_gg1


def mean_wait(theta, key):
    s = simulate_gg1(
        key, Exponential(theta[0]), Exponential(theta[1]), n_customers=20_000, replications=256
    )
    return s.mean_wait(warmup=2_000).mean()


theta = jnp.array([1.0, 0.5])  # mean interarrival, mean service
jax.grad(mean_wait)(theta, jax.random.PRNGKey(0))  # ≈ [-1.0, 3.0] (M/M/1 theory)
```

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin/python elsewhere
python -m pytest
python examples/gate0_mm1_gradients.py           # Gate 0 report
```

GPU (CUDA) runs in WSL2, because JAX publishes CUDA wheels only for Linux. From WSL:

```bash
bash scripts/setup_wsl_gpu.sh   # venv in ~/.venvs/gyoretsu, installs .[dev,gpu], runs tests on GPU
```

## License

MIT

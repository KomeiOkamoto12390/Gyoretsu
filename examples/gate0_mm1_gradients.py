"""Gate 0 report: gradients of M/M/1 mean wait from every estimator, side by side.

Run:  python examples/gate0_mm1_gradients.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from gyoretsu import Exponential, simulate_gg1
from gyoretsu.grad import central_difference, cosine_similarity
from gyoretsu.stats import mean_ci
from gyoretsu.theory import mm1_wait, mm1_wait_grad

N, WARMUP, B = 20_000, 2_000, 256
THETA = jnp.array([1.0, 0.5])  # [mean interarrival, mean service]
KEY = jax.random.PRNGKey(0)


@partial(jax.jit, static_argnames=("tau", "reps"))
def mean_wait(theta, key, tau=0.0, reps=B):
    s = simulate_gg1(
        key,
        Exponential(theta[0]),
        Exponential(theta[1]),
        n_customers=N,
        replications=reps,
        tau=tau,
    )
    return s.mean_wait(WARMUP).mean()


grad = jax.jit(jax.grad(mean_wait), static_argnames=("tau", "reps"))


def timed(f):
    jax.block_until_ready(f())  # compile
    t0 = time.perf_counter()
    out = jax.block_until_ready(f())
    return out, time.perf_counter() - t0


def main():
    print(f"M/M/1, rho = {THETA[1] / THETA[0]:.2f}, {B} replications x {N} customers\n")
    w, t_fwd = timed(lambda: mean_wait(THETA, KEY))
    print(f"mean wait  sim {float(w):.4f}   theory {float(mm1_wait(1.0, 0.5)):.4f}")

    truth = mm1_wait_grad(1.0, 0.5)
    g, t_bwd = timed(lambda: grad(THETA, KEY))
    fd = central_difference(lambda t: mean_wait(t, KEY), THETA)
    per_rep = jax.vmap(lambda k: grad(THETA, k, reps=1))(jax.random.split(KEY, B))
    _, half = mean_ci(per_rep)

    print("\n                       dW/da      dW/ds")
    print(f"closed form        {truth[0]:9.4f}  {truth[1]:9.4f}")
    ci = f"(95% CI +/- {half[0]:.3f}, {half[1]:.3f})"
    print(f"pathwise (exact)   {g[0]:9.4f}  {g[1]:9.4f}   {ci}")
    print(f"finite difference  {fd[0]:9.4f}  {fd[1]:9.4f}")
    for tau in (0.1, 0.01, 0.001):
        gs = grad(THETA, KEY, tau=tau)
        print(f"smoothed tau={tau:<6} {gs[0]:9.4f}  {gs[1]:9.4f}")

    cos = float(cosine_similarity(g, fd))
    print(f"\ncosine(pathwise, FD) = {cos:.5f}  -> gate {'PASS' if cos >= 0.95 else 'FAIL'}")
    print(
        f"forward {t_fwd * 1e3:.1f} ms, gradient {t_bwd * 1e3:.1f} ms "
        f"({t_bwd / t_fwd:.1f}x forward)"
    )


if __name__ == "__main__":
    main()

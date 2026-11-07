"""Heavy benchmark: SimPy vs Gyoretsu on the same M/M/1 model.

Both simulators estimate the steady-state mean wait in queue from ``--reps`` independent
replications of ``--customers`` customers each, discarding ``--warmup`` customers.

* SimPy is written idiomatically (one process per customer, ``Resource`` for the server,
  ``random.expovariate`` draws) and run in parallel across all CPU cores with
  multiprocessing, which is the fastest way a SimPy user can run replications.
* Gyoretsu runs the same replications batched with ``vmap`` on every available JAX device,
  in chunks so a 6 GB GPU is never exceeded. Compile time is reported separately.

Gradients: SimPy has none, so its cost for a 2-parameter gradient is derived as the cost
of central finite differences (4 extra full runs). Gyoretsu's gradient is measured.

Run inside WSL for the GPU:
    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/simpy_vs_gyoretsu.py
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------- SimPy


def _simpy_replication(args: tuple[int, int, int, float, float]) -> float:
    import simpy

    seed, n, warmup, mean_inter, mean_service = args
    rng = random.Random(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    acc = [0.0, 0]

    def customer(i: int):
        arrived = env.now
        with server.request() as req:
            yield req
            if i >= warmup:
                acc[0] += env.now - arrived
                acc[1] += 1
            yield env.timeout(rng.expovariate(1.0 / mean_service))

    def source():
        for i in range(n):
            if i > 0:
                yield env.timeout(rng.expovariate(1.0 / mean_inter))
            env.process(customer(i))

    env.process(source())
    env.run()
    return acc[0] / acc[1]


def run_simpy(reps, n, warmup, a, s, workers, seed=12345):
    jobs = [(seed + r, n, warmup, a, s) for r in range(reps)]
    t0 = time.perf_counter()
    if workers == 1:
        out = [_simpy_replication(j) for j in jobs]
    else:
        # spawn, not fork: forking a process that has started JAX threads can deadlock.
        with mp.get_context("spawn").Pool(workers) as pool:
            out = pool.map(_simpy_replication, jobs, chunksize=max(1, reps // (workers * 8)))
    return np.asarray(out), time.perf_counter() - t0


# ---------------------------------------------------------------------------- Gyoretsu


def run_gyoretsu(device, reps, n, warmup, a, s, chunk):
    import jax
    import jax.numpy as jnp

    from gyoretsu import Exponential, simulate_gg1

    def per_rep_means(theta, key, size):
        sample = simulate_gg1(
            key, Exponential(theta[0]), Exponential(theta[1]), n_customers=n, replications=size
        )
        return sample.mean_wait(warmup)

    fwd = jax.jit(per_rep_means, static_argnums=2)
    grad = jax.jit(jax.grad(lambda th, k, size: per_rep_means(th, k, size).sum()), static_argnums=2)
    with jax.default_device(device):
        return _time_gyoretsu(fwd, grad, jnp.array([a, s]), reps, chunk)


def _time_gyoretsu(fwd, grad, theta, reps, chunk):
    import jax

    sizes = [min(chunk, reps - i) for i in range(0, reps, chunk)]
    keys = jax.random.split(jax.random.PRNGKey(0), len(sizes))

    t0 = time.perf_counter()
    for size in sorted(set(sizes)):  # compile every distinct chunk shape
        jax.block_until_ready(fwd(theta, keys[0], size))
        jax.block_until_ready(grad(theta, keys[0], size))
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    means = [fwd(theta, k, size) for k, size in zip(keys, sizes, strict=True)]
    means = np.concatenate([np.asarray(m) for m in means])
    fwd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    g = sum(grad(theta, k, size) for k, size in zip(keys, sizes, strict=True)) / reps
    g = np.asarray(g)
    grad_s = time.perf_counter() - t0
    return means, fwd_s, g, grad_s, compile_s


# ---------------------------------------------------------------------------- report


def ci(x):
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / math.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--customers", type=int, default=20_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--chunk", type=int, default=2_500, help="Gyoretsu replications per call")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument(
        "--simpy-reps",
        type=int,
        default=None,
        help="run SimPy on this many replications and scale its time (default: all)",
    )
    p.add_argument(
        "--simpy-single-reps",
        type=int,
        default=60,
        help="replications for the single-core SimPy rate, scaled to --reps (0 to skip)",
    )
    p.add_argument("--skip-simpy", action="store_true")
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    a, s = 1.0, args.rho
    wq = s * s / (a - s)
    wq_grad = [-(s**2) / (a - s) ** 2, s * (2 * a - s) / (a - s) ** 2]
    total = args.reps * args.customers
    print(
        f"M/M/1 rho={args.rho}  reps={args.reps:,}  customers/rep={args.customers:,}  "
        f"total={total / 1e6:,.0f}M customers  theory Wq={wq:.4f}  "
        f"dWq/d[a,s]={np.round(wq_grad, 3)}\n",
        flush=True,
    )

    rows, results = [], {}

    # SimPy first, before JAX starts any threads.
    if not args.skip_simpy:
        sreps = args.simpy_reps or args.reps
        means, t = run_simpy(sreps, args.customers, args.warmup, a, s, args.workers)
        scale = args.reps / sreps
        fwd_s = t * scale
        m, h = ci(means)
        note = "measured" if scale == 1 else f"{sreps:,} reps measured, scaled x{scale:g}"
        results["simpy"] = dict(
            mean_wait=m,
            ci95=h,
            forward_s=fwd_s,
            measured_s=t,
            reps_run=sreps,
            workers=args.workers,
            grad_fd_s=4 * fwd_s,
        )
        rows.append((f"SimPy ({args.workers} cores)", m, h, fwd_s, 4 * fwd_s, note))
        print(f"SimPy {args.workers} cores: {t:.1f}s for {sreps:,} reps", flush=True)

        if args.simpy_single_reps:
            n1 = args.simpy_single_reps
            means1, t1 = run_simpy(n1, args.customers, args.warmup, a, s, 1, seed=999_000)
            f1 = t1 * args.reps / n1
            results["simpy_1core"] = dict(measured_s=t1, reps_run=n1, forward_s=f1)
            m1, h1 = ci(means1)
            rows.append(("SimPy (1 core)", m1, h1, f1, 4 * f1, f"{n1} reps measured, scaled"))
            print(f"SimPy 1 core: {t1:.1f}s for {n1} reps", flush=True)

    import jax

    for platform_name in ("gpu", "cpu"):
        try:
            device = jax.devices(platform_name)[0]
        except RuntimeError:
            continue
        means, fwd_s, g, grad_s, comp_s = run_gyoretsu(
            device, args.reps, args.customers, args.warmup, a, s, args.chunk
        )
        m, h = ci(means)
        results[f"gyoretsu_{platform_name}"] = dict(
            mean_wait=m,
            ci95=h,
            forward_s=fwd_s,
            grad_s=grad_s,
            compile_s=comp_s,
            grad=g.tolist(),
            device=str(device),
        )
        rows.append((f"Gyoretsu {platform_name.upper()}", m, h, fwd_s, grad_s, "measured"))
        print(
            f"Gyoretsu {platform_name}: forward {fwd_s:.2f}s, gradient {grad_s:.2f}s, "
            f"compile {comp_s:.1f}s, grad={g.round(3)}",
            flush=True,
        )

    print(
        f"\n{'':22}{'mean wait':>16}{'forward':>11}{'gradient':>11}"
        f"   speedup vs SimPy 12-core / 1-core"
    )
    base = results.get("simpy", {}).get("forward_s")
    base1 = results.get("simpy_1core", {}).get("forward_s")
    for name, m, h, f, gr, note in rows:
        sp = f"{base / f:7.0f}x" if base else ""
        sp1 = f" / {base1 / f:6.0f}x" if base1 else ""
        print(f"{name:22}{m:9.4f}±{h:.4f}{f:10.2f}s{gr:10.2f}s   {sp}{sp1}  ({note})")
    print(f"{'theory':22}{wq:9.4f}   gradient dWq/d[a,s] = {np.round(wq_grad, 3)}")
    print(
        "\nSimPy gradient = central finite differences over 2 parameters = 4 x forward (derived)."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"simpy_vs_gyoretsu_{stamp}.json"
    path.write_text(
        json.dumps(
            dict(
                config=vars(args) | {"out": str(args.out)},
                theory_wq=wq,
                theory_grad=wq_grad,
                machine=dict(
                    platform=platform.platform(), cpus=os.cpu_count(), jax=jax.__version__
                ),
                results=results,
            ),
            indent=2,
        )
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()

"""Gate 0 study: do pathwise gradients survive event-order swaps?

Two-class M/M/1 non-preemptive priority queue. Compares the Jacobian of the per-class
mean waits [W0, W1] with respect to theta = [a0, a1, s0, s1] (mean interarrival and mean
service per class) from:

* Cobham's closed form (ground truth),
* central finite differences with common random numbers (second reference),
* exact-mode autodiff (tau = 0), which ignores the jumps at event-order swaps,
* straight-through smoothing at several temperatures tau.

Run (GPU, inside WSL):
    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/priority_gradient_study.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gyoretsu import Exponential, simulate_priority2
from gyoretsu.theory import priority_waits

NAMES = ["a0", "a1", "s0", "s1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=2048)
    p.add_argument("--customers", type=int, default=5_000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--taus", type=float, nargs="+", default=[1.0, 0.3, 0.1, 0.03, 0.01])
    p.add_argument("--fd-step", type=float, default=0.02, help="relative FD step")
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args()

    theta = jnp.array([1 / 0.3, 2.5, 1.0, 1.0])  # rho0 = 0.3, rho1 = 0.4

    def waits(th, key, tau, reps):
        sample = simulate_priority2(
            key,
            (Exponential(th[0]), Exponential(th[1])),
            (Exponential(th[2]), Exponential(th[3])),
            n_customers=args.customers,
            replications=reps,
            tau=tau,
        )
        return sample.mean_waits(args.warmup).mean(0)  # (2,)

    fwd = jax.jit(waits, static_argnums=(2, 3))
    jac1 = jax.jit(jax.jacrev(waits), static_argnums=(2, 3))
    keys = jax.random.split(jax.random.PRNGKey(0), args.reps)

    truth_w = np.asarray(priority_waits(theta[:2], theta[2:]))
    truth_j = np.asarray(jax.jacrev(lambda t: priority_waits(t[:2], t[2:]))(theta))

    print(f"device {jax.devices()[0]}, {args.reps} reps x {args.customers} customers")
    w = np.asarray(fwd(theta, keys[0], 0.0, args.reps))
    print(f"mean waits sim {w.round(4)}  theory {truth_w.round(4)}\n")

    def per_rep_jac(tau):
        js = jax.vmap(lambda k: jac1(theta, k, tau, 1))(keys)  # (reps, 2, 4)
        js = np.asarray(js)
        return js.mean(0), 1.96 * js.std(0, ddof=1) / np.sqrt(len(js))

    # Finite differences with CRN on the full batch.
    t0 = time.perf_counter()
    big = jax.random.PRNGKey(1)
    fd = np.zeros((2, 4))
    for i in range(4):
        h = args.fd_step * float(theta[i])
        e = jnp.zeros(4).at[i].set(h)
        up = np.asarray(fwd(theta + e, big, 0.0, args.reps))
        dn = np.asarray(fwd(theta - e, big, 0.0, args.reps))
        fd[:, i] = (up - dn) / (2 * h)
    fd_s = time.perf_counter() - t0

    rows = {"theory": (truth_j, np.zeros_like(truth_j)), "finite diff": (fd, None)}
    for tau in [0.0, *args.taus]:
        rows["exact (tau=0)" if tau == 0 else f"ST tau={tau:g}"] = per_rep_jac(tau)

    def rel_err(j):
        return float(np.linalg.norm(j - truth_j) / np.linalg.norm(truth_j))

    for cls in (0, 1):
        print(f"d W{cls} / d theta" + "".join(f"{n:>16}" for n in NAMES))
        for name, (j, ci) in rows.items():
            cells = "".join(
                f"{j[cls, i]:>9.3f}" + (f"±{ci[cls, i]:<6.3f}" if ci is not None else " " * 7)
                for i in range(4)
            )
            print(f"  {name:16}{cells}")
        print()
    print("relative error of full Jacobian vs theory:")
    for name, (j, _) in rows.items():
        if name != "theory":
            print(f"  {name:16}{rel_err(j):8.3f}")
    print(f"(FD took {fd_s:.1f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"priority_gradient_study_{stamp}.json"
    path.write_text(
        json.dumps(
            dict(
                config={k: str(v) for k, v in vars(args).items()},
                device=str(jax.devices()[0]),
                theta=np.asarray(theta).tolist(),
                params=NAMES,
                theory_waits=truth_w.tolist(),
                sim_waits=w.tolist(),
                jacobians={
                    k: dict(
                        mean=j.tolist(),
                        ci95=None if c is None else c.tolist(),
                        rel_err=None if k == "theory" else rel_err(j),
                    )
                    for k, (j, c) in rows.items()
                },
            ),
            indent=2,
        )
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()

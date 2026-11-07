# Roadmap

Each gate is a review with the faculty advisor. The next phase starts only when the gate's
criteria are met. Dates after Gate 0 are targets aligned to quarterly reviews and will be
confirmed at kickoff.

| Phase | Target | Gate criterion | Status |
|---|---|---|---|
| 0. Prototype | Nov–Dec 2026 | Pathwise gradients of a single-server queue agree with finite differences. If not, narrow scope to GPU forward simulation plus classic IPA. | In progress; criterion met on M/M/1 (`tests/test_gate0_gradients.py`) |
| 1. v0.1 alpha | ~Q1 2027 | Forward results match queueing theory; tandem lines and networks run batched on GPU | |
| 2. v0.5 beta | ~Q2 2027 | Gradient search beats brute force on the staffing study within 10% of its budget; calibration recovers synthetic parameters | |
| 3. Feature freeze | ~Q3 2027 | Quantum module agrees with SeQUeNCe on small repeater chains | |
| 4. v1.0 | Dec 2027 | Documentation, ≥ 5 worked examples, green test suite | |

## Gate 0 checklist

- [x] Differentiable G/G/1 engine with exact, smoothed and straight-through modes
- [x] Forward results match M/M/1, M/D/1 and M/G/1 (Pollaczek–Khinchine) theory
- [x] Waits match SimPy and a plain departure-time recursion on identical inputs
- [x] Pathwise gradient matches the closed-form M/M/1 derivative
- [x] Pathwise gradient matches finite differences (cosine ≥ 0.95)
- [x] Autodiff reproduces hand-written IPA
- [x] Smoothing bias shrinks monotonically as tau → 0
- [x] Multi-server FIFO queue (G/G/c) via the sorted Kiefer–Wolfowitz workload recursion:
      waits match SimPy exactly (c = 2, 3, 5), steady state matches Erlang C (c = 2, 4, 8),
      gradients match the Erlang C derivative and finite differences (`tests/test_mmc.py`)
- [ ] Continuous relaxation of the server count `c` (needed for staffing optimization)
- [ ] First model with genuine event-order discontinuities (priorities, heterogeneous
      servers, probabilistic routing or blocking): this is where smoothing is actually needed
- [ ] Heavy-traffic stress test (rho → 1): gradient variance vs batch size
- [ ] Tandem line via the departure recursion `D[k,n] = max(D[k-1,n], D[k,n-1]) + S[k,n]`
- [x] GPU runs in WSL2 (RTX 4050, 6 GB). Scan unroll=32 is required: 10,000 reps × 5,000
      customers takes ~16–38 ms on GPU vs ~1,250 ms on CPU (unroll=1 on GPU: ~940 ms)
- [x] GPU timing vs SimPy (`benchmarks/simpy_vs_gyoretsu.py`, results in `benchmarks/results/`).
      M/M/1, rho = 0.9, 10,000 reps × 20,000 customers (200M): SimPy 287 s on 12 cores,
      Gyoretsu 0.19 s forward / 0.25 s gradient on GPU (1,483x faster than 12-core SimPy, ~6,000x
      faster than single-core SimPy, which is extrapolated from 60 reps). At 1B customers:
      SimPy 1,366 s, Gyoretsu 3.75 s. All estimates agree with theory within their 95% CIs.
      Caveat: this compares a specialized recursion with a general process-based simulator.
- [ ] Larger GPU chunks for long paths (the 1B run used 500-rep chunks and was latency-bound
      at ~270M customers/s, versus ~1,000M/s with 2,500-rep chunks)
- [ ] Reproduce a benchmark from Che, Dong & Namkoong (2024)

## Design findings

**Soft-min server selection is biased when resources tie (2026-11-07).** The proposal
smooths next-event selection by replacing the hard minimum with a soft-min. Idle identical
servers all have exactly zero remaining work, so a soft choice splits each new job evenly
across them. That bias does not vanish as tau → 0. On M/M/3 at rho = 0.7 the naive
version converges to a mean wait of 1.648 against an exact 1.158 (+42%). Smoothing the
sorted workload recursion converges correctly (1.1576 at tau = 1e-3). Rule: exchangeable
resources are represented as sorted vectors, never as a soft choice among tied candidates.
Soft-min applies only where the candidates are distinguishable.

**Single-server and identical-multi-server FIFO queues need no smoothing for gradients.**
Their sample paths are continuous in the parameters, so exact autodiff already gives
unbiased pathwise (IPA) gradients. Gate 0 therefore validates the engine and gradient
machinery. The smoothing layer gets its real test in the first model with event-order
discontinuities.

## Success criteria (by Dec 2027)

| Objective | Criterion |
|---|---|
| Correct simulation | Steady-state results match M/M/1, M/M/c and Jackson networks within 95% CIs |
| Correct gradients | Cosine similarity ≥ 0.95 with finite differences on all benchmark models |
| Speed at scale | ≥ 10x faster than SimPy for 10,000 replications on one consumer GPU |
| Optimization value | Matches brute force with ≤ 10% of its budget on a 50+ variable staffing problem |
| Calibration | Recovers synthetic parameters; fits one real dataset with uncertainty |
| Adoption | v1.0 on PyPI, docs, ≥ 5 examples, external issues or contributions |

## Planned package layout

`model/` (builders, `gy.param`), `dist/`, `compile/` (graph → arrays), `engine/`,
`smooth/`, `grad/`, `optimize/` (Optax), `calibrate/` (NumPyro/BlackJAX), `report/`,
`cli/`, `domains/quantum/`. Today these are single modules under `src/gyoretsu/` and will be
split into packages as they grow, without changing import paths.

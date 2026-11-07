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
- [ ] Multi-server queue (M/M/c): event ordering is where smoothing becomes necessary
- [ ] Heavy-traffic stress test (rho → 1): gradient variance vs batch size
- [ ] Tandem line via the departure recursion `D[k,n] = max(D[k-1,n], D[k,n-1]) + S[k,n]`
- [x] GPU runs in WSL2 (RTX 4050, 6 GB). Scan unroll=32 is required: 10,000 reps × 5,000
      customers takes ~16–38 ms on GPU vs ~1,250 ms on CPU (unroll=1 on GPU: ~940 ms)
- [ ] GPU timing vs SimPy (10,000 replications), as a committed benchmark script
- [ ] Reproduce a benchmark from Che, Dong & Namkoong (2024)

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

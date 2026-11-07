# CLAUDE.md

Gyoretsu: differentiable discrete-event simulation in JAX. Package `gyoretsu` under `src/`.
The project plan and gates are in ROADMAP.md.

## Commands (Windows; Python comes from Anaconda, which is not on PATH)

- Tests: `.venv\Scripts\python.exe -m pytest`
- Lint: `.venv\Scripts\python.exe -m ruff check .` and `ruff format --check .`
- Gate 0 report: `.venv\Scripts\python.exe examples\gate0_mm1_gradients.py`
- Recreate the venv: `C:\Users\komei\anaconda3\python.exe -m venv .venv`, then `pip install -e ".[dev]"`

## Conventions

- Distributions split sampling into parameter-free `noise()` plus a differentiable
  `transform()`. Keep it that way: it is what gives common random numbers and pathwise gradients.
- `tau == 0` as a Python number means exact mode. `tau > 0` may be traced. The exact mode
  must always be available, because reported results come from exact simulation.
- Tests use fixed PRNG keys and are deterministic. Statistical assertions compare against
  closed-form theory with explicit confidence intervals, not arbitrary tolerances.
- Every new model type needs: a forward check against theory or SimPy, and a gradient
  check against finite differences.

## GPU (WSL2)

CUDA work runs in WSL2 Ubuntu, never in native Windows. Setup: `bash scripts/setup_wsl_gpu.sh`.
The WSL venv lives at `~/.venvs/gyoretsu` (Linux filesystem); the repo stays on `/mnt/c/...`.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (done by `activate`; export it yourself when calling the
venv python directly), otherwise JAX tries to grab 75% of VRAM and logs OOM errors.
From Git Bash: `MSYS_NO_PATHCONV=1 wsl.exe -- bash <script on /mnt/c>`; nested quoting via PowerShell breaks.
Performance: event loops are launch-latency bound on GPU; keep `lax.scan(..., unroll=DEFAULT_UNROLL)`.
Hardware: RTX 4050 Laptop, 6 GB VRAM, so size GPU batches accordingly.

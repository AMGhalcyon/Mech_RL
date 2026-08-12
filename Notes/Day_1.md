# Day 1 — Got the project scaffolding working

## Goal
Set up Mech_RL so it installs clean and has a tested foundation (utils + domain). No physics, no env, no training yet.

## What I did

**Project structure**
- `pyproject.toml` — package definition, all deps, ruff + pytest config
- `requirements.txt`, `requirements-dev.txt` — pinned runtime + dev deps
- `.gitignore` — Python + project conventions
- `.venv/` — isolated Python 3.13 environment with everything installed
- `src/mech_rl/` — installable package (editable), version `0.1.0`
- All 9 subpackages scaffolded with `__init__.py`: `domain`, `physics`, `environment`, `reward`, `rl`, `evaluation`, `tracking`, `visualization`, `utils`

**mech_rl/utils/**
- `paths.py` — `find_project_root()` walks up from cwd until it sees `pyproject.toml` or `.git`. Conventional subdirs (`CONFIG_DIR`, `MODELS_DIR`, etc.) resolved once at import.
- `reproducibility.py` — `set_seed(seed, deterministic_torch=False)` seeds Python `random`, NumPy, and PyTorch (CPU + CUDA). Lazy-imports torch so non-torch code paths don't pay the import cost.
- `logging.py` — stdlib logging with one formatter, idempotent setup.
- `config.py` — `BaseConfig` (frozen, extra-forbid) + `load_config()` that validates YAML against a Pydantic model. `ConfigError` raised with precise messages on malformed input.

**mech_rl/domain/**
- `types.py` — `Radians`, `NewtonMeters`, `Meters`, etc. as NewTypes for documentation. `as_array()` helper normalizes to float64.
- `state.py` — `RobotState` (frozen dataclass, q + qdot, shape-validated) and `EndEffectorPose` (x, y, theta).
- `parameters.py` — `RobotParams`, `SimParams`, `RewardParams` as Pydantic models with unit-tagged fields (lengths >0, masses >0, friction ≥0, dt bounded, integrator in allowed set).

**Tests**
- `tests/conftest.py` — fixtures for default params + sample states
- `tests/unit/test_paths.py` — 7 tests
- `tests/unit/test_reproducibility.py` — 5 tests
- `tests/unit/test_domain.py` — 20 tests

**Result:** 32 passed in 1.49s. Ruff clean.

## Why I did it this way
1. **Python ≥3.11** as the floor (we have 3.13.7 locally). Enough headroom for modern type-hint syntax without `from __future__ import annotations` everywhere.
2. **`frozen=True, extra="forbid"`** on all config Pydantic models. Prevents accidental mutation and catches YAML typos.
3. **Lazy torch import** in `reproducibility.py` — torch is heavy; only pay the cost if seeding is called.
4. **`ConfigError` as the single config-failure exception** — one exception type to catch at the script boundary.
5. **No tests for `config.py` yet** — first test arrives when we have an actual YAML config to validate (Day 4 with Hydra).

## What's broken / annoying
- `pyproject.toml` has 2 `TODO` markers for author name and GitHub URL. Update before any "release" milestone (Day 14 wrap-up at latest).
- `RewardParams` field naming: `distance_coef`, `effort_coef`, etc. If we later add multiplicative reward terms (e.g. logistic shaping), the naming might need a small rename. Defer until needed.
- `integrator: str` in `SimParams` — currently a string. Could be an Enum. Keeping as string for YAML simplicity; revisit if it bites us.
- `_forearm_shorter_than_total` validator in `RobotParams` is a no-op placeholder. Removed silently; if we add geometric sanity checks later, this is the spot.

## How to check it works
```bash
.venv/Scripts/python.exe -m pytest tests/unit -v   # 32 passed
.venv/Scripts/python.exe -m ruff check src/mech_rl tests/   # clean
.venv/Scripts/python.exe -c "import mech_rl; print(mech_rl.__version__)"
# mech_rl v0.1.0
```

## Next up
**Day 2 — Physics core.** Kinematics, dynamics, integrator — fully tested without any RL dependency.
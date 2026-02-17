# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Numerical solver for bifurcational analysis of the Bratu PDE: `-Δu(x,y) + λ·e^(u(x,y)) = 0` on the domain [-1, 1]² with zero Dirichlet boundary conditions. The solution is represented as a sum of separable terms `u(x,y) = Σ fᵢ(x)·gᵢ(y)` using sine basis functions, and solved via an alternating iterative method with lambda continuation.

## Commands

```bash
# Run the solver (entry point)
python pde_solver.py

# All unit + integration tests
pytest tests/ -v

# Unit and integration tests only (skip benchmarks)
pytest tests/ --ignore=tests/test_performance.py -v

# Performance benchmarks only
pytest tests/test_performance.py --benchmark-only --benchmark-sort=mean

# Math-focused tests only
pytest tests/test_ode_solver.py tests/test_bvp_solver.py tests/test_pde_solver.py -v
```

Dependencies: `torch` (2.4.0+cu121), `matplotlib` (3.8.0), `pytest>=8.0`, `pytest-benchmark>=4.0` (see `requirements.txt`).

## Architecture

Three-layer hierarchical solver, each layer depending on the one below:

```
PDESolver (pde_solver.py)
  Alternating direction iteration for f(x) and g(y) with lambda continuation.
  Assembles Galerkin-style mass/stiffness matrices via ODE integration,
  then delegates each 1D subproblem to BVPSolver.

  └── BVPSolver (bvp_solver.py)
        Shooting method from midpoint m=0 toward both boundaries.
        Newton-Raphson correction using autograd Jacobian of boundary residual.

        └── ODESolver (ode_solver.py)
              Classical RK4 integrator (Butcher tableau).
              Two modes: solve_by_segment (full trajectory) and solve_by_end (final state only).
```

All numerics use PyTorch tensors in float64 (`torch.set_default_dtype(torch.float64)`). Automatic differentiation (`backward()`) computes the Jacobian in the BVP shooting method — no finite differences.

### Key algorithmic details

- **Grid indexing**: `_idx(t)` maps a coordinate t ∈ [-1, 1] to a half-grid index via `int((t+1) / half_grid_size)`. Both a main grid (`grid_size`) and a half-grid (`grid_size/2`) are used.
- **Normalization**: After each f or g BVP solve, amplitudes are rebalanced so that `max|fᵢ| ≈ max|gᵢ|` for each component i, stabilizing the alternating iteration.
- **Lambda continuation**: Starts from λ close to 0 and steps toward `target_lambda` in increments of 1/40.
- **Convergence**: Measured by max-norm difference of f and g between successive iterations, threshold set by `convergence_tolerance`.

## Tests

```
tests/
    conftest.py           # sys.path setup, matplotlib stub, tolerance constants, shared fixtures
    test_ode_solver.py    # 7 unit tests: RK4 accuracy, order, shape, backward integration
    test_bvp_solver.py    # 8 unit tests: Newton convergence, Jacobian correctness, SVD check
    test_pde_solver.py    # 7 unit tests: basis functions, Gram/stiffness matrices, normalization
    test_integration.py   # 5 end-to-end tests: nonlinear BVP, small-lambda PDE, symmetry
    test_performance.py   # 3 benchmarks: ODE throughput, BVP memory, one PDE lambda step
```

Note: matplotlib is stubbed in `conftest.py` to allow testing even when the installed matplotlib is incompatible with the system NumPy version.

## Specs

No specification files yet.

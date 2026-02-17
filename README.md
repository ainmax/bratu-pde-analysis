# Bratu PDE — Bifurcation Analysis

Numerical solver for the Bratu PDE on the unit square with bifurcation analysis via lambda continuation.

**Equation**: `-Δu(x,y) + λ·e^(u(x,y)) = 0` on `[-1, 1]²`, zero Dirichlet boundary conditions.

## Method

The solution is represented as a separable sum:

```
u(x, y) = Σ fᵢ(x) · gᵢ(y)
```

using sine basis functions. The system is solved by an alternating iterative method:

1. Fix `g`, solve a Galerkin-projected 1D BVP for `f`.
2. Fix `f`, solve a Galerkin-projected 1D BVP for `g`.
3. Repeat until convergence (`max‖Δf‖, ‖Δg‖ < tolerance`).

Lambda continuation steps from `λ ≈ 0` toward the target value, using the previous solution as the initial guess at each step.

Each 1D BVP is solved by a shooting method (from the domain midpoint toward both boundaries) with Newton-Raphson correction. The Jacobian is computed via PyTorch autograd — no finite differences.

## Architecture

```
PDESolver (pde_solver.py)
  └── BVPSolver (bvp_solver.py)
        └── ODESolver (ode_solver.py)   ← classical RK4
```

All numerics use `torch.float64`.

## Usage

```bash
python pde_solver.py
```

Produces `plot*.png` files showing the `fᵢ` and `gᵢ` components at each iteration and lambda step.

## Testing

```bash
# Install dependencies
pip install -r requirements.txt

# All unit + integration tests
pytest tests/ -v

# Performance benchmarks
pytest tests/test_performance.py --benchmark-only --benchmark-sort=mean
```

30 tests across five files cover RK4 accuracy and order, Newton convergence, autograd Jacobian correctness, Galerkin matrix properties, PDE symmetry, and timing regression baselines.

## Dependencies

- Python 3.11+
- `torch >= 2.4.0`
- `matplotlib >= 3.8.0`
- `pytest >= 8.0`, `pytest-benchmark >= 4.0` (testing only)

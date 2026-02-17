"""Unit tests for ODESolver (RK4 integrator)."""
import math

import torch
import pytest

from ode_solver import ODESolver
from conftest import (
    ODE_GLOBAL_ERROR_TOL,
    ODE_MATCH_TOL,
    decay_ode,
    harmonic_ode,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_decay_solver() -> ODESolver:
    return ODESolver(decay_ode)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_decay_output_shape():
    """solve_by_segment must return shape [steps+1, state_dim]."""
    solver = make_decay_solver()
    init = torch.tensor([1.0])
    a, b, h = 0.0, 1.0, 0.1
    steps = int(abs(b - a) / h)          # 10

    result = solver.solve_by_segment(init_state=init, a=a, b=b, grid_size=h)

    assert result.shape == (steps + 1, 1), (
        f"Expected ({steps + 1}, 1), got {result.shape}"
    )


def test_rk4_accuracy_exponential_decay():
    """Global error at h=0.01 on y'=-y must be < 1e-8."""
    solver = make_decay_solver()
    result = solver.solve_by_end(
        init_state=torch.tensor([1.0]),
        a=0.0, b=1.0, grid_size=0.01,
    )
    exact = math.exp(-1.0)
    error = abs(result[0].item() - exact)
    assert error < ODE_GLOBAL_ERROR_TOL, f"RK4 global error {error:.2e} >= {ODE_GLOBAL_ERROR_TOL}"


def test_rk4_order_of_convergence():
    """Error ratio between successive grid refinements must be ≈ 16 (order 4)."""
    solver = make_decay_solver()
    exact = math.exp(-1.0)
    grid_sizes = [0.1, 0.05, 0.025]

    errors = []
    for h in grid_sizes:
        result = solver.solve_by_end(
            init_state=torch.tensor([1.0]),
            a=0.0, b=1.0, grid_size=h,
        )
        errors.append(abs(result[0].item() - exact))

    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert 12.0 < ratio < 20.0, (
            f"Convergence ratio {ratio:.2f} not near 16 (step {i}: h={grid_sizes[i]}, h/2={grid_sizes[i+1]})"
        )


def test_solve_by_end_matches_solve_by_segment():
    """solve_by_end and solve_by_segment must agree on the final state."""
    solver = make_decay_solver()
    init = torch.tensor([1.0])
    kwargs = dict(init_state=init, a=0.0, b=1.0, grid_size=0.05)

    end_only = solver.solve_by_end(**kwargs)
    full_traj = solver.solve_by_segment(**kwargs)

    diff = (end_only - full_traj[-1]).abs().max().item()
    assert diff < ODE_MATCH_TOL, f"Mismatch between solve_by_end and solve_by_segment[-1]: {diff:.2e}"


def test_solve_by_end_zero_steps():
    """When a == b, solve_by_end must return the initial state without crashing."""
    solver = make_decay_solver()
    init = torch.tensor([3.14])
    result = solver.solve_by_end(init_state=init, a=0.5, b=0.5, grid_size=0.1)
    diff = (result - init).abs().max().item()
    assert diff == 0.0, f"Zero-step integration should return init_state exactly; got diff {diff}"


def test_backward_integration():
    """Integrating y'=-y from t=1 (y=e^{-1}) back to t=0 must recover y(0)=1."""
    solver = make_decay_solver()
    y1 = torch.tensor([math.exp(-1.0)])
    traj = solver.solve_by_segment(init_state=y1, a=1.0, b=0.0, grid_size=0.01)

    # Last entry is the state at t=0
    error = abs(traj[-1][0].item() - 1.0)
    assert error < ODE_GLOBAL_ERROR_TOL, f"Backward-integration error {error:.2e} >= {ODE_GLOBAL_ERROR_TOL}"


def test_vector_state_ode():
    """
    Simple harmonic oscillator [u', v'] = [v, -u], u(0)=1, v(0)=0.
    Exact: u(t) = cos(t).  At t=1.0, u(1) = cos(1).

    Use rational grid_size=0.001 so int(1.0/0.001)=1000 steps exactly —
    no floating-point alignment error, only RK4 truncation error.
    """
    def cosine_ode(t, state):
        """[u', v'] = [v, -u]"""
        return torch.stack([state[1], -state[0]])

    solver = ODESolver(cosine_ode)
    init = torch.tensor([1.0, 0.0])          # u(0)=1, u'(0)=0
    result = solver.solve_by_end(
        init_state=init,
        a=0.0,
        b=1.0,
        grid_size=0.001,
    )
    exact = math.cos(1.0)
    error = abs(result[0].item() - exact)
    assert error < ODE_GLOBAL_ERROR_TOL, f"|u(1) - cos(1)| = {error:.2e} >= {ODE_GLOBAL_ERROR_TOL}"

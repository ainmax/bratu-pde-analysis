"""
Performance benchmarks via pytest-benchmark.

Run with:
    pytest tests/test_performance.py --benchmark-only --benchmark-sort=mean
"""
import tracemalloc
from unittest.mock import patch

import torch
import pytest

from ode_solver import ODESolver
from bvp_solver import BVPSolver
from pde_solver import PDESolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def linear_bvp_solver():
    def f_ode(t, state):
        return torch.stack([state[1], torch.tensor(1.0, dtype=torch.float64)])

    def boundary_fn(a, b, a_state, b_state):
        return torch.stack([a_state[0], b_state[0]])

    return BVPSolver(f=f_ode, boundary=boundary_fn, a=-1.0, b=1.0, m=0.0, grid_size=0.02)


@pytest.fixture(scope="module")
def large_ode_solver():
    """6D harmonic system: coupled oscillators."""
    def f_6d(t, state):
        # Pairs: (u_i, v_i), v_i' = -u_i
        pairs = state.reshape(3, 2)
        return torch.stack([pairs[:, 1], -pairs[:, 0]], dim=1).reshape(6)

    return ODESolver(f_6d)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def test_bench_ode_solve_by_segment_large_grid(benchmark, large_ode_solver):
    """2000-step, 6D state integration must complete in < 1 s on CPU."""
    init = torch.zeros(6, dtype=torch.float64)
    init[1] = 1.0  # non-trivial start

    result = benchmark(
        large_ode_solver.solve_by_segment,
        init_state=init, a=0.0, b=10.0, grid_size=0.005,
    )
    assert result.shape == (2001, 6)
    assert benchmark.stats["mean"] < 1.0, (
        f"Mean time {benchmark.stats['mean']:.3f}s >= 1.0s threshold"
    )


def test_bench_bvp_shoot_memory(benchmark, linear_bvp_solver):
    """
    Memory must remain bounded over 100 consecutive _shoot calls.
    Peak usage must stay below 200 MB.
    """
    solver = linear_bvp_solver
    state = torch.tensor([-0.3, 0.1], dtype=torch.float64)

    def run_100_shoots():
        s = state
        for _ in range(100):
            r, J = solver._shoot(s)
            s = s.detach()

    tracemalloc.start()
    benchmark(run_100_shoots)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 ** 2)
    assert peak_mb < 200, f"Peak memory {peak_mb:.1f} MB >= 200 MB threshold"


def test_bench_pde_solver_one_lambda_step(benchmark):
    """
    One _pass_iterations call with n=2, grid_size=1/16 must complete in < 30 s.
    Serves as a regression baseline.
    """
    s = PDESolver(
        grid_size=1 / 16,
        bvp_tolerance=1e-4,
        convergence_tolerance=1e-1,
        target_lambda=-0.001,
    )
    s.n = 2
    s._setup()

    with patch("pde_solver.make_plot"):
        benchmark(s._pass_iterations, -0.001)

    assert benchmark.stats["mean"] < 30.0, (
        f"Mean time {benchmark.stats['mean']:.2f}s >= 30s threshold"
    )

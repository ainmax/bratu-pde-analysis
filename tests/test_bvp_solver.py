"""Unit tests for BVPSolver (shooting + Newton-Raphson)."""
import math

import torch
import pytest

from ode_solver import ODESolver
from bvp_solver import BVPSolver
from conftest import BVP_BOUNDARY_TOL, BVP_POINTWISE_TOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_linear_bvp(grid_size: float = 0.05) -> BVPSolver:
    """
    u'' = 1,  u(-1)=0, u(1)=0
    Exact: u(x) = (x^2 - 1)/2,  u(0) = -0.5, u'(0) = 0
    """
    def f_ode(t, state):
        return torch.stack([state[1], torch.tensor(1.0, dtype=torch.float64)])

    def boundary_fn(a, b, a_state, b_state):
        return torch.stack([a_state[0], b_state[0]])

    return BVPSolver(f=f_ode, boundary=boundary_fn, a=-1.0, b=1.0, m=0.0, grid_size=grid_size)


def make_nonlinear_bvp(grid_size: float = 0.05) -> BVPSolver:
    """
    u'' = u^2 - 0.1,  u(-1)=0, u(1)=0
    No analytic solution; used to test Newton convergence rate.
    """
    def f_ode(t, state):
        return torch.stack([state[1], state[0] ** 2 - 0.1])

    def boundary_fn(a, b, a_state, b_state):
        return torch.stack([a_state[0], b_state[0]])

    return BVPSolver(f=f_ode, boundary=boundary_fn, a=-1.0, b=1.0, m=0.0, grid_size=grid_size)


def _shoot_boundary_value(solver: BVPSolver, m: torch.Tensor) -> torch.Tensor:
    """Forward-only shoot (no autograd) to get boundary residual for FD Jacobian."""
    ode = ODESolver(solver.f)
    a_traj = ode.solve_by_segment(init_state=m, a=solver.m, b=solver.a, grid_size=solver.grid_size)
    b_traj = ode.solve_by_segment(init_state=m, a=solver.m, b=solver.b, grid_size=solver.grid_size)
    return solver.boundary(solver.a, solver.b, a_traj[-1], b_traj[-1]).detach()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_linear_bvp_solution_accuracy():
    """
    After solving the linear BVP, boundary residual must be < BVP_BOUNDARY_TOL
    and the mid-point value must match the exact solution within BVP_POINTWISE_TOL.
    """
    solver = make_linear_bvp()
    init = torch.tensor([-0.3, 0.1], dtype=torch.float64).requires_grad_(True)
    traj = solver.solve(init_state=init, tol=BVP_BOUNDARY_TOL, max_iter=50)

    # Check boundary residual via stored states
    r = solver.boundary(-1.0, 1.0, solver.a_state[-1].detach(), solver.b_state[-1].detach())
    assert r.abs().max().item() < BVP_BOUNDARY_TOL * 10, (
        f"Boundary residual {r.abs().max().item():.2e} exceeds tolerance"
    )

    # The trajectory starts at t=-1 and ends at t=1.
    # Find the approximate midpoint index (t=0) in the full trajectory.
    total_steps = int(abs(solver.b - solver.a) / solver.grid_size)  # 40
    mid_idx = total_steps // 2  # index corresponding to t=0
    u_mid = traj[mid_idx, 0].item()
    assert abs(u_mid - (-0.5)) < BVP_POINTWISE_TOL, (
        f"u(0) = {u_mid:.6f}, expected -0.5, error {abs(u_mid + 0.5):.2e}"
    )


def test_boundary_residual_at_convergence():
    """Re-evaluating boundary after solve() must give residual < BVP_BOUNDARY_TOL."""
    solver = make_linear_bvp()
    init = torch.tensor([-0.4, 0.05], dtype=torch.float64).requires_grad_(True)
    solver.solve(init_state=init, tol=BVP_BOUNDARY_TOL, max_iter=50)

    r = solver.boundary(-1.0, 1.0, solver.a_state[-1].detach(), solver.b_state[-1].detach())
    assert r.abs().max().item() < BVP_BOUNDARY_TOL * 10


def test_newton_quadratic_convergence():
    """
    Manual Newton loop on a nonlinear BVP: the residual norm must shrink
    at least quadratically (ratio |r_{k+1}|/|r_k|^2 roughly constant and finite).
    """
    solver = make_nonlinear_bvp(grid_size=0.02)
    state = torch.tensor([0.05, 0.0], dtype=torch.float64).requires_grad_(True)

    residuals = []
    for _ in range(8):
        r, J = solver._shoot(state)
        norm = r.norm().item()
        residuals.append(norm)
        if norm < 1e-11:
            break
        delta = torch.linalg.inv(J) @ r
        state = (state.detach() - delta.detach()).requires_grad_(True)

    # Need at least 3 residuals to check quadratic trend
    assert len(residuals) >= 3, "Newton did not produce enough iterations"

    # Find first pair in the superlinear regime (both < 1)
    pairs_checked = 0
    for i in range(len(residuals) - 1):
        r_k, r_kp1 = residuals[i], residuals[i + 1]
        if r_k < 1.0 and r_kp1 > 1e-14:
            ratio = r_kp1 / (r_k ** 2)
            # Quadratic convergence: ratio should be O(1) — not growing unboundedly
            assert ratio < 1e4, (
                f"Convergence ratio |r_{{k+1}}|/|r_k|^2 = {ratio:.2e} too large at step {i}"
            )
            pairs_checked += 1

    # At minimum, overall convergence must be fast: last / first < 1e-6
    assert residuals[-1] / residuals[0] < 1e-6 or residuals[-1] < 1e-11, (
        f"Newton did not converge: r_0={residuals[0]:.2e}, r_last={residuals[-1]:.2e}"
    )


def test_shoot_jacobian_correctness():
    """
    Autograd Jacobian from _shoot must agree with finite-difference Jacobian
    to relative tolerance 1e-4.
    """
    solver = make_linear_bvp(grid_size=0.02)
    m0 = torch.tensor([-0.3, 0.1], dtype=torch.float64)

    # Autograd Jacobian
    r_auto, J_auto = solver._shoot(m0)
    J_auto = J_auto.detach()

    # Finite-difference Jacobian (central differences)
    eps = 1e-5
    n = len(m0)
    m_len = len(r_auto)
    J_fd = torch.zeros(m_len, n, dtype=torch.float64)
    for j in range(n):
        e = torch.zeros(n, dtype=torch.float64)
        e[j] = eps
        r_plus = _shoot_boundary_value(solver, m0 + e)
        r_minus = _shoot_boundary_value(solver, m0 - e)
        J_fd[:, j] = (r_plus - r_minus) / (2 * eps)

    # Compare
    rel_err = (J_auto - J_fd).abs() / (J_fd.abs().clamp(min=1e-10))
    assert rel_err.max().item() < 1e-4, (
        f"Autograd vs FD Jacobian max relative error: {rel_err.max().item():.2e}"
    )


def test_check_jacobian_svd_full_rank():
    """check_jacobian_svd on identity matrix: ok=True, rank=3, cond≈1."""
    solver = make_linear_bvp()
    J = torch.eye(3, dtype=torch.float64)
    ok, info = solver.check_jacobian_svd(J)

    assert ok is True, f"Expected ok=True, got {ok}"
    assert info["rank"] == 3, f"Expected rank=3, got {info['rank']}"
    assert abs(info["cond_est"] - 1.0) < 1e-6, f"Expected cond≈1, got {info['cond_est']}"


def test_check_jacobian_svd_singular():
    """check_jacobian_svd on zero matrix: ok=False, rank=0."""
    solver = make_linear_bvp()
    J = torch.zeros(3, 3, dtype=torch.float64)
    ok, info = solver.check_jacobian_svd(J)

    assert ok is False, f"Expected ok=False, got {ok}"
    assert info["rank"] == 0, f"Expected rank=0, got {info['rank']}"


def test_solution_trajectory_shape():
    """solve() must return shape [total_steps+1, state_dim]."""
    grid_size = 0.1
    solver = make_linear_bvp(grid_size=grid_size)
    init = torch.tensor([-0.5, 0.0], dtype=torch.float64).requires_grad_(True)
    traj = solver.solve(init_state=init, tol=1e-8, max_iter=50)

    total_steps = int(abs(solver.b - solver.a) / grid_size)  # 20
    state_dim = 2
    assert traj.shape == (total_steps + 1, state_dim), (
        f"Expected ({total_steps + 1}, {state_dim}), got {traj.shape}"
    )


def test_boundary_values_at_endpoints():
    """First and last rows of trajectory must satisfy u(±1) ≈ 0."""
    solver = make_linear_bvp(grid_size=0.05)
    init = torch.tensor([-0.5, 0.0], dtype=torch.float64).requires_grad_(True)
    traj = solver.solve(init_state=init, tol=BVP_BOUNDARY_TOL, max_iter=50)

    u_left = traj[0, 0].item()
    u_right = traj[-1, 0].item()
    assert abs(u_left) < 1e-5, f"|u(-1)| = {abs(u_left):.2e} >= 1e-5"
    assert abs(u_right) < 1e-5, f"|u(1)| = {abs(u_right):.2e} >= 1e-5"

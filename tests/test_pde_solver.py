"""Unit tests for PDESolver internals."""
import math
from unittest.mock import patch

import torch
import pytest

from ode_solver import ODESolver
from pde_solver import PDESolver
from conftest import PDE_BOUNDARY_TOL


# ---------------------------------------------------------------------------
# Shared fixture: minimal solver with n=2, coarse grid, after _setup()
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_solver():
    s = PDESolver(
        grid_size=1 / 8,
        bvp_tolerance=1e-4,
        convergence_tolerance=1e-2,
        target_lambda=-0.001,
    )
    s.n = 2
    s._setup()
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_idx_mapping_endpoints(minimal_solver):
    """_idx(-1) must be 0 and _idx(1) must be the last valid half-grid index."""
    s = minimal_solver
    idx_left = s._idx(-1.0)
    idx_right = s._idx(1.0)

    assert idx_left == 0, f"_idx(-1) = {idx_left}, expected 0"

    last_valid = len(s.half_grid) - 1
    assert idx_right == last_valid, (
        f"_idx(1) = {idx_right}, expected {last_valid} (last valid index)"
    )

    # Neither index must raise IndexError when used to read f
    _ = s.f[idx_left]
    _ = s.f[idx_right]


def test_basis_functions_boundary_conditions(minimal_solver):
    """φᵢ(±1) must be 0 for all modes i."""
    s = minimal_solver
    for name, pts, arr in [("f", [-1.0, 1.0], s.f), ("g", [-1.0, 1.0], s.g)]:
        for t in pts:
            vals = arr[s._idx(t)]
            for i, v in enumerate(vals):
                assert abs(v.item()) < PDE_BOUNDARY_TOL, (
                    f"{name}[_idx({t})][{i}] = {v.item():.2e}, expected 0"
                )


def test_basis_function_orthogonality():
    """
    ∫_{-1}^{1} φᵢ(t) φⱼ(t) dt must be ≈ δᵢⱼ.
    Verified numerically via ODESolver.
    """
    n = 3
    grid_size = 1 / 256

    def gram_integrand(t, state):
        phi = torch.tensor([
            math.sin((i + 1) * math.pi * (t + 1) / 2) for i in range(n)
        ], dtype=torch.float64)
        return torch.outer(phi, phi)

    ode = ODESolver(gram_integrand)
    G = ode.solve_by_end(
        init_state=torch.zeros(n, n, dtype=torch.float64),
        a=-1.0, b=1.0, grid_size=grid_size,
    )

    # Diagonal should be ≈ 1, off-diagonal ≈ 0
    eye = torch.eye(n, dtype=torch.float64)
    diag_err = (G.diag() - 1.0).abs().max().item()
    off_diag_err = (G - G.diag().diag()).abs().max().item()

    assert diag_err < 0.01, f"Diagonal entries deviate from 1: max error {diag_err:.4f}"
    assert off_diag_err < 1e-4, f"Off-diagonal entries non-zero: max {off_diag_err:.4f}"


def test_gram_matrix_near_identity(minimal_solver):
    """
    ∫ g⊗g dy computed via ODESolver after _setup() must be near identity:
    off-diagonal entries < 0.05.
    """
    s = minimal_solver

    def integrand(y_, state):
        return torch.outer(s.g[s._idx(y_)], s.g[s._idx(y_)])

    ode = ODESolver(integrand)
    G = ode.solve_by_end(
        init_state=torch.zeros(s.n, s.n, dtype=torch.float64),
        a=-1.0, b=1.0, grid_size=s.grid_size,
    )

    off_diag = G - G.diag().diag()
    assert off_diag.abs().max().item() < 0.05, (
        f"Gram matrix off-diagonal max: {off_diag.abs().max().item():.4f}"
    )


def test_stiffness_matrix_diagonal_values(minimal_solver):
    """
    matrix_b diagonal entries must match ((i+1)π/2)² within 5%.
    """
    s = minimal_solver
    expected = torch.tensor(
        [((i + 1) * math.pi / 2) ** 2 for i in range(s.n)], dtype=torch.float64
    )

    def integrand(y_, state):
        return -torch.outer(s.g[s._idx(y_)], s.gpp[s._idx(y_)])

    ode = ODESolver(integrand)
    B = ode.solve_by_end(
        init_state=torch.zeros(s.n, s.n, dtype=torch.float64),
        a=-1.0, b=1.0, grid_size=s.grid_size,
    )

    diag = B.diag()
    rel_err = ((diag - expected) / expected).abs()
    assert rel_err.max().item() < 0.05, (
        f"Stiffness diagonal relative errors: {rel_err.tolist()}"
    )


def test_calc_divergence_norm_trivial_solution(minimal_solver):
    """_calc_divergence_norm at λ=0 must return a finite float without crashing."""
    s = minimal_solver
    norm = s._calc_divergence_norm(0.0)
    assert math.isfinite(norm), f"_calc_divergence_norm returned non-finite: {norm}"


def test_normalization_preserves_product(minimal_solver):
    """
    The amplitude rebalancing must preserve f[i][k] * g[i][k] for every grid
    point i and mode k.
    """
    s = minimal_solver
    # Perturb f and g so f_max != g_max (triggers actual rescaling)
    for i in range(len(s.half_grid)):
        s.f[i] = s.f[i] * 2.0
        s.g[i] = s.g[i] * 0.5

    # Snapshot the products before normalization
    products_before = [
        s.f[i].clone() * s.g[i].clone() for i in range(len(s.half_grid))
    ]

    # Replicate the normalization logic from _pass_iterations
    f_tensor = torch.abs(torch.stack(s.f, dim=0))
    g_tensor = torch.abs(torch.stack(s.g, dim=0))
    for i in range(len(s.half_grid)):
        for k in range(s.n):
            f_max = torch.max(f_tensor[:, k]).item()
            g_max = torch.max(g_tensor[:, k]).item()
            f_mult = math.sqrt(g_max / f_max)
            g_mult = math.sqrt(f_max / g_max)
            s.f[i][k] *= f_mult
            s.fp[i][k] *= f_mult
            s.fpp[i][k] *= f_mult
            s.g[i][k] *= g_mult
            s.gp[i][k] *= g_mult
            s.gpp[i][k] *= g_mult

    products_after = [s.f[i] * s.g[i] for i in range(len(s.half_grid))]

    for i, (pb, pa) in enumerate(zip(products_before, products_after)):
        denom = pb.abs().clamp(min=1e-15)
        rel_err = ((pa - pb) / denom).abs().max().item()
        assert rel_err < 1e-12, (
            f"Normalization changed f*g at half_grid index {i}: rel_err={rel_err:.2e}"
        )

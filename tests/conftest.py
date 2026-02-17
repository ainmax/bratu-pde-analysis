import sys
import os
import math
from unittest.mock import MagicMock

# Make the project root importable from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub out matplotlib before pde_solver is imported.
# This allows the test suite to run even when matplotlib is not installed
# or is incompatible with the current NumPy version.
for _mod in ("matplotlib", "matplotlib.pyplot", "matplotlib.backends",
             "matplotlib.backends.backend_agg"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import torch
import pytest

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
ODE_GLOBAL_ERROR_TOL = 1e-8
ODE_MATCH_TOL = 1e-14
BVP_BOUNDARY_TOL = 1e-6
BVP_POINTWISE_TOL = 1e-4
PDE_BOUNDARY_TOL = 1e-14


# ---------------------------------------------------------------------------
# Shared ODE definitions (reused across test modules)
# ---------------------------------------------------------------------------

def decay_ode(t: float, state: torch.Tensor) -> torch.Tensor:
    """y' = -y"""
    return -state


def harmonic_ode(t: float, state: torch.Tensor) -> torch.Tensor:
    """[u', v'] = [v, -u]  (simple harmonic oscillator, u'' + u = 0)"""
    return torch.stack([state[1], -state[0]])


# ---------------------------------------------------------------------------
# Shared BVP fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_bvp_components():
    """
    Returns (f_ode, boundary_fn, a, b, m, grid_size, exact_u0, exact_u0p)
    for the analytic BVP:  u'' = 1,  u(-1)=0, u(1)=0
    Exact solution: u(x) = (x^2 - 1) / 2
    Exact midpoint: u(0) = -0.5, u'(0) = 0
    """
    c = 1.0

    def f_ode(t: float, state: torch.Tensor) -> torch.Tensor:
        return torch.stack([state[1], torch.tensor(c, dtype=torch.float64)])

    def boundary_fn(a, b, a_state, b_state):
        return torch.stack([a_state[0], b_state[0]])

    return dict(
        f_ode=f_ode,
        boundary_fn=boundary_fn,
        a=-1.0,
        b=1.0,
        m=0.0,
        grid_size=0.05,
        exact_u0=-0.5,
        exact_u0p=0.0,
        c=c,
    )

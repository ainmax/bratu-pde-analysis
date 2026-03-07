import math

import torch
from typing import Callable, Tuple

from ode_solver import ODESolver

torch.set_default_dtype(torch.float64)

# s' = f(t, s)
# boundary(a, b, s_a, s_b) = 0

class BVPSolver:
    def __init__(
            self,
            *,
            f: Callable,
            boundary: Callable,
            init_boundary: Callable | None = None,
            a: float,
            b: float,
            m: float,
            grid_size: float
    ):
        self.f: Callable = f
        self.boundary: Callable = boundary
        self.init_boundary: Callable | None = init_boundary
        self.a: float = a
        self.b: float = b
        self.m: float = m
        self.a_state: torch.Tensor = torch.tensor([])
        self.b_state: torch.Tensor = torch.tensor([])
        self.grid_size = grid_size
        self.steps_count: int = int(abs(b - a) / grid_size)

    def _shoot(self, init_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ode_solver = ODESolver(self.f)
        mid_state = init_state.detach().clone().requires_grad_(True)
        self.a_state = ode_solver.solve_by_segment(
            init_state = mid_state,
            a = self.m,
            b = self.a,
            grid_size = self.grid_size
        )
        self.b_state = ode_solver.solve_by_segment(
            init_state=mid_state,
            a=self.m,
            b=self.b,
            grid_size=self.grid_size
        )

        boundary_value: torch.Tensor = torch.tensor([])
        if self.init_boundary is not None:
            boundary_value = torch.cat(
                (
                    self.boundary(self.a, self.b, self.a_state[-1], self.b_state[-1]),
                    self.init_boundary(mid_state)
                ),
                dim=0
            )
        else:
            boundary_value: torch.Tensor = self.boundary(self.a, self.b, self.a_state[-1], self.b_state[-1])

        jacobian = torch.zeros(len(boundary_value), len(mid_state))
        for i in range(len(mid_state)):
            boundary_value[i].backward(retain_graph=True)
            jacobian[i] = mid_state.grad
            mid_state.grad.data.zero_()

        return boundary_value, jacobian

    def solve(
            self,
            *,
            init_state: torch.Tensor,
            tol: float = 1e-6,
            max_iter: int = 100
    ) -> torch.Tensor:
        state: torch.Tensor = init_state
        # log
        print("BVP solving started")

        for iteration in range(max_iter):
            # log
            print(f"Iteration {iteration} started")

            boundary_value, boundary_jacobian = self._shoot(state)

            if torch.max(torch.abs(boundary_value)) < tol:
                print(f"Converged in {iteration + 1} iterations")
                print("Max real part of Jacobian matrix eigen value:", torch.max(torch.linalg.eigvals(boundary_jacobian).real))
                break

            if math.isnan(boundary_value[0].item()):
                raise ValueError(f"Невязка слишком большая на итерации {iteration}.")

            if torch.max(torch.abs(boundary_jacobian)) < 1e-12:
                raise ValueError(f"Якобиан почти нулевой на итерации {iteration}.")

            state = state - torch.linalg.solve(boundary_jacobian, boundary_value)
        else:
            raise RuntimeError(f"За {max_iter} итераций не сошлось")

        return torch.cat((self.a_state.flip(dims=[0]), self.b_state[1::]), dim=0).detach()

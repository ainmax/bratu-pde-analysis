import math

import torch
from typing import Callable, Tuple

from ode_solver import ODESolver

torch.set_default_dtype(torch.float64)

class BVPSolver:
    """
    Classic Newton BVP solver:

    s'(t) = f(t, s(t)),
    boundary(a, b, s(a), s(b)) = 0
    """

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
        """
        Fixes all parameters except the initial state of the function
        :param f: rhs for ODE; s'(t) = f(t, s(t)), where t in R and s(t) is a tensor shaped with (n, 1)
        :param boundary: boundary for solution at the ends of segment; boundary(a, b, s(a), s(b)) = 0
        :param init_boundary: optional boundary for initial state
        :param a: begin of the segment
        :param b: end of the segment
        :param m: middle of the segment
        :param grid_size: distance between neighboring grid points
        """

        self.f: Callable = f
        self.boundary: Callable = boundary
        self.init_boundary: Callable | None = init_boundary
        self.a: float = a
        self.b: float = b
        self.m: float = m
        self.a_state: torch.Tensor = torch.tensor([])
        self.b_state: torch.Tensor = torch.tensor([])
        self.grid_size = grid_size
        self.steps_count: int = round(abs(b - a) / grid_size)

    def _shoot(self, init_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates current jacoby matrix and boundary value
        :param init_state: current initial state
        :return: tuple (boundary value, jacoby matrix); tuples of tensors shaped with ((n, 1), (n, n)) or ((n + 1, 1), (n + 1, n + 1))
        """

        ode_solver = ODESolver(self.f)
        mid_state = init_state.clone().detach().requires_grad_(True)
        self.a_state = ode_solver.solve_by_segment(
            init_state=mid_state,
            a=self.m,
            b=self.a,
            grid_size=self.grid_size
        )
        self.b_state = ode_solver.solve_by_segment(
            init_state=mid_state,
            a=self.m,
            b=self.b,
            grid_size=self.grid_size
        )

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

        jac_matrix = torch.zeros(len(boundary_value), len(mid_state))
        for i in range(len(boundary_value)):
            boundary_value[i].backward(retain_graph=True)
            jac_matrix[i] = mid_state.grad
            mid_state.grad.data.zero_()

        return boundary_value, jac_matrix

    def solve(
            self,
            *,
            init_state: torch.Tensor,
            tol: float = 1e-6,
            max_iter: int = 100
    ) -> torch.Tensor:
        """
        Calculates solution approximation with boundary compliance
        :param init_state: initial state; tensor shaped with (n, 1)
        :param tol: tolerance of boundary compliance
        :param max_iter: the maximal number of iterations
        :return: solution with boundary compliance; tensor shaped with (grid_cardinality, 1)
        """

        state: torch.Tensor = init_state.detach()
        # log
        print('BVP solving started')

        for iteration in range(max_iter):
            # log
            print(f'Iteration {iteration} started')

            boundary_value, boundary_jacobian = self._shoot(state)

            if torch.max(torch.abs(boundary_value)) < tol:
                ode_solver = ODESolver(self.f)
                mid_state = state.clone().detach().requires_grad_(True)
                self.a_state = ode_solver.solve_on_subgrid(
                    init_state=mid_state,
                    a=self.m,
                    b=self.a,
                    grid_size=self.grid_size
                )
                self.b_state = ode_solver.solve_on_subgrid(
                    init_state=mid_state,
                    a=self.m,
                    b=self.b,
                    grid_size=self.grid_size
                )

                print(f'Converged in {iteration + 1} iterations')
                u, s, vh = torch.linalg.svd(boundary_jacobian)
                sigma_min = s[-1]
                print('Minimal Jacobian matrix singular value:', sigma_min)
                print('Jacobian matrix determinant:', torch.linalg.det(boundary_jacobian))
                break

            if math.isnan(boundary_value[0].item()):
                raise ValueError(f'Невязка слишком большая на итерации {iteration}.')

            if torch.max(torch.abs(boundary_jacobian)) < 1e-12:
                raise ValueError(f'Якобиан почти нулевой на итерации {iteration}.')

            state = state - torch.linalg.solve(boundary_jacobian, boundary_value)

            if iteration == max_iter - 1:
                u, s, vh = torch.linalg.svd(boundary_jacobian)
                sigma_min = s[-1]
                print('Minimal Jacobian matrix singular value:', sigma_min)
                print('Jacobian matrix determinant:', torch.linalg.det(boundary_jacobian))
        else:
            raise RuntimeError(f'За {max_iter} итераций не сошлось')

        return torch.cat((self.a_state.flip(dims=[0]), self.b_state[1:]), dim=0).detach()

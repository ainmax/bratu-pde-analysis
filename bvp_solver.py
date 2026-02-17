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
            a: float,
            b: float,
            m: float,
            grid_size: float
    ):
        self.f: Callable = f
        self.boundary: Callable = boundary
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

        boundary_value: torch.Tensor = self.boundary(self.a, self.b, self.a_state[-1], self.b_state[-1])

        jacobian = torch.zeros(len(boundary_value), len(mid_state))
        for i in range(len(mid_state)):
            boundary_value[i].backward(retain_graph=True)
            jacobian[i] = mid_state.grad
            mid_state.grad.data.zero_()

        return boundary_value, jacobian

    def check_jacobian_svd(self, J: torch.Tensor, *, rtol=1e-12, atol=1e-14):
        """
        Проверяет вырожденность/плохую обусловленность по сингулярным числам.
        rtol — относительный порог к максимальной сингулярной.
        atol — абсолютный порог.
        Возвращает (ok, info_dict).
        """
        # SVDvals устойчивее, чем det/cond через inverse
        s = torch.linalg.svdvals(J)  # sorted desc
        s_max = s[0]
        s_min = s[-1]

        # Численный ранг: сколько сингулярных чисел "существенные"
        tol = torch.maximum(atol * torch.ones((), dtype=J.dtype, device=J.device),
                            rtol * s_max)
        rank = int((s > tol).sum().item())

        # оценка cond; если s_min==0, cond=inf
        cond = (s_max / s_min).item() if s_min > 0 else float("inf")

        ok = (rank == min(J.shape)) and torch.isfinite(s).all().item()

        info = {
            "shape": tuple(J.shape),
            "s_max": s_max.item(),
            "s_min": s_min.item(),
            "rank": rank,
            "tol": tol.item(),
            "cond_est": cond,
        }
        return ok, info

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
            print(boundary_value)

            if torch.max(torch.abs(boundary_value)) < tol:
                print(f"Converged in {iteration + 1} iterations")
                break

            # ok, info = self.check_jacobian_svd(boundary_jacobian, rtol=1e-12, atol=1e-14)
            # if not ok or info["cond_est"] > 1e12:
            #     raise ValueError(f"Bad Jacobian: {info}")

            if torch.max(torch.abs(boundary_jacobian)) < 1e-12:
                raise ValueError(f"Якобиан почти вырожденный на итерации {iteration}")

            state = state - torch.linalg.inv(boundary_jacobian) @ boundary_value
        else:
            raise RuntimeError(f"За {max_iter} итераций не сошлось")

        return torch.cat((self.a_state.flip(dims=[0]), self.b_state[1::]), dim=0).detach()

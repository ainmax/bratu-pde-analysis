import copy
import math

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bvp_solver import BVPSolver

torch.set_default_dtype(torch.float64)


def make_plot(
        *,
        x_grid: list[float],
        f_values: list[list[float]],
        g_values: list[list[float]],
        n: int,
        file_index: int
):
    fig, axes = plt.subplots(n + 1, 2, figsize=(14, 3 * (n + 1)))

    for i in range(n):
        axes[i][0].plot(x_grid, f_values[i], 'b-', label='f', linewidth=2)
        axes[i][0].legend(fontsize=10)
        axes[i][0].grid(True, alpha=0.3)
        axes[i][0].set_title(f'f{i + 1} approximation', fontsize=14)

        axes[i][1].plot(x_grid, g_values[i], 'b-', label='Решение', linewidth=2)
        axes[i][1].legend(fontsize=10)
        axes[i][1].grid(True, alpha=0.3)
        axes[i][1].set_title(f'g{i + 1} approximation', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"plot{file_index}.png", dpi=150)
    print(f"График сохранен: plot{file_index}.png")
    plt.close()


# Equation -\Delta u(x, y) + \lambda e^{u(x, y) = 0}
class PDESolver:
    def __init__(
            self,
            *,
            grid_size: float,
            bvp_tolerance: float,
            convergence_tolerance: float,
            initial_lambda: float,
            default_lambda_step: float,
            target_lambda: float,
            n: int
    ):
        self.grid: list[float] = []
        self.half_grid: list[float] = []
        self.grid_size: float = grid_size
        self.half_grid_size: float = grid_size / 2

        self.n: int = n

        self.f_tensor: list[list[torch.Tensor]] = [[], [], []]
        self.f_tensor_prev: list[list[torch.Tensor]] = [[], [], []]
        self.f_tensor_prev_prev: list[list[torch.Tensor]] = [[], [], []]
        self.g_tensor: list[list[torch.Tensor]] = [[], [], []]
        self.g_tensor_prev: list[list[torch.Tensor]] = [[], [], []]
        self.g_tensor_prev_prev: list[list[torch.Tensor]] = [[], [], []]

        self.bvp_tolerance: float = bvp_tolerance
        self.convergence_tolerance = convergence_tolerance

        self.target_lambda: float = target_lambda
        self.default_lambda_step: float = default_lambda_step
        self.initial_lambda: float = initial_lambda

    def _setup(self):
        self.grid = [
            x * self.grid_size
            for x in range(
                int(-1 / self.grid_size),
                int(1 / self.grid_size) + 1
            )
        ]
        self.half_grid = [
            x * self.half_grid_size
            for x in range(
                -int(1 / self.half_grid_size),
                int(1 / self.half_grid_size) + 1
            )
        ]

        def calc_basis(t: float):
            return torch.tensor([
                math.sin((i + 1) * math.pi * (t + 1) / 2)
                for i in range(self.n)
            ])

        def calc_basis_p(t: float):
            return torch.tensor([
                (i + 1) * math.pi / 2 * math.cos((i + 1) * math.pi * (t + 1) / 2)
                for i in range(self.n)
            ])

        def calc_basis_pp(t: float):
            return torch.tensor([
                -((i + 1) * math.pi / 2)**2 * math.sin((i + 1) * math.pi * (t + 1) / 2)
                for i in range(self.n)
            ])

        self.f_tensor[0] = [calc_basis(x) for x in self.half_grid]
        self.g_tensor[0] = [calc_basis(y) for y in self.half_grid]
        self.f_tensor[1] = [calc_basis_p(x) for x in self.half_grid]
        self.g_tensor[1] = [calc_basis_p(y) for y in self.half_grid]
        self.f_tensor[2] = [calc_basis_pp(x) for x in self.half_grid]
        self.g_tensor[2] = [calc_basis_pp(y) for y in self.half_grid]

        self.f_tensor_prev = copy.deepcopy(self.f_tensor)
        self.g_tensor_prev = copy.deepcopy(self.g_tensor)
        self.f_tensor_prev_prev = copy.deepcopy(self.f_tensor)
        self.g_tensor_prev_prev = copy.deepcopy(self.g_tensor)

        # log
        print("Setup completed")

    def _calc_divergence_norm(self, lambda_: float):
        integral: float = 0.0
        for x in self.grid:
            for y in self.grid:
                integral += (
                    (
                        torch.dot(self.f_tensor[2][self._idx(x)], self.g_tensor[0][self._idx(y)]) +
                        torch.dot(self.g_tensor[2][self._idx(y)], self.f_tensor[0][self._idx(x)]) -
                        lambda_ * torch.exp(torch.dot(
                            self.f_tensor[0][self._idx(x)],
                            self.g_tensor[0][self._idx(y)]
                        )
                    )
                ).item())**2

        return math.sqrt(integral / (len(self.grid)**2)) / 2

    def solve(self):
        # log
        print(f"Process with n = {self.n} started")

        self._setup()
        self._do_lambda_continuation()

    def _do_lambda_continuation(self):
        lambda_ = self.initial_lambda
        step = self.default_lambda_step
        step_prev = self.default_lambda_step
        step_prev_prev = self.default_lambda_step

        is_target_achieved: bool = False
        iteration_index: int = 0

        f_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor)
        f_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev)
        f_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev_prev)
        g_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor)
        g_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev)
        g_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev_prev)

        while not is_target_achieved:
            # log
            print("=" * 50)
            print(f"Process for lambda = {lambda_} and step = {step} is started.")

            self._extrapolate_variables(
                delta=step,
                delta_prev=step_prev,
                delta_prev_prev=step_prev_prev
            )

            try:
                self._pass_iterations(lambda_)
            except Exception as e:
                print(e)
                if lambda_ == self.initial_lambda:
                    raise ValueError("Bad initial lambda value.")
                lambda_ -= step
                step /= 1.5
                lambda_ += step
                self.f_tensor = copy.deepcopy(f_backup)
                self.f_tensor_prev = copy.deepcopy(f_prev_backup)
                self.f_tensor_prev_prev = copy.deepcopy(f_prev_prev_backup)
                self.g_tensor = copy.deepcopy(g_backup)
                self.g_tensor_prev = copy.deepcopy(g_prev_backup)
                self.g_tensor_prev_prev = copy.deepcopy(g_prev_prev_backup)

                continue

            f_backup = copy.deepcopy(self.f_tensor)
            f_prev_backup = copy.deepcopy(self.f_tensor_prev)
            f_prev_prev_backup = copy.deepcopy(self.f_tensor_prev_prev)
            g_backup = copy.deepcopy(self.g_tensor)
            g_prev_backup = copy.deepcopy(self.g_tensor_prev)
            g_prev_prev_backup = copy.deepcopy(self.g_tensor_prev_prev)

            make_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
                n=self.n,
                file_index=-(iteration_index + 1),
            )

            lambda_ += step

            step_prev_prev = step_prev
            step_prev = step
            step *= 1.1
            step = max(step, self.target_lambda - lambda_)

            iteration_index += 1

    def _extrapolate_variables(
            self,
            *,
            delta: float,
            delta_prev: float,
            delta_prev_prev: float
    ):
        new_f_tensor = copy.deepcopy(self.f_tensor)
        new_g_tensor = copy.deepcopy(self.g_tensor)

        for i in range(len(self.half_grid)):
            for j in range(3):
                f_delta = (self.f_tensor[j][i] - self.f_tensor_prev[j][i]) / delta_prev
                f_delta_prev = (self.f_tensor_prev[j][i] - self.f_tensor_prev_prev[j][i]) / delta_prev_prev
                f_delta_delta = 2 * (f_delta - f_delta_prev) / (delta_prev + delta_prev_prev)
                new_f_tensor[j][i] += f_delta * delta + f_delta_delta / 2 * delta**2

                g_delta = (self.g_tensor[j][i] - self.g_tensor_prev[j][i]) / delta_prev
                g_delta_prev = (self.g_tensor_prev[j][i] - self.g_tensor_prev_prev[j][i]) / delta_prev_prev
                g_delta_delta = 2 * (g_delta - g_delta_prev) / (delta_prev + delta_prev_prev)
                new_g_tensor[j][i] += g_delta * delta + g_delta_delta / 2 * delta**2


        self.f_tensor_prev_prev = copy.deepcopy(self.f_tensor_prev)
        self.g_tensor_prev_prev = copy.deepcopy(self.g_tensor_prev)

        self.f_tensor_prev = copy.deepcopy(self.f_tensor)
        self.g_tensor_prev = copy.deepcopy(self.g_tensor)

        self.f_tensor = new_f_tensor
        self.g_tensor = new_g_tensor

    def _pass_iterations(self, lambda_: float):
        def boundary(a: float, b: float, a_state: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_state[:self.n], b_state[:self.n]), dim=0)

        iteration_index = -1
        while True:
            iteration_index += 1
            # log
            print("Iteration by f started")

            make_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
                n=self.n,
                file_index=2 * iteration_index,
            )

            last_f: torch.Tensor = torch.stack(self.f_tensor[0], dim=0)
            last_g: torch.Tensor = torch.stack(self.g_tensor[0], dim=0)

            # solve bvp by f

            # def func_a_g(y_: float, state_a: torch.Tensor):
            #     return torch.outer(self.g_tensor[0][self._idx(y_)], self.g_tensor[0][self._idx(y_)])
            #
            # solver_a = ODESolver(func_a_g)
            # matrix_a: torch.Tensor = solver_a.solve_by_end(
            #     init_state=torch.zeros(self.n, self.n),
            #     a=-1,
            #     b=1,
            #     grid_size=self.grid_size
            # )
            # matrix_a = torch.linalg.inv(matrix_a)
            #
            # def func_b_g(y_: float, state_a: torch.Tensor):
            #     return -torch.outer(self.g_tensor[0][self._idx(y_)], self.g_tensor[2][self._idx(y_)])
            #
            # solver_b = ODESolver(func_b_g)
            # matrix_b: torch.Tensor = solver_b.solve_by_end(
            #     init_state=torch.zeros(self.n, self.n),
            #     a=-1,
            #     b=1,
            #     grid_size=self.grid_size
            # )

            G_tensor = torch.stack(self.g_tensor[0])
            Gpp_tensor = torch.stack(self.g_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(G_tensor.unsqueeze(2), G_tensor.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a)

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(G_tensor.unsqueeze(2), Gpp_tensor.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_f(x_: float, state: torch.Tensor) -> torch.Tensor:
                f_at_x = state[:self.n]
                u_all_y = torch.mv(G_tensor, f_at_x)
                integrand = G_tensor * torch.exp(u_all_y).unsqueeze(1)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                primes = matrix_a @ (matrix_b @ f_at_x + lambda_ * matrix_c)

                return torch.cat((state[self.n:], primes), dim=0)

            bvp_solver_ = BVPSolver(
                f=func_f,
                boundary=boundary,
                a=-1,
                b=1,
                m=0,
                grid_size=self.half_grid_size
            )
            bvp_init_state = torch.cat(
                (
                    self.f_tensor[0][self._idx(0)],
                    self.f_tensor[1][self._idx(0)]
                ),
                dim=0
            ).requires_grad_(True)
            f_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.f_tensor[0][i] = f_solution[i][:self.n]
                self.f_tensor[1][i] = f_solution[i][self.n:]
                self.f_tensor[2][i] = func_f(self.half_grid[i], f_solution[i])[self.n:]

            # log
            print("f variables updated")

            # solve bvp by g

            f_tensor = torch.abs(torch.stack(self.f_tensor[0], dim=0))
            g_tensor = torch.abs(torch.stack(self.g_tensor[0], dim=0))

            for i in range(len(self.half_grid)):
                for k in range(self.n):
                    f_max: float = torch.max(f_tensor[:, k]).item()
                    g_max: float = torch.max(g_tensor[:, k]).item()
                    f_multiplier = math.sqrt(g_max / f_max)
                    g_multiplier = math.sqrt(f_max / g_max)
                    self.f_tensor[0][i][k] *= f_multiplier
                    self.f_tensor[1][i][k] *= f_multiplier
                    self.f_tensor[2][i][k] *= f_multiplier
                    self.g_tensor[0][i][k] *= g_multiplier
                    self.g_tensor[1][i][k] *= g_multiplier
                    self.g_tensor[2][i][k] *= g_multiplier

            make_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
                n=self.n,
                file_index=2 * iteration_index + 1,
            )

            # log
            print("Iteration by g started")

            # def func_a_f(x_: float, state_a: torch.Tensor):
            #     return torch.outer(self.f_tensor[0][self._idx(x_)], self.f_tensor[0][self._idx(x_)])
            #
            # solver_a = ODESolver(func_a_f)
            # matrix_a: torch.Tensor = solver_a.solve_by_end(
            #     init_state=torch.zeros(self.n, self.n),
            #     a=-1,
            #     b=1,
            #     grid_size=self.grid_size
            # )
            # print(torch.max(torch.abs(matrix_a - matrix_a_new)))
            #
            # def func_b_f(x_: float, state_a: torch.Tensor):
            #     return -torch.outer(self.f_tensor[0][self._idx(x_)], self.f_tensor[2][self._idx(x_)])
            #
            # solver_b = ODESolver(func_b_f)
            # matrix_b: torch.Tensor = solver_b.solve_by_end(
            #     init_state=torch.zeros(self.n, self.n),
            #     a=-1,
            #     b=1,
            #     grid_size=self.grid_size
            # )
            # print(torch.max(torch.abs(matrix_b_new - matrix_b)))

            F_tensor = torch.stack(self.f_tensor[0])
            Fpp_tensor = torch.stack(self.f_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(F_tensor.unsqueeze(2), F_tensor.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a)

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(F_tensor.unsqueeze(2), Fpp_tensor.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_g(y_: float, state: torch.Tensor) -> torch.Tensor:
                g_at_y = state[:self.n]
                u_all_x = torch.mv(F_tensor, g_at_y)
                integrand = F_tensor * torch.exp(u_all_x).unsqueeze(1)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                primes = matrix_a @ (matrix_b @ g_at_y + lambda_ * matrix_c)

                return torch.cat((state[self.n:], primes), dim=0)

            bvp_solver_ = BVPSolver(
                f=func_g,
                boundary=boundary,
                a=-1,
                b=1,
                m=0,
                grid_size=self.half_grid_size
            )
            bvp_init_state = torch.cat(
                (
                    self.g_tensor[0][self._idx(0)],
                    self.g_tensor[1][self._idx(0)]
                ),
                dim=0
            ).requires_grad_(True)
            g_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.g_tensor[0][i] = g_solution[i][:self.n]
                self.g_tensor[1][i] = g_solution[i][self.n:]
                self.g_tensor[2][i] = func_g(self.half_grid[i], g_solution[i])[self.n:]

            f_tensor = torch.abs(torch.stack(self.f_tensor[0], dim=0))
            g_tensor = torch.abs(torch.stack(self.g_tensor[0], dim=0))

            for i in range(len(self.half_grid)):
                for k in range(self.n):
                    f_max: float = torch.max(f_tensor[:, k]).item()
                    g_max: float = torch.max(g_tensor[:, k]).item()
                    f_multiplier = math.sqrt(g_max / f_max)
                    g_multiplier = math.sqrt(f_max / g_max)
                    self.f_tensor[0][i][k] *= f_multiplier
                    self.f_tensor[1][i][k] *= f_multiplier
                    self.f_tensor[2][i][k] *= f_multiplier
                    self.g_tensor[0][i][k] *= g_multiplier
                    self.g_tensor[1][i][k] *= g_multiplier
                    self.g_tensor[2][i][k] *= g_multiplier

            # log
            print("g variables updated")

            f_difference = torch.max(torch.abs(torch.stack(self.f_tensor[0], dim=0) - last_f)).item()
            g_difference = torch.max(torch.abs(torch.stack(self.g_tensor[0], dim=0) - last_g)).item()

            print(
                f_difference,
                g_difference,
                self._calc_divergence_norm(lambda_)
            )

            if max(f_difference, g_difference) < self.convergence_tolerance:
                break


    def _idx(self, t: float):
        return int((t + 1) / self.half_grid_size)


if __name__ == "__main__":
    pde_solver = PDESolver(
        grid_size=1 / 32,
        bvp_tolerance=1e-6,
        convergence_tolerance=1e-4,
        initial_lambda=-0.001,
        default_lambda_step=-0.5,
        target_lambda=-6,
        n=2
    )

    pde_solver.solve()

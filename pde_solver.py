import copy
import math

import torch

import matplotlib
import matplotlib.pyplot as plt

from bvp_solver import BVPSolver

matplotlib.use('Agg')
torch.set_default_dtype(torch.float64)


def make_approximation_plot(
        *,
        x_grid: list[float],
        f_values: list[list[float]],
        g_values: list[list[float]],
        n: int,
        suffix: str,
        file_index: int
):
    fig, axes = plt.subplots(n + 1, 2, figsize=(14, 3 * (n + 1)))

    for i in range(n):
        axes[i][0].plot(x_grid, f_values[i], 'b-', label='f', linewidth=1)
        axes[i][0].legend(fontsize=10)
        axes[i][0].grid(True, alpha=0.3)
        axes[i][0].set_title(f'f{i + 1} approximation', fontsize=10)

        axes[i][1].plot(x_grid, g_values[i], 'b-', label='Решение', linewidth=1)
        axes[i][1].legend(fontsize=10)
        axes[i][1].grid(True, alpha=0.3)
        axes[i][1].set_title(f'g{i + 1} approximation', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"plots/plot-{suffix}{file_index}.png", dpi=150)
    print(f"График сохранен: plot{file_index}.png")
    plt.close()


def make_continuation_plot(data: list[tuple[float, float]]):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.plot([e[0] for e in data], [e[1] for e in data], 'b-', label='f', linewidth=1)
    axes.set_xlabel("lambda", fontsize=12)
    axes.set_ylabel("norm", fontsize=12)
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    axes.set_title(f'Continuation plot', fontsize=14)
    axes.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"plots/plot-c.png", dpi=300)
    print(f"График сохранен: plot-c.png")
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
            initial_lambda_step: float,
            max_lambda_step: float,
            limit_tolerance: float,
            initial_norm_step: float,
            max_norm_step: float,
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

        self.bvp_tolerance: float = abs(bvp_tolerance)
        self.convergence_tolerance: float = abs(convergence_tolerance)

        self.initial_lambda_step: float = initial_lambda_step
        self.max_lambda_step: float = abs(max_lambda_step)
        self.initial_lambda: float = initial_lambda
        self.limit_tolerance: float = abs(limit_tolerance)

        self.initial_norm_step: float = initial_norm_step
        self.max_norm_step: float = abs(max_norm_step)

        self.continuation_data: list[tuple[float, float]] = []

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
                math.sin((i + 1) * math.pi * (t + 1) / 2) / 10**i
                for i in range(self.n)
            ])

        def calc_basis_p(t: float):
            return torch.tensor([
                (i + 1) * math.pi / 2 * math.cos((i + 1) * math.pi * (t + 1) / 2) / 10**i
                for i in range(self.n)
            ])

        def calc_basis_pp(t: float):
            return torch.tensor([
                -((i + 1) * math.pi / 2)**2 * math.sin((i + 1) * math.pi * (t + 1) / 2) / 10**i
                for i in range(self.n)
            ])

        self.f_tensor[0] = [calc_basis(x) for x in self.half_grid]
        self.g_tensor[0] = [calc_basis(y) for y in self.half_grid]
        self.f_tensor[1] = [calc_basis_p(x) for x in self.half_grid]
        self.g_tensor[1] = [calc_basis_p(y) for y in self.half_grid]
        self.f_tensor[2] = [calc_basis_pp(x) for x in self.half_grid]
        self.g_tensor[2] = [calc_basis_pp(y) for y in self.half_grid]

        self.f_tensor_prev = [[torch.tensor([0] * self.n) for _ in self.half_grid] for _ in range(3)]
        self.g_tensor_prev = [[torch.tensor([0] * self.n) for _ in self.half_grid] for _ in range(3)]
        self.f_tensor_prev_prev = [[torch.tensor([0] * self.n) for _ in self.half_grid] for _ in range(3)]
        self.g_tensor_prev_prev = [[torch.tensor([0] * self.n) for _ in self.half_grid] for _ in range(3)]

        # log
        print("Setup is completed.")

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
        self._do_parameter_continuation()

    def _do_parameter_continuation(self):
        lambda_ = self.initial_lambda
        step = self.initial_lambda_step
        step_prev = step
        step_prev_prev = step

        is_limit_achieved: bool = False
        iteration_index: int = 0

        for n in range(1, self.n + 1):
            self._pass_lambda_iterations(lambda_, n)
        lambda_ += step

        f_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor)
        f_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev)
        f_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev_prev)
        g_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor)
        g_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev)
        g_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev_prev)

        while not is_limit_achieved:
            # log
            print("=" * 50)
            print(f"Process for lambda = {lambda_} and step = {step} is started.")

            if abs(step) < self.limit_tolerance:
                is_limit_achieved = True
                print(200 * "=")
                print("Limit by lambda is achieved")
                break

            self._extrapolate_variables(
                delta=step,
                delta_prev=step_prev,
                delta_prev_prev=step_prev_prev
            )

            try:
                for n in range(1, self.n + 1):
                    self._pass_lambda_iterations(lambda_, n)
            except Exception as e:
                print(e)

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

            make_approximation_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
                n=self.n,
                suffix="subcrit",
                file_index=-(iteration_index + 1)
            )

            step_prev_prev = step_prev
            step_prev = step
            step *= 1.5
            step = math.copysign(min(abs(step), self.max_lambda_step), step)

            # log
            print("=" * 100)
            print(f"lambda = {lambda_} is passed.")

            self.continuation_data.append(
                (
                    lambda_,
                    torch.inner(self.f_tensor[0][self._idx(0)], self.g_tensor[0][self._idx(0)]).item()
                )
            )

            make_continuation_plot(self.continuation_data)

            lambda_ += step
            iteration_index += 1

        # Change continuation variable
        print("Starting norm continuation")

        lambda_ = torch.tensor([self.continuation_data[-1][0]])
        norm = torch.tensor([self.continuation_data[-1][1]])

        step = self.initial_norm_step
        step_prev = step
        step_prev_prev = step

        iteration_index: int = 0

        self._pass_norm_iterations(lambda_, norm)
        norm += step

        f_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor)
        f_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev)
        f_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.f_tensor_prev_prev)
        g_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor)
        g_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev)
        g_prev_prev_backup: list[list[torch.Tensor]] = copy.deepcopy(self.g_tensor_prev_prev)
        lambda_backup: torch.Tensor = lambda_.clone()

        while True:
            # log
            print("=" * 50)
            print(f"Process for lambda = {lambda_}, norm = {norm} and step = {step} is started.")

            self._extrapolate_variables(
                delta=step,
                delta_prev=step_prev,
                delta_prev_prev=step_prev_prev
            )

            try:
                lambda_ = self._pass_norm_iterations(lambda_, norm).clone()
            except Exception as e:
                print(e)

                norm -= step
                step /= 1.5
                norm += step

                self.f_tensor = copy.deepcopy(f_backup)
                self.f_tensor_prev = copy.deepcopy(f_prev_backup)
                self.f_tensor_prev_prev = copy.deepcopy(f_prev_prev_backup)
                self.g_tensor = copy.deepcopy(g_backup)
                self.g_tensor_prev = copy.deepcopy(g_prev_backup)
                self.g_tensor_prev_prev = copy.deepcopy(g_prev_prev_backup)
                lambda_ = lambda_backup.clone()

                continue

            f_backup = copy.deepcopy(self.f_tensor)
            f_prev_backup = copy.deepcopy(self.f_tensor_prev)
            f_prev_prev_backup = copy.deepcopy(self.f_tensor_prev_prev)
            g_backup = copy.deepcopy(self.g_tensor)
            g_prev_backup = copy.deepcopy(self.g_tensor_prev)
            g_prev_prev_backup = copy.deepcopy(self.g_tensor_prev_prev)
            lambda_backup = lambda_.clone()

            make_approximation_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
                n=self.n,
                suffix="supercrit",
                file_index=-(iteration_index + 1)
            )

            step_prev_prev = step_prev
            step_prev = step
            step *= 1.5
            step = math.copysign(min(abs(step), self.max_norm_step), step)

            # log
            print("=" * 100)
            print(f"lambda = {lambda_} and norm = {norm} are passed.")

            self.continuation_data.append(
                (
                    lambda_.item(),
                    torch.inner(self.f_tensor[0][self._idx(0)], self.g_tensor[0][self._idx(0)]).item()
                )
            )

            make_continuation_plot(self.continuation_data)

            norm += step
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

    def _pass_lambda_iterations(self, lambda_: float, n: int):
        def boundary(a: float, b: float, a_state: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_state[0:1], b_state[0:1]), dim=0)

        iteration_index = -1
        while True:
            iteration_index += 1
            # log
            print("Iteration by f started")

            # make_approximation_plot(
            #     x_grid=self.half_grid,
            #     f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
            #     g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
            #     n=self.n,
            #     suffix="subcrit",
            #     file_index=2 * iteration_index
            # )

            last_f: torch.Tensor = torch.stack(self.f_tensor[0], dim=0)
            last_g: torch.Tensor = torch.stack(self.g_tensor[0], dim=0)

            # solve bvp by f

            g_tensor_stack = torch.stack(self.g_tensor[0])
            gpp_tensor_stack = torch.stack(self.g_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(g_tensor_stack.unsqueeze(2), g_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a[:n, :n])

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(g_tensor_stack.unsqueeze(2), gpp_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_f(x_: float, state: torch.Tensor) -> torch.Tensor:
                if n == 1:
                    u_all_y = g_tensor_stack[:, 0] * state[0]
                    integrand = g_tensor_stack[:, 0] * torch.exp(u_all_y)
                    matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                    prime = matrix_a[0, 0:1] * (matrix_b[0, 0] * state[0] + lambda_ * matrix_c)
                    return torch.cat((state[1:2], prime), dim=0)
                else:
                    f_at_x = torch.cat((self.f_tensor[0][self._idx(x_)][:n - 1], state[0:1]), dim=0)
                    u_all_y = torch.mv(g_tensor_stack[:, :n], f_at_x)
                    integrand = g_tensor_stack[:, :n] * torch.exp(u_all_y).unsqueeze(1)
                    matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                    prime = matrix_a[n - 1:, :] @ (torch.mv(matrix_b[:n, :n], f_at_x).unsqueeze(1) + lambda_ * matrix_c.unsqueeze(1))
                    return torch.cat((state[1:2], prime[0]), dim=0)

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
                    self.f_tensor[0][self._idx(0)][n - 1:n],
                    self.f_tensor[1][self._idx(0)][n - 1:n]
                ),
                dim=0
            )
            f_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.f_tensor[0][i][n - 1] = f_solution[i][:1]
                self.f_tensor[1][i][n - 1] = f_solution[i][1:]
                self.f_tensor[2][i][n - 1] = func_f(self.half_grid[i], f_solution[i])[1:]

            # log
            print("f variables updated")

            # solve bvp by g

            f_tensor = torch.abs(torch.stack(self.f_tensor[0], dim=0))
            g_tensor = torch.abs(torch.stack(self.g_tensor[0], dim=0))

            for i in range(len(self.half_grid)):
                f_max: float = torch.max(f_tensor[:, n - 1]).item()
                g_max: float = torch.max(g_tensor[:, n - 1]).item()
                f_multiplier = math.sqrt(g_max / f_max)
                g_multiplier = math.sqrt(f_max / g_max)
                self.f_tensor[0][i][n - 1] *= f_multiplier
                self.f_tensor[1][i][n - 1] *= f_multiplier
                self.f_tensor[2][i][n - 1] *= f_multiplier
                self.g_tensor[0][i][n - 1] *= g_multiplier
                self.g_tensor[1][i][n - 1] *= g_multiplier
                self.g_tensor[2][i][n - 1] *= g_multiplier

            # make_approximation_plot(
            #     x_grid=self.half_grid,
            #     f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
            #     g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
            #     n=self.n,
            #     suffix="subcrit",
            #     file_index=2 * iteration_index + 1
            # )

            # log
            print("Iteration by g started")

            f_tensor_stack = torch.stack(self.f_tensor[0])
            fpp_tensor_stack = torch.stack(self.f_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(f_tensor_stack.unsqueeze(2), f_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a[:n, :n])

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(f_tensor_stack.unsqueeze(2), fpp_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_g(y_: float, state: torch.Tensor) -> torch.Tensor:
                if n == 1:
                    u_all_x = f_tensor_stack[:, 0] * state[0]
                    integrand = f_tensor_stack[:, 0] * torch.exp(u_all_x)
                    matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                    prime = matrix_a[0, 0:1] * (matrix_b[0, 0] * state[0] + lambda_ * matrix_c)
                    return torch.cat((state[1:2], prime), dim=0)
                else:
                    g_at_y = torch.cat((self.g_tensor[0][self._idx(y_)][:n - 1], state[0:1]), dim=0)
                    u_all_x = torch.mv(f_tensor_stack[:, :n], g_at_y)
                    integrand = f_tensor_stack[:, :n] * torch.exp(u_all_x).unsqueeze(1)
                    matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                    prime = matrix_a[n - 1:, :] @ (torch.mv(matrix_b[:n, :n], g_at_y).unsqueeze(1) + lambda_ * matrix_c.unsqueeze(1))
                    return torch.cat((state[1:2], prime[0]), dim=0)

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
                    self.g_tensor[0][self._idx(0)][n - 1:n],
                    self.g_tensor[1][self._idx(0)][n - 1:n]
                ),
                dim=0
            )
            g_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.g_tensor[0][i][n - 1] = g_solution[i][:1]
                self.g_tensor[1][i][n - 1] = g_solution[i][1:]
                self.g_tensor[2][i][n - 1] = func_g(self.half_grid[i], g_solution[i])[1:]

            f_tensor = torch.abs(torch.stack(self.f_tensor[0], dim=0))
            g_tensor = torch.abs(torch.stack(self.g_tensor[0], dim=0))

            for i in range(len(self.half_grid)):
                f_max: float = torch.max(f_tensor[:, n - 1]).item()
                g_max: float = torch.max(g_tensor[:, n - 1]).item()
                f_multiplier = math.sqrt(g_max / f_max)
                g_multiplier = math.sqrt(f_max / g_max)
                self.f_tensor[0][i][n - 1] *= f_multiplier
                self.f_tensor[1][i][n - 1] *= f_multiplier
                self.f_tensor[2][i][n - 1] *= f_multiplier
                self.g_tensor[0][i][n - 1] *= g_multiplier
                self.g_tensor[1][i][n - 1] *= g_multiplier
                self.g_tensor[2][i][n - 1] *= g_multiplier

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

    def _pass_norm_iterations(self, lambda_: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        def boundary(a: float, b: float, a_state: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_state[:self.n], b_state[:self.n]), dim=0)

        iteration_index = -1
        while True:
            iteration_index += 1
            # log
            print("Iteration by f started")

            # make_approximation_plot(
            #     x_grid=self.half_grid,
            #     f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
            #     g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
            #     n=self.n,
            #     suffix="supercrit",
            #     file_index=2 * iteration_index
            # )

            last_f: torch.Tensor = torch.stack(self.f_tensor[0], dim=0)
            last_g: torch.Tensor = torch.stack(self.g_tensor[0], dim=0)

            # solve bvp by f

            g_tensor_stack = torch.stack(self.g_tensor[0])
            gpp_tensor_stack = torch.stack(self.g_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(g_tensor_stack.unsqueeze(2), g_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a)

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(g_tensor_stack.unsqueeze(2), gpp_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_f(x_: float, state: torch.Tensor) -> torch.Tensor:
                f_at_x = state[:self.n]
                u_all_y = torch.mv(g_tensor_stack, f_at_x)
                integrand = g_tensor_stack * torch.exp(u_all_y).unsqueeze(1)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                primes = matrix_a @ (matrix_b @ f_at_x + state[-1] * matrix_c)

                return torch.cat((state[self.n:-1], primes, torch.tensor([0])), dim=0)

            def f_init_boundary(state: torch.Tensor) -> torch.Tensor:
                return torch.inner(state[:self.n], g_tensor_stack[self._idx(0)]).unsqueeze(0) - norm

            bvp_solver_ = BVPSolver(
                f=func_f,
                boundary=boundary,
                init_boundary=f_init_boundary,
                a=-1,
                b=1,
                m=0,
                grid_size=self.half_grid_size
            )
            bvp_init_state = torch.cat(
                (
                    self.f_tensor[0][self._idx(0)],
                    self.f_tensor[1][self._idx(0)],
                    lambda_
                ),
                dim=0
            )
            f_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.f_tensor[0][i] = f_solution[i][:self.n]
                self.f_tensor[1][i] = f_solution[i][self.n:-1]
                self.f_tensor[2][i] = func_f(self.half_grid[i], f_solution[i])[self.n:-1]

            lambda_ = f_solution[0][-1].unsqueeze(0)

            # log
            print(f"New lambda value = {lambda_}")
            print("f variables updated")

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

            # make_approximation_plot(
            #     x_grid=self.half_grid,
            #     f_values=[[e[k].item() for e in self.f_tensor[0]] for k in range(self.n)],
            #     g_values=[[e[k].item() for e in self.g_tensor[0]] for k in range(self.n)],
            #     n=self.n,
            #     suffix="supercrit",
            #     file_index=2 * iteration_index + 1
            # )

            # log
            print("Iteration by g started")

            f_tensor_stack = torch.stack(self.f_tensor[0])
            fpp_tensor_stack = torch.stack(self.f_tensor[2])

            matrix_a: torch.Tesor = torch.sum(torch.bmm(f_tensor_stack.unsqueeze(2), f_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size
            matrix_a = torch.linalg.inv(matrix_a)

            matrix_b: torch.Tesor = -1 * torch.sum(torch.bmm(f_tensor_stack.unsqueeze(2), fpp_tensor_stack.unsqueeze(1)), dim=0) * self.half_grid_size

            def func_g(y_: float, state: torch.Tensor) -> torch.Tensor:
                g_at_y = state[:self.n]
                u_all_x = torch.mv(f_tensor_stack, g_at_y)
                integrand = f_tensor_stack * torch.exp(u_all_x).unsqueeze(1)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size
                primes = matrix_a @ (matrix_b @ g_at_y + state[-1] * matrix_c)

                return torch.cat((state[self.n:-1], primes, torch.tensor([0])), dim=0)

            def g_init_boundary(state: torch.Tensor) -> torch.Tensor:
                return torch.inner(state[:self.n], f_tensor_stack[self._idx(0)]).unsqueeze(0) - norm

            bvp_solver_ = BVPSolver(
                f=func_g,
                boundary=boundary,
                init_boundary=g_init_boundary,
                a=-1,
                b=1,
                m=0,
                grid_size=self.half_grid_size
            )
            bvp_init_state = torch.cat(
                (
                    self.g_tensor[0][self._idx(0)],
                    self.g_tensor[1][self._idx(0)],
                    lambda_
                ),
                dim=0
            )
            g_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=10
            )

            for i in range(len(self.half_grid)):
                self.g_tensor[0][i] = g_solution[i][:self.n]
                self.g_tensor[1][i] = g_solution[i][self.n:-1]
                self.g_tensor[2][i] = func_g(self.half_grid[i], g_solution[i])[self.n:-1]

            lambda_ = g_solution[0][-1].unsqueeze(0)

            # log
            print(f"New lambda value = {lambda_}")
            print("g variables updated")

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

            f_difference = torch.max(torch.abs(torch.stack(self.f_tensor[0], dim=0) - last_f)).item()
            g_difference = torch.max(torch.abs(torch.stack(self.g_tensor[0], dim=0) - last_g)).item()

            print(
                f_difference,
                g_difference,
                self._calc_divergence_norm(lambda_.item())
            )

            if max(f_difference, g_difference) < self.convergence_tolerance:
                break

        return lambda_


    def _idx(self, t: float):
        return int((t + 1) / self.half_grid_size)


if __name__ == "__main__":
    pde_solver = PDESolver(
        grid_size=1 / 64,
        bvp_tolerance=1e-6,
        convergence_tolerance=1e-4,
        initial_lambda=-0.001,
        initial_lambda_step=-0.1,
        max_lambda_step=0.12,
        limit_tolerance=1e-4,
        initial_norm_step=1e-2,
        max_norm_step=0.25,
        n=3
    )

    pde_solver.solve()

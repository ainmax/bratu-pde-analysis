import math
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bvp_solver import BVPSolver
from ode_solver import ODESolver

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
            target_lambda: float
    ):
        self.grid: list[float] = []
        self.half_grid: list[float] = []
        self.grid_size: float = grid_size
        self.half_grid_size: float = grid_size / 2

        self.n: int = 0
        self.f: list[torch.Tensor] = []
        self.fp: list[torch.Tensor] = []
        self.fpp: list[torch.Tensor] = []
        self.g: list[torch.Tensor] = []
        self.gp: list[torch.Tensor] = []
        self.gpp: list[torch.Tensor] = []

        self.bvp_tolerance: float = bvp_tolerance
        self.convergence_tolerance = convergence_tolerance
        self.target_lambda: float = target_lambda

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

        self.f = [calc_basis(x) for x in self.half_grid]
        self.g = [calc_basis(y) for y in self.half_grid]
        self.fp = [calc_basis_p(x) for x in self.half_grid]
        self.gp = [calc_basis_p(y) for y in self.half_grid]
        self.fpp = [calc_basis_pp(x) for x in self.half_grid]
        self.gpp = [calc_basis_pp(y) for y in self.half_grid]

        # log
        print("Setup completed")

    def _calc_divergence_norm(self, lambda_: float):
        integral: float = 0.0
        for x in self.grid:
            for y in self.grid:
                integral += (
                    (
                        torch.dot(self.fpp[self._idx(x)], self.g[self._idx(y)]) +
                        torch.dot(self.gpp[self._idx(y)], self.f[self._idx(x)]) -
                        lambda_ * torch.exp(torch.dot(
                            self.f[self._idx(x)],
                            self.g[self._idx(y)]
                        )
                    )
                ).item())**2

        return math.sqrt(integral / (len(self.grid)**2)) / 2

    def solve(self):
        for n in range(3, 4):
            # log
            print(f"Process with n = {n} started")

            self.n = n
            self._setup()
            self._do_lambda_continuation()

    def _do_lambda_continuation(self):
        lambda_ = max(-0.01, self.target_lambda)
        steps_count = round(abs(self.target_lambda - lambda_) * 40) + 1
        step = (self.target_lambda - lambda_) / steps_count
        for i in range(steps_count):
            # log
            print("=" * 20)
            print(f"Process for lambda = {lambda_} started")

            self._pass_iterations(lambda_)
            make_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g] for k in range(self.n)],
                n=self.n,
                file_index=-(i + 1),
            )
            lambda_ += step

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
                f_values=[[e[k].item() for e in self.f] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g] for k in range(self.n)],
                n=self.n,
                file_index=2 * iteration_index,
            )

            last_f: torch.Tensor = torch.stack(self.f, dim=0)
            last_g: torch.Tensor = torch.stack(self.g, dim=0)

            # solve bvp by f

            def func_a_g(y_: float, state_a: torch.Tensor):
                return torch.outer(self.g[self._idx(y_)], self.g[self._idx(y_)])

            solver_a = ODESolver(func_a_g)
            matrix_a: torch.Tensor = solver_a.solve_by_end(
                init_state=torch.zeros(self.n, self.n),
                a=-1,
                b=1,
                grid_size=self.grid_size
            )
            matrix_a = torch.linalg.inv(matrix_a)

            def func_b_g(y_: float, state_a: torch.Tensor):
                return -torch.outer(self.g[self._idx(y_)], self.gpp[self._idx(y_)])

            solver_b = ODESolver(func_b_g)
            matrix_b: torch.Tensor = solver_b.solve_by_end(
                init_state=torch.zeros(self.n, self.n),
                a=-1,
                b=1,
                grid_size=self.grid_size
            )

            G_tensor = torch.stack(self.g)

            def func_f(x_: float, state: torch.Tensor) -> torch.Tensor:
                # 1. Извлекаем текущее состояние f(x) из вектора состояния RK4
                # state[:n] — значения f_i, state[n:] — значения f_i'
                f_at_x = state[:self.n]

                # 2. Вычисляем u(x, y) = sum(f_i(x) * g_i(y)) для ВСЕХ y сразу
                # Результат: тензор формы (GridSizeY,)
                u_all_y = torch.mv(G_tensor, f_at_x)

                # 3. Вычисляем нелинейный член (Квадратура интеграла g * exp(u) dy)
                # torch.exp(u_all_y).unsqueeze(1) создает столбец для вещания (broadcasting)
                # (GridSizeY, n) * (GridSizeY, 1) -> (GridSizeY, n)
                integrand = G_tensor * torch.exp(u_all_y).unsqueeze(1)

                # Суммируем по оси y и умножаем на шаг (метод трапеций/прямоугольников)
                # matrix_c — вектор формы (n,)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size

                # 4. Формируем вторые производные f''(x)
                # Используем ваши предосчитанные матрицы A и B
                # Уравнение: f'' = matrix_a @ (matrix_b @ f + lambda * Phi)
                # ВНИМАНИЕ: Проверьте знак минус перед matrix_a, если стрельба будет улетать
                primes = matrix_a @ (matrix_b @ f_at_x + lambda_ * matrix_c)

                # 5. Возвращаем [f', f''] для следующего шага RK4
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
                    self.f[self._idx(0)],
                    self.fp[self._idx(0)]
                ),
                dim=0
            ).requires_grad_(True)
            f_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=100
            )

            for i in range(len(self.half_grid)):
                self.f[i] = f_solution[i][:self.n]
                self.fp[i] = f_solution[i][self.n:]
                self.fpp[i] = func_f(self.half_grid[i], f_solution[i])[self.n:]

            # log
            print("f variables updated")

            # solve bvp by g

            f_tensor = torch.abs(torch.stack(self.f, dim=0))
            g_tensor = torch.abs(torch.stack(self.g, dim=0))

            for i in range(len(self.half_grid)):
                for k in range(self.n):
                    f_max: float = torch.max(f_tensor[:, k]).item()
                    g_max: float = torch.max(g_tensor[:, k]).item()
                    f_multiplier = math.sqrt(g_max / f_max)
                    g_multiplier = math.sqrt(f_max / g_max)
                    self.f[i][k] *= f_multiplier
                    self.fp[i][k] *= f_multiplier
                    self.fpp[i][k] *= f_multiplier
                    self.g[i][k] *= g_multiplier
                    self.gp[i][k] *= g_multiplier
                    self.gpp[i][k] *= g_multiplier

            make_plot(
                x_grid=self.half_grid,
                f_values=[[e[k].item() for e in self.f] for k in range(self.n)],
                g_values=[[e[k].item() for e in self.g] for k in range(self.n)],
                n=self.n,
                file_index=2 * iteration_index + 1,
            )

            # log
            print("Iteration by g started")


            def func_a_f(x_: float, state_a: torch.Tensor):
                return torch.outer(self.f[self._idx(x_)], self.f[self._idx(x_)])

            solver_a = ODESolver(func_a_f)
            matrix_a: torch.Tensor = solver_a.solve_by_end(
                init_state=torch.zeros(self.n, self.n),
                a=-1,
                b=1,
                grid_size=self.grid_size
            )
            matrix_a = torch.linalg.inv(matrix_a)

            def func_b_f(x_: float, state_a: torch.Tensor):
                return -torch.outer(self.f[self._idx(x_)], self.fpp[self._idx(x_)])

            solver_b = ODESolver(func_b_f)
            matrix_b: torch.Tensor = solver_b.solve_by_end(
                init_state=torch.zeros(self.n, self.n),
                a=-1,
                b=1,
                grid_size=self.grid_size
            )

            F_tensor = torch.stack(self.f)

            def func_g(y_: float, state: torch.Tensor) -> torch.Tensor:
                # 1. Извлекаем текущее состояние f(x) из вектора состояния RK4
                # state[:n] — значения f_i, state[n:] — значения f_i'
                g_at_y = state[:self.n]

                # 2. Вычисляем u(x, y) = sum(f_i(x) * g_i(y)) для ВСЕХ y сразу
                # Результат: тензор формы (GridSizeY,)
                u_all_x = torch.mv(F_tensor, g_at_y)

                # 3. Вычисляем нелинейный член (Квадратура интеграла g * exp(u) dy)
                # torch.exp(u_all_y).unsqueeze(1) создает столбец для вещания (broadcasting)
                # (GridSizeY, n) * (GridSizeY, 1) -> (GridSizeY, n)
                integrand = F_tensor * torch.exp(u_all_x).unsqueeze(1)

                # Суммируем по оси y и умножаем на шаг (метод трапеций/прямоугольников)
                # matrix_c — вектор формы (n,)
                matrix_c = torch.sum(integrand, dim=0) * self.half_grid_size

                # 4. Формируем вторые производные f''(x)
                # Используем ваши предосчитанные матрицы A и B
                # Уравнение: f'' = matrix_a @ (matrix_b @ f + lambda * Phi)
                # ВНИМАНИЕ: Проверьте знак минус перед matrix_a, если стрельба будет улетать
                primes = matrix_a @ (matrix_b @ g_at_y + lambda_ * matrix_c)

                # 5. Возвращаем [f', f''] для следующего шага RK4
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
                    self.g[self._idx(0)],
                    self.gp[self._idx(0)]
                ),
                dim=0
            ).requires_grad_(True)
            g_solution: torch.Tensor = bvp_solver_.solve(
                init_state=bvp_init_state,
                tol=self.bvp_tolerance,
                max_iter=100
            )

            for i in range(len(self.half_grid)):
                self.g[i] = g_solution[i][:self.n]
                self.gp[i] = g_solution[i][self.n:]
                self.gpp[i] = func_g(self.half_grid[i], g_solution[i])[self.n:]

            f_tensor = torch.abs(torch.stack(self.f, dim=0))
            g_tensor = torch.abs(torch.stack(self.g, dim=0))

            for i in range(len(self.half_grid)):
                for k in range(self.n):
                    f_max: float = torch.max(f_tensor[:, k]).item()
                    g_max: float = torch.max(g_tensor[:, k]).item()
                    f_multiplier = math.sqrt(g_max / f_max)
                    g_multiplier = math.sqrt(f_max / g_max)
                    self.f[i][k] *= f_multiplier
                    self.fp[i][k] *= f_multiplier
                    self.fpp[i][k] *= f_multiplier
                    self.g[i][k] *= g_multiplier
                    self.gp[i][k] *= g_multiplier
                    self.gpp[i][k] *= g_multiplier

            # log
            print("g variables updated")

            f_difference = torch.max(torch.abs(torch.stack(self.f, dim=0) - last_f)).item()
            g_difference = torch.max(torch.abs(torch.stack(self.g, dim=0) - last_g)).item()

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
        convergence_tolerance=1e-3,
        target_lambda=-0.2
    )

    pde_solver.solve()

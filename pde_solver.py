import copy
import dataclasses
import math

import torch

import matplotlib
import matplotlib.pyplot as plt

from bvp_solver import BVPSolver
from function_representation import FuncRepr

matplotlib.use('Agg')
torch.set_default_dtype(torch.float64)


@dataclasses.dataclass(kw_only=True)
class PDEState:
    lambda_: float
    repr: FuncRepr


class PDESolver:
    """
    PDESolver for equation:
    -Delta u(x, y) + lambda e^{u(x, y) = 0}
    """

    def __init__(
            self,
            *,
            grid_size_2exp: int,
            bvp_tolerance: float,
            convergence_tolerance: float,
            bvp_max_iter: int,
            initial_lambda: float,
            initial_lambda_step: float,
            max_lambda_step: float,
            limit_tolerance: float,
            initial_norm_step: float,
            max_norm_step: float,
            n: int,
            t0: float,
            t1: float
    ):
        self.n: int = n

        self.state: PDEState = PDEState(
            lambda_=initial_lambda,
            repr=FuncRepr(
                grid_size_2exp=grid_size_2exp,
                n=n,
                t0=t0,
                t1=t1
            )
        )
        self.last_states: list[PDEState] = [
            copy.deepcopy(self.state),
            copy.deepcopy(self.state),
            copy.deepcopy(self.state)
        ]

        self.bvp_tolerance: float = abs(bvp_tolerance)
        self.convergence_tolerance: float = abs(convergence_tolerance)
        self.bvp_max_iter = bvp_max_iter

        self.initial_lambda_step: float = initial_lambda_step
        self.max_lambda_step: float = abs(max_lambda_step)
        self.initial_lambda: float = initial_lambda
        self.limit_tolerance: float = abs(limit_tolerance)

        self.initial_norm_step: float = initial_norm_step
        self.max_norm_step: float = abs(max_norm_step)

        self.continuation_data: list[tuple[float, float]] = []

    def solve(self):
        # log
        print(f'Process with n = {self.n} started.')
        self._do_lambda_continuation()
        self._do_norm_continuation()

    def _calc_divergence_norm(self, n: int):
        integral: float = 0.0
        for ix in range(self.state.repr.subgrid_cardinality):
            for iy in range(self.state.repr.subgrid_cardinality):
                integral += (
                    (
                        torch.dot(self.state.repr.functions['h']['vpp'][ix][:n], self.state.repr.functions['g']['v'][iy][:n]) +
                        torch.dot(self.state.repr.functions['g']['vpp'][iy][:n], self.state.repr.functions['h']['v'][ix][:n]) +
                        self.state.lambda_ * torch.exp(torch.dot(
                            self.state.repr.functions['h']['v'][ix][:n],
                            self.state.repr.functions['g']['v'][iy][:n]
                        )
                    )
                ).item())**2
        return math.sqrt(integral / (self.state.repr.subgrid_cardinality**2))

    def _extrapolate_state(self):
        delta: float = self.state.lambda_ - self.last_states[0].lambda_
        delta_last: float = self.last_states[0].lambda_ - self.last_states[1].lambda_
        delta_b_last: float = self.last_states[1].lambda_ - self.last_states[2].lambda_

        for name in FuncRepr.names:
            for prime in FuncRepr.primes:
                func_delta = 0 if delta_last == 0 \
                    else (self.last_states[0].repr.functions[name][prime] - self.last_states[1].repr.functions[name][prime]) / delta_last
                func_b_delta = 0 if delta_b_last == 0 \
                    else (self.last_states[1].repr.functions[name][prime] - self.last_states[2].repr.functions[name][prime]) / delta_b_last
                func_2delta = 0 if delta_last + delta_b_last == 0 \
                    else (func_delta - func_b_delta) / (delta_last + delta_b_last)
                self.state.repr.functions[name][prime] = \
                    self.last_states[0].repr.functions[name][prime] + func_delta * delta + func_2delta * delta * (delta + delta_last)

    def _shift_states(self):
        self.last_states[2] = self.last_states[1]
        self.last_states[1] = self.last_states[0]
        self.last_states[0] = self.state
        self.state = copy.deepcopy(self.last_states[0])

    def _do_lambda_continuation(self):
        # log
        print('Start lambda continuation.')

        step = self.initial_lambda_step
        is_limit_achieved: bool = False
        iteration_index: int = 0

        for n in range(1, self.n + 1):
            # log
            print(f'Start iterations for n = {n}.')

            self._pass_lambda_iterations(n)

        while not is_limit_achieved:
            if abs(step) < self.limit_tolerance:
                # log
                print(200 * '=')
                print('Limit by lambda is achieved.')

                break

            self.state.lambda_ += step
            self._extrapolate_state()

            # log
            print('=' * 50)
            print(f'Process for lambda = {self.state.lambda_} and step = {step} is started.')

            try:
                for n in range(1, self.n + 1):
                    self._pass_lambda_iterations(n)
            except Exception as e:
                print(e)

                step /= 1.5
                self.state.lambda_ = self.last_states[0].lambda_ + step
                continue

            self.state.repr.make_functions_plot(suffix='sub-critical', file_index=-(iteration_index + 1))
            self.state.repr.make_3d_plot(suffix='sub-critical', file_index=-(iteration_index + 1))

            step *= 1.5
            step = math.copysign(min(abs(step), self.max_lambda_step), step)

            # log
            print('=' * 100)
            print(f'lambda = {self.state.lambda_} is passed.')

            self.continuation_data.append(
                (
                    self.state.lambda_,
                    self.state.repr.calc(
                        (self.state.repr.t0 + self.state.repr.t1) / 2,
                        (self.state.repr.t0 + self.state.repr.t1) / 2
                    ).item()
                )
            )

            self._make_continuation_plot()

            self._shift_states()
            iteration_index += 1

    def _do_norm_continuation(self):
        # log
        print('Starting norm continuation.')

        self.state.lambda_ = self.continuation_data[-1][0]
        norm = torch.tensor([self.continuation_data[-1][1]])

        step = self.initial_norm_step
        iteration_index: int = 0

        self._pass_norm_iterations(norm)

        while True:
            norm += step
            self._extrapolate_state()

            # log
            print('=' * 50)
            print(f'Process for lambda = {self.state.lambda_}, norm = {norm} and step = {step} is started.')

            try:
                self._pass_norm_iterations(norm)
            except Exception as e:
                print(e)

                norm -= step
                step /= 1.5
                norm += step

                continue

            self.state.repr.make_functions_plot(suffix='supercritical', file_index=-(iteration_index + 1))
            self.state.repr.make_3d_plot(suffix='supercritical', file_index=-(iteration_index + 1))

            step *= 1.5
            step = math.copysign(min(abs(step), self.max_norm_step), step)

            # log
            print('=' * 100)
            print(f'lambda = {self.state.lambda_} and norm = {norm} are passed.')

            self.continuation_data.append(
                (
                    self.state.lambda_,
                    self.state.repr.calc(
                        (self.state.repr.t0 + self.state.repr.t1) / 2,
                        (self.state.repr.t0 + self.state.repr.t1) / 2
                    ).item()
                )
            )

            self._make_continuation_plot()
            iteration_index += 1

    def _pass_lambda_iterations(self, n: int):
        def boundary(a: float, b: float, a_state: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_state[0:1], b_state[0:1]), dim=0)

        iteration_index = -1
        while True:
            iteration_index += 1
            last_h: torch.Tensor = self.state.repr.functions['h']['v'].clone()
            last_g: torch.Tensor = self.state.repr.functions['g']['v'].clone()

            for name_i in range(2):
                main_func = FuncRepr.names[name_i]
                const_func = FuncRepr.names[(name_i + 1) % 2]

                # log
                print(f'Iteration by {main_func} started.')

                const_tensor_stack = self.state.repr.functions[const_func]['v'].clone()
                const_pp_tensor_stack = self.state.repr.functions[const_func]['vpp'].clone()

                matrix_a: torch.Tensor = torch.sum(torch.bmm(const_tensor_stack.unsqueeze(2), const_tensor_stack.unsqueeze(1)), dim=0) * self.state.repr.subgrid_size
                matrix_a = torch.linalg.inv(matrix_a[:n, :n])

                matrix_b: torch.Tensor = -1 * torch.sum(torch.bmm(const_tensor_stack.unsqueeze(2), const_pp_tensor_stack.unsqueeze(1)), dim=0) * self.state.repr.subgrid_size

                def main_func_rhs(x_: float, state: torch.Tensor) -> torch.Tensor:
                    if n == 1:
                        u_all_y = const_tensor_stack[:, 0] * state[0]
                        integrand = const_tensor_stack[:, 0] * torch.exp(u_all_y)
                        matrix_c = torch.sum(integrand, dim=0) * self.state.repr.subgrid_size
                        prime = matrix_a[0, 0:1] * (matrix_b[0, 0] * state[0] - self.state.lambda_ * matrix_c)
                        return torch.cat((state[1:2], prime), dim=0)
                    else:
                        main_func_at_x = torch.cat((self.state.repr.functions[main_func]['v'][self.state.repr.idx(x_)][:n - 1], state[0:1]), dim=0)
                        u_all_y = torch.mv(const_tensor_stack[:, :n], main_func_at_x)
                        integrand = const_tensor_stack[:, :n] * torch.exp(u_all_y).unsqueeze(1)
                        matrix_c = torch.sum(integrand, dim=0) * self.state.repr.subgrid_size
                        prime = matrix_a[n - 1:, :] @ (torch.mv(matrix_b[:n, :n], main_func_at_x).unsqueeze(1) - self.state.lambda_ * matrix_c.unsqueeze(1)).squeeze(dim=1)
                        return torch.cat((state[1:2], prime), dim=0)

                mid_point = (self.state.repr.t0 + self.state.repr.t1) / 2
                bvp_solver_ = BVPSolver(
                    f=main_func_rhs,
                    boundary=boundary,
                    a=self.state.repr.t0,
                    b=self.state.repr.t1,
                    m=mid_point,
                    grid_size=self.state.repr.grid_size
                )
                bvp_init_state = torch.cat(
                    (
                        self.state.repr.functions[main_func]['v'][self.state.repr.idx(mid_point)][n - 1:n],
                        self.state.repr.functions[main_func]['vp'][self.state.repr.idx(mid_point)][n - 1:n]
                    ),
                    dim=0
                )
                main_solution: torch.Tensor = bvp_solver_.solve(
                    init_state=bvp_init_state,
                    tol=self.bvp_tolerance,
                    max_iter=self.bvp_max_iter
                )

                for i in range(self.state.repr.subgrid_cardinality):
                    self.state.repr.functions[main_func]['v'][i][n - 1] = main_solution[i][:1].detach()
                    self.state.repr.functions[main_func]['vp'][i][n - 1] = main_solution[i][1:].detach()
                    self.state.repr.functions[main_func]['vpp'][i][n - 1] = main_func_rhs(self.state.repr.subgrid[i], main_solution[i])[1:].detach()

                self.state.repr.balance_norms(n - 1)

                # log
                print(f'{main_func} variables updated.')

            h_difference = torch.max(torch.abs(self.state.repr.functions['h']['v'] - last_h)).item()
            g_difference = torch.max(torch.abs(self.state.repr.functions['g']['v'] - last_g)).item()

            print(
                h_difference,
                g_difference,
                self._calc_divergence_norm(n)
            )

            if max(h_difference, g_difference) < self.convergence_tolerance:
                break

    def _pass_norm_iterations(self, norm: torch.Tensor):
        def boundary(a: float, b: float, a_state: torch.Tensor, b_state: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_state[:self.n], b_state[:self.n]), dim=0)

        iteration_index = -1
        while True:
            iteration_index += 1

            last_h: torch.Tensor = self.state.repr.functions['h']['v'].clone()
            last_g: torch.Tensor = self.state.repr.functions['g']['v'].clone()

            for name_i in range(2):
                main_func = FuncRepr.names[name_i]
                const_func = FuncRepr.names[(name_i + 1) % 2]

                # log
                print(f'Iteration by {main_func} started.')

                const_tensor_stack = self.state.repr.functions[const_func]['v']
                const_pp_tensor_stack = self.state.repr.functions[const_func]['vpp']

                matrix_a: torch.Tensor = torch.sum(torch.bmm(const_tensor_stack.unsqueeze(2), const_tensor_stack.unsqueeze(1)), dim=0) * self.state.repr.subgrid_size
                matrix_a = torch.linalg.inv(matrix_a)

                matrix_b: torch.Tensor = -1 * torch.sum(torch.bmm(const_tensor_stack.unsqueeze(2), const_pp_tensor_stack.unsqueeze(1)), dim=0) * self.state.repr.subgrid_size

                def main_func_rhs(x_: float, state: torch.Tensor) -> torch.Tensor:
                    main_func_at_x = state[:self.n]
                    u_all_y = torch.mv(const_tensor_stack, main_func_at_x)
                    integrand = const_tensor_stack * torch.exp(u_all_y).unsqueeze(1)
                    matrix_c = torch.sum(integrand, dim=0) * self.state.repr.subgrid_size
                    primes = matrix_a @ (matrix_b @ main_func_at_x - state[-1] * matrix_c)

                    return torch.cat((state[self.n:-1], primes, torch.tensor([0])), dim=0)

                def main_func_init_boundary(state: torch.Tensor) -> torch.Tensor:
                    return torch.inner(
                        state[:self.n],
                        const_tensor_stack[self.state.repr.idx((self.state.repr.t1 + self.state.repr.t0) / 2)]
                    ).unsqueeze(0) - norm

                mid_point = (self.state.repr.t1 + self.state.repr.t0) / 2
                bvp_solver_ = BVPSolver(
                    f=main_func_rhs,
                    boundary=boundary,
                    init_boundary=main_func_init_boundary,
                    a=self.state.repr.t0,
                    b=self.state.repr.t1,
                    m=mid_point,
                    grid_size=self.state.repr.grid_size
                )
                bvp_init_state = torch.cat(
                    (
                        self.state.repr.functions[main_func]['v'][self.state.repr.idx(mid_point)],
                        self.state.repr.functions[main_func]['vp'][self.state.repr.idx(mid_point)],
                        torch.tensor([self.state.lambda_])
                    ),
                    dim=0
                )
                main_solution: torch.Tensor = bvp_solver_.solve(
                    init_state=bvp_init_state,
                    tol=self.bvp_tolerance,
                    max_iter=self.bvp_max_iter
                )

                for i in range(self.state.repr.subgrid_cardinality):
                    self.state.repr.functions[main_func]['v'][i] = main_solution[i][:self.n]
                    self.state.repr.functions[main_func]['vp'][i] = main_solution[i][self.n:-1]
                    self.state.repr.functions[main_func]['vpp'][i] = main_func_rhs(self.state.repr.subgrid[i], main_solution[i])[self.n:-1]

                self.state.lambda_ = main_solution[0][-1].item()

                # log
                print(f'New lambda value = {self.state.lambda_}.')
                print(f'{main_func} variables updated.')

                for k in range(self.n):
                    self.state.repr.balance_norms(k)

            h_difference = torch.max(torch.abs(self.state.repr.functions['h']['v'] - last_h)).item()
            g_difference = torch.max(torch.abs(self.state.repr.functions['g']['v'] - last_g)).item()

            print(
                h_difference,
                g_difference,
                self._calc_divergence_norm(self.n)
            )

            if max(h_difference, g_difference) < self.convergence_tolerance:
                break

    def _make_continuation_plot(self):
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))

        axes.plot([e[1] for e in self.continuation_data], [e[0] for e in self.continuation_data], 'b-', label='f', linewidth=1)
        axes.set_xlabel('u(0, 0)', fontsize=12)
        axes.set_ylabel('lambda', fontsize=12)
        axes.legend(fontsize=10)
        axes.grid(True, alpha=0.3)
        axes.set_title(f'Continuation plot', fontsize=14)
        axes.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'plots/plot-c.png', dpi=300)
        print(f'График сохранен: plot-c.png')
        plt.close()

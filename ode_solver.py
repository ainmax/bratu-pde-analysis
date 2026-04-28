import math

import torch
from typing import Callable

torch.set_default_dtype(torch.float64)

class ODESolver:
    """
    Classic RK4 ODE solver

    s'(t) = f(t, s(t)), t in [a, b], s(a) = init_s
    """

    def __init__(self, f: Callable[[float, torch.Tensor], torch.Tensor]):
        """
        Rhs function definition
        :param f: function with domain R and values represented as tensors shaped with (n, 1)
        """

        self.f = f

    def _calc_next_state(self, t: float, last_state: torch.Tensor, h: float) -> torch.Tensor:
        """
        Does one RK4 step with given parameters
        :param t: last predicted point
        :param last_state: state which obtained from previous call or initial state; tensor shaped with (n, 1)
        :param h: step size
        :return: approximated state at point t + h; tensor shaped with (n, 1)
        """

        a = [0.5, 0.5, 1.0]
        b = [[0.5], [0.0, 0.5], [0.0, 0.0, 1.0]]
        p = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        k_1 = h * self.f(t, last_state)
        k_2 = h * self.f(t + a[0] * h, last_state + b[0][0] * k_1)
        k_3 = h * self.f(t + a[1] * h, last_state + b[1][0] * k_1 + b[1][1] * k_2)
        k_4 = h * self.f(t + a[2] * h, last_state + b[2][0] * k_1 + b[2][1] * k_2 + b[2][2] * k_3)
        next_state = last_state + p[0] * k_1 + p[1] * k_2 + p[2] * k_3 + p[3] * k_4
        return next_state

    def solve_by_segment(
            self,
            *,
            init_state: torch.Tensor,
            a: float,
            b: float,
            grid_size: float
    ) -> torch.Tensor:
        """
        Calculates ODE solution RK4 approximation on grid at segment [a, b]
        :param init_state: initial state; tensor shaped with (n, 1)
        :param a: segment begin (orientation matters)
        :param b: segment end (orientation matters)
        :param grid_size: distance between neighboring grid points
        :return: tensor with states at each point of grid; tensor shaped with (M, n), where M - number of grid points at segment
        """

        steps_count: int = round(abs(b - a) / grid_size)
        states: list[torch.Tensor] = [init_state]
        h: float = grid_size * math.copysign(1.0, b - a)
        for i in range(steps_count):
            t = a + h * i
            next_state = self._calc_next_state(t, states[-1], h)
            states.append(next_state)
        return torch.stack(states, dim=0)

    def solve_by_end(
            self,
            *,
            init_state: torch.Tensor,
            a: float,
            b: float,
            grid_size: float
    ) -> torch.Tensor:
        """
        Calculates ODE solution RK4 approximation on end of the segment
        :param init_state: initial state; tensor shaped with (n, 1)
        :param a: segment begin (orientation matters)
        :param b: segment end (orientation matters)
        :param grid_size: distance between neighboring grid points
        :return: tensor with state at end of the segment; tensor shaped with (n, 1)
        """

        steps_count: int = round(abs(b - a) / grid_size)
        states: list[torch.Tensor] = [init_state, None]
        h: float = grid_size * math.copysign(1.0, b - a)
        for i in range(steps_count):
            t: float = a + h * i
            next_state = self._calc_next_state(t, states[i % 2], h)
            states[(i + 1) % 2] = next_state
        return states[steps_count % 2]

    def solve_on_subgrid(
        self,
        *,
        init_state: torch.Tensor,
        a: float,
        b: float,
        grid_size: float
    ) -> torch.Tensor:
        """
        Calculates ODE solution RK4 approximation on subgrid at segment [a, b]
        :param init_state: initial state; tensor shaped with (n, 1)
        :param a: segment begin (orientation matters)
        :param b: segment end (orientation matters)
        :param grid_size: distance between neighboring grid points
        :return: tensor with states at each point of subgrid; tensor shaped with (M, n), where M - number of subgrid points at segment
        """

        steps_count: int = round(abs(b - a) / grid_size)
        states: list[torch.Tensor] = [init_state]
        h: float = grid_size * math.copysign(1.0, b - a)
        for i in range(steps_count):
            t = a + h * i
            states.append(states[-1] + h * self.f(t, states[-1]) / 2)
            next_state = self._calc_next_state(t, states[-2], h)
            states.append(next_state)
        return torch.stack(states, dim=0)

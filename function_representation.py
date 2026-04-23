import math

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
torch.set_default_dtype(torch.float64)


class FuncRepr:
    names: list[str] = ['h', 'g']
    primes: list[str] = ['v', 'vp', 'vpp']

    def __init__(
            self,
            *,
            grid_size_2exp: int,
            n: int,
            t0: float,
            t1: float
    ):
        self.n = n
        self.t0 = t0
        self.t1 = t1

        self.grid_size: float = (t1 - t0) / 2**grid_size_2exp
        self.grid: list[float] = []
        self.grid_cardinality = 2**grid_size_2exp + 1
        self.subgrid_size: float = self.grid_size / 2
        self.subgrid: list[float] = []
        self.subgrid_cardinality = 2 * 2**grid_size_2exp + 1

        self._setup_grid()

        self.functions: dict[str, dict[str, torch.Tensor]] = {
            'h': {
                'v': torch.tensor([]),
                'vp': torch.tensor([]),
                'vpp': torch.tensor([]),
            },
            'g': {
                'v': torch.tensor([]),
                'vp': torch.tensor([]),
                'vpp': torch.tensor([]),
            },
        }

        self._setup_functions()

    def _setup_grid(self):
        self.grid = [
            self.t0 + t * self.grid_size
            for t in range(self.grid_cardinality)
        ]
        self.subgrid = [
            self.t0 + t * self.subgrid_size
            for t in range(self.subgrid_cardinality)
        ]

    def _setup_functions(self):
        def calc_basis(t: float):
            return torch.tensor([
                math.sin((i + 1) * math.pi * (t - self.t0) / (self.t1 - self.t0)) / 10 ** i
                for i in range(self.n)
            ])

        def calc_basis_p(t: float):
            return torch.tensor([
                (i + 1) * math.pi / (self.t1 - self.t0) * math.cos((i + 1) * math.pi * (t - self.t0) / (self.t1 - self.t0)) / 10 ** i
                for i in range(self.n)
            ])

        def calc_basis_pp(t: float):
            return torch.tensor([
                -((i + 1) * math.pi / (self.t1 - self.t0))**2 * math.sin((i + 1) * math.pi * (t - self.t0) / (self.t1 - self.t0)) / 10 ** i
                for i in range(self.n)
            ])

        self.functions['h']['v'] = torch.stack([calc_basis(x) for x in self.subgrid], dim=0)
        self.functions['h']['vp'] = torch.stack([calc_basis_p(x) for x in self.subgrid], dim=0)
        self.functions['h']['vpp'] = torch.stack([calc_basis_pp(x) for x in self.subgrid], dim=0)
        self.functions['g']['v'] = torch.stack([calc_basis(y) for y in self.subgrid], dim=0)
        self.functions['g']['vp'] = torch.stack([calc_basis_p(y) for y in self.subgrid], dim=0)
        self.functions['g']['vpp'] = torch.stack([calc_basis_pp(y) for y in self.subgrid], dim=0)

    def balance_norms(self, k: int):
        h_abs = torch.abs(self.functions['h']['v'])
        g_abs = torch.abs(self.functions['g']['v'])
        h_max: float = torch.max(h_abs[:, k]).item()
        g_max: float = torch.max(g_abs[:, k]).item()
        multipliers: dict[str, float] = {
            'h': math.sqrt(g_max / h_max),
            'g': math.sqrt(h_max / g_max),
        }
        for name in FuncRepr.names:
            for prime in FuncRepr.primes:
                self.functions[name][prime][:, k] *= multipliers[name]

    def calc(self, x: float, y: float):
        h_x = self.functions['h']['v'][self.idx(x)]
        g_y = self.functions['g']['v'][self.idx(y)]
        return torch.dot(h_x, g_y)

    def idx(self, t: float):
        return round((t - self.t0) / self.subgrid_size)

    def make_functions_plot(
        self,
        *,
        suffix: str,
        file_index: int
    ):
        fig, axes = plt.subplots(self.n + 1, 2, figsize=(14, 3 * (self.n + 1)))

        for i in range(self.n):
            for name_i in range(2):
                name = FuncRepr.names[name_i]
                axes[i][name_i].plot(self.subgrid, self.functions[name]['v'][:, i].tolist(), 'b-', label=name, linewidth=1)
                axes[i][name_i].legend(fontsize=10)
                axes[i][name_i].grid(True, alpha=0.3)
                axes[i][name_i].set_title(f'{name}{i + 1} approximation', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'plots/plot-{suffix}{file_index}.png', dpi=150)
        print(f'График сохранен: plot{file_index}.png')
        plt.close()

    def make_3d_plot(
            self,
            *,
            suffix: str,
            file_index: int
    ):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection='3d')

        x = np.zeros((self.subgrid_cardinality, self.subgrid_cardinality))
        y = np.zeros((self.subgrid_cardinality, self.subgrid_cardinality))
        z = np.zeros((self.subgrid_cardinality, self.subgrid_cardinality))
        for i in range(self.subgrid_cardinality):
            for j in range(self.subgrid_cardinality):
                x[i, j] = self.subgrid[i]
                y[i, j] = self.subgrid[j]
                z[i, j] = self.calc(x[i, j].item(), y[i, j].item())

        surf = ax.plot_surface(x, y, z, linewidth=0, cmap=matplotlib.colormaps['magma'], antialiased=False)
        ax.set_title('Surface Plot')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(f'3d-plots/3d-plot-{suffix}{file_index}.png')
        plt.close()

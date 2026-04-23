from pde_solver import PDESolver

if __name__ == '__main__':
    pde_solver = PDESolver(
        grid_size_2exp=6,
        bvp_tolerance=1e-9,
        convergence_tolerance=1e-4,
        bvp_max_iter=20,
        initial_lambda=-0.001,
        initial_lambda_step=-0.1,
        max_lambda_step=0.1,
        limit_tolerance=1e-5,
        initial_norm_step=1e-2,
        max_norm_step=0.2,
        n=3,
        t0=-1,
        t1=1
    )

    pde_solver.solve()
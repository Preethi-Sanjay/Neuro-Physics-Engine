import numpy as np


def solve_heat(
    nx=100,
    nt=100,
    alpha=1.0,
):
    """
    Explicit Finite Difference Solver
    for

        u_t = alpha*u_xx
    """

    L = 1.0
    T = 1.0

    dx = L / (nx - 1)
    dt = T / (nt - 1)

    r = alpha * dt / dx**2

    x = np.linspace(0, L, nx)

    t = np.linspace(0, T, nt)

    U = np.zeros((nt, nx))

    # Initial condition
    U[0] = np.sin(np.pi * x)

    for n in range(nt - 1):

        for i in range(1, nx - 1):

            U[n + 1, i] = (
                U[n, i]
                + r
                * (
                    U[n, i + 1]
                    - 2 * U[n, i]
                    + U[n, i - 1]
                )
            )

        U[n + 1, 0] = 0
        U[n + 1, -1] = 0

    return x, t, U
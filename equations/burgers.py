import torch


def residual(model, x, t, nu=0.01 / torch.pi):
    """
    1D Burgers Equation

        u_t + u*u_x = nu*u_xx

    Residual:
        R = u_t + u*u_x - nu*u_xx
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    inputs = torch.cat((x, t), dim=1)

    u = model(inputs)

    # -----------------------------
    # First Derivatives
    # -----------------------------
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    # -----------------------------
    # Second Derivative
    # -----------------------------
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    return u_t + u * u_x - nu * u_xx


def analytical_solution(x, t):
    """
    Placeholder analytical solution.

    Burgers equation usually uses
    numerical reference solutions.

    Replace later with dataset
    or numerical solver.
    """

    return torch.zeros_like(x)


def initial_condition(x):
    """
    Initial Condition

        u(x,0) = -sin(pi*x)
    """

    return -torch.sin(torch.pi * x)


def boundary_condition_left(t):
    return torch.zeros_like(t)


def boundary_condition_right(t):
    return torch.zeros_like(t)
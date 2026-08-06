import torch


def residual(model, x, t):
    """
    Computes the residual of the 1D Heat Equation.

        u_t = u_xx

    Residual:
        R = u_t - u_xx
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    inputs = torch.cat((x, t), dim=1)

    u = model(inputs)

    # First derivative with respect to x
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    # First derivative with respect to t
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative with respect to x
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    return u_t - u_xx


def analytical_solution(x, t):
    """
    Exact analytical solution:

        u(x,t)=exp(-pi²t)sin(pi x)
    """

    return torch.exp(-(torch.pi ** 2) * t) * torch.sin(torch.pi * x)


def initial_condition(x):
    """
    Initial condition:

        u(x,0)=sin(pi x)
    """

    return torch.sin(torch.pi * x)


def boundary_condition_left(t):
    """
    u(0,t)=0
    """

    return torch.zeros_like(t)


def boundary_condition_right(t):
    """
    u(1,t)=0
    """

    return torch.zeros_like(t)
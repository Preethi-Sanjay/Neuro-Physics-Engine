import torch


def residual(model, x, t, c=1.0):
    """
    Computes the residual of the 1D Wave Equation:

        u_tt = c² * u_xx

    Residual:
        R = u_tt - c² * u_xx

    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    inputs = torch.cat((x, t), dim=1)

    u = model(inputs)

    # First derivatives
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

    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_tt = torch.autograd.grad(
        u_t,
        t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = u_tt - (c ** 2) * u_xx

    return residual


def analytical_solution(x, t, c=1.0):
    """
    Exact solution:

        u(x,t)=sin(pi*x)*cos(pi*c*t)
    """

    return torch.sin(torch.pi * x) * torch.cos(torch.pi * c * t)


def initial_condition(x):
    """
    Initial displacement:

        u(x,0)=sin(pi*x)
    """

    return torch.sin(torch.pi * x)


def initial_velocity(x):
    """
    Initial velocity:

        u_t(x,0)=0
    """

    return torch.zeros_like(x)


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
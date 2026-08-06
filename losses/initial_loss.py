import torch


def initial_condition_loss(model, x, equation_module):
    """
    Computes the initial condition loss for any equation.

    The equation module must implement:

        initial_condition(x)

    """

    device = x.device

    t0 = torch.zeros_like(x).to(device)

    inputs = torch.cat((x, t0), dim=1)

    prediction = model(inputs)

    target = equation_module.initial_condition(x)

    loss = torch.mean((prediction - target) ** 2)

    return loss


def initial_velocity_loss(model, x, equation_module):
    """
    Computes the initial velocity loss.

    Used only by equations like the Wave Equation.

    Heat Equation does not require this.
    """

    if not hasattr(equation_module, "initial_velocity"):
        return torch.tensor(
            0.0,
            device=x.device,
            requires_grad=True
        )

    x = x.clone().detach().requires_grad_(True)

    t0 = torch.zeros_like(x).requires_grad_(True)

    inputs = torch.cat((x, t0), dim=1)

    prediction = model(inputs)

    u_t = torch.autograd.grad(
        prediction,
        t0,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
    )[0]

    target = equation_module.initial_velocity(x)

    loss = torch.mean((u_t - target) ** 2)

    return loss
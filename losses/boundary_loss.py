import torch


def boundary_condition_loss(model, t, equation_module):
    """
    Generic boundary condition loss.

    Supports every equation that defines:

        boundary_condition_left()
        boundary_condition_right()

    """

    device = t.device

    # -------------------------
    # Left Boundary (x = 0)
    # -------------------------
    x_left = torch.zeros_like(t).to(device)

    left_inputs = torch.cat((x_left, t), dim=1)

    left_prediction = model(left_inputs)

    left_target = equation_module.boundary_condition_left(t)

    left_loss = torch.mean(
        (left_prediction - left_target) ** 2
    )

    # -------------------------
    # Right Boundary (x = 1)
    # -------------------------
    x_right = torch.ones_like(t).to(device)

    right_inputs = torch.cat((x_right, t), dim=1)

    right_prediction = model(right_inputs)

    right_target = equation_module.boundary_condition_right(t)

    right_loss = torch.mean(
        (right_prediction - right_target) ** 2
    )

    return left_loss + right_loss
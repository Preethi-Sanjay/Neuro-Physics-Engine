import torch


def physics_loss(residual):

    return torch.mean(residual ** 2)
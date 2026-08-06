import torch


def relative_l2_error(prediction, target, eps=1e-12):
    """
    Computes Relative L2 Error safely.

    Prevents division by zero.
    """

    numerator = torch.norm(prediction - target)

    denominator = torch.norm(target)

    if denominator < eps:
        return numerator.item()

    return (numerator / denominator).item()


def mae(prediction, target):
    return torch.mean(torch.abs(prediction - target)).item()


def mse(prediction, target):
    return torch.mean((prediction - target) ** 2).item()


def rmse(prediction, target):
    return torch.sqrt(torch.mean((prediction - target) ** 2)).item()
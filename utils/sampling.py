import torch


def sample_collocation_points(num_points):

    x = torch.rand(num_points, 1)

    t = torch.rand(num_points, 1)

    return x, t
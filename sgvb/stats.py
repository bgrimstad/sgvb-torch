"""
Created 05 March 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import torch
import torch.nn.functional as F


def kld_normals(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor):
    """
    Compute the Kullback-Leibler divergence between the variational and prior, KL(q || p).

    Assuming that the variational distribution is a mean field normal:
        q = Normal(mean_q, std_q^2),
    where mean_q and std_q are tensors. Likewise, the prior is assumed to be a mean-field normal
        p = Normal(mean_p, std_p^2),
    where mean_p and std_p are tensors.

    :param mean_q: Mean of approximation (tensor)
    :param std_q: Scale of approximation (tensor)
    :param mean_p: Mean of prior (tensor)
    :param std_p: Scale of prior (tensor)
    :return:
    """
    kld = (((mean_q - mean_p) / std_p).pow(2) - 1 + (std_q / std_p).pow(2) - 2 * torch.log(std_q / std_p)).sum()

    return 0.5 * kld


def map_real_to_positive(x: torch.Tensor):
    """
    Map a real number 'x' to a strictly positive real 'y' using the softplus mapping:
        y = log(1 + exp(x)) + e,
    where e is a small number is added to allow gradient flow.
    :param x: Number on the real line (for example variational parameters)
    :return: Strictly positive numbers (for example standard deviations)
    """
    return F.softplus(x) + 1e-6


def map_positive_to_real(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of softplus mapping used to map a positive number to the real line:
        y = log(exp(x-e) - 1),
    where e is a small number (see map_real_to_positive). As x -> 1e-6 from above, y -> inf
    :param x: Strictly positive numbers
    :return: Numbers on the real line
    """
    return torch.log(torch.exp(x - 1e-6) - 1.0)


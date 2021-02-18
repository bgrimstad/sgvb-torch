"""
Created 05 March 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import typing as ty
import torch
from torch.distributions.normal import Normal
from sgvb.stats import kld_normals, map_real_to_positive, map_positive_to_real


class FixedNoise(torch.nn.Module):
    """
    Fixed noise model:
        noise ~ Normal(0, sigma**2),
    where sigma is a constant.
    """
    def __init__(self, sigma: float = 1.0):
        """
        Fixed noise model
        :param sigma: fixed noise level
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert sigma > 0, "FixedNoise: 'sigma' must be a positive number"
        self.sigma = sigma

    def forward(self, x: torch.Tensor, sample: bool = True) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-pass returning a tensor of size (batch_size, 1) filled with sigma
        :param x: Inputs. Tensors of size (batch_size, n_in)
        :param sample: Included for compatibility (not used since sigma is fixed)
        :return:
        """
        batch_size = x.size()[0]
        s = torch.tensor(data=[self.sigma], device=self.device).expand(batch_size, 1)
        kld_cost = torch.tensor(data=[0.], device=self.device)
        assert s.size(0) == batch_size, 'FixedNoise.forward: Wrong size on tensor'
        return s, kld_cost

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample noise from posterior conditioned on the input x

        :param x: input of size (batch_size, n_inputs)
        :param n_samples: number of samples
        :return: sampled noise as tensor of size (n_samples, n_batch, 1)
        """
        return sample_noise(self, x, n_samples)


class HomoscedasticNoise(torch.nn.Module):
    """
    A homoscedastic noise model:
        noise ~ Normal(0, s**2),
        s ~ LogNormal(mu_s, sigma_s),
    where mu_s and sigma_s > 0 are learnable parameters.

    The output of the model is the noise standard deviation 's'.

    In the implementation, the model is reparameterized to:
        s = exp(psi_s),
        psi_s = mu_s + sigma_s * eps,
        eps ~ Normal(0, 1).

    NOTE: This implements a learnable log-normal distribution using reparameterization.
    """

    def __init__(self, mu_s: float = 1.0, sigma_s: float = 1.0):
        """
        Homoscedastic noise model
        :param mu_s: Prior mean of psi_s
        :param sigma_s: Prior standard deviation of psi_s
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # *************************************************************************
        # Prior
        assert sigma_s > 0, "HomoscedasticNoise: 'sigma_s' must be a positive number"

        # Prior p(psi_s) = Normal(mean_p, std_p ** 2)
        self.mean_p = mu_s * torch.ones(1, device=self.device)
        self.std_p = sigma_s * torch.ones(1, device=self.device)

        # *************************************************************************
        # Variational parameters: psi_s ~ Normal(mean_q, std_q ** 2)

        # Initial values of variational parameters
        std_p_tensor = torch.tensor([self.std_p], device=self.device)
        scale_init = map_positive_to_real(std_p_tensor)  # Inverse of softplus
        mean_init = torch.tensor([self.mean_p], device=self.device)

        self.mean_q = torch.nn.Parameter(mean_init, requires_grad=True)
        self.scale_q = torch.nn.Parameter(scale_init, requires_grad=True)

    @property
    def std_q(self):
        """
        Map scale parameter to a strictly positive standard deviation
        :return: standard deviation
        """
        return map_real_to_positive(self.scale_q)

    def forward(self, x: torch.Tensor, sample: bool = True) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-pass using the reparameterization
        :param x: Inputs. Tensors of size (batch_size, n_in)
        :param sample: If True, sample parameters, otherwise, use mean parameters (W_mean_q, b_mean_q)
        :return:
        """

        batch_size = x.size(0)

        if not self.training and not sample:
            s = torch.exp(self.mean_q).expand(batch_size, 1)
            assert s.size(0) == batch_size, 'HomoscedasticNoise.forward: Wrong size on tensor'
            return s, torch.tensor(data=[0.], device=self.device)

        else:

            std_q = map_real_to_positive(self.scale_q)

            # Draw noise from a standard normal distribution
            eps = x.new_empty((batch_size, 1)).normal_(mean=0, std=1)  # (batch_size, 1)
            psi_s = self.mean_q + std_q * eps  # (batch_size, 1)
            s = torch.exp(psi_s)

            # Check tensor sizes
            assert eps.size(0) == batch_size and psi_s.size(0) == batch_size and s.size(0) == batch_size, \
                'HomoscedasticNoise.forward: Wrong size on tensors'

            # Compute KL divergence cost, KL(q || p)
            kld_cost = kld_normals(mean_q=self.mean_q, std_q=std_q, mean_p=self.mean_p, std_p=self.std_p)

            return s, kld_cost

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample noise from posterior conditioned on the input x

        :param x: input of size (batch_size, n_inputs)
        :param n_samples: number of samples
        :return: sampled noise as tensor of size (n_samples, n_batch, 1)
        """
        return sample_noise(self, x, n_samples)


def sample_noise(noise_model: torch.nn.Module, x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Sample noise from posterior conditioned on the input x

    :param noise_model: model to sample from
    :param x: input of size (batch_size, n_inputs)
    :param n_samples: number of samples
    :return: sampled noise as tensor of size (n_samples, n_batch, 1)
    """
    assert n_samples > 0, "'n_samples' must be a positive integer"
    x = x.to(noise_model.device)

    # Initialize tensor with size (n_samples, n_batch, 1) to hold predictions
    n_batch = x.shape[0]
    samples = x.data.new_zeros((n_samples, n_batch, 1))

    for i in range(n_samples):
        sigma, _ = noise_model.forward(x, sample=True)
        mu = torch.zeros_like(sigma)
        noise_dist = Normal(mu, sigma)
        noise = noise_dist.sample()
        samples[i] = noise

    return samples

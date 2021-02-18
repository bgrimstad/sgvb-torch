"""
Created 09 February 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import typing as ty
import torch
from sgvb.stats import kld_normals, map_real_to_positive, map_positive_to_real
from math import sqrt


class BayesLinearLayer(torch.nn.Module):
    """
    A Bayesian linear layer represents an affine transformation of an input vector 'x' to an output vector 'y'. The
    posterior uncertainty in the weights and biases are encoded by a fully factorized (mean-field) Normal variational
    distribution. The prior uncertainty is represented by a fully factorized Normal distribution.

    The class implements the following generative process:
        W ~ Normal(weight_mean_q, weight_std_q^2)
        b ~ Normal(bias_mean_q, bias_std_q^2)
        y ~ Wx + b

    We use subscript q and p to refer to the variational and prior distribution. The KL divergence between the
    variational (q) and prior (p) distribution is obtained in closed form. This is possible since both are mean-field
    normals.

    The 'local reparameterization trick' is used to reduce variance (see reference below). In effect, this means the
    weights are never sampled directly; instead one samples in the output space.

    Reference:
    Kingma, Diederik P., Tim Salimans, and Max Welling.
    "Variational dropout and the local reparameterization trick."
    Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, n_in: int, n_out: int, weight_prior_std: float = 1.0, bias_prior_std: float = 1.0):
        """
        Initialization of variational and prior distribution.

        To enforce strictly positive standard deviations we map the scale parameters with the softplus operator.
        That is, we set std = softplus(scale).

        We initialize the variational distribution to be the same as the prior distribution.

        :param n_in: Number of layer inputs
        :param n_out: Number of layer outputs
        :param weight_prior_std: Prior standard deviation of weights
        :param bias_prior_std: Prior standard deviation of biases
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_in = n_in
        self.n_out = n_out

        # *************************************************************************
        # Prior mean and standard deviation for weights
        # Initially, we set the same prior on all weights
        self.weight_mean_p = torch.zeros((self.n_in, self.n_out), device=self.device)
        self.weight_std_p = weight_prior_std * torch.ones((self.n_in, self.n_out), device=self.device)

        # *************************************************************************
        # Prior mean and standard deviation for biases
        # Initially, we set the same prior on all biases
        self.bias_mean_p = torch.zeros(self.n_out, device=self.device)
        self.bias_std_p = bias_prior_std * torch.ones(self.n_out, device=self.device)

        # *************************************************************************
        # Variational parameters for the weights: W ~ Normal(weight_mean_q, weight_std_q^2)
        # Note that we initialize the variational to be equal to the prior
        self.weight_mean_q = torch.nn.Parameter(self.weight_mean_p, requires_grad=True)
        weight_scale_q_init = map_positive_to_real(self.weight_std_p)
        self.weight_scale_q = torch.nn.Parameter(weight_scale_q_init, requires_grad=True)

        # *************************************************************************
        # Variational parameters for the biases:  b ~ Normal(bias_mean_q, bias_std_q^2)
        self.bias_mean_q = torch.nn.Parameter(self.bias_mean_p, requires_grad=True)
        bias_scale_q_init = map_positive_to_real(self.bias_std_p)
        self.bias_scale_q = torch.nn.Parameter(bias_scale_q_init, requires_grad=True)

    @property
    def weight_std_q(self) -> torch.Tensor:
        """
        Map scale parameters to strictly positive standard deviations
        :return: weight standard deviation
        """
        return map_real_to_positive(self.weight_scale_q)

    @property
    def bias_std_q(self) -> torch.Tensor:
        """
        Map scale parameters to strictly positive standard deviations
        :return: weight standard deviation
        """
        return map_real_to_positive(self.bias_scale_q)

    def forward(self, x: torch.Tensor, sample: bool = True) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-pass using the local reparameterization trick.

        The distribution for the pre-activation 'a' is given as:
            a = weight_pre + bias_pre,
        where
            weight_pre = gamma + sqrt(delta) * epsilon,
            bias_pre = bias_mean_q + bias_std_q * epsilon,
        and
            epsilon ~ N(0,1),
            gamma = x*weight_mean_q,  (matrix multiplication)
            delta = x^2 * weight_std_q^2.  (element-wise squaring and matrix multiplication)

        :param x: Inputs. Tensors of size (batch_size, n_in)
        :param sample: If True, sample parameters, otherwise, use mean parameters (W_mean_q, b_mean_q)
        :return:
        """

        batch_size = x.size(0)

        if not self.training and not sample:  # Used for testing
            output = torch.mm(x, self.weight_mean_q) + self.bias_mean_q.expand(batch_size, self.n_out)
            return output, torch.tensor(data=[0.], device=self.device)

        else:

            # Compute gamma and delta_sqrt
            gamma = torch.mm(x, self.weight_mean_q)
            delta = torch.mm(x.pow(2), self.weight_std_q.pow(2))
            delta_sqrt = torch.sqrt(delta + 1e-6)  # Add a small constant to avoid division by zero in gradient

            # Check tensor sizes
            assert x.size(0) == gamma.size(0) and self.weight_mean_q.size(1) == gamma.size(1), 'Wrong size on gamma'
            assert x.size(0) == delta_sqrt.size(0) and self.weight_mean_q.size(1) == delta_sqrt.size(1), \
                'Wrong size on delta_sqrt'

            # Draw noise vectors from a standard normal distribution for each element in the mini-batch.
            # Tensor.new_empty() constructs a new tensor of the same data type and device as self tensor.
            eps_weight = delta_sqrt.new_empty(delta_sqrt.size()).normal_(mean=0, std=1)  # (n_batch, n_out)
            eps_bias = self.bias_std_q.new_empty(self.bias_std_q.size()).normal_(mean=0, std=1)

            # Compute output
            weight_out = gamma + delta_sqrt * eps_weight  # (batch_size, n_output)
            bias_out = self.bias_mean_q + self.bias_std_q * eps_bias

            # Bias term is made two dimensional (unsqueeze) and expanded to batch size (expand) -> stacked
            output = weight_out + bias_out.unsqueeze(0).expand(batch_size, -1)

            # Check tensor sizes
            assert output.size(0) == x.size(0) and output.size(1) == self.n_out, 'Wrong output size'

            # Compute KL divergence cost, KL(q || p)
            kld = kld_normals(mean_q=self.weight_mean_q, std_q=self.weight_std_q,
                              mean_p=self.weight_mean_p, std_p=self.weight_std_p)
            kld += kld_normals(mean_q=self.bias_mean_q, std_q=self.bias_std_q,
                               mean_p=self.bias_mean_p, std_p=self.bias_std_p)

            return output, kld

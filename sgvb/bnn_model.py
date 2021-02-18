"""
Created 10 February 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import numpy as np
import typing as ty
from math import pi, sqrt
import torch
from torch.utils.data import TensorDataset

from sgvb.bnn_layer import BayesLinearLayer
from sgvb.noise_models import FixedNoise


class BayesNet(torch.nn.Module):
    """
    Implementation of a Bayesian neural network (BNN). The BNN is composed of a series Bayesian linear layers and
    activation functions. The weights and biases of the linear layers are represented by mean-field normal
    distributions.
    """

    def __init__(self, layers: ty.List[int], weight_prior_std: float = None, bias_prior_std: float = None,
                 noise_model: torch.nn.Module = None):
        """
        Constructor for BNN. If no prior is set for the weights and biases, a prior based on He initialization is used.

        :param layers: Layer sizes given as list of integers (incl. input and output layer). For example, [2, 10, 1]
                       constructs a network with two inputs, one hidden layer with 10 nodes, and one output.
        :param weight_prior_std: Standard deviation of prior on weights (all weights get the same prior std. dev.)
        :param bias_prior_std: Standard deviation of prior on biases (all biases get the same prior std. dev.)
        :param noise_model: Noise model (see noise_models.py)
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running on device:', self.device)

        assert len(layers) >= 2, "At least two layers are required (incl. input and output layer)"
        self.train_size = None
        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]
        self.n_hidden = layers[1:-1]  # Hidden layers
        self.layers = layers

        # Priors for each layer
        self.weight_prior_std = [weight_prior_std] * (len(self.layers) - 1)
        self.bias_prior_std = [bias_prior_std] * (len(self.layers) - 1)

        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]

            if not self.weight_prior_std[i]:
                if i == 0:
                    self.weight_prior_std[i] = np.sqrt(1 / n_in).item()
                else:
                    self.weight_prior_std[i] = np.sqrt(2 / n_in).item()  # He initialization

            if not self.bias_prior_std[i]:
                self.bias_prior_std[i] = 0.01  # Using a small number to avoid large output variances in deep networks

        # Fully connected linear layers
        linear_layers = []
        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]

            layer = BayesLinearLayer(n_in, n_out, weight_prior_std=self.weight_prior_std[i],
                                     bias_prior_std=self.bias_prior_std[i])
            linear_layers.append(layer)

        self.linear_layers = torch.nn.ModuleList(linear_layers)  # Lists of modules/layers must be registered

        # Non-linearity (e.g. ReLU, ELU, or SELU)
        self.act = torch.nn.ReLU(inplace=False)

        # Noise model
        if not noise_model:
            self.noise_model = FixedNoise(sigma=1.0)
        else:
            self.noise_model = noise_model

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor, sample: bool = True) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-pass through model (not applying noise).
        For each hidden layer we compute the pre-activation and KL cost.
        The pre-activation is then fed through a non-linearity (activation function).

        :param x: Input of size (batch_size, input_dim)
        :param sample: If True, sample parameters, otherwise, use mean parameters (W_mean_q, b_mean_q).
        :return: (output, kl_cost), where output is of size (batch_size, output_dim), and kl_cost is the KL cost.
        """
        assert not torch.isnan(x).any(), f'BayesNet.forward: Found NaN in input {x}'

        # Input given with size (batch_size, input_dim)
        x = x.view(-1, self.n_inputs)
        x = x.to(self.device)

        # KL cost for lambda parameters
        kl_cost = torch.tensor(data=[.0], device=self.device)

        # Loop through layers
        for i, fc in enumerate(self.linear_layers):
            x, kl_qp = fc(x, sample)
            kl_cost += kl_qp
            # Skip activation on output layer
            if i < len(self.linear_layers) - 1:
                x = self.act(x)

        # Return output
        return x, kl_cost

    def _loglikelihood(self, output: torch.Tensor, target: torch.Tensor, sigma_n: torch.Tensor) \
            -> torch.Tensor:
        """
        Compute log-likelihood of mini-batch of size M from a training set of size N.

        For regression with a normal noise model N(0, sigma_n**2) the expected log-likelihood is:
            -0.5 * (N/M) * sum_i[ log(2 * pi * sigma_i**2) + ((target_i - output_i) / sigma_i)**2 ].

        NOTE: We scale the log-likelihood by (N / M) to approximate the log-likelihood on the full dataset.
              Some implementations equivalently scale the KL divergence term by (M / N).

        :param output: model outputs
        :param target: target values
        :param sigma_n: Standard deviation of measurements
        :return: log-likelihood
        """
        batch_size = target.size()[0]
        ell = torch.log(2 * pi * sigma_n.pow(2)).sum() + torch.div(target - output, sigma_n).pow(2).sum()
        return -0.5 * (self.train_size / batch_size) * ell

    def batch_elbo(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 1, kl_factor: float = 1.0) \
            -> ty.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss on a batch (x, y). The loss is used to fit the model/infer the model parameters.
        :param x: input of size (batch_size, input_dim)
        :param y: output target of size (batch_size, output_dim)
        :param n_samples: Number of Monte-Carlo samples
        :param kl_factor: Factor used to weight Kullback-Leibler divergence term
        :return: loss, log-likelihood, KL cost
        """

        assert n_samples >= 1, "MC sample size must be one or larger"

        # MC sampling
        ell_mc = torch.zeros((1,), device=self.device)  # MC estimator of the (mini-batch) expected log-likelihood
        kl_mc = torch.zeros((1,), device=self.device)  # MC estimator of the KL cost
        for i in range(n_samples):

            # Do a forward-pass to compute the output and KL cost
            output, kl_cost_i = self.forward(x, sample=True)

            # Compute noise standard deviation and KL cost
            noise_std, kl_cost_noise_i = self.noise_model(x)

            # Compute log-likelihood of mini-batch
            ell_i = self._loglikelihood(output, y, sigma_n=noise_std)

            # Add to the MC estimators
            ell_mc += ell_i
            kl_mc += kl_cost_i + kl_cost_noise_i

        # Average to get the MC estimate
        ell_mc /= n_samples
        kl_mc /= n_samples

        # Compute the ELBO (the KL factor can be used to put less weight on the KL cost)
        elbo = ell_mc - kl_factor * kl_mc

        return elbo, ell_mc, kl_mc

    def infer_parameters(self, train_set: TensorDataset, val_set: TensorDataset,
                         n_epochs: int, batch_size: int = 1, n_samples: int = 1, kl_factor: float = 1.0,
                         ) -> ty.Dict:
        """
        Approximate inference using stochastic gradient variational Bayes
        (training Bayesian neural network parameters)

        TODO: Make it possible to do sequential inference, using previous posterior as prior

        :param train_set: Training dataset
        :param val_set: Validation dataset
        :param n_epochs: Number of epochs to train
        :param batch_size: Batch size (default: 1)
        :param n_samples: Number of Monte-Carlo samples to use in inference (default: 1)
        :param kl_factor: Factor used to weight the KL term (default: 1.0)
        :return: Dictionary with training statistics
        """

        self.train_size = len(train_set)  # Used to compute log-likelihood

        # Dataset loader
        if str(self.device) == 'cuda':
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       pin_memory=True, num_workers=0)
        else:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       pin_memory=False, num_workers=3)

        # Training stats
        train_stats = {
            'elbo': [],
            'log_likelihood': [],
            'kl_div': [],
            'mse_train': [],
            'mae_train': [],
            'mse_val': [],
            'mae_val': [],
        }

        # Train model
        for epoch in range(1, n_epochs + 1):

            # Lists to hold batch losses
            elbo_epoch = []
            ll_epoch = []
            kl_epoch = []

            # Iterate over training data in batches
            for data in train_loader:
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                # Zero parameter gradients
                self.optimizer.zero_grad()

                # Forward pass to compute loss = -ELBO
                elbo, ell, kl = self.batch_elbo(x, y, n_samples=n_samples, kl_factor=kl_factor)

                # We want to maximize the ELBO, which is equivalent to minimizing the negative ELBO
                loss = -elbo

                # Compute gradients wtr. parameters
                # with torch.autograd.detect_anomaly():  # Use for debugging
                #     loss.backward()
                loss.backward()

                # Update parameters
                self.optimizer.step()

                # Compute loss
                elbo_epoch.append(elbo.detach().item())
                ll_epoch.append(ell.detach().item())
                kl_epoch.append(kl.detach().item())

            # Compute stats
            mean_elbo = sum(elbo_epoch) / len(elbo_epoch)
            mean_ll = sum(ll_epoch) / len(ll_epoch)
            mean_kld = sum(kl_epoch) / len(kl_epoch)

            train_stats['elbo'].append(mean_elbo)
            train_stats['log_likelihood'].append(mean_ll)
            train_stats['kl_div'].append(mean_kld)

            # Compute train error
            x_train, y_train = train_set.tensors
            pred_train, _ = self.forward(x_train.to(self.device), sample=False)
            pred_train = pred_train.detach()

            mse_train = (pred_train - y_train.to(self.device)).pow(2).mean().item()
            mae_train = torch.abs(pred_train - y_train.to(self.device)).mean().item()

            train_stats['mse_train'].append(mse_train)
            train_stats['mae_train'].append(mae_train)

            # Compute validation error
            if val_set:
                x_val, y_val = val_set.tensors
                pred_val, _ = self.forward(x_val.to(self.device), sample=False)
                pred_val = pred_val.detach()

                mse_val = (pred_val - y_val.to(self.device)).pow(2).mean().item()
                mae_val = torch.abs(pred_val - y_val.to(self.device)).mean().item()

                train_stats['mse_val'].append(mse_val)
                train_stats['mae_val'].append(mae_val)

                # Print progress
                print('[%d] ELBO: %.8f, log-likelihood: %.8f, KL: %.8f, Val MSE: %.8f, Val MAE: %.8f'
                      % (epoch, mean_elbo, mean_ll, mean_kld, mse_val, mae_val))

            else:
                # Print progress
                print('[%d] ELBO: %.8f, log-likelihood: %.8f, KL: %.8f, Train MSE: %.8f, Train MAE: %.8f'
                      % (epoch, mean_elbo, mean_ll, mean_kld, mse_train, mae_train))

        return train_stats

    def sample_predict(self, x: torch.Tensor, n_samples: int, add_noise: bool = True) -> torch.Tensor:
        """
        Sample predictions conditioned on the input x
        TODO: add measurement noise to prediction (make it optional)

        :param x: input of size (batch_size, n_inputs)
        :param n_samples: number of samples
        :param add_noise: simulate noise on prediction
        :return: tensor of size (n_samples, n_batch, n_outputs) holding predictions
        """
        assert n_samples > 0, "'n_samples' must be a positive integer"
        x = x.to(self.device)

        # Initialize tensor with size (n_samples, n_batch, n_outputs) to hold predictions
        n_batch = x.shape[0]
        predictions = x.data.new_zeros((n_samples, n_batch, self.n_outputs))

        for i in range(n_samples):
            y, _ = self.forward(x, sample=True)
            if add_noise:
                y += self.noise_model.sample(y, n_samples=1)[0]  # Get first sample
            predictions[i] = y.detach()

        return predictions

    def prediction_uncertainty(self, x: torch.Tensor, n_samples: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction mean and variance
        :param x: input of size (batch_size, n_inputs)
        :param n_samples: number of samples
        :return: prediction mean and variance
        """
        assert n_samples > 0
        x = x.to(self.device)

        # Create tensors with size (n_samples, n_batch, n_outputs) hold to predictions and noise.
        n_batch = x.shape[0]
        model_output = x.data.new_zeros((n_samples, n_batch, self.n_outputs))
        noise_std = x.data.new_zeros((n_samples, n_batch, self.n_outputs))

        for i in range(n_samples):
            y, _ = self.forward(x, sample=True)
            model_output[i] = y.detach()
            e, _ = self.noise_model(y, sample=True)
            noise_std[i] = e.detach()

        # Estimate mean and variance
        mean = torch.mean(model_output, 0)
        second_moment = torch.mean(noise_std.pow(2) + model_output.pow(2), 0)
        variance = second_moment - mean.pow(2)

        return mean, variance

    def get_num_parameters(self):
        return np.sum(p.numel() for p in self.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.noise_model.train(mode)

    def save(self, path: str):
        """
        Save model and optimizer state
        :param path: Path to save checkpoint
        :return:
        """
        torch.save({
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load model from file
        :param path: Path to checkpoint
        :return:
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


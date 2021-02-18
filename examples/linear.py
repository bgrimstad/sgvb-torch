"""
Created 05 March 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from sgvb.bnn_model import BayesNet
from sgvb.noise_models import FixedNoise, HomoscedasticNoise
from examples.plotting import plot_training_stats, plot_prediction


def load_dataset():
    # True parameters of linear model
    a = 1.
    b = .5
    sigma_n = .5

    # Create some data
    n_data = 1000
    x = np.linspace(-5, 5, n_data).reshape((-1, 1))
    noise = np.random.normal(0, sigma_n, (n_data, 1))

    y = a*x + b + noise

    return x, y


def mean_weight_std(model: BayesNet):
    mean = 0
    for l in model.linear_layers:
        mean += torch.mean(l.weight_std_q)
    return (mean / len(model.linear_layers)).item()


if __name__ == '__main__':

    # Set seed
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    x_train_np, y_train_np = load_dataset()
    train_size = y_train_np.shape[0]  # Number of training datapoints (N)
    x_train = torch.from_numpy(x_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()
    train_data = TensorDataset(x_train, y_train)

    # Build linear model
    n_inputs = 1
    n_outputs = 1
    layers = [n_inputs, n_outputs]
    # noise_model = FixedNoise(sigma=.5)
    noise_model = HomoscedasticNoise(mu_s=0, sigma_s=1.0)
    model = BayesNet(layers=layers, noise_model=noise_model)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f'Total params: {model.get_num_parameters()}')

    mean_weight_std_before_train = mean_weight_std(model)

    # Train model
    train_stats = model.infer_parameters(train_data, val_set=None, n_epochs=100, batch_size=10, n_samples=5)
    model.train(mode=False)  # Set model training mode to False so that we can perform predictions

    # Plot training stats
    plot_training_stats(train_stats)

    # Compute MAE on training data
    pred_train, _ = model(x_train, sample=False)
    pred_train = pred_train.cpu()
    mae_train = (pred_train - y_train).abs().mean().detach().numpy()
    print('MAE train:', mae_train)

    # Mean weight std before and after training
    mean_weight_std_after_train = mean_weight_std(model)
    print(f'Mean weight std before training: {mean_weight_std_before_train}')
    print(f'Mean weight std after training: {mean_weight_std_after_train}')

    # Plot prediction on training data
    plt.figure()
    plt.plot(x_train_np.flatten().tolist(), y_train_np, '-*', label='Data')
    plt.plot(x_train_np.flatten().tolist(), pred_train.detach().numpy(), label='Model')
    plt.title('Mean prediction of model (weights are not sampled)')
    plt.legend()

    # Print model weights (compare with model)
    print('Printing layers')
    for l in model.linear_layers:
        print(l.weight_mean_q)
        print(l.weight_std_q)
        print(l.bias_mean_q)
        print(l.bias_std_q)
        print()

    # Homoscedastic noise model parameters
    if type(noise_model) == 'HomoscedasticNoise':
        print('Noise parameters')
        print(model.noise_model.mean_q)
        print(model.noise_model.std_q)

    # Plot uncertainty on training data
    plot_prediction(model, x_train, y_train, n_samples=100)
    plt.show()

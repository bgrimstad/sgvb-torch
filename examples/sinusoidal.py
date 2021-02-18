"""
Created 05 March 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from math import pi

from sgvb.bnn_model import BayesNet
from sgvb.noise_models import FixedNoise, HomoscedasticNoise


def load_dataset():
    # Noise level
    # Note: the noise is heteroscedastic
    sigma_n = .02

    # Create some data
    n_data = 500
    x = np.linspace(.0, .5, n_data).reshape((-1, 1))
    noise = np.random.normal(0, sigma_n, (n_data, 1))
    y = x + noise + 0.3*np.sin(2*pi*(x + noise)) + 0.3*np.sin(4*pi*(x + noise))

    return x, y


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

    # Create model
    # NOTE: We use a learned homoscedastic noise model which cannot capture the heteroscedastic noise of the model
    layers = [1, 100, 100, 1]  # Layer sizes, includes input and output layer size
    # noise_model = FixedNoise(sigma=0.1)
    noise_model = HomoscedasticNoise(mu_s=0, sigma_s=1.0)
    model = BayesNet(layers=layers, noise_model=noise_model)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f'Total params: {model.get_num_parameters()}')

    # Train model
    train_stats = model.infer_parameters(train_data, val_set=None, n_epochs=500, batch_size=100, n_samples=3)
    model.train(mode=False)  # Set model training mode to False so that we can perform predictions

    # Learned parameters of homoscedastic noise model
    if type(noise_model) == 'HomoscedasticNoise':
        print('Noise parameters')
        print(model.noise_model.mean_q)
        print(model.noise_model.std_q)

    # Plot training stats
    elbo = train_stats['elbo']
    plt.figure()
    plt.plot(range(len(elbo)), elbo)
    plt.xlabel("Number of epochs")
    plt.ylabel("ELBO")

    # Plot prediction
    n_data_test = 500
    x_test_np = np.linspace(-0.5, 1., n_data_test)
    x_test = torch.from_numpy(x_test_np).float()

    mean_test, var_test = model.prediction_uncertainty(x_test, n_samples=1000)
    mean_test = mean_test.cpu().detach().numpy().reshape(-1,)
    var_test = var_test.cpu().detach().numpy()
    std_test = np.sqrt(var_test).reshape(-1,)

    fig, ax = plt.subplots()
    ax.scatter(x_train_np, y_train_np, marker='x', c='red')
    ax.plot(x_test_np, mean_test, '-', c='black', label='Mean prediction')
    ax.fill_between(x_test_np, mean_test - std_test * 2, mean_test + std_test * 2,
                    facecolor='gray', alpha=0.3, label='2 sigma CI')

    plt.legend()
    plt.show()


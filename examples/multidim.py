"""
Created 29 February 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import torch
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from sgvb.bnn_model import BayesNet
from sgvb.noise_models import FixedNoise


def unknown_function(x1, x2):
    z1 = (15 * x1 - 5)
    z2 = (10 * x2 - 2)
    y = 1 / (1 + np.exp(-z1)) + 1 / (1 + np.exp(-z2)) + z1 * z2
    return y


def create_dataset():
    np.random.seed(1234)
    n_data = 100

    x1 = np.linspace(-1, 2, n_data).reshape((-1, 1))
    x2 = 0.1*np.random.randn(n_data).reshape((-1, 1))
    x = np.hstack([x1, x2])
    noise = 0.1 * np.random.randn(n_data).reshape((-1, 1))
    y = unknown_function(x1, x2) + noise

    return x, y


if __name__ == '__main__':

    x_data_np, y_data_np = create_dataset()
    train_size = y_data_np.shape[0]  # Number of training datapoints (N)

    x_data = torch.from_numpy(x_data_np).float()
    y_data = torch.from_numpy(y_data_np).float()
    train_data = TensorDataset(x_data, y_data)

    # Set seed
    # seed = 123
    # torch.manual_seed(seed)

    # Build model
    layers = [2, 100, 100, 100, 1]
    noise_model = FixedNoise(sigma=0.1)
    model = BayesNet(layers=layers, noise_model=noise_model)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f'Total params: {model.get_num_parameters()}')

    # Train model
    train_stats = model.infer_parameters(train_data, val_set=None, n_epochs=100, batch_size=10, n_samples=5)

    # Compute MAE
    model.train(mode=False)  # Set model training mode to False so that we can perform predictions
    y_model, _ = model(x_data, sample=False)
    y_model = y_model.cpu().detach()
    error = y_data - y_model
    np_error = error.detach().numpy()
    np_error = np.abs(np_error)
    np_error = np_error[~np.isnan(np_error)]
    print('MAE:', np.mean(np_error))

    # Plot prediction on training data
    plt.figure()
    plt.plot(range(len(y_data_np)), y_data_np, '-*', label='Data')
    plt.plot(range(len(x_data_np)), y_model.detach().numpy(), label='Model')
    plt.legend()

    # Make predictions on new data
    n_sites = 5
    x1 = np.array([0, 1, 0, 1, 5]).reshape(-1, 1)
    x2 = np.array([0, 0, 1, 1, 5]).reshape(-1, 1)
    x_new_np = np.hstack([x1, x2])
    noise_new_np = 0.1*np.random.randn(n_sites).reshape(-1, 1)
    y_new_np = unknown_function(x1, x2) + noise_new_np
    x_new = torch.tensor(data=x_new_np, dtype=torch.float)
    y_new = torch.tensor(data=y_new_np, dtype=torch.float)

    n_samples = 100
    pred = model.sample_predict(x_new, n_samples=n_samples).cpu().detach().numpy()
    pred_mean = pred.mean(axis=0)  # Compute mean prediction

    plt.figure()
    for i in range(n_samples):
        plt.plot(range(n_sites), pred[i], c='gray', alpha=0.3)

    plt.plot(range(n_sites), pred_mean, '--k', lw=2, label='Mean prediction (sample mean)')
    plt.plot(range(n_sites), y_new_np, '-.r', lw=2, label='Data')
    plt.legend()

    plt.show()


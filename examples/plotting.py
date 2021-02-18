"""
Created 22 April 2020
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""


import torch
import matplotlib.pyplot as plt
from sgvb.bnn_model import BayesNet


def prediction_summary(predictions: torch.Tensor):
    """
    Compute summary of predictions
    :param predictions: prediction tensor of size (n_samples, batch_size, n_outputs)
    :return: dictionary with a mean tensor
    """
    n_samples = predictions.size(0)
    site_stats = {
        "mean": torch.mean(predictions, 0),
        "std": torch.std(predictions, 0),
        "5%": predictions.kthvalue(int(n_samples * 0.05), dim=0)[0],
        "95%": predictions.kthvalue(int(n_samples * 0.95), dim=0)[0],
    }
    return site_stats


def plot_prediction(model: BayesNet, x: torch.Tensor, y: torch.Tensor = None, n_samples: int = 100):
    """
    Plot model predictions at data sites x.

    :param model: Bayesian neural network (BayesNet)
    :param x: Input data
    :param y: Output data
    :param n_samples: Number of samples to draw for each input x_i
    :return: list of figures
    """

    # Tensors to numpy for easy plotting
    x_np = x.detach().numpy()
    if x_np.ndim == 1:
        x_axis = x_np.tolist()
    elif x_np.shape[1] == 1:
        x_axis = x_np[:, 0].tolist()
    else:
        # Multi-dimensional input - use sample numbers on x-axis
        x_axis = list(range(len(x_np)))

    y_np = None
    if y is not None:
        y_np = y.detach().numpy()

    # Sample network
    pred = model.sample_predict(x, n_samples=n_samples)
    pred = pred.cpu()
    pred_np = pred.detach().numpy()

    # Compute mean prediction
    pred_mean = pred_np.mean(axis=0)

    # We can also compute a mean prediction using only the mean values of the weights
    # Note that this will not add measurement noise to the prediction since the mean of the noise is zero
    pred_mean_weights, _ = model.forward(x, sample=False)
    pred_mean_weights = pred_mean_weights.cpu()
    pred_mean_weights_np = pred_mean_weights.detach().numpy()

    # List of figures
    figures = []

    # Plot predictions
    fig1 = plt.figure()

    for i in range(n_samples):
        # This will plot all predictions we have drawn from the posterior as separate lines
        plt.plot(x_axis, pred_np[i], c='gray', alpha=0.2)

    plt.plot(x_axis, pred_mean, '--k', lw=2, label='Mean prediction (weights sampled)')
    plt.plot(x_axis, pred_mean_weights_np, '-.k', lw=2, label='Mean prediction (weights not sampled)')
    if y_np is not None:
        plt.scatter(x_axis, y_np, marker='x', c='k', s=20, zorder=1e3, label='Data')
    plt.legend()
    figures.append(fig1)

    # Compute preduction summary
    summary = prediction_summary(pred)
    summary_mean = summary['mean'].detach().numpy().reshape(-1, )
    summary_std = summary['std'].detach().numpy().reshape(-1, )
    summary_5pct = summary['5%'].detach().numpy().reshape(-1, )
    summary_95pct = summary['95%'].detach().numpy().reshape(-1, )

    # Plot prediction with two standard deviations
    fig2, ax = plt.subplots()
    ax.fill_between(x_axis, summary_mean - summary_std * 2, summary_mean + summary_std * 2,
                    facecolor='gray', alpha=0.3, label='2 sigma')
    ax.plot(x_axis, summary_mean, c='black', label='Mean prediction')
    if y_np is not None:
        ax.scatter(x_axis, y_np, marker='x', c='k', s=20, zorder=1e3, label='Data')
    plt.legend()
    figures.append(fig2)

    # Plot prediction with 90% confidence interval
    fig3, ax = plt.subplots()
    ax.fill_between(x_axis, summary_5pct, summary_95pct, facecolor='gray', alpha=0.3, label='90% CI')
    ax.plot(x_axis, summary_mean, c='black', label='Mean prediction')  # Mean of sampled predictions
    if y_np is not None:
        ax.scatter(x_axis, y_np, marker='x', c='k', s=20, zorder=1e3, label='Data')
    plt.legend()
    figures.append(fig3)

    return figures

def plot_training_stats(train_stats):
    """
    Plot training statistics
    :param train_stats: dictionary with training statistics
    :return:
    """

    elbo = train_stats['elbo']
    mse_train = train_stats['mse_train']
    mae_train = train_stats['mae_train']

    plt.figure()
    plt.plot(range(len(elbo)), elbo)
    plt.xlabel("Number of epochs")
    plt.ylabel("ELBO")

    plt.figure()
    plt.plot(range(len(mse_train)), mse_train)
    plt.xlabel("Number of epochs")
    plt.ylabel("Train MSE")

    plt.figure()
    plt.plot(range(len(mae_train)), mae_train)
    plt.xlabel("Number of epochs")
    plt.ylabel("Train MAE")

    plt.show()

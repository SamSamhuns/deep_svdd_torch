from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_in_out_dist(in_df: pd.DataFrame,
                     out_df: pd.DataFrame,
                     savepath: str = "plots/outlier_scores_deep_svdd.jpg"):
    """
    Plot the in_vs_out KDE distributions
    """
    _, ax = plt.subplots()
    in_df.plot.kde(ax=ax, legend=True, title="Outliers vs Inliers (Deep SVDD)")
    out_df.plot.kde(ax=ax, legend=True)
    plt.xlim(-0.05, 0.08)
    ax.grid(axis="x")
    ax.grid(axis="y")

    print(f"Saving in_vs_out dist to {savepath}")
    plt.savefig(savepath)
    plt.clf()
    plt.close()


def plot_metric(metrics_lists: List[List[Union[int, float]]],
                labels: List[str] = None,
                title: str = "Metrics over epoch",
                xlabel: str = "Epochs",
                savepath: str = "plots/ametric.jpg"):
    """
    Plot the metrics over iterations/epochs.
    """
    if not labels:
        labels = ["metric"]
    plt.figure(figsize=(10, 6))
    max_metric_len = 0
    for i,  metric_list in enumerate(metrics_lists):
        plt.plot(metric_list, label=labels[i])
        max_metric_len = max(max_metric_len, len(metric_list))

    plt.xlabel(xlabel)
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Set x-ticks to show only whole numbers
    plt.xticks(np.arange(0, max_metric_len + 1, step=1))
    plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close()


def plot_scatter_loss(
        y_score: np.ndarray,
        y_true: np.ndarray,
        savepath: str = "ae_scatter_loss.png",
        title: str = "Scatter plot",
        xlabel: str = "Sequence Points",
        ylabel: str = "AE loss") -> None:
    """Plot the loss scores of the autoencoder."""
    fig = plt.figure(figsize=(10, 10))

    idx_dict = [
        {"idx": np.where(y_true == 0)[0], "color": "blue",
         "mkr": 'o', "label": "Normal"},
        {"idx": np.where(y_true == 1)[0], "color": "red", "mkr": 'x', "label": "Anomaly"}]
    for idict in idx_dict:
        plt.scatter(
            idict["idx"], y_score[idict["idx"]],
            color=idict["color"], s=5, alpha=0.6,
            marker=idict["mkr"], label=idict["label"])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')

    print("Saving scatter plot to", savepath)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.close(fig)


def plot_scatter_2d(
        X: np.ndarray,
        y: np.ndarray,
        savepath: str,
        title: str = "Scatter plot",
        xlabel: str = "t-SNE Dimension 1",
        ylabel: str = "t-SNE Dimension 2") -> None:
    """Create a scatter plot
    Parameters:
        X (np.ndarray): The input data points, 2D array of shape (n_samples, 2).
        y (np.ndarray): The labels for each data point, 1D array of shape (n_samples,).
        savepath (str): The filename to save the scatter plot image.
        title (str, optional): The title of the plot. Defaults to "Scatter plot".
        xlabel (str, optional): The label for the x-axis. Defaults to "t-SNE Dimension 1".
        ylabel (str, optional): The label for the y-axis. Defaults to "t-SNE Dimension 2".
    """
    fig = plt.figure(figsize=(15, 10))
    for label in np.unique(y):
        indices = (y == label).squeeze()
        plt.scatter(X[indices, 0],
                    X[indices, 1],
                    label=f"Class: {label}",
                    marker="x", alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)


def plot_tsne_2d(
        X: np.ndarray, y: np.ndarray,
        savepath: str,
        learning_rate: Union[str, float] = "auto",
        title: str = "t-SNE plot",
        xlabel: str = "t-SNE dimension 1",
        ylabel: str = "t-SNE dimension 2",
        n_components: int = 2,
        perplexity: int = 5,
        random_state: int = 42) -> None:
    """
    Generate a t-SNE plot and save a scatter plot with matplotlib.
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    # Fit and transform the data with the t-SNE model
    x_embedded = TSNE(n_components=n_components,
                      learning_rate=learning_rate,
                      perplexity=perplexity,
                      init='random',
                      random_state=random_state).fit_transform(X)

    print(f"Saving t-SNE plot to {savepath}")
    # Create a scatter plot
    plot_scatter_2d(x_embedded, y, savepath,
                    title=title, xlabel=xlabel, ylabel=ylabel)

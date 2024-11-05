from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_in_out_dist(in_df: pd.DataFrame,
                     out_df: pd.DataFrame,
                     savepath: str = "outlier_scores_deep_svdd.jpg"):
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

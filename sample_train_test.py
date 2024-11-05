from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from train import TrainerDeepSVDD
from preprocess import get_mnist_dls


@dataclass
class Args:
    """Deep SVDD arguments."""
    num_epochs: int = 150
    num_epochs_ae: int = 150
    patience: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.5e-6
    weight_decay_ae: float = 0.5e-3
    lr_ae: float = 1e-4
    lr_milestones: List[int] = field(default_factory=lambda: [50])
    batch_size: int = 200
    pretrain: bool = True
    latent_dim: int = 32
    normal_class: int = 0


def plot_in_out_dist(in_df, out_df, savepath: str = "outlier_scores_deep_svdd.jpg"):
    """
    Plot the in vs the out distributions
    """
    _, ax = plt.subplots()
    in_df.plot.kde(ax=ax, legend=True, title='Outliers vs Inliers (Deep SVDD)')
    out_df.plot.kde(ax=ax, legend=True)
    plt.xlim(-0.05, 0.08)
    ax.grid(axis='x')
    ax.grid(axis='y')

    print(f"Saving in_vs_out dist to {savepath}")
    plt.savefig(savepath)


def main():
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = get_mnist_dls(args)

    deep_SVDD = TrainerDeepSVDD(args, dls, device)

    # pretrain with Encoder and Decoder
    if args.pretrain:
        deep_SVDD.pretrain()
    # train encoder
    deep_SVDD.train()

    # evaluate encoder
    thres = deep_SVDD.calc_threshold_score(deep_SVDD.train_loader, deep_SVDD.encoder)
    labels, scores = deep_SVDD.eval_enc(thres)

    scores_in = scores[np.where(labels == 0)[0]]
    scores_out = scores[np.where(labels == 1)[0]]

    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])
    plot_in_out_dist(in_, out_)


if __name__ == "__main__":
    main()

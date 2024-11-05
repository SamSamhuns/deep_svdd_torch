import argparse
import torch
import pandas as pd
import numpy as np

from deepsvdd.trainer import TrainerDeepSVDD
from deepsvdd.preprocess import get_mnist_dls
from deepsvdd.utils.plots import plot_in_out_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="number of epochs (default: %(default)s)")
    parser.add_argument("--num_epochs_ae", type=int, default=150,
                        help="number of epochs for the pretraining (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Patience for Early Stopping (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate (default: %(default)s)")
    parser.add_argument("--weight_decay", type=float, default=0.5e-6,
                        help="Weight decay hyperparameter for the L2 reg (default: %(default)s)")
    parser.add_argument("--weight_decay_ae", type=float, default=0.5e-3,
                        help="Weight decay hyperparameter for the L2 reg in AE (default: %(default)s)")
    parser.add_argument("--lr_ae", type=float, default=1e-4,
                        help="learning rate for autoencoder (default: %(default)s)")
    parser.add_argument("--lr_milestones", type=list, default=[50],
                        help="Milestones at which the scheduler mults the lr by 0.1 (default: %(default)s)")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--pretrain", type=bool, default=True,
                        help="Pretrain the network using an autoencoder (default: %(default)s)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of the latent variable z (default: %(default)s)")
    parser.add_argument("--normal_class", type=int, default=0,
                        help="Class to be treated as normal; the rest is set as anomalous (default: %(default)s)")
    # parsing arguments.
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # check if cuda is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dls = get_mnist_dls(args)
    deep_SVDD = TrainerDeepSVDD(args, dls, device)
    # pretrain with encoder and decoder
    if args.pretrain:
        deep_SVDD.pretrain()
    # train encoder only
    deep_SVDD.train()

    # evaluate encoder
    thres = deep_SVDD.calc_threshold_score(
        deep_SVDD.train_loader, deep_SVDD.encoder)
    labels, scores = deep_SVDD.eval_enc(thres)

    # plot the distribution of the inlier/outlier scores
    scores_in = scores[np.where(labels == 0)[0]]
    scores_out = scores[np.where(labels == 1)[0]]

    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])
    plot_in_out_dist(in_, out_)

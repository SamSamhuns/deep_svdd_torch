import argparse
import torch
import pandas as pd
import numpy as np

from trainer import TrainerDeepSVDD
from preprocess import get_mnist_dls
from utils.utils import plot_in_out_dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=150,
                        help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50,
                        help="Patience for Early Stopping")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.5e-6,
                        help="Weight decay hyperparameter for the L2 regularization")
    parser.add_argument("--weight_decay_ae", type=float, default=0.5e-3,
                        help="Weight decay hyperparameter for the L2 regularization")
    parser.add_argument("--lr_ae", type=float, default=1e-4,
                        help="learning rate for autoencoder")
    parser.add_argument("--lr_milestones", type=list, default=[50],
                        help="Milestones at which the scheduler multiply the lr by 0.1")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size")
    parser.add_argument("--pretrain", type=bool, default=True,
                        help="Pretrain the network using an autoencoder")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of the latent variable z")
    parser.add_argument("--normal_class", type=int, default=0,
                        help="Class to be treated as normal. The rest will be set as anomalous.")
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

    scores_in = scores[np.where(labels == 0)[0]]
    scores_out = scores[np.where(labels == 1)[0]]

    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])
    plot_in_out_dist(in_, out_)

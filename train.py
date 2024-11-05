import os
import argparse
import json
from datetime import datetime
import torch
import pandas as pd
import numpy as np

from deepsvdd.trainer import TrainerDeepSVDD
from deepsvdd.preprocess import get_mnist_dls
from deepsvdd.utils.plots import plot_in_out_dist, plot_tsne_2d, plot_scatter_loss


def save_args(args, save_path: str):
    """
    Convert args to a dictionary and save to JSON and save to file
    """
    args_dict = vars(args)  # Convert argparse Namespace to dictionary

    # Save args_dict to a JSON file
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Train+test settings saved to {save_path}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # check if cuda is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = "logs/" + datetime.now().strftime("%Y_%m_%d-%H_%M")
    os.makedirs(log_dir, exist_ok=True)
    # save the train/test args
    save_args(args, save_path=f"{log_dir}/args.json")

    dls = get_mnist_dls(args)
    deep_SVDD = TrainerDeepSVDD(args, dls, log_dir=log_dir, device=device)
    # pretrain with autoencoder
    if args.pretrain:
        deep_SVDD.pretrain()
        # eval autoencoder
        recons_losses, y_trues = deep_SVDD.eval_ae()
        plot_scatter_loss(recons_losses, y_trues,
                          savepath=f"{log_dir}/recons_test_losses_ae.jpg")

    # train encoder only
    deep_SVDD.train()

    # evaluate encoder
    thres = deep_SVDD.calc_threshold_score(
        deep_SVDD.train_loader, deep_SVDD.encoder)
    y_trues, y_scores, y_preds, z_embs = deep_SVDD.eval_enc(thres)

    # plot the distribution of the inlier/outlier scores
    scores_in = y_scores[np.where(y_trues == 0)[0]]
    scores_out = y_scores[np.where(y_trues == 1)[0]]

    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])
    plot_in_out_dist(in_, out_,
                     savepath=f"{log_dir}/outlier_scores_deep_svdd.jpg")
    # plot the t-sne plots
    plot_tsne_2d(z_embs, y_trues,
                 savepath=f"{log_dir}/tsne_mnist_true.jpg",
                 title="t-SNE - MNIST")
    plot_tsne_2d(z_embs, y_preds,
                 savepath=f"{log_dir}/tsne_mnist_pred.jpg",
                 title="t-SNE - MNIST")

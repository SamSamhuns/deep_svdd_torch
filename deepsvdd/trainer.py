import os
import torch
from torch import optim
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import numpy as np

from deepsvdd.model import autoencoder, encoder
from deepsvdd.utils.common import weights_init_normal
from deepsvdd.utils.plots import plot_metric


class TrainerDeepSVDD:
    def __init__(self, args, dls, log_dir, device):
        self.args = args
        self.train_loader, self.test_loader = dls
        self.device = device
        self.ae = autoencoder(self.args.latent_dim).to(self.device)
        self.encoder = encoder().to(self.device)
        self.c = None
        self.log_dir = log_dir

    def save_ae_weights(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        self.encoder = encoder(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        self.encoder.load_state_dict(state_dict, strict=False)
        torch.save({"center": c.cpu().data.numpy().tolist(),
                    "net_dict": self.encoder.state_dict()}, f"{self.log_dir}/pretrained_ae.pth")

    def save_enc_weights(self, model, epoch):
        """Saving the Deep SVDD encoder model weights"""
        torch.save(model.state_dict(), f"{self.log_dir}/model_epoch_{epoch}.pth")

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def calc_threshold_score(self, train_loader, model, percentile: float = 0.99) -> float:
        """Calculate the anomaly threshold score for the model after training.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            model (BaseModel): The Deep SVDD network.
            percentile (float): Confidence percentile for threshold calculation.

        Returns:
            float: Anomaly threshold score.
        """
        model.eval()
        dists = torch.tensor([], device=self.device)
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                outputs = model(x)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                dists = torch.cat((dists, dist), dim=0)

        return np.percentile(dists.cpu().numpy(), percentile * 100)

    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        self.ae.apply(weights_init_normal)
        optimizer = optim.Adam(self.ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

        epoch_loss, epoch_lr = [], []
        self.ae.train()
        print("Pretraining Autoencoder...")
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = self.ae(x)
                reconst_loss = torch.mean(
                    torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            avg_epoch_loss = total_loss / len(self.train_loader)
            epoch_loss.append(avg_epoch_loss)
            epoch_lr.append(scheduler.get_last_lr()[0])

            print(f"AE Epoch: {epoch}, Loss: {avg_epoch_loss:.3f}")
        self.save_ae_weights(self.ae, self.train_loader)
        # plot training loss and lr
        os.makedirs(self.log_dir, exist_ok=True)
        plot_metric(
            [epoch_loss], labels=["AE Loss"],
            title="Train AE Loss over epochs", savepath=f"{self.log_dir}/ae_loss.jpg")
        plot_metric(
            [epoch_lr], labels=["AE LR"],
            title="AE LR over epochs", savepath=f"{self.log_dir}/ae_lr.jpg")

    def eval_ae(self):
        """Evaluate the autoencoder"""
        self.ae.eval()
        print("Testing Autoencoder model...")
        recons_losses = []
        y_trues = []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float().to(self.device)
                x_hat = self.ae(x)
                recons_loss = torch.sum(
                    (x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
                recons_losses.append(recons_loss.cpu())
                y_trues.append(y)
        recons_losses = torch.cat(recons_losses).numpy()
        y_trues = torch.cat(y_trues).numpy()
        return recons_losses, y_trues

    def train(self):
        """Training the Deep SVDD model"""

        if self.args.pretrain is True:
            state_dict = torch.load(f"{self.log_dir}/pretrained_ae.pth")
            self.encoder.load_state_dict(state_dict["net_dict"])
            c = torch.Tensor(state_dict["center"]).to(self.device)
        else:
            self.encoder.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

        epoch_loss, epoch_lr = [], []
        self.encoder.train()
        print("Training Deep SVDD...")
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.encoder(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print(f"Enc Epoch: {epoch}, Loss: {
                  total_loss/len(self.train_loader):.3f}")
        self.c = c
        self.save_enc_weights(self.encoder, epoch)

        # plot training loss and lr
        os.makedirs(self.log_dir, exist_ok=True)
        plot_metric(
            [epoch_loss], labels=["AE Loss"],
            title="Train Enc Loss over epochs", savepath=f"{self.log_dir}/ae_loss.jpg")
        plot_metric(
            [epoch_lr], labels=["AE LR"],
            title="AE Enc over epochs", savepath=f"{self.log_dir}/ae_lr.jpg")

    def eval_enc(self, threshold: float = 0.5):
        """Evaluate the Deep SVDD encoder model"""

        y_scores = []
        y_trues = []
        y_preds = []
        z_embs = []
        self.encoder.eval()
        print("Testing Deep SVDD...")
        print(f"Using a clsf threshold of {threshold}")
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float().to(self.device)
                z = self.encoder(x)
                y_score = torch.sum((z - self.c) ** 2, dim=1)

                y_trues.append(y.cpu())
                y_scores.append(y_score.cpu())
                y_preds.append(y_score.cpu() > threshold)
                z_embs.append(z.cpu())

        y_scores = torch.cat(y_scores).numpy()
        y_trues = torch.cat(y_trues).numpy()
        y_preds = torch.cat(y_preds).numpy()
        z_embs = torch.cat(z_embs).numpy()
        print(f"ROC AUC score: {
            roc_auc_score(y_trues, y_scores)*100:.2f}")
        print(f"Accuracy score: {
            accuracy_score(y_trues, y_preds)*100:.2f}")
        print(f"Balanced accuracy score: {
            balanced_accuracy_score(y_trues, y_preds)*100:.2f}")

        return y_trues, y_scores, y_preds, z_embs

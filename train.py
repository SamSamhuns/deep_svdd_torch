import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from model import autoencoder, encoder
from utils.utils import weights_init_normal


class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.encoder = encoder().to(self.device)

    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        print("Pretraining Autoencoder...")
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(
                    torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            print(f"Epoch: {epoch}, Loss: {total_loss/len(self.train_loader):.3f}")
        self.save_ae_weights(ae, self.train_loader)

    def save_ae_weights(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        os.makedirs("weights", exist_ok=True)
        c = self.set_c(model, dataloader)
        self.encoder = encoder(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        self.encoder.load_state_dict(state_dict, strict=False)
        torch.save({"center": c.cpu().data.numpy().tolist(),
                    "net_dict": self.encoder.state_dict()}, "weights/pretrained_parameters.pth")
        
    def save_enc_weights(self, model, epoch):
        """Saving the Deep SVDD encoder model weights"""
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/model_epoch_{epoch}.pth")

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

    def calc_threshold_score(
            self, train_loader: DataLoader, model: torch.nn.Module, percentile: float = 0.99) -> float:
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
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                dists = torch.cat((dists, dist), dim=0)

        return np.percentile(dists.cpu().numpy(), percentile * 100)

    def train(self):
        """Training the Deep SVDD model"""

        if self.args.pretrain is True:
            state_dict = torch.load("weights/pretrained_parameters.pth")
            self.encoder.load_state_dict(state_dict["net_dict"])
            c = torch.Tensor(state_dict["center"]).to(self.device)
        else:
            self.encoder.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

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
            print(f"Epoch: {epoch}, Loss: {total_loss/len(self.train_loader):.3f}")
        self.c = c
        self.save_enc_weights(self.encoder, epoch)
    
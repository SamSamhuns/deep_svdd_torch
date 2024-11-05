import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def eval(encoder, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    encoder.eval()
    print("Testing...")
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = encoder(x)
            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    threshold = np.percentile(scores, 95)
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print(f"ROC AUC score: {roc_auc_score(labels, scores)*100:.2f}")
    print(f"Accuracy score: {accuracy_score(labels, scores > threshold)*100:.2f}")

    return labels, scores

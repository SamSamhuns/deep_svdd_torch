import torch
import matplotlib.pyplot as plt


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


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

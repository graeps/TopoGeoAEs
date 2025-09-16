from sklearn.decomposition import PCA
import numpy as np


def scatter_datapoints(ax, data, title, dot_size=2, colors=None, color_norm=None, cmap='hsv', apply_pca=True,
                        pca_dim=3):
    d = data.shape[1]
    pca_applied = False

    if d == 1:
        sc = ax.scatter(data[:, 0], np.zeros_like(data[:, 0]), c=colors, cmap=cmap, norm=color_norm, s=dot_size,
                        alpha=0.7)
        ax.set_yticks([])
    elif d == 2:
        sc = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, norm=color_norm, s=dot_size, alpha=0.7)
    elif d == 3 and apply_pca is False:
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap, norm=color_norm, s=dot_size, alpha=0.7)
    else:
        data = PCA(n_components=pca_dim).fit_transform(data)
        if pca_dim == 2:
            sc = ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=cmap, norm=color_norm, s=dot_size, alpha=0.7)
        else:
            sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=cmap, norm=color_norm, s=dot_size,
                            alpha=0.7)
        pca_applied = True

    if title is not None:
        title_suffix = " (PCA)" if pca_applied else ""
        ax.set_title(f"{title}{title_suffix}")
    ax.set_aspect('equal', adjustable='datalim')

    return sc


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

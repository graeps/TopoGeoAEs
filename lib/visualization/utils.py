from sklearn.decomposition import PCA
import numpy as np


def scatter_datapoints(ax, data, title, dot_size=2, colors=None, color_norm=None, cmap='hsv', apply_pca=True,
                        pca_dim=3):
    """
    Plots a scatter plot of input data. Can handle 1D, 2D, and 3D data, with optional PCA reduction to 2/3 dimensions. 
    Customizable dot size, colors, color normalization, and colormap.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis where the scatter plot will be rendered.
        data (numpy.ndarray): The input data for the scatter plot. Should have shape (n_samples, n_features).
        title (str): The title of the scatter plot. A " (PCA)" suffix will be applied if PCA is used.
        dot_size (float, optional): The size of scatter points. Defaults to 2.
        colors (array-like, optional): The colors for each data point. Defaults to None.
        color_norm (matplotlib.colors.Normalize, optional): Normalization for color mapping. Defaults to None.
        cmap (str, optional): The colormap name for the scatter points. Defaults to 'hsv'.
        apply_pca (bool, optional): Whether to apply PCA for dimensionality reduction if the data has more than 
            three dimensions. Defaults to True.
        pca_dim (int, optional): Number of dimensions for PCA projection if applied. Should be 2 or 3. Defaults to 3.

    Returns:
        matplotlib.collections.PathCollection: The scatter plot object created by matplotlib.

    Raises:
        ValueError: If the data dimensionality is higher than 3 and `apply_pca` is False.
    """
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

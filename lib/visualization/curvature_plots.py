import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata
import re

from ..curvature.curvature_pipeline import compute_all_curvatures
from .utils import scatter_datapoints
from ..utils.vectors import get_vectors
from ..datasets.lookup import get_dataset_category
from ..models.lookup import is_euclidean_model, is_spherical_model

def _sanitize_filename(name: str) -> str:
    """
    Sanitizes a given filename by modifying it to meet specified criteria.

    This function ensures the filename is safe for use by replacing certain characters
    and sequences. It removes directory separator characters, replaces disallowed
    characters with underscores, trims unwanted underscores, and converts the
    string to lowercase.

    Args:
        name (str): The filename to sanitize.

    Returns:
        str: A sanitized version of the input filename with disallowed
        characters and sequences replaced.
    """
    s = str(name)
    s = s.replace(os.sep, "_")
    # Replace any sequence of disallowed chars with underscore
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_").lower()


def plot_curvature_norms(angles, curvature_norms, config, norm_val, profile_type, title=None, small_text=None):
    """
    Generates and plots curvature norms on various manifolds and dataset configurations. The visualization
    produced can range from 2D plots to 3D scatter plots, depending on the dataset type and the underlying
    manifold geometry. Optionally, annotated text and customized titles are included based on provided
    parameters. Results may be saved to a specified directory in a designated format.

    Args:
        angles: Array of angles used for representing points on the manifold.
        curvature_norms: Array of curvature norm values corresponding to each angle.
        config: Configuration object containing dataset details and other plotting parameters.
        norm_val: Optional normalization value for scaling curvature norms.
        profile_type: Type of profile for which the curvature norms are being plotted.
        title: Optional title for the plot.
        small_text: Optional small annotation displayed on the plot, typically in the upper-left corner.
    """
    colormap = plt.get_cmap("hsv")

    if norm_val is not None and norm_val > 0:
        color_norm = mpl.colors.Normalize(0.0, norm_val)
    else:
        eps = 1e-3
        color_norm = mpl.colors.Normalize(0.0, max(curvature_norms) + eps)

    if config.dataset_name in {"s1_low", "s1_high"}:
        fig = plt.figure(figsize=(12, 6))  # increased size for clarity
        ax1 = fig.add_subplot(121)
        ax1.plot(angles, curvature_norms, linewidth=2)

        ax2 = fig.add_subplot(122, projection="polar")
        sc = ax2.scatter(
            angles,
            np.ones_like(angles),
            c=curvature_norms,
            s=50,
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax2.set_yticks([])
        if title:
            ax1.set_title(title)
        # Optional small annotation in upper-left of the polar panel
        if small_text:
            try:
                ax2.text(0.01, 0.01, small_text, transform=ax2.transAxes, va='bottom', ha='left', fontsize=14,
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
            except Exception:
                pass

    elif config.dataset_name in {"s2_low", "t2_low", "s2_high", "t2_high"}:
        if config.dataset_name in {"s2_low", "s2_high"}:
            x = config.radius * np.sin(angles[:, 0]) * np.cos(angles[:, 1])
            y = config.radius * np.sin(angles[:, 0]) * np.sin(angles[:, 1])
            z = config.radius * np.cos(angles[:, 0])
        else:  # t2_low / t2_high
            theta = angles[:, 0]
            phi = angles[:, 1]
            x = (config.major_radius - config.minor_radius * np.cos(theta)) * np.cos(phi)
            y = (config.major_radius - config.minor_radius * np.cos(theta)) * np.sin(phi)
            z = config.minor_radius * np.sin(theta)
        points = np.stack([x, y, z], axis=1)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        sc = scatter_datapoints(
            ax=ax,
            data=points,
            colors=curvature_norms,
            dot_size=20,
            cmap="rainbow",
            apply_pca=False,
            pca_dim=3,
            title=title,
        )
        fig.colorbar(sc, ax=ax, shrink=0.7)

        if config.dataset_name in {"t2_low", "t2_high"}:
            r = config.major_radius + config.minor_radius
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
        elif config.dataset_name in {"s2_low", "s2_high"}:
            r = config.radius
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-2 * r / 3, 2 * r / 3)
        else:
            raise NotImplementedError
        plt.axis("off")
        # Optional small annotation in upper-left corner of the 3D panel
        if small_text:
            try:
                ax.text2D(0.01, 0.01, small_text, transform=ax.transAxes, va='bottom', ha='left', fontsize=14,
                          bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
            except Exception:
                pass
    else:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    plt.tight_layout()

    if config.log_dir is not None:
        # Use title in filename if provided, otherwise use profile_type, sanitized
        filename_base = _sanitize_filename(title) if title else _sanitize_filename(profile_type)
        fname = f"heatmaps_template_manifold_{filename_base}_{_sanitize_filename(config.model_type)}.png"
        path = os.path.join(config.log_dir, fname)
        plt.savefig(path)
        print(f"Saved curvature norm plot to: {path}")

    plt.show()
    plt.close()


def scatter_curvature_heatmaps(config, pts, curv, title=None, small_text=None):
    """
    Visualizes curvature data as a scatter plot heatmap on the input data, latents or recons of euclidean models.
    The function adjusts dimensionality using principal component
    analysis (PCA) based on the dataset and creates either a 2D or 3D scatter plot.
    Color gradients represent curvature magnitudes.

    Args:
        config: Configuration object containing settings and metadata for visualization.
        pts: List or array-like of points to be used for scatter visualization.
        curv: List or array-like containing curvature magnitudes corresponding to the
            points.
        title: Optional. Title for the heatmap plot.
        small_text: Optional. Additional text annotation to be displayed in the plot.
    """
    if pts is None or curv is None:
        return
    try:
        points = np.asarray(pts, dtype=float)
        curvature_norms = np.asarray(curv, dtype=float).reshape(-1)
    except Exception:
        return

    pca_dim = 2 if get_dataset_category(config.dataset_name) == "1d" else 3

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(
        111,
        projection='3d' if points.shape[1] >= 3 and pca_dim == 3 else None
    )
    sc = scatter_datapoints(ax=ax, data=points, title=title, colors=curvature_norms, cmap='rainbow', pca_dim=pca_dim)
    fig.colorbar(sc, ax=ax, shrink=0.7)

    # Optional small annotation in upper-left corner
    if small_text:
        try:
            is_3d = getattr(ax, "name", "") == "3d"
            if is_3d:
                # For 3D axes, use text2D with axes transform
                ax.text2D(0.01, 0.01, small_text, transform=ax.transAxes, va='bottom', ha='left', fontsize=14,
                          bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
            else:
                ax.text(0.01, 0.01, small_text, transform=ax.transAxes, va='bottom', ha='left', fontsize=14,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))
        except Exception:
            pass

    plt.tight_layout()

    # Save individual heatmap
    if getattr(config, "log_dir", None) is not None:
        safe_title = _sanitize_filename(title if title else "heatmap")
        fname_indiv = f"heatmap_{safe_title}.png"
        fig.savefig(os.path.join(config.log_dir, fname_indiv))

    plt.show()
    plt.close(fig)


def plot_curvatures_1d(angles, curvature_norms, config, title):
    """
    Plots 1-dimensional curvature norms against specified angles.

    Args:
        angles: array-like
            A 1D array of angles (in radians) against which curvature norms are plotted.
        curvature_norms: dict, list, tuple, or array-like
            Curvature norms to be plotted. Can be a dictionary with series names as keys, a list
            or tuple of series, or a single array-like object representing one series.
        config: object
            Configuration object that may contain attributes such as `log_dir`. Used for saving
            the plot to a file if a `log_dir` is specified.
        title: str
            Title of the plot to display and potentially use for file naming if saved.
    """
    try:
        ang = np.asarray(angles).reshape(-1)
    except Exception:
        raise ValueError("angles must be array-like and 1D")

    series = []
    if curvature_norms is None:
        raise ValueError("curvature_norms cannot be None")

    if isinstance(curvature_norms, dict):
        for k, v in curvature_norms.items():
            if v is None:
                continue
            y = np.asarray(v).reshape(-1)
            if y.shape[0] != ang.shape[0]:
                print(f"plot_curvatures_1d: Skipping '{k}' due to length mismatch "
                      f"(angles={ang.shape[0]}, series={y.shape[0]})")
                continue
            series.append((str(k), y))
    elif isinstance(curvature_norms, (list, tuple)):
        for i, item in enumerate(curvature_norms):
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], (list, tuple, np.ndarray)):
                lbl, arr = item
                y = np.asarray(arr).reshape(-1)
                label = str(lbl)
            else:
                y = np.asarray(item).reshape(-1)
                label = f"Series {i+1}"
            if y.shape[0] != ang.shape[0]:
                print(f"plot_curvatures_1d: Skipping '{label}' due to length mismatch "
                      f"(angles={ang.shape[0]}, series={y.shape[0]})")
                continue
            series.append((label, y))
    else:
        # Single array-like
        y = np.asarray(curvature_norms).reshape(-1)
        if y.shape[0] != ang.shape[0]:
            raise ValueError(f"Length mismatch: angles={ang.shape[0]}, series={y.shape[0]}")
        series.append(("Curvature", y))

    if not series:
        raise ValueError("No valid curvature series to plot after validation.")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    for i, (lbl, y) in enumerate(series):
        color = colors[i % len(colors)] if colors else None
        linestyle = '--' if 'true' in lbl.lower() else '-'
        ax.plot(ang, y, label=lbl, color=color, linestyle=linestyle, linewidth=2, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel('Angle')
    # Common tick setup for [0, 2π]
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Save if requested
    if getattr(config, "log_dir", None) is not None:
        safe_title = _sanitize_filename(title)
        fname = f"curvplot_{safe_title}.png"
        path = os.path.join(config.log_dir, fname)
        fig.savefig(path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_curvatures_2d(angles, curvature_norms, config, title):
    """
    Plots 2-dimensional curvature norms against specified angles.
    It performs cubic interpolation to render smooth plots and enables dataset-specific
    grid resolutions for optimal visualization.

    Args:
        angles (array-like): An array of shape (N, 2) containing angle coordinates in radians.
        curvature_norms (dict, list, tuple, or array-like): Curvature norms associated with the
            angles. Can be of multiple types: a dictionary mapping labels to values, a list/tuple
            of curvature values, or a single array-like curvature series.
        config (object): A configuration object containing dataset name and optional log directory.
        title (str): The title for the overall plot, used as a prefix for individual surface plots.

    Raises:
        ValueError: If `angles` shape is not (N, 2).
        ValueError: If `curvature_norms` is None.
        ValueError: Multiple reasons like mismatch in the number of angles and curvature norms,
            or no valid curvature series after validation.
    """
    # Validate angles
    A = np.asarray(angles)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError(f"angles must be (N, 2); got shape {A.shape}")

    grid_res = 100
    if config.dataset_name in {"s2_high", "s2_low", "sphere_high_dim", "nested_spheres", "nested_spheres_high_dim"}:
        grid_x, grid_y = np.meshgrid(np.linspace(0, np.pi, int(grid_res // 2)), np.linspace(0, 2 * np.pi, grid_res))
    else:
        grid_x, grid_y = np.meshgrid(np.linspace(0, 2 * np.pi, grid_res), np.linspace(0, 2 * np.pi, grid_res))

    def interpolate(lbls, values):
        return griddata(lbls, values, (grid_x, grid_y), method="cubic")

    def pi_formatter(x, pos):
        if np.isclose(x, 0):
            return "0"
        elif np.isclose(x, np.pi / 2):
            return r"$\frac{\pi}{2}$"
        elif np.isclose(x, np.pi):
            return r"$\pi$"
        elif np.isclose(x, 2 * np.pi):
            return r"$2\pi$"
        else:
            return ""

    # Normalize series
    series = []
    if curvature_norms is None:
        raise ValueError("curvature_norms cannot be None")

    if isinstance(curvature_norms, dict):
        for k, v in curvature_norms.items():
            if v is None:
                continue
            y = np.asarray(v).reshape(-1)
            if y.shape[0] != A.shape[0]:
                print(f"plot_curvatures_2d: Skipping '{k}' due to length mismatch "
                      f"(angles={A.shape[0]}, series={y.shape[0]})")
                continue
            series.append((str(k), y))
    elif isinstance(curvature_norms, (list, tuple)):
        for i, item in enumerate(curvature_norms):
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], (list, tuple, np.ndarray)):
                lbl, arr = item
                y = np.asarray(arr).reshape(-1)
                label = str(lbl)
            else:
                y = np.asarray(item).reshape(-1)
                label = f"Series {i+1}"
            if y.shape[0] != A.shape[0]:
                print(f"plot_curvatures_2d: Skipping '{label}' due to length mismatch "
                      f"(angles={A.shape[0]}, series={y.shape[0]})")
                continue
            series.append((label, y))
    else:
        y = np.asarray(curvature_norms).reshape(-1)
        if y.shape[0] != A.shape[0]:
            raise ValueError(f"Length mismatch: angles={A.shape[0]}, series={y.shape[0]}")
        series.append((str(title), y))

    if not series:
        raise ValueError("No valid curvature series to plot after validation.")

    # Render one surface per series
    for lbl, y in series:
        surface = interpolate(A, y)
        if surface is None or not np.any(np.isfinite(surface)):
            print(f"plot_curvatures_2d: Skipping '{lbl}' due to failed interpolation or all-NaN surface.")
            continue

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x, grid_y, surface, cmap='viridis')
        ax.set_title(f"{title} - {lbl}", fontsize=14)
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_zlabel('Curvature')
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

        if getattr(config, "log_dir", None) is not None:
            safe = _sanitize_filename(f"{title}_{lbl}")
            fname = f"curvplot_{safe}.png"
            fig.savefig(os.path.join(config.log_dir, fname), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def plot_all_curvatures(config, model, data_loader):
    """
    Generates and visualizes curvature-related metrics for data, latents, and reconstructions in a model.

    This function computes and plots various curvatures and related metrics derived from inputs, latent spaces, and
    reconstructed data generated by a given model. The visualization approach is based on
    the intrinsic dimension (1d or 2d) and the model type (euclidean or non-euclidean latent space).

    Args:
        config: Configuration object containing settings for curvature computation, dataset properties, model type,
            and plotting options.
        model: Model object whose behavior, including curvature properties, is being evaluated and visualized.
        data_loader: DataLoader object providing input data for the model and relevant labels for subsequent
            computations and plots.
    """
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_points_emp_curv)
    results_dict = compute_all_curvatures(config, model, recons, latents, inputs, labels)

    points = (inputs, latents, recons)
    metrics = results_dict.get("metrics", {})

    def _compose_small_text():
        lines = []
        d = metrics.get("quadric_inputs_vs_latents")
        if isinstance(d, dict):
            try:
                mse = d.get("mse", float("nan"))
                smape = d.get("smape", float("nan"))
                lines.append(f"data/latents \nMSE={mse:.3g}, SMAPE={smape:.3g}%")
            except Exception:
                pass
        d2 = metrics.get("true_vs_learned_rotated_sub")
        if isinstance(d2, dict):
            try:
                mse = d2.get("mse", float("nan"))
                smape = d2.get("smape", float("nan"))
                lines.append(f"true/pullback \nMSE={mse:.3g}, SMAPE={smape:.3g}%")
            except Exception:
                pass
        return "\n".join(lines) if lines else None

    # Extract empirical quadric curvatures (full)
    curv_in = results_dict["curvatures"]["inputs"]
    curv_lat = results_dict["curvatures"]["latents"]
    curv_rec = results_dict["curvatures"]["recons"]

    # Pullback / true / rotated from results_dict
    curv_true = results_dict["curvatures"]["true_sub"]
    curv_learned = results_dict["curvatures"]["learned_sub"]
    curv_learned_rotated = results_dict["curvatures"]["learned_rotated_sub"]
    z_grid = results_dict["curvatures"]["z_grid"]

    is_spherical = is_spherical_model(config.model_type)
    is_euclidean = is_euclidean_model(config.model_type)
    has_true = bool(getattr(config, "compute_true_curv", False) and (curv_true is not None) and (z_grid is not None))
    learned_source = curv_learned_rotated if is_spherical else curv_learned
    has_learned = bool(getattr(config, "compute_learned_curv", False) and (learned_source is not None) and (z_grid is not None))
    dataset_category = get_dataset_category(config.dataset_name)

    if dataset_category == "1d":
        if is_spherical:
            if has_true:
                plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None,
                                     profile_type="true", title="Data")
            if has_learned:
                plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned, config=config, norm_val=None,
                                     profile_type="learned", title="Latents", small_text=_compose_small_text())
                plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned_rotated, config=config, norm_val=None,
                                     profile_type="learned", title="Latents Transformed", small_text=_compose_small_text())
        else:
            # Plot Inputs vs Latents over labels
            if (curv_in is not None) and (curv_lat is not None):
                plot_curvatures_1d(
                    angles=labels,
                    curvature_norms={
                        'Data Curvature Quadric Estimate': curv_in,
                        'Latent Curvature Quadric Estimate': curv_lat,
                    },
                    config=config,
                    title='Inputs vs Latents'
                )
            # Plot Inputs vs Reconstructions over labels (if both available)
            if (curv_in is not None) and (curv_rec is not None):
                plot_curvatures_1d(
                    angles=labels,
                    curvature_norms={
                        'Data Curvature Quadric Estimate': curv_in,
                        'Reconstruction Curvature Quadric Estimate': curv_rec,
                    },
                    config=config,
                    title='Inputs vs Reconstructions'
                )
            # Plot True vs Learned over z_grid
            if has_true and (curv_learned is not None):
                plot_curvatures_1d(
                    angles=z_grid,
                    curvature_norms={
                        'True Curvature': curv_true,
                        'Decoder Curvature Pullback Estimate': curv_learned,
                    },
                    config=config,
                    title='True vs Learned'
                )
            # Individual heatmaps
            if curv_in is not None:
                scatter_curvature_heatmaps(config, points[0], curv_in, title="Inputs Heatmap")
            if curv_lat is not None:
                scatter_curvature_heatmaps(config, points[1], curv_lat, title="Latents Heatmap", small_text=_compose_small_text())
            if curv_rec is not None:
                scatter_curvature_heatmaps(config, points[2], curv_rec, title="Recons Heatmap")

    elif dataset_category == "2d":
        if not is_euclidean:
            tl_series = {}
            if has_true and (curv_true is not None):
                tl_series["True Curvature"] = curv_true
            if has_learned and (curv_learned is not None):
                tl_series["Learned Curvature"] = curv_learned
            if tl_series:
                plot_curvatures_2d(angles=z_grid, curvature_norms=tl_series, config=config, title="Estimated Curvature (Pullback)")

        # Spherical variants (rotated learned)
        if is_spherical and (curv_learned_rotated is not None):
            plot_curvatures_2d(
                angles=z_grid,
                curvature_norms={"Learned Curvature (Rotated)": curv_learned_rotated},
                config=config,
                title="Surfaces: Learned Rotated"
            )

        # Keep auxiliary visualizations
        if is_euclidean:
            # Plot surfaces over data labels for available empirical estimates
            label_series = {}
            if (curv_in is not None) and (
                    getattr(config, "compute_curv_inputs", False) or getattr(config, "compute_quadric_curv_inputs",
                                                                             False)):
                label_series["Inputs"] = curv_in
            if (curv_lat is not None) and (
                    getattr(config, "compute_curv_latents", False) or getattr(config, "compute_quadric_curv_latents",
                                                                              False)):
                label_series["Latents"] = curv_lat
            if (curv_rec is not None) and (
                    getattr(config, "compute_curv_recons", False) or getattr(config, "compute_rec_curv",
                                                                             False) or getattr(config,
                                                                                               "compute_quadric_curv_recons",
                                                                                               False)):
                label_series["Recons"] = curv_rec
            if label_series:
                plot_curvatures_2d(angles=labels, curvature_norms=label_series, config=config,
                                   title="Estimated Curvature (Quadric)")
            if curv_in is not None:
                scatter_curvature_heatmaps(config, points[0], curv_in, title="Inputs Heatmap")
            if curv_lat is not None:
                scatter_curvature_heatmaps(config, points[1], curv_lat, title="Latents Heatmap", small_text=_compose_small_text())
            if curv_rec is not None:
                scatter_curvature_heatmaps(config, points[2], curv_rec, title="Recons Heatmap")
        elif not is_euclidean and not is_spherical:
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None, profile_type="true", title="Data")
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned, config=config, norm_val=None, profile_type="learned", title="Latents", small_text=_compose_small_text())
        elif is_spherical:
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None, profile_type="true", title="Data")
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned_rotated, config=config, norm_val=None, profile_type="learned", title="Latents Transformed", small_text=_compose_small_text())
        else:
            raise NotImplementedError

    elif dataset_category == "multi_entity":
        entity_indices = labels[:, 0]
        entity_indices_sub = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        unique_entities = unique_entities[unique_entities != 100]
        for multi_entity in unique_entities:
            mask = (entity_indices == multi_entity)
            mask_sub = (entity_indices_sub == multi_entity)
            angels = labels[mask][:, 1:]

            inputs_entity = inputs[mask]
            latents_entity = latents[mask]
            recons_entity = recons[mask]
            points_entity = (inputs_entity, latents_entity, recons_entity)

            curv_true_entity = curv_true[mask_sub]
            curv_in_entity = curv_in[mask]
            curv_rec_entity = curv_rec[mask]
            curv_lat_entity = curv_lat[mask]
            curv_learned_entity = curv_learned[mask_sub]

            series_entity = {}
            if curv_true_entity is not None:
                series_entity["True Curvature"] = curv_true_entity
            if curv_in_entity is not None:
                series_entity["Inputs"] = curv_in_entity
            if curv_rec_entity is not None:
                series_entity["Recons"] = curv_rec_entity
            if curv_lat_entity is not None:
                series_entity["Latents"] = curv_lat_entity
            if curv_learned_entity is not None:
                series_entity["Learned Curvature"] = curv_learned_entity

            if series_entity:
                plot_curvatures_2d(angles=angels, curvature_norms=series_entity, config=config, title=f"Entity {int(multi_entity)} Surfaces")

            # Individual heatmaps per entity
            if curv_in_entity is not None:
                scatter_curvature_heatmaps(config, inputs_entity, curv_in_entity, title=f"Entity {int(multi_entity)} Inputs Heatmap")
            if curv_lat_entity is not None:
                scatter_curvature_heatmaps(config, latents_entity, curv_lat_entity, title=f"Entity {int(multi_entity)} Latents Heatmap", small_text=_compose_small_text())
            if curv_rec_entity is not None:
                scatter_curvature_heatmaps(config, recons_entity, curv_rec_entity, title=f"Entity {int(multi_entity)} Recons Heatmap")
    else:
        raise NotImplementedError("Label dimension not supported for curvature plotting.")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import json
from scipy.interpolate import griddata

from ..curvature.curvature_pipeline import compute_all_curvatures
from .utils import scatter_datapoints
from ..utils.vectors import get_vectors
from ..curvature.curvature_metrics import compute_curvature_error_smape, compute_curvature_error_mse

def plot_curvature_norms(angles, curvature_norms, config, norm_val, profile_type):
    fig = plt.figure(figsize=(12, 6))  # increased size for clarity
    colormap = plt.get_cmap("hsv")

    # Handle curvature normalization safely
    if norm_val is not None and norm_val > 0:
        color_norm = mpl.colors.Normalize(0.0, norm_val)
    else:
        eps = 1e-3
        color_norm = mpl.colors.Normalize(0.0, max(curvature_norms) + eps)

    if config.dataset_name in {"s1_low", "s1_high"}:
        ax1 = fig.add_subplot(121)
        ax1.plot(angles, curvature_norms, linewidth=2)
        # ax1.set_xlabel("angle", fontsize=12)
        # ax1.set_ylabel("mean curvature norm", fontsize=12)
        # ax1.set_title(f"{profile_type} profile", fontsize=14, pad=20)

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
        # ax2.set_title(f"{profile_type} profile", fontsize=14, pad=20)

    elif config.dataset_name in {"s2_low", "t2_low", "s2_high", "t2_high"}:
        if config.dataset_name in {"s2_low", "s2_high"}:
            x = config.radius * np.sin(angles[:, 0]) * np.cos(angles[:, 1])
            y = config.radius * np.sin(angles[:, 0]) * np.sin(angles[:, 1])
            z = config.radius * np.cos(angles[:, 0])
        else:  # t2_low
            theta = angles[:, 0]
            phi = angles[:, 1]
            x = (config.major_radius - config.minor_radius * np.cos(theta)) * np.cos(phi)
            y = (config.major_radius - config.minor_radius * np.cos(theta)) * np.sin(phi)
            z = config.minor_radius * np.sin(theta)
        points = np.stack([x, y, z], axis=1)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        sc = scatter_datapoints(ax=ax, data=points, colors=curvature_norms, dot_size=20,
                                 cmap="rainbow", apply_pca=False, pca_dim=3, title=None)
        fig.colorbar(sc, ax=ax, shrink=0.7)
        plt.tight_layout()

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

    else:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    plt.tight_layout()

    if config.log_dir is not None:
        fname = f"heatmaps_template_manifold_{profile_type}_{config.model_type}.png"
        path = os.path.join(config.log_dir, fname)
        plt.savefig(path)
        print(f"Saved curvature norm plot to: {path}")

    plt.show()
    plt.close(fig)


def scatter_curvature_heatmaps(config, points, points_sub, z_grid, curv_true, curv_in, curv_rec, curv_lat,
                               curv_learned, entity=None):
    inputs, latents, recons = points
    inputs_sub, latents_sub = points_sub
    color_map = 'rainbow'

    def plot_heatmap_group(heatmaps, suptitle, tag):
        num_heatmaps = len(heatmaps)
        num_cols = 3
        num_rows = (num_heatmaps + num_cols - 1) // num_cols

        if config.dataset_name == "s1_low":
            pca_dim = 2
        else:
            pca_dim = 3

        # Main figure with all subplots
        fig = plt.figure(figsize=(18, 6 * num_rows))
        for i, (title, (curv, pts)) in enumerate(heatmaps.items(), 1):
            ax = fig.add_subplot(num_rows, num_cols, i,
                                 projection='3d' if pts.shape[1] >= 3 and pca_dim == 3 else None)
            sc = scatter_datapoints(ax=ax, data=pts, title=title, colors=curv, cmap=color_map, pca_dim=pca_dim)
            fig.colorbar(sc, ax=ax, shrink=0.7)

            # Individual figure for each heatmap
            if getattr(config, "log_dir", None) is not None:
                fig_indiv = plt.figure(figsize=(6, 6))
                ax_indiv = fig_indiv.add_subplot(111,
                                                 projection='3d' if pts.shape[1] >= 3 and pca_dim == 3 else None)
                sc_indiv = scatter_datapoints(ax=ax_indiv, data=pts, title=None, colors=curv, cmap=color_map,
                                               pca_dim=pca_dim)
                fig_indiv.colorbar(sc_indiv, ax=ax_indiv, shrink=0.7)
                plt.tight_layout()
                safe_title = title.replace(" ", "_").lower()
                fname_indiv = f"heatmap_{safe_title}.png"
                fig_indiv.savefig(os.path.join(config.log_dir, fname_indiv))
                plt.close(fig_indiv)

        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if getattr(config, "log_dir", None) is not None:
            fname = f"heatmap_{tag}.png"
            fig.savefig(os.path.join(config.log_dir, fname))
        plt.show()

    heatmaps = {
        "Inputs": (curv_in, inputs),
        "Latents": (curv_lat, latents),
    }

    if config.compute_rec_curv:
        heatmaps["Recons"] = (curv_rec, recons)

    if entity is not None:
        plot_heatmap_group(
            heatmaps,
            f'Curvature Heatmap Comparison - Connected Components {int(entity)}',
            'curvature_heatmaps'
        )
    else:
        plot_heatmap_group(
            heatmaps,
            'Curvature Heatmap Comparison',
            'curvature_heatmaps'
        )


def plot_curvatures_1d(labels, curv_true, curv_in, curv_rec,
                       curv_lat, curv_lat_norm, curv_learned, z_grid, config):
    curve_groups = []
    if config.compute_true_curv:
        curve_groups.append([
            ('True Curvature', curv_true, z_grid, 'tab:green'),
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Reconstructed Curvature', curv_rec, labels, 'tab:red')
        ])
    else:
        curve_groups.append([
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Reconstructed Curvature', curv_rec, labels, 'tab:red')
        ])

    if config.compute_learned_curv and config.compute_true_curv and config.compute_emp_curv:
        curve_groups.append([
            ('True Curvature', curv_true, z_grid, 'tab:green'),
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Learned Curvature', curv_learned, z_grid, 'tab:pink'),
            ('Latent Curvature', curv_lat, labels, 'tab:orange')
        ])
    elif config.compute_learned_curv and config.compute_true_curv:
        curve_groups.append([
            ('True Curvature', curv_true, z_grid, 'tab:green'),
            ('Learned Curvature', curv_learned, labels, 'tab:pink'),
        ])
    elif config.compute_true_curv:
        curve_groups.append([
            ('True Curvature', curv_true, z_grid, 'tab:green'),
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Latent Curvature', curv_lat, labels, 'tab:orange')
        ])
    else:
        curve_groups.append([
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Latent Curvature', curv_lat, labels, 'tab:orange')
        ])

    if config.compute_true_curv:
        curve_groups.append([
            ('True Curvature', curv_true, z_grid, 'tab:green'),
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Normalized Latent Curvature', curv_lat_norm, labels, 'tab:orange')
        ])
    else:
        curve_groups.append([
            ('Input Curvature', curv_in, labels, 'tab:blue'),
            ('Normalized Latent Curvature', curv_lat_norm, labels, 'tab:orange')
        ])

    titles = ['Sanity Check', 'Compare to Inputs I', 'Compare to Inputs II']

    if len(curve_groups) != len(titles):
        raise ValueError("curve_groups and titles must have the same length")

    fig, axes = plt.subplots(1, len(curve_groups), figsize=(6 * len(curve_groups), 5), sharex=True)
    if len(curve_groups) == 1:
        axes = [axes]

    for idx, (ax, curves, title) in enumerate(zip(axes, curve_groups, titles)):
        for label, curve_values, x_points, color in curves:
            linestyle = '--' if label == 'True Curvature' else '-'
            ax.plot(x_points, curve_values, label=label,
                    color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Angle')
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=8)

        # Save individual subplot if log_dir is specified
        if getattr(config, "log_dir", None) is not None:
            indiv_path = os.path.join(config.log_dir, f"curvplot_{title}.png")
            fig_indiv, ax_indiv = plt.subplots(figsize=(6, 5))
            for label, curve_values, x_points, color in curves:
                linestyle = '--' if label == 'True Curvature' else '-'
                ax_indiv.plot(x_points, curve_values, label=label,
                              color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
            ax_indiv.set_xticks([0, np.pi, 2 * np.pi])
            ax_indiv.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
            ax_indiv.grid(False)
            ax_indiv.legend(loc='upper right', fontsize=16)
            plt.tight_layout()
            fig_indiv.savefig(indiv_path)
            plt.close(fig_indiv)

    axes[0].set_ylabel('Curvature')
    plt.tight_layout()
    if getattr(config, "log_dir", None) is not None:
        fig.savefig(os.path.join(config.log_dir, "curvature_grid_plot.png"))
    plt.show()


def plot_curvatures_2d(labels, labels_sub, curv_true, curv_in, curv_rec, curv_lat, curv_learned, z_grid, config,
                       entity=None):
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

    def plot_surface_group(surfaces, suptitle, tag):
        num_surfaces = len(surfaces)
        num_cols = 3
        num_rows = (num_surfaces + num_cols - 1) // num_cols
        fig = plt.figure(figsize=(6 * num_cols, 5 * num_rows))

        for i, (title, surface) in enumerate(surfaces.items(), 1):
            ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')
            ax.plot_surface(grid_x, grid_y, surface, cmap='viridis')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            ax.set_zlabel('Curvature')
            ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
            ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

        fig.suptitle(suptitle, fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if config.log_dir is not None:
            fname = f"surface_group_{tag}.png"
            fig.savefig(os.path.join(config.log_dir, fname), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_individual_surface(title, surface, tag):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x, grid_y, surface, cmap='viridis')

        # Remove title and axis labels
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # Retain tick formatters for clarity
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

        if config.log_dir is not None:
            fname = f"curvplot_{tag}.png"
            fig.savefig(os.path.join(config.log_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Define surface groups similar to curve_groups
    surface_groups = []

    # Group 1: Sanity Check / Basic
    basic_group = {
        "Inputs": interpolate(labels, curv_in),
        "Latents": interpolate(labels, curv_lat)
    }
    if config.compute_true_curv:
        basic_group["True"] = interpolate(z_grid, curv_true)
    surface_groups.append(("Sanity Check", "emp", basic_group))

    # Group 2: Learned Curvatures (if available)
    if config.compute_learned_curv:
        learned_group = {
            "Pullback": interpolate(z_grid, curv_learned),
        }
        if config.compute_true_curv:
            learned_group["True Curvature on Input Data"] = interpolate(z_grid, curv_true)
        surface_groups.append(("Learned vs True", "learned", learned_group))

    # Group 3: Reconstructed (if available)
    if config.compute_rec_curv:
        rec_group = {
            "Inputs": interpolate(labels, curv_in),
            "Recons": interpolate(labels, curv_rec)
        }
        if config.compute_true_curv:
            rec_group["True"] = interpolate(z_grid, curv_true)
        surface_groups.append(("Reconstructed vs Input", "reconstructed", rec_group))

    # Plot and optionally save grouped and individual surfaces
    for suptitle, tag, group in surface_groups:
        plot_surface_group(group, f"{suptitle} - {entity if entity is not None else 'All'}", tag)
        if config.log_dir is not None:
            for surface_title, surface in group.items():
                safe_title = surface_title.lower().replace(' ', '_').replace('(', '').replace(')', '')
                plot_individual_surface(surface_title, surface, f"{tag}_{safe_title}")


def compute_all_error_metrics(pairs):
    names = [name for name, _, _ in pairs]
    smape = [compute_curvature_error_smape(a, b) for _, a, b in pairs]
    mse = [compute_curvature_error_mse(a, b) for _, a, b in pairs]
    linf = [compute_curvature_error_linf(a, b) for _, a, b in pairs]
    return names, mse, smape, linf


def compute_stds(*arrays):
    stds = [np.std(np.asarray(arr)) for arr in arrays]
    labels = [name for name in ['True', 'Input', 'Latent', 'Reconstructed', 'Learned'][:len(stds)]]
    return stds, labels


def plot_error_bars(ax, names, mse, linf, bar_width=0.2):
    x = np.arange(len(['MSE', '$L^\infty$']))
    for i, name in enumerate(names):
        offsets = x + (i - len(names) / 2) * bar_width
        heights = [mse[i], linf[i]]
        ax.bar(offsets, heights, bar_width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(['MSE', '$L^\infty$'], rotation=15)
    ax.set_title('Error Metrics')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_smape(ax, names, smape):
    ax.bar(np.arange(len(smape)), smape, color='tab:cyan', width=0.4)
    ax.set_xticks(np.arange(len(smape)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_title('SMAPE')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_std(ax, labels, stds):
    ax.bar(np.arange(len(stds)), stds, color='tab:gray', width=0.4)
    ax.set_xticks(np.arange(len(stds)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title('Curvature STD')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_curvature_errors_and_stats(curv_true, curv_in, curv_rec,
                                    curv_lat, curv_lat_norm, curv_learned, config):
    pairs = [
        ("Empirical Inputs vs Latents", curv_in, curv_lat),
        ("Empirical Inputs vs Latents Normalized", curv_in, curv_lat_norm),
    ]
    std_arrays = [curv_in, curv_lat, curv_rec]
    std_labels = ['Input', 'Latent', 'Reconstructed']

    if config.compute_true_curv:
        pairs += [
            ("True vs Empirical on Inputs", curv_true, curv_in),
            ("True vs Empirical on Latent", curv_true, curv_lat),
            ("True vs Normalized Empirical on Latent", curv_true, curv_lat_norm),
        ]
        std_arrays.append(curv_true)
        std_labels.append('True')

    if config.compute_learned_curv:
        pairs += [
            ("Learned vs Empirical on Inputs", curv_learned, curv_in),
        ]
        std_arrays.append(curv_learned)
        std_labels.append('Learned')

    if config.compute_true_curv and config.compute_learned_curv:
        pairs += [
            ("True vs Learned", curv_true, curv_learned),
        ]

    names, mse, smape, linf = compute_all_error_metrics(pairs)
    stds, _ = compute_stds(*std_arrays)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_error_bars(axes[0], names, mse, linf, bar_width=0.1)
    plot_smape(axes[1], names, smape)
    plot_std(axes[2], std_labels, stds)

    if config.log_dir is not None:
        plt.savefig(os.path.join(config.log_dir, "curvature_errors_combined.png"))
    plt.tight_layout()
    plt.show()

    results = {
        "error_comparisons": names,
        "errors": {"MSE": mse, "SMAPE_percent": smape, "L_inf": linf},
        "curvature_std": {"labels": std_labels, "values": stds}
    }

    if config.log_dir is not None:
        with open(os.path.join(config.log_dir, "curvature_errors_stats.json"), "w") as f:
            json.dump(results, f, indent=4)


def _plot_all_curvatures_from_vectors(config, model, recons, latents, inputs, labels):
    points_sub, curvatures_sub, curvatures_emp_full, points = compute_all_curvatures(config, model, recons, latents,
                                                                                     inputs,
                                                                                     labels)
    points = (inputs, latents, recons)
    inputs_sub, latents_sub, recons_sub = points_sub
    labels_sub, curv_in_sub, curv_rec_sub, curv_lat_sub, curv_lat_norm_sub, curv_true, curv_learned, curv_learned_rotated, z_grid = curvatures_sub
    labels, curv_in, curv_lat, curv_lat_norm, curv_rec = curvatures_emp_full

    # Plot curvatures over angles
    if labels.ndim == 1:
        points_sub = (inputs_sub, latents_sub)
        if config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
            plot_curvatures_1d(labels=labels_sub, curv_true=curv_true, curv_in=curv_in_sub, curv_rec=curv_rec_sub,
                               curv_lat=curv_lat_sub,
                               curv_lat_norm=curv_lat_norm_sub,
                               curv_learned=curv_learned_rotated, z_grid=z_grid, config=config)
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None,
                                 profile_type="true")
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned_rotated, config=config, norm_val=None,
                                 profile_type="learned")
        else:
            plot_curvatures_1d(labels=labels_sub, curv_true=curv_true, curv_in=curv_in_sub, curv_rec=curv_rec_sub,
                               curv_lat=curv_lat_sub,
                               curv_lat_norm=curv_lat_norm_sub,
                               curv_learned=curv_learned, config=config, z_grid=z_grid)
            scatter_curvature_heatmaps(config, points=points, points_sub=points_sub, z_grid=z_grid, curv_true=curv_true,
                                       curv_in=curv_in, curv_rec=curv_rec, curv_learned=curv_learned, curv_lat=curv_lat)
    elif labels.ndim == 2 and labels.shape[1] == 2:
        points_sub = (inputs_sub, latents_sub)
        if config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
            plot_curvatures_2d(labels=labels, labels_sub=labels_sub, curv_true=curv_true, curv_in=curv_in,
                               curv_rec=curv_rec, curv_lat=curv_lat, curv_learned=curv_learned_rotated, z_grid=z_grid,
                               config=config)
        else:
            plot_curvatures_2d(labels=labels, labels_sub=labels_sub, curv_true=curv_true, curv_in=curv_in,
                               curv_rec=curv_rec, curv_lat=curv_lat, curv_learned=curv_learned, z_grid=z_grid,
                               config=config)
        if config.model_type in {"EuclideanVAE", "EuclideanAE"}:
            scatter_curvature_heatmaps(config, points=points, points_sub=points_sub, z_grid=z_grid, curv_true=curv_true,
                                       curv_in=curv_in, curv_rec=curv_rec, curv_learned=curv_learned, curv_lat=curv_lat,
                                       )
        elif config.model_type in {"VMFToroidalVAE", "VMToroidalVAE", "ToroidalAE"}:
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None,
                                 profile_type="true")
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned, config=config, norm_val=None,
                                 profile_type="learned")
        elif config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_true, config=config, norm_val=None,
                                 profile_type="true")
            plot_curvature_norms(angles=z_grid, curvature_norms=curv_learned_rotated, config=config, norm_val=None,
                                 profile_type="learned")
        else:
            raise NotImplementedError

    elif labels.ndim == 2 and labels.shape[1] == 3:
        entity_indices = labels[:, 0]
        entity_indices_sub = labels_sub[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        unique_entities = unique_entities[unique_entities != 100]
        for multi_entity in unique_entities:
            mask = (entity_indices == multi_entity)
            mask_sub = (entity_indices_sub == multi_entity)
            angels = labels[mask][:, 1:]
            angels_sub = labels_sub[mask_sub][:, 1:]

            inputs_entity = inputs[mask]
            latents_entity = latents[mask]
            recons_entity = recons[mask]
            points_entity = (inputs_entity, latents_entity, recons_entity)

            inputs_sub_entity = inputs_sub[mask_sub]
            latents_sub_entity = latents_sub[mask_sub]
            points_sub_entity = (inputs_sub_entity, latents_sub_entity)

            curv_true_entity = curv_true[mask_sub]
            curv_in_entity = curv_in[mask]
            curv_rec_entity = curv_rec[mask]
            curv_lat_entity = curv_lat[mask]
            curv_learned_entity = curv_learned[mask_sub]
            plot_curvatures_2d(labels=angels, labels_sub=angels_sub, curv_true=curv_true_entity,
                               curv_in=curv_in_entity, curv_rec=curv_rec_entity, curv_lat=curv_lat_entity,
                               curv_learned=curv_learned_entity, z_grid=z_grid, config=config, entity=multi_entity)
            scatter_curvature_heatmaps(config, points=points_entity, points_sub=points_sub_entity, z_grid=z_grid,
                                       curv_true=curv_true_entity,
                                       curv_in=curv_in_entity, curv_rec=curv_rec_entity,
                                       curv_learned=curv_learned_entity, curv_lat=curv_lat_entity,
                                       entity=multi_entity)
    else:
        raise NotImplementedError("Label dimension not supported for curvature plotting.")

    if config.model_type in {"VMFSphericalVAE", "SphericalAE"}:
        plot_curvature_errors_and_stats(curv_true=curv_true, curv_in=curv_in_sub, curv_rec=curv_rec_sub,
                                        curv_lat=curv_lat_sub, curv_lat_norm=curv_lat_norm_sub,
                                        curv_learned=curv_learned_rotated, config=config)
    else:
        plot_curvature_errors_and_stats(curv_true=curv_true, curv_in=curv_in_sub, curv_rec=curv_rec_sub,
                                        curv_lat=curv_lat_sub, curv_lat_norm=curv_lat_norm_sub,
                                        curv_learned=curv_learned, config=config)


def plot_all_curvatures(config, model, data_loader):
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_points_emp_curv)
    _plot_all_curvatures_from_vectors(config=config, model=model, recons=recons, latents=latents, inputs=inputs,
                                      labels=labels)

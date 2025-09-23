import os
import numpy as np
import matplotlib as plt
import time
import json

from ..topology.persistence import compare_persistent_homology
from ..utils.vectors import get_vectors

def plot_persistence_diagrams(config, suptitle, diagrams, homology_dimensions):
    """Plot two persistence diagrams side-by-side and print bottleneck distance matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), )
    fig.suptitle(suptitle, fontsize=16)
    titles = ["Persistence Diagram for Input Data", "Persistence Diagram for Latent Representation"]
    colors = {0: 'red', 1: 'blue', 2: 'green'}

    for i, ax in enumerate(axes):
        diagram = diagrams[i]
        diagram = diagram[diagram[:, 0] != diagram[:, 1]]
        bd = diagram[:, :2]

        posinf_mask = np.isposinf(bd)
        neginf_mask = np.isneginf(bd)

        if bd.size:
            max_val = np.max(np.where(posinf_mask, -np.inf, bd))
            min_val = np.min(np.where(neginf_mask, np.inf, bd))
        else:
            max_val, min_val = 1.0, 0.0

        value_range = max_val - min_val
        buffer = 0.05 * value_range
        min_val_display = min_val - buffer
        max_val_display = max_val + buffer

        ax.plot([min_val_display, max_val_display], [min_val_display, max_val_display],
                linestyle='--', color='black', linewidth=1)

        for dim in homology_dimensions:
            subdiagram = diagram[diagram[:, 2] == dim]
            births = subdiagram[:, 0]
            deaths = subdiagram[:, 1]
            deaths = np.where(np.isposinf(deaths), max_val_display + 0.05 * value_range, deaths)

            label = f"$H_{dim}$"
            ax.scatter(births, deaths, label=label, color=colors.get(dim, 'gray'), s=20)

        ax.set_title(titles[i])
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_xlim([min_val_display, max_val_display])
        ax.set_ylim([min_val_display, max_val_display])
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    if config.log_dir is not None:
        fname = "persistence_diagrams.png"
        plt.savefig(os.path.join(config.log_dir, fname))
    plt.show()


def plot_betti_curves(config, suptitle, betti_curves, homology_dimensions=None):
    betti_numbers, samplings = betti_curves
    n_plots = len(betti_numbers)
    titles = ["Normalized Betti Curves Input Data", "Normalized Betti Curves Latent Space"]
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    for i, betti_numbers in enumerate(betti_numbers):
        if homology_dimensions is None:
            dims = list(range(betti_numbers.shape[0]))
        else:
            dims = homology_dimensions

        ax = axes[0, i]
        for dim in dims:
            # Normalize each Betti number curve
            normalized_betti = betti_numbers[dim] / np.max(betti_numbers[dim])  # Normalize to [0, 1]
            ax.plot(samplings[dim], normalized_betti, label=f"H{dim}", linewidth=1.5)

        ax.set_title(titles[i])
        ax.set_xlabel("Filtration parameter")
        ax.set_ylabel("Normalized Betti number")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    if config.log_dir is not None:
        fname = "betti_curves.png"
        plt.savefig(os.path.join(config.log_dir, fname))
    plt.show()


def plot_persistence(config, model, data_loader):
    recons, latents, inputs, labels = get_vectors(config, model, data_loader, config.n_points_emp_curv)

    start_time = time.time()

    if config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori"}:
        n = config.n_points_pers_hom
        idx = np.random.choice(inputs.shape[0], n, replace=False)
        inputs_subsampled = inputs[idx]
        latents_subsampled = latents[idx]
        diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                        config.homology_dimensions, scale=config.scale)
        title_diagram = f"Persistence Diagram Comparison - Full Dataset"
        title_betti_curve = f"Betti Curve Comparison - Full Dataset"
        plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
        plot_betti_curves(config, title_betti_curve, betti_curves)

        entity_indices = labels[:, 0]
        unique_entities = entity_indices.unique(sorted=True)
        for entity in unique_entities:
            mask = (entity_indices == entity)
            inputs_for_entity = inputs[mask]
            latents_for_entity = latents[mask]
            n = config.n_points_pers_hom
            idx = np.random.choice(inputs_for_entity.shape[0], n, replace=False)
            inputs_subsampled = inputs_for_entity[idx]
            latents_subsampled = latents_for_entity[idx]
            diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                            config.homology_dimensions,
                                                                            scale=config.scale)
            title_diagram = f"Persistence Diagram Comparison - Connected Component {entity}"
            title_betti_curve = f"Betti Curve Comparison - Connected Component {entity}"
            plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
            plot_betti_curves(config, title_betti_curve, betti_curves)
        print("Bottleneck Distances", distances)
    else:
        n = config.n_points_pers_hom
        idx = np.random.choice(inputs.shape[0], n, replace=False)
        inputs_subsampled = inputs[idx]
        latents_subsampled = latents[idx]
        diagrams, betti_curves, distances = compare_persistent_homology((inputs_subsampled, latents_subsampled),
                                                                        config.homology_dimensions, scale=config.scale)
        title_diagram = f"Persistence Diagram Comparison"
        title_betti_curve = f"Betti Curve Comparison"
        plot_persistence_diagrams(config, title_diagram, diagrams, config.homology_dimensions)
        plot_betti_curves(config, title_betti_curve, betti_curves)
        print("Bottleneck Distances", distances)
    end_time = time.time()
    print(f"Execution time persistent homology: {end_time - start_time:.4f} seconds")

    results = {
        "bottleneck distances": distances.tolist()
    }

    if config.log_dir is not None:
        with open(os.path.join(config.log_dir, "distances.json"), "w") as f:
            json.dump(results, f, indent=4)



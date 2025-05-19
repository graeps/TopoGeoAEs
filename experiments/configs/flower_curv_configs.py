import os
from types import SimpleNamespace

base_config = {
    # Experiment
    "experiment": "flower_curve",
    "random_seed": 42,

    # Dataset
    "dataset_name": "s1_synthetic",
    "batch_size": 64,
    "rotation": "random",
    "translation": "random",
    "n_times": 10000,
    "embedding_dim": 10,
    "radius": 2,
    "noise_var": 0.001,
    "n_wiggles": 5,
    "geodesic_distortion_amp": 0.1,
    "geodesic_distortion_func": "wiggles",  # Accepted values "wiggles" or "bump"

    # Model
    'model_type': 'EuclideanVAE',
    "data_dim": 10,
    'latent_dim': 2,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 50, 50],
    'decoder_widths': [32, 32, 32],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 10,
    'log_interval': 100,
    'recon_loss': "MSE",
    'topo_loss': True,
    'dim_topo_loss': 0,  # Max feature dimension topological loss
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "quadric_dim": 1,
    "n_plot_points": 1000,
    "n_grid_points": 800,  # Number of points to compute the curvature for
    "k": 110,
}

param_grid = {
    "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0] * 3,
    "gamma": [0.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0] * 3,
    "dim_topo_loss": ["_", 0, 1, 0, 1, 0, 1] * 3,
    "geodesic_amp": [0.1] * 7 + [0.5] * 14,
    "noise_var": [0.001] * 14 + [0.01] * 7,
}


def describe_experiment(overrides):
    desc_lines = []
    for k, v in overrides.items():
        if k != "experiment":
            desc_lines.append(f"{k}={v}")
    return ", ".join(desc_lines)


def generate_experiments(base_configuration, parameter_grid):
    # Ensure all lists are of equal length
    lengths = [len(v) for v in parameter_grid.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All parameter lists in param_grid must have the same length for synchronized iteration.")

    n = lengths[0]
    experiments = {}

    for i in range(n):
        overrides = {k: v[i] for k, v in parameter_grid.items() if v[i] != "_"}
        name = f"exp{i:02d}_{overrides.get('experiment', 'default')}"
        overrides["experiment"] = name

        cfg = base_configuration.copy()
        cfg.update(overrides)
        cfg["description"] = describe_experiment(overrides)

        default_root_log_dir = "./results"  # Or any preferred base path

        if cfg.get("log_dir") is None:
            log_dir = os.path.join(default_root_log_dir, cfg["dataset_name"], f"results_{name}")
        else:
            log_dir = os.path.join(cfg["log_dir"], cfg["dataset_name"], f"results_{name}")

        os.makedirs(log_dir, exist_ok=True)
        cfg["log_dir"] = log_dir

        experiments[name] = SimpleNamespace(**cfg)

    return experiments


all_configs = generate_experiments(base_config, param_grid)

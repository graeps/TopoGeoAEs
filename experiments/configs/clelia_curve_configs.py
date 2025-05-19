from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "flower_curve",
    "random_seed": 42,

    # Dataset
    "dataset_name": "clelia_curve",
    "batch_size": 64,
    "rotation": "random",
    "translation": "random",
    "n_times": 10000,
    "radius": 2.0,
    "embedding_dim": 10,
    "noise_var": 0.001,
    "clelia_c": 1,

    # Model
    'model_type': 'EuclideanVAE',
    "data_dim": 10,
    'latent_dim': 2,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 50, 50],
    'decoder_widths': [50, 50, 50],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 15,
    'log_interval': 100,
    'recon_loss': "MSE",
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "quadric_dim": 1,
    "n_plot_points": 1000,
    "n_grid_points": 800,  # to compute curvature
    "k": 160,
}

param_grid = {
    "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0] * 3,
    "gamma": [0.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0] * 3,
    "dim_topo_loss": ["_", 0, 1, 0, 1, 0, 1] * 3,
    "clelia_c": [0.3] * 7 + [1.0] * 7 + [3.0] * 7
}

all_configs = generate_experiments(base_config, param_grid)

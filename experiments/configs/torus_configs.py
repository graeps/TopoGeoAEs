from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "torus",
    "random_seed": 42,

    # Dataset
    "dataset_name": "torus",
    "batch_size": 64,
    "rotation": "random",
    "translation": "random",
    "deformation_amp": 0.1,
    "n_times": 30000,
    "major_radius": 2.5,
    "minor_radius": 1,
    "filled": False,
    "embedding_dim": 10,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 10,
    'latent_dim': 3,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [100, 100, 100],
    'decoder_widths': [50, 50, 50],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 20,
    'log_interval': 100,
    'recon_loss': "MSE",
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "quadric_dim": 2,
    "n_plot_points": 1000,
    "n_grid_points": 2000,  # to compute curvature
    "k": 200,
}

param_grid = {
    "alpha": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 3,
    "gamma": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0] * 3,
    "dim_topo_loss": ["_", 0, 1, 2, 0, 1, 2, 0, 1, 2] * 3,
    "deformation_amp": [0.1] * 20 + [0.4] * 10,
    "noise_var": [0.001] * 10 + [0.01] * 10 + [0.001] * 10,
}

all_configs = generate_experiments(base_config, param_grid)


from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "flower_scrunchy",
    "random_seed": 42,

    # Dataset
    "dataset_name": "flower_scrunchy",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "n_times": 10000,
    "radius": 2.0,
    "n_wiggles": 4,
    "geodesic_distortion_amp": 0.1,
    "embedding_dim": 10,
    "noise_var": 0.001,

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
    "n_curv_estimation_points": 800,  # to compute curvature
    "n_curv_evaluation_points": 800,
    "k": 110,

    # Persistent homology
    "scale": False,
    "homology_dimensions": [0,1]
}

param_grid = {
    "alpha": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0] * 3,
    "gamma": [0.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0] * 3,
    "dim_topo_loss": ["_", 0, 1, 0, 1, 0, 1] * 3,
    "geodesic_distortion_amp": [0.5] * 7 + [2.5] * 14,
    "noise_var": [0.001] * 14 + [0.01] * 7,
    "n_times": [10000] * 7 + [20000] * 14,
}

all_configs = generate_experiments(base_config, param_grid)

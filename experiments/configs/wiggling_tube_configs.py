from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "wiggling_tube",
    "random_seed": 40,
    "logging": True,

    # Dataset
    "dataset_name": "wiggling_tube",
    "batch_size": 64,
    "deformation_amp": 0.3,
    "n_phi": 300,
    "n_theta": 30,
    "minor_radius": 0.6,
    "embedding_dim": 10,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 10,
    'latent_dim': 3,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [100, 100, 100],
    'decoder_widths': [100, 100, 100],

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
    "n_plot_points": 8000,
    "n_curv_estimation_points": 8000,  # to compute curvature
    "n_curv_evaluation_points": 8000,
    # heuristics (estimation_points, k)
    #   n_phi=200, n_theta=20, minor_radius = 0.5: (5000, 100)
    "k": 500,
    "smoothing": False,

    # Persistent homology
    "persistent_homology": True,
    "n_points_pers_hom": 200,
    "scale": False,
    "homology_dimensions": [0, 1, 2]
}

param_grid = {
    "alpha": [1] * 1 + [1, 1, 0] * 3,
    "gamma": [0] + ([1] * 1 + [1000] * 1 + [1] * 1) * 3,
    "deformation_amp": [0.3] * 10,
    "dim_topo_loss": [0] + [0] * 3 + [1] * 3 + [2] * 3,
}

all_configs = generate_experiments(base_config, param_grid)

from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "torus",
    "random_seed": 42,
    "logging": True,

    # Dataset
    "dataset_name": "torus",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "deformation_amp": 0.0,
    "n_times": 9000,
    "major_radius": 1,
    "minor_radius": 0.9,
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
    "compute_true_curv": False,
    "compute_learned_curv": False,
    "compute_rec_curv": True,
    "n_plot_points": 2000,
    "n_points_emp_curv": 8000,  # to compute curvature
    "n_points_pullback_curv": 8000,
    # heuristics (estimation_points, k)
    #   major_radius = 2.5, minor_radius = 1, n_times = 6000: (5000, 460)
    #   major_radius = 4, minor_radius = 1, n_times = 10000: (5000, 250)
    #   major_radius = 4, minor_radius = 1, n_times = 11000: (8000, 440)
    "k": 500,
    "smoothing": False,

    # Persistent homology
    "persistent_homology": True,
    "n_points_pers_hom": 2000,
    "scale": False,
    "homology_dimensions": [0, 1, 2]
}

param_grid = {
    "alpha": [1] * 1 + [1, 1, 0] * 3,
    "gamma": [0] + [1, 1000, 1] * 3,
    "deformation_amp": [0.5] * 10,
    "dim_topo_loss": [0] + [0] * 3 + [1] * 3 + [2] * 3,
}


all_configs = generate_experiments(base_config, param_grid)

from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "interlocked_tori",
    "random_seed": 42,
    "logging": True,

    # Dataset
    "dataset_name": "interlocked_tori",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "deformation_amp": 0.1,
    "n_times": 9000,
    "major_radius": 4,
    "minor_radius": 1,
    "embedding_dim": 10,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 10,
    'latent_dim': 4,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 50, 50],
    'decoder_widths': [20, 20, 20],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 1,
    'log_interval': 100,
    'recon_loss': "MSE",
    'topo_loss': True,
    'dim_topo_loss': 2,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "n_plot_points": 4000,
    "n_curv_estimation_points": 16000,
    "n_curv_evaluation_points": 600,
    "k": 400,
    "smoothing": True,

    # Persistent homology
    "persistent_homology": True,
    "n_points_pers_hom": 1000,
    "scale": False,
    "homology_dimensions": [0, 1, 2]
}

param_grid = {
    "alpha": [1] * 7,
    "gamma": [0] + [1] * 3 + [1000] * 3,
    "deformation_amp": [0.5] * 7,
    "dim_topo_loss": [0] + [0, 1, 2] * 2,
}

# param_grid = {
#     "alpha": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] * 2,
#     "gamma": [0.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0] * 2,
#     "dim_topo_loss": [0, 0, 1, 2, 0, 1, 2] * 2,
#     "deformation_amp": [3.5] * 14,
#     "k": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750, 800, 850],
# }

all_configs = generate_experiments(base_config, param_grid)

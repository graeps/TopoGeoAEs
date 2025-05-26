from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "nested_spheres",
    "random_seed": 42,
    "logging": False,

    # Dataset
    "dataset_name": "nested_spheres",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "deformation_amp": 0.0001,
    "n_times": 5000,
    "major_radius": 10.0,
    "mid_radius": 5.0,
    "minor_radius": 3.0,
    "embedding_dim": 4,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 4,
    'latent_dim': 3,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 50, 50],
    'decoder_widths': [30, 30, 30],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 2,
    'log_interval': 100,
    'recon_loss': "MSE",
    'topo_loss': True,
    'dim_topo_loss': 2,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "n_plot_points": 1000,
    "n_curv_estimation_points": 5000,  # to compute curvature
    "n_curv_evaluation_points": 500,
    "k": 180,
    "smoothing": True,

    # Persistent homology
    "scale": False,
    "homology_dimensions": [0, 1, 2]
}

param_grid = {
    "alpha": [1.0, ] * 14,
    "gamma": [100.0, ] * 14,
    "dim_topo_loss": [0, 0, 1, 2, 0, 1, 2] * 2,
    "deformation_amp": [0] * 14,
    "k": [40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350, 400]
}

# param_grid = {
#     "alpha": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] * 2,
#     "gamma": [0.0, 1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0] * 2,
#     "dim_topo_loss": [0, 0, 1, 2, 0, 1, 2] * 2,
#     "deformation_amp": [3.5] * 14,
#     "k": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 750, 800, 850],
# }

all_configs = generate_experiments(base_config, param_grid)

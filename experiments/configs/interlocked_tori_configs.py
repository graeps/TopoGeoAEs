from scripts.experiment_utils import generate_experiments

base_config = {
    # Experiment
    "experiment": "interlocked_tori",
    "random_seed": 40,

    # Dataset
    "dataset_name": "interlocked_tori",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "deformation_amp": 0.0,
    "n_times": 2500,
    "major_radius": 2.5,
    "minor_radius": 1,
    "filled": False,
    "embedding_dim": 4,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 4,
    'latent_dim': 3,
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
    'dim_topo_loss': 2,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss

    # Curvature computation
    "quadric_dim": 2,
    "n_plot_points": 1000,
    "n_curv_estimation_points": 1000,  # to compute curvature
    "n_curv_evaluation_points": 1000,
    "k": 180,

    # Persistent homology
    "scale": False,
    "homology_dimensions": [0, 1, 2]
}

param_grid = {
    "alpha": [1.0, ] * 14,
    "gamma": [100.0, ] * 14,
    "dim_topo_loss": [0, 0, 1, 2, 0, 1, 2] * 2,
    "deformation_amp": [3] * 14,
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

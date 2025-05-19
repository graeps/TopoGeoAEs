from types import SimpleNamespace

base_config = {
    # Dataset
    "dataset_name": "entangled_tori",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "n_times": 10000,
    "major_radius": 5.0,
    "minor_radius": 1.0,
    "embedding_dim": 10,
    "noise_var": 0.0,
    "filled1": False,
    "filled2": False,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 10,
    'latent_dim': 3,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 50, 50],
    'decoder_widths': [32, 32, 32],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 15,
    'log_interval': 100,
    'recon_loss': "MSE",
    'dim_topo_loss': 0,  # Max feature dimension topological loss

    # Curvature computation
    "quadric_dim": 2,
    "n_plot_points": 1000,
    "n_grid_points": 800,  # to compute curvature
    "k": 160,
}

# Low embedding dim + no noise
exp1a_config = {
    "experiment": "exp1a_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': False,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

exp1b_config = {
    "experiment": "exp1b_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp1c_config = {
    "experiment": "exp1c_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp1d_config = {
    "experiment": "exp1d_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 2,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp1e_config = {
    "experiment": "exp1e_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp1f_config = {
    "experiment": "exp1f_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp1g_config = {
    "experiment": "exp1g_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 2,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}


def make_config(overrides):
    cfg = base_config.copy()
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


all_configs = {
    "exp1a_entangled_tori": make_config(exp1a_config),
    "exp1b_entangled_tori": make_config(exp1b_config),
    "exp1c_entangled_tori": make_config(exp1c_config),
    "exp1d_entangled_tori": make_config(exp1d_config),
    "exp1e_entangled_tori": make_config(exp1e_config),
    "exp1f_entangled_tori": make_config(exp1f_config),
    "exp1g_entangled_tori": make_config(exp1g_config),
}

from types import SimpleNamespace

base_config = {
    # Experiment
    "experiment": "8_curve",
    "random_seed": 42,

    # Dataset
    "dataset_name": "8_curve",
    "batch_size": 64,
    "rotation": "random",
    "translation": "random",
    "n_times": 10000,
    "embedding_dim": 3,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    "data_dim": 3,
    'latent_dim': 2,
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
    "quadric_dim": 1,
    "n_plot_points": 1000,
    "n_grid_points": 800,  # to compute curvature
    "k": 160,
}

# Low embedding dim + no noise
exp1a_config = {
    "experiment": "exp1a_scrunchy: low embedding dim + noise",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp1a + only topological loss
exp1b_config = {
    "experiment": "exp1b_scrunchy: exp1a + only topological loss",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp1a + mix topo and recon loss
exp1c_config = {
    "experiment": "exp1c_scrunchy: exp1a + mix topo and recon loss",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp1d_config = {
    "experiment": "exp1d_scrunchy: exp1a + mix topo and recon loss",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp1e_config = {
    "experiment": "exp1e_scrunchy: exp1a + mix topo and recon loss",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp2a_config = {
    "experiment": "exp1a_scrunchy: low embedding dim + noise",

    # Dataset
    "clelia_c": 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp1a + only topological loss
exp2b_config = {
    "experiment": "exp1b_scrunchy: exp1a + only topological loss",

    # Dataset
    "clelia_c": 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp1a + mix topo and recon loss
exp2c_config = {
    "experiment": "exp1c_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp2d_config = {
    "experiment": "exp1d_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp2e_config = {
    "experiment": "exp1e_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp3a_config = {
    "experiment": "exp1a_scrunchy: low embedding dim + noise",

    # Dataset
    "clelia_c": 3,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp1a + only topological loss
exp3b_config = {
    "experiment": "exp1b_scrunchy: exp1a + only topological loss",

    # Dataset
    "clelia_c": 3,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp1a + mix topo and recon loss
exp3c_config = {
    "experiment": "exp1c_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 3,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp3d_config = {
    "experiment": "exp1d_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 3,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp3e_config = {
    "experiment": "exp1e_scrunchy: exp1a + mix topo and recon loss",

    # Dataset
    "clelia_c": 3,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}


def make_config(overrides):
    cfg = base_config.copy()
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


all_configs = {
    "exp1a_scrunchy": make_config(exp1a_config),
    "exp1b_scrunchy": make_config(exp1b_config),
    "exp1c_scrunchy": make_config(exp1c_config),
    "exp1d_scrunchy": make_config(exp1d_config),
    "exp1e_scrunchy": make_config(exp1e_config),
    "exp2a_scrunchy": make_config(exp2a_config),

}

from types import SimpleNamespace

base_config = {
    # Dataset
    "dataset_name": "scrunchy",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "n_times": 10000,
    "radius": 2.0,
    "n_wiggles": 4,
    "geodesic_distortion_amp": 1.0,
    "embedding_dim": 3,
    "noise_var": 0.0,

    # Model
    'model_type': 'EuclideanVAE',
    'latent_dim': 2,
    'sftbeta': 4.5,
    'device': "cpu",
    'encoder_widths': [50, 32, 32],
    'decoder_widths': [32, 32, 32],

    # Optimizer
    "learning_rate": 0.001,

    # Trainer
    'verbose': False,
    'num_epochs': 10,
    'log_interval': 100,
    'recon_loss': "MSE",
    'dim_topo_loss': 0,  # Max feature dimension topological loss

    # Curvature computation
    "n_grid_points": 800,  # to compute curvature
}

# Low embedding dim + no noise
exp1a_config = {
    "experiment": "exp1a_scrunchy: low embedding dim + noise",

    # Model
    'data_dim': 3,

    # Trainer
    'topo_loss': False,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp1a + only topological loss
exp1b_config = {
    "experiment": "exp1b_scrunchy: exp1a + only topological loss",

    # Model
    'data_dim': 3,

    # Trainer
    'topo_loss': True,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp1a + mix topo and recon loss
exp1c_config = {
    "experiment": "exp1c_scrunchy: exp1a + mix topo and recon loss",

    # Model
    'data_dim': 3,

    # Trainer
    'topo_loss': True,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# High embedding dimension
exp2a_config = {
    "experiment": "exp2a_scrunchy: High embedding dimension",

    # Dataset
    "embedding_dim": 100,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': False,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# High embedding dimension + only topo loss
exp2b_config = {
    "experiment": "exp2b_scrunchy: High embedding dimension + only topo loss",

    # Dataset
    "embedding_dim": 100,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp2a + mix topo and recon loss
exp2c_config = {
    "experiment": "exp2c_scrunchy: exp2a + mix topo and recon loss",

    # Dataset
    "embedding_dim": 100,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# High embedding dim + noise
exp3a_config = {
    "experiment": "exp3a_scrunchy: High embedding dim + noise",

    # Dataset
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': False,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp3a + only topo loss
exp3b_config = {
    "experiment": "exp3b_scrunchy: exp3a + only topo loss",

    # Dataset
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp3a + mix topo and recon loss
exp3c_config = {
    "experiment": "exp3c_scrunchy: exp3a + mix topo and recon loss",

    # Dataset
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# High embedding dim + noise + large radius + many wiggles
exp4a_config = {
    "experiment": "exp4a_scrunchy: High embedding dim + moderate noise + large radius + many wiggles",

    # Dataset
    "radius": 10.0,
    "n_wiggles": 100,
    "geodesic_distortion_amp": 4.0,
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': False,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

# exp4a + only topo loss
exp4b_config = {
    "experiment": "exp4b_scrunchy: exp4a + only topo loss",

    # Dataset
    "radius": 10.0,
    "n_wiggles": 100,
    "geodesic_distortion_amp": 4.0,
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# exp4a + mix topo and recon loss
exp4c_config = {
    "experiment": "exp4c_scrunchy: exp4a + mix topo and recon loss",

    # Dataset
    "radius": 10.0,
    "n_wiggles": 100,
    "geodesic_distortion_amp": 4.0,
    "embedding_dim": 100,
    "noise_var": 0.01,

    # Model
    'data_dim': 100,

    # Trainer
    'topo_loss': True,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}


# Final experiment configs (Base + Override)
def make_config(overrides):
    cfg = base_config.copy()
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


all_configs = {
    "exp1a_scrunchy": make_config(exp1a_config),
    "exp1b_scrunchy": make_config(exp1b_config),
    "exp1c_scrunchy": make_config(exp1c_config),
    "exp2a_scrunchy": make_config(exp2a_config),
    "exp2b_scrunchy": make_config(exp2b_config),
    "exp2c_scrunchy": make_config(exp2c_config),
    "exp3a_scrunchy": make_config(exp3a_config),
    "exp3b_scrunchy": make_config(exp3b_config),
    "exp3c_scrunchy": make_config(exp3c_config),
    "exp4a_scrunchy": make_config(exp4a_config),
    "exp4b_scrunchy": make_config(exp4b_config),
    "exp4c_scrunchy": make_config(exp4c_config),
}

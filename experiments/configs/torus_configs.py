from types import SimpleNamespace

base_config = {
    # Dataset
    "dataset_name": "torus",
    "batch_size": 64,
    "rotation": "random",
    "translation": None,
    "n_times": 40000,
    "major_radius": 2.5,
    "minor_radius": 2,
    "filled": False,
    "embedding_dim": 3,
    "noise_var": 0.001,

    # Model
    'model_type': 'EuclideanVAE',
    'data_dim': 3,
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

exp1a_config = {
    "experiment": "exp1a_torus: low embedding dim + low noise + only recon loss",

    # Trainer
    'topo_loss': False,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

exp1b_config = {
    "experiment": "exp1b_torus: low embedding dim + low noise + only topo loss + dim_topo_loss 0",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp1c_config = {
    "experiment": "exp1c_torus: low embedding dim + low noise + mixed loss + dim_topo_loss 0",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp1d_config = {
    "experiment": "exp1d_torus: low embedding dim + low noise + only topo loss + dim_topo_loss 1",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp1e_config = {
    "experiment": "exp1e_torus: low embedding dim + low noise + mixed loss + dim_topo_loss 1",

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

exp2a_config = {
    "experiment": "exp2a_torus: small minor radius + low embedding dim + low noise + only recon loss",

    # Dataset
    'minor_radius': 0.5,

    # Trainer
    'topo_loss': False,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 0.0,  # Weight for topological loss
}

exp2b_config = {
    "experiment": "exp2b_torus: small minor radius  + low embedding dim + low noise + only topo loss + dim_topo_loss 0",

    # Dataset
    'minor_radius': 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

exp2c_config = {
    "experiment": "exp2c_torus: small minor radius  + low embedding dim + low noise + mixed loss + dim_topo_loss 0",

    # Dataset
    'minor_radius': 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 0,
    'alpha': 1.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 100.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp2d_config = {
    "experiment": "exp2d_torus: small minor radius + low embedding dim + low noise + only topo loss + dim_topo_loss 1",

    # Dataset
    'major_radius': 5.0,
    'minor_radius': 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
    'alpha': 0.0,  # Weight for reconstruction loss
    'beta': 0.0,  # Weight for KL loss
    'gamma': 1.0,  # Weight for topological loss
}

# Low embedding dim + no noise
exp2e_config = {
    "experiment": "exp2e_torus: small minor radius + low embedding dim + low noise + mixed loss + dim_topo_loss 1",

    # Dataset
    'minor_radius': 0.5,

    # Trainer
    'topo_loss': True,
    'dim_topo_loss': 1,
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
    "exp1a_torus": make_config(exp1a_config),
    "exp1b_torus": make_config(exp1b_config),
    "exp1c_torus": make_config(exp1c_config),
    "exp1d_torus": make_config(exp1d_config),
    "exp1e_torus": make_config(exp1e_config),
    "exp2a_torus": make_config(exp2a_config),
    "exp2b_torus": make_config(exp2b_config),
    "exp2c_torus": make_config(exp2c_config),
    "exp2d_torus": make_config(exp2d_config),
    "exp2e_torus": make_config(exp2e_config),
}

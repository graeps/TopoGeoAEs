import os
import sys
import torch
import numpy as np
import torch.optim as optim

# Set paths
mvae_dir = os.path.split(os.getcwd())[0]
if mvae_dir not in sys.path:
    sys.path.append(mvae_dir)

import lib.dataloaders.synthetic_loader as dataloader
import lib.models.vae.euclidean_vae as model
import lib.trainer as trainer
import lib.utils as utils

from .experiment_utils import generate_experiment_report

# Optional: Map your config names to their import paths
CONFIG_MODULES = {
    "scrunchy": "configs.scruchy_configs",
    "flower_curve": "configs.flower_curve_configs",
    "flower_scrunchy": "configs.flower_scrunchy_configs",
    "clelia_curve": "configs.clelia_curve_configs",
    "torus": "configs.torus_configs",
    "interlocked_tori": "configs.interlocked_tori_configs",
    "nested_spheres": "configs.nested_spheres_configs",
}


def run_experiment(name):
    assert name in CONFIG_MODULES, f"Unknown config name: {name}"

    # Dynamically import the right config module
    config_module = __import__(CONFIG_MODULES[name], fromlist=["all_configs"])
    all_configs = config_module.all_configs

    for config_name, config in all_configs.items():
        print("----------------------------------------------------------------")
        print(f"Running {config_name}. Description: {config.description}")
        print("----------------------------------------------------------------")

        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        data_loader = dataloader.load_synthetic_ds(config)
        train_loader, test_loader = data_loader

        vae_model = model.EuclideanVAE(config)
        optimizer = optim.Adam(vae_model.parameters(), lr=config.learning_rate)

        history = trainer.MVAETrainer(vae_model, data_loader, optimizer, config).train()

        utils.show_training_history(config, history)
        utils.plot_data_latents_recon(config, vae_model, train_loader)
        utils.plot_all_curvatures(config=config, model=vae_model, data_loader=train_loader)
        # utils.plot_curvature_persistence(config=config, model=vae_model, data_loader=train_loader)

        generate_experiment_report(config)

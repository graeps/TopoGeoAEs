import os
import sys
import torch
import numpy as np
import torch.optim as optim

import time

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
    "wiggling_tube": "configs.wiggling_tube_configs",
    "interlocked_tori": "configs.interlocked_tori_configs",
    "nested_spheres": "configs.nested_spheres_configs",
}


def run_experiment(name=None, all_configs=None):
    start_time = time.time()

    if all_configs is None:
        assert name in CONFIG_MODULES, f"Unknown config name: {name}"
        config_module = __import__(CONFIG_MODULES[name], fromlist=["all_configs"])
        all_configs = config_module.all_configs
    else:
        assert isinstance(all_configs, dict), "all_configs must be a dict if provided directly"

    for config_name, config in all_configs.items():
        print("\n======================================================================================")
        print("======================================================================================")
        print(f"Running {config_name}. \nDescription: {config.description}")
        print("--------------------------------------------------------------------------------------")

        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        data_loader = dataloader.load_synthetic_ds(config)
        train_loader, test_loader = data_loader

        vae_model = model.EuclideanVAE(config)
        optimizer = optim.Adam(vae_model.parameters(), lr=config.learning_rate)

        history = trainer.MVAETrainer(vae_model, data_loader, optimizer, config).train()

        utils.show_training_history(config, history)
        utils.plot_data_latents_recon(config, vae_model, train_loader)

        if config.persistent_homology:
            utils.plot_curvature_persistence(config=config, model=vae_model, data_loader=train_loader)
        else:
            utils.plot_all_curvatures(config=config, model=vae_model, data_loader=train_loader)

        if config.logging:
            generate_experiment_report(config)

        end_time = time.time()
        print(f"Execution time {config_name}: {end_time - start_time:.4f} seconds")

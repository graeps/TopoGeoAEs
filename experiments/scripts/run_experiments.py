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

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs  # noqa: E402

import lib.dataloaders.synthetic_loader as synthetic_loader
import lib.models.vae.euclidean_vae as e_vae_model
import lib.models.vae.vmf_toroidal_vae as vmf_toroidal_vae_model
import lib.models.vae.vmf_spherical_vae as vmf_spherical_vae_model
import lib.models.ae.euclidean_ae as ae_model
import lib.models.ae.spherical_ae as spherical_model
import lib.models.ae.toroidal_ae as toroidal_model
import lib.trainer as trainer
import lib.visualization as visual

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
        gs.random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        data_loader = synthetic_loader.load_synthetic_ds(config)
        train_loader, test_loader = data_loader

        if config.model_type == "EuclideanVAE":
            model = e_vae_model.EuclideanVAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.MVAETrainer(model, data_loader, optimizer, config).train()
        elif config.model_type == "VMFToroidalVAE":
            model = vmf_toroidal_vae_model.VMFToroidalVAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.MVAETrainer(model, data_loader, optimizer, config).train()
        elif config.model_type == "VMFSphericalVAE":
            model = vmf_spherical_vae_model.VMFSphericalVAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.MVAETrainer(model, data_loader, optimizer, config).train()
        elif config.model_type == "EuclideanAE":
            model = ae_model.EuclideanAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.AETrainer(model, data_loader, optimizer, config).train()
        elif config.model_type == "SphericalAE":
            model = spherical_model.SphericalAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.AETrainer(model, data_loader, optimizer, config).train()
        elif config.model_type == "ToroidalAE":
            model = toroidal_model.ToroidalAE(config)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            history = trainer.AETrainer(model, data_loader, optimizer, config).train()
        else:
            raise NotImplementedError

        visual.training_history.show_training_history(config, history)
        visual.plot_data_latents_recon(config, model, train_loader)

        if config.compute_persistent_homology:
            visual.plot_persistence(config=config, model=model, data_loader=train_loader)
        elif config.plot_curvatures:
            visual.plot_all_curvatures(config=config, model=model, data_loader=train_loader)

        if config.logging:
            generate_experiment_report(config)

        end_time = time.time()
        print(f"Execution time {config_name}: {end_time - start_time:.4f} seconds")
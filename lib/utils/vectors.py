import os
import numpy as np
import torch
import json


def get_vectors(config, model, data_loader, n_samples = None, save = False, save_title = None):
    """
    Computes latent representations, reconstructions, and organizes data from a dataset using
    a specified model. Handles data preprocessing, saving, and sorting of results as required.

    Args:
        config: Configuration object that includes model settings, device information, and
            other experimental parameters.
        model: The neural network model used to compute latent representations and reconstructions.
        data_loader: DataLoader object to iterate over the dataset.
        n_samples: Number of samples to return after random selection.
        save: If True, saves the vectors (inputs, latents, recons, labels) as JSON to config.log_dir.
        save_title: Optional base filename (without extension) for the saved JSON. If None, a default
            name "vectors_{experiment}.json" is used.

    Returns:
        A tuple containing reconstructions, latent representations, original inputs, and
        sorted labels in the following order:
        - `recons`: Tensor with reconstructed data.
        - `latents`: Tensor containing latent representations computed by the model.
        - `inputs`: Tensor with the original input data.
        - `labels`: Tensor containing sorted labels corresponding to the input data.

    Raises:
        NotImplementedError: If the model type is not supported or if labels have an
            unexpected dimensional structure.
    """
    if config.verbose:
        print("Forwarding data through model to compute latents and recons...")

    model.eval()
    inputs, latents, recons, labels = [], [], [], []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            if config.model_type in {"EuclideanVAE", "VMToroidalVAE", "VMFToroidalVAE", "VMFSphericalVAE"}:
                z, x_recon, _ = model(x)
            elif config.model_type in {"EuclideanAE", "ParamAE", "SphericalAE", "ToroidalAE"}:
                angles, z, x_recon = model(x)
            else:
                raise NotImplementedError

            inputs.append(x.cpu())
            latents.append(z.cpu())
            recons.append(x_recon.cpu())
            labels.append(y.cpu())

    inputs = torch.cat(inputs, dim=0)
    latents = torch.cat(latents, dim=0)
    recons = torch.cat(recons, dim=0)
    labels = torch.cat(labels, dim=0)

    if n_samples is not None:
        n_total = latents.shape[0]
        n_samples = min(n_samples, n_total)
        indices = torch.randperm(n_total)[:n_samples]
    
        # Apply random sampling
        inputs = inputs[indices]
        latents = latents[indices]
        recons = recons[indices]
        labels = labels[indices]
    
        # Reshaping labels if 1 dim array
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)  # Shape: (N, 1)
    
        # Sort by label if label dim = 1
        if labels.shape[1] == 1:
            labels = labels.squeeze()
            sort_idx = torch.argsort(labels)
    
        # lexicographic sort if label dim = 2
        elif labels.shape[1] == 2:
            sort_idx = np.lexsort((labels[:, 1].numpy(), labels[:, 0].numpy()))
            sort_idx = torch.from_numpy(sort_idx)
    
        # lexicographic sort if label dim = 3, for nested_spheres and interlocked_tori.
        elif labels.shape[1] == 3:
            sort_idx = np.lexsort((
                labels[:, 2].numpy(),  # phi (3rd column)
                labels[:, 1].numpy(),  # theta (2nd column)
                labels[:, 0].numpy(),  # entity index (1st column)
            ))
    
        else:
            raise NotImplementedError(f"Labels should either be one-dimensional or two-dimensional")
    
        labels = labels[sort_idx]
        inputs = inputs[sort_idx]
        latents = latents[sort_idx]
        recons = recons[sort_idx]

    # Optional JSON save
    if save and getattr(config, "log_dir", None) is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        filename_base = save_title if save_title else f"vectors_{getattr(config, 'experiment', 'exp')}"
        save_path = os.path.join(config.log_dir, f"{filename_base}.json")

        def _to_serializable(obj):
            if obj is None:
                return None
            # Native JSON types
            if isinstance(obj, (bool, int, float, str)):
                return obj
            # NumPy scalars
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            # NumPy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Torch tensors
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            # Dicts
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            # Lists/Tuples
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            # Fallback to string
            return str(obj)

        results_dict = {
            "experiment": getattr(config, "experiment", "unknown"),
            "points": {
                "labels": labels,
                "inputs": inputs,
                "latents": latents,
                "recons": recons,
            },
        }
        results_json = _to_serializable(results_dict)
        with open(save_path, "w") as f:
            json.dump(results_json, f, indent=2)
        if getattr(config, "verbose", False):
            print(f"Saved vectors (JSON) to {save_path}")

    return recons, latents, inputs, labels

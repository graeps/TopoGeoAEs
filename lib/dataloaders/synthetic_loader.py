from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

from ..datasets.datasets_main import load_s1_low, load_s1_high, load_s2_low, load_s2_high, load_t2_low, load_t2_high

from ..datasets.datasets_other import generate_genus3, \
    load_nested_spheres, load_clelia_curve, load_8_curve, load_interlocked_tori, load_wiggling_tube, \
    load_nested_spheres_high_dim_bump, load_interlocked_tubes, load_sphere_high_dim_bump, load_scrunchy


class InvalidConfigError(Exception):
    pass


def load_synthetic_ds(config):
    """
    Load a synthetic dataset and return training and test DataLoaders.

    The function dispatches to the appropriate dataset generator depending on
    ``config.dataset_name`` and converts the returned tensors/labels to a
    TensorDataset suitable for PyTorch.

    Args:
        config: Configuration object containing all dataset-specific parameters,
            including ``dataset_name``, geometry parameters, ``embedding_dim``,
            ``n_points``, ``noise_var``, and ``batch_size``.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            * train_loader: DataLoader for the training split.
            * test_loader:  DataLoader for the test split.
    """

    # 1. Dispatch to the appropriate dataset loader
    if config.dataset_name == "s1_low":
        data, labels = load_s1_low(
            rotation=config.rotation,
            n_points=config.n_points,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            deformation_type=config.deformation_type,
        )
    elif config.dataset_name == "s1_high":
        data, labels = load_s1_high(
            n_points=config.n_points,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            translation=config.translation,
            rotation=config.rotation,
        )
    elif config.dataset_name == "s2_low":
        data, labels = load_s2_low(
            rotation=config.rotation,
            n_points=config.n_points,
            radius=config.radius,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "t2_low":
        data, labels = load_t2_low(
            rotation=config.rotation,
            n_points=config.n_points,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "t2_high":
        data, labels = load_t2_high(
            n_points=config.n_points,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            noise_var=config.noise_var,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            translation=config.translation,
            rotation=config.rotation,
        )
    elif config.dataset_name == "s2_high":
        data, labels = load_s2_high(
            n_points=config.n_points,
            radius=config.radius,
            noise_var=config.noise_var,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            translation=config.translation,
            rotation=config.rotation,
        )
    elif config.dataset_name == "scrunchy":
        data, labels = load_scrunchy(
            rotation=config.rotation,
            n_points=config.n_points,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "nested_spheres":
        data, labels = load_nested_spheres(
            n_points=config.n_points,
            minor_radius=config.minor_radius,
            mid_radius=config.mid_radius,
            major_radius=config.major_radius,
            noise_var=config.noise_var,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
        )
    elif config.dataset_name == "sphere_high_dim":
        data, labels = load_sphere_high_dim_bump(
            n_points=config.n_points,
            radius=config.radius,
            noise_var=config.noise_var,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
        )
    elif config.dataset_name == "nested_spheres_high_dim":
        data, labels = load_nested_spheres_high_dim_bump(
            n_points=config.n_points,
            minor_radius=config.minor_radius,
            mid_radius=config.mid_radius,
            major_radius=config.major_radius,
            noise_var=config.noise_var,
            deformation_amp=config.deformation_amp,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
            enclosing_sphere=config.enclosing_sphere,
        )
    elif config.dataset_name == "clelia_curve":
        data, labels = load_clelia_curve(
            n_points=config.n_points,
            r=config.radius,
            c=config.clelia_c,
            noise_var=config.noise_var,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
        )
    elif config.dataset_name == "8_curve":
        data, labels = load_8_curve(
            n_points=config.n_points,
            noise_var=config.noise_var,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
        )
    elif config.dataset_name == "interlocked_tori":
        data, labels = load_interlocked_tori(
            n_points=config.n_points,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            noise_var=config.noise_var,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            rotation=config.rotation,
        )
    elif config.dataset_name == "interlocked_tubes":
        data, labels = load_interlocked_tubes(
            n_phi=config.n_phi,
            n_theta=config.n_theta,
            minor_radius=config.minor_radius,
            noise_var=config.noise_var,
            wiggling_dim=config.wiggling_dim,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            rotation=config.rotation,
        )
    elif config.dataset_name == "genus_3":
        data, labels = generate_genus3(
            n_points=config.n_points,
            noise_var=config.noise_var,
            embedding_dim=config.embedding_dim,
            rotation=config.rotation,
        )
    elif config.dataset_name == "wiggling_tube":
        data, labels = load_wiggling_tube(
            n_phi=config.n_phi,
            n_theta=config.n_theta,
            minor_radius=config.minor_radius,
            noise_var=config.noise_var,
            wiggling_dim=config.wiggling_dim,
            embedding_dim=config.embedding_dim,
            deformation_amp=config.deformation_amp,
            rotation=config.rotation,
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

    # 2. Convert labels to a tensor if necessary
    if config.dataset_name in {
        "nested_spheres",
        "nested_spheres_high_dim",
        "interlocked_tori",
        "interlocked_tubes",
    }:
        # These datasets return a tuple (entity_index, angles)
        entity_index, angles = labels
        entity_index = entity_index.unsqueeze(1).float()  # shape [N,1]
        combined_label = torch.cat([entity_index, angles], dim=1)
        dataset = TensorDataset(data, combined_label)

    elif isinstance(labels, torch.Tensor):
        dataset = TensorDataset(data, labels.float())

    else:
        # labels is a pandas.DataFrame
        dataset = TensorDataset(data, torch.tensor(labels.values, dtype=torch.float32))

    # 3. Train/test split and DataLoader construction
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, test_loader

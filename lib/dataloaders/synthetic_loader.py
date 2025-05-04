from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

from ..datasets.synthetic_sphere_like import load_s1_synthetic
from ..datasets.synthetic_sphere_like import load_s1_in_s1_synthetic
from ..datasets.synthetic_sphere_like import load_scrunchy_synthetic
from ..datasets.synthetic_sphere_like import load_interlocking_rings_synthetic
from ..datasets.synthetic_sphere_like import load_s2_synthetic

from ..datasets.synthetic_sphere_like import load_t2_synthetic


def load_synthetic_ds(config):
    if config.dataset_name == "s1_synthetic":
        dataset, labels = load_s1_synthetic(
            rotation=config.rotation,
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            geodesic_distortion_func=config.geodesic_distortion_func,
        )
    elif config.dataset_name == "s1_in_s1_synthetic":
        dataset, labels = load_s1_in_s1_synthetic(
            rotation=config.rotation,
            n_times=config.n_times,
            radius_inner=config.radius_inner,
            radius_outer=config.radius_outer,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            geodesic_distortion_func=config.geodesic_distortion_func,
        )
    elif config.dataset_name == "scrunchy_synthetic":
        dataset, labels = load_scrunchy_synthetic(
            rotation=config.rotation,
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "interlocking_rings_synthetic":
        dataset, labels = load_interlocking_rings_synthetic(
            rotation=config.rotation,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "s2_synthetic":
        dataset, labels = load_s2_synthetic(
            rotation=config.rotation,
            n_times=config.n_times,
            radius=config.radius,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "t2_synthetic":
        dataset, labels = load_t2_synthetic(
            rotation=config.rotation,
            n_times=config.n_times,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    else:
        raise InvalidConfigError(f"Unknown dataset: {config['dataset_name']}")

    dataset = TensorDataset(dataset, torch.tensor(labels.values).float())

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


class InvalidConfigError(Exception):
    pass

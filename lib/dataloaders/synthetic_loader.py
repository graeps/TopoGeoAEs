from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

from ..datasets.synthetic_sphere_like import load_s1_synthetic, load_s1_in_s1_synthetic, load_scrunchy, \
    load_interlocking_rings_synthetic, load_s2_synthetic, load_t2_synthetic, load_flower_scrunchy

from ..datasets.topo_datasets import generate_sphere, generate_genus3, generate_three_manifolds, \
    generate_nested_spheres, load_clelia_curve, load_8_curve, generate_entangled_tori, load_torus


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
            random_seed=config.random_seed,
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
    elif config.dataset_name == "scrunchy":
        dataset, labels = load_scrunchy(
            rotation=config.rotation,
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            random_seed=config.random_seed,
        )
    elif config.dataset_name == "flower_scrunchy":
        dataset, labels = load_flower_scrunchy(
            rotation=config.rotation,
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            random_seed=config.random_seed,
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
            random_seed=config.random_seed,
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
            random_seed=config.random_seed,
        )
    elif config.dataset_name == "torus":
        dataset, labels = load_torus(n_points=config.n_times, major_radius=config.major_radius,
                                     minor_radius=config.minor_radius,
                                     noise_var=config.noise_var,
                                     deformation_amp=config.deformation_amp,
                                     embedding_dim=config.embedding_dim,
                                     translation=config.translation, rotation=config.rotation,
                                     random_seed=config.random_seed, )
    elif config.dataset_name == "entangled_tori":
        dataset, labels = generate_entangled_tori(n_points=config.n_times, R=config.major_radius, r=config.minor_radius,
                                                  filled1=config.filled1, filled2=config.filled2,
                                                  noise_var=config.noise_var, embedding_dim=config.embedding_dim,
                                                  translation=config.translation, rotation=config.rotation,
                                                  random_seed=config.random_seed, )
    elif config.dataset_name == "genus_3":
        dataset, labels = generate_genus3(n_points=config.n_times, noise_var=config.noise_var,
                                          embedding_dim=config.embedding_dim, translation=config.translation,
                                          rotation=config.rotation, random_seed=config.random_seed, )
    elif config.dataset_name == "sphere_filled":
        dataset, labels = generate_sphere(n_points=config.n_times, radius=config.radius, filled=True,
                                          noise_var=config.noise_var, embedding_dim=config.embedding_dim,
                                          translation=config.translation, rotation=config.rotation,
                                          random_seed=config.random_seed, )
    elif config.dataset_name == "sphere_hollowed":
        dataset, labels = generate_sphere(n_points=config.n_times, radius=config.radius, filled=False,
                                          noise_var=config.noise_var, embedding_dim=config.embedding_dim,
                                          translation=config.translation, rotation=config.rotation,
                                          random_seed=config.random_seed, )
    elif config.dataset_name == "nested_spheres":
        dataset, labels = generate_nested_spheres(n_points=config.n_times, radii=config.radii,
                                                  noise_var=config.noise_var, embedding_dim=config.embedding_dim,
                                                  translation=config.translation, rotation=config.rotation,
                                                  random_seed=config.random_seed, )
    elif config.dataset_name == "clelia_curve":
        dataset, labels = load_clelia_curve(n_points=config.n_times, r=config.radius, c=config.clelia_c,
                                            noise_var=config.noise_var,
                                            embedding_dim=config.embedding_dim, translation=config.rotation,
                                            rotation=config.rotation, random_seed=config.random_seed, )
    elif config.dataset_name == "8_curve":
        dataset, labels = load_8_curve(n_points=config.n_times, noise_var=config.noise_var,
                                       embedding_dim=config.embedding_dim, translation=config.rotation,
                                       rotation=config.rotation, random_seed=config.random_seed, )
    elif config.dataset_name == "three_entangled_tori":
        dataset, labels = generate_three_manifolds("entangled_tori")
    elif config.dataset_name == "three_nested_spheres":
        dataset, labels = generate_three_manifolds("three_nested_spheres")
    elif config.dataset_name == "spheres":
        dataset, labels = generate_three_manifolds("spheres")
    else:
        raise InvalidConfigError(f"Unknown dataset: {config['dataset_name']}")
    if isinstance(labels, torch.Tensor):
        dataset = TensorDataset(dataset, labels.float())
    else:
        dataset = TensorDataset(dataset, torch.tensor(labels.values).float())

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader


class InvalidConfigError(Exception):
    pass

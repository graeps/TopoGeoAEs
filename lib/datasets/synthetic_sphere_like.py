"""Generate and load synthetic datasets."""

import os

# Set Geomstats backend before importing it
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal  # noqa: E402

import geomstats.backend as gs  # noqa: E402
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402


def load_s1_synthetic(rotation, n_times=1500, radius=1.0, n_wiggles=6, geodesic_distortion_amp=0.4, embedding_dim=10,
                      noise_var=0.01, geodesic_distortion_func="wiggles", random_seed=42,
                      ):
    """Generate noisy S¹-immersed data in R^embedding_dim."""
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s1_synthetic_immersion(geodesic_distortion_func=geodesic_distortion_func, radius=radius,
                                           n_wiggles=n_wiggles, geodesic_distortion_amp=geodesic_distortion_amp,
                                           embedding_dim=embedding_dim, rot=rot, )

    angles = gs.linspace(0, 2 * gs.pi, n_times)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_times,))
        data = data + radius * noise

    return data, angles


def load_s1_in_s1_synthetic(
        rotation,
        n_times=1500,
        radius_outer=3.0,
        radius_inner=1.0,
        n_wiggles=6,
        geodesic_distortion_amp=0.4,
        embedding_dim=10,
        noise_var=0.01,
        geodesic_distortion_func="wiggles",
):
    """Generate noisy S¹-immersed data in R^embedding_dim."""
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion_outer = get_s1_synthetic_immersion(
        geodesic_distortion_func=geodesic_distortion_func,
        radius=radius_outer,
        n_wiggles=n_wiggles,
        geodesic_distortion_amp=geodesic_distortion_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    immersion_inner = get_s1_synthetic_immersion(
        geodesic_distortion_func=geodesic_distortion_func,
        radius=radius_inner,
        n_wiggles=n_wiggles,
        geodesic_distortion_amp=geodesic_distortion_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    angles_outer = gs.linspace(0, 2 * gs.pi, n_times)
    angles_inner = gs.linspace(0, 2 * gs.pi, n_times)
    data_outer = torch.stack([immersion_outer(angle) for angle in angles_outer])
    data_inner = torch.stack([immersion_inner(angle) for angle in angles_inner])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=noise_var * torch.eye(embedding_dim),
    ).sample((n_times,))

    noisy_data_outer = data_outer + radius_outer * noise
    noisy_data_inner = data_inner + radius_inner * noise

    noisy_data = torch.cat([noisy_data_outer, noisy_data_inner], dim=0)

    labels = pd.DataFrame({
        "circle_id": [0] * n_times + [1] * n_times,
        "angles": torch.cat([angles_outer, angles_inner]).numpy()
    })

    return noisy_data, labels


def load_s2_synthetic(
        rotation,
        n_times,
        radius,
        geodesic_distortion_amp,
        embedding_dim,
        noise_var,
        random_seed=42,
):
    """Generate noisy S²-immersed data in R^embedding_dim."""
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s2_synthetic_immersion(radius, geodesic_distortion_amp, embedding_dim, rot)

    sqrt_ntimes = int(gs.sqrt(n_times))
    thetas = gs.linspace(0.01, gs.pi, sqrt_ntimes)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        data = data + noise

    return data, angle_grid


def load_t2_synthetic(
        rotation,
        n_times,
        major_radius,
        minor_radius,
        geodesic_distortion_amp,
        embedding_dim,
        noise_var,
        random_seed=42,
):
    """Generate noisy T²-immersed data in R^embedding_dim."""
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_t2_synthetic_immersion(
        major_radius, minor_radius, geodesic_distortion_amp, embedding_dim, rot
    )

    sqrt_ntimes = int(gs.sqrt(n_times))
    thetas = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        data = data + noise

    return data, angle_grid


def load_scrunchy(rotation,
                  n_times=1500,
                  radius=1.0,
                  n_wiggles=6,
                  geodesic_distortion_amp=0.4,
                  embedding_dim=10,
                  noise_var=0.01, random_seed=42,
                  ):
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_scrunchy_immersion(
        radius=radius,
        n_wiggles=n_wiggles,
        distortion_amp=geodesic_distortion_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )
    angles = gs.linspace(0, 2 * gs.pi, n_times)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_times,))
        noisy_data = data + radius * noise
    else:
        noisy_data = data

    labels = angles.unsqueeze(dim=1)
    return noisy_data, labels


def load_interlocking_rings_synthetic(rotation,
                                      n_times=1500,
                                      radius=1.0,
                                      embedding_dim=10,
                                      noise_var=0.01,
                                      ):
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    immersion = get_interlocking_rings_immersion(
        radius=radius,
        embedding_dim=embedding_dim,
        rot=rot,
    )
    angles = gs.linspace(0, 4 * gs.pi, n_times)
    data = torch.stack([immersion(angle) for angle in angles])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=noise_var * torch.eye(embedding_dim),
    ).sample((n_times,))

    noisy_data = data + radius * noise
    labels = pd.DataFrame({"angles": angles})
    return noisy_data, labels


def get_s1_synthetic_immersion(
        geodesic_distortion_func,
        radius,
        n_wiggles,
        geodesic_distortion_amp,
        embedding_dim,
        rot,
):
    def immersion(angle):
        if geodesic_distortion_func == "wiggles":
            amp = radius * (1 + geodesic_distortion_amp * gs.cos(n_wiggles * angle))
        elif geodesic_distortion_func == "bump":
            amp = radius * (
                    1
                    + geodesic_distortion_amp * gs.exp(-5 * (angle - gs.pi / 2) ** 2)
                    + geodesic_distortion_amp * gs.exp(-5 * (angle - 3 * gs.pi / 2) ** 2)
            )
        else:
            raise NotImplementedError(f"Unknown distortion: {geodesic_distortion_func}")

        base_point = amp * gs.array([gs.cos(angle), gs.sin(angle)])
        base_point = gs.squeeze(base_point, axis=-1)
        if embedding_dim > 2:
            base_point = gs.concatenate([base_point, gs.zeros(embedding_dim - 2)])

        return gs.einsum("ij,j->i", rot, base_point)

    return immersion


def get_s2_synthetic_immersion(radius, geodesic_distortion_amp, embedding_dim, rot):
    """Returns a function mapping S² angles to R^embedding_dim with distortion."""

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1
                + geodesic_distortion_amp * gs.exp(-5 * theta ** 2)
                + geodesic_distortion_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def get_t2_synthetic_immersion(major_radius, minor_radius, geodesic_distortion_amp, embedding_dim, rot):
    """Returns a function mapping T² angles to R^embedding_dim with distortion."""

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1
                + geodesic_distortion_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
                + geodesic_distortion_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
        )

        x = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def get_scrunchy_immersion(radius, n_wiggles, distortion_amp, embedding_dim, rot):
    def immersion(angle):
        x = radius * gs.cos(angle)
        y = radius * gs.sin(angle)
        z = distortion_amp * gs.cos(n_wiggles * angle)
        point = gs.squeeze(gs.array([x, y, z]), axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def get_interlocking_rings_immersion(radius, embedding_dim, rot):
    def immersion(angle: float) -> torch.Tensor:
        assert 0 <= angle <= 4 * gs.pi
        if angle < 2 * gs.pi:
            # First Ring
            x = radius * gs.cos(angle)
            y = radius * gs.sin(angle)
            z = 0
            point = gs.squeeze(gs.array([x, y, z]), axis=-1)

        else:
            # Second Ring
            x = radius * gs.sin(angle) + radius
            y = 0
            z = radius * gs.cos(angle)
            point = gs.squeeze(gs.array([x, y, z]), axis=-1)

        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        return gs.einsum("ij,j->i", rot, point)

    return immersion

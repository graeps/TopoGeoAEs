"""Generate and load synthetic datasets."""

import os

# Set Geomstats backend before importing it
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal  # noqa: E402

import geomstats.backend as gs  # noqa: E402
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402

def load_s1_synthetic(
        synthetic_rotation,
        n_times=1500,
        radius=1.0,
        n_wiggles=6,
        geodesic_distortion_amp=0.4,
        embedding_dim=10,
        noise_var=0.01,
        geodesic_distortion_func="wiggles",
):
    """Generate noisy S¹-immersed data in R^embedding_dim."""
    rot = torch.eye(embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s1_synthetic_immersion(
        geodesic_distortion_func=geodesic_distortion_func,
        radius=radius,
        n_wiggles=n_wiggles,
        geodesic_distortion_amp=geodesic_distortion_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    angles = gs.linspace(0, 2 * gs.pi, n_times)
    data = torch.stack([immersion(angle) for angle in angles])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=noise_var * torch.eye(embedding_dim),
    ).sample((n_times,))

    noisy_data = data + radius * noise
    labels = pd.DataFrame({"angles": angles})
    return noisy_data, labels


def load_s2_synthetic(
        synthetic_rotation,
        n_times,
        radius,
        geodesic_distortion_amp,
        embedding_dim,
        noise_var,
):
    """Generate noisy S²-immersed data in R^embedding_dim."""
    rot = torch.eye(embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s2_synthetic_immersion(radius, geodesic_distortion_amp, embedding_dim, rot)

    sqrt_ntimes = int(gs.sqrt(n_times))
    thetas = gs.linspace(0.01, gs.pi, sqrt_ntimes)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
    ).sample((sqrt_ntimes ** 2,))

    noisy_data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return noisy_data, labels


def load_t2_synthetic(
        synthetic_rotation,
        n_times,
        major_radius,
        minor_radius,
        geodesic_distortion_amp,
        embedding_dim,
        noise_var,
):
    """Generate noisy T²-immersed data in R^embedding_dim."""
    rot = torch.eye(embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_t2_synthetic_immersion(
        major_radius, minor_radius, geodesic_distortion_amp, embedding_dim, rot
    )

    sqrt_ntimes = int(gs.sqrt(n_times))
    thetas = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
    ).sample((sqrt_ntimes ** 2,))

    noisy_data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return noisy_data, labels


def get_s1_synthetic_immersion(
        geodesic_distortion_func: str,
        radius: float,
        n_wiggles: int,
        geodesic_distortion_amp: float,
        embedding_dim: int,
        rot: torch.Tensor,
):
    """Return immersion S¹ → ℝ^N producing a distorted high-dimensional circle.

    Parameters
    ----------
    geodesic_distortion_func : {"wiggles", "bump"}
        Type of geodesic distortion.
    radius : float
        Base radius of the circle.
    n_wiggles : int
        Number of oscillations (used if distortion is "wiggles").
    geodesic_distortion_amp : float
        Amplitude of distortion.
    embedding_dim : int
        Target embedding dimension.
    rot : torch.Tensor
        Rotation matrix in SO(N) applied after immersion.

    Returns
    -------
    synth_immersion : callable
        Function mapping angle ∈ [0, 2π] to ℝ^embedding_dim.
    """

    def synth_immersion(angle: float) -> torch.Tensor:
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
        if embedding_dim > 2:
            base_point = gs.concatenate([base_point, gs.zeros(embedding_dim - 2)])

        return gs.einsum("ij,j->i", rot, base_point)

    return synth_immersion


def get_s2_synthetic_immersion(radius, geodesic_distortion_amp, embedding_dim, rot):
    """Returns a function mapping S² angles to R^embedding_dim with distortion."""

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = radius * (
                1
                + geodesic_distortion_amp * gs.exp(-5 * theta ** 2)
                + geodesic_distortion_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)

        point = amplitude * gs.array([x, y, z])
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
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion

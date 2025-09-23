"""Synthetic datasets used for the experiments described in the main text of the diploma thesis."""

import os
import pandas as pd

# Set Geomstats backend before importing it
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal  # noqa: E402

import geomstats.backend as gs  # noqa: E402
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402
import numpy as np

from .utils import embedd, rotate_translate, embedd_rotate_translate


def get_s1_low_immersion(deformation_type, radius, n_wiggles, deformation_amp, embedding_dim, rot):
    """
    Returns the immersion of the manifold where the s1_low dataset concentrates.  The immersion parameterizes a deformed circle.

    Args:
        deformation_type (str): The type of deformation to apply. Choices are "wiggles" and "bump".
        radius (float): The radius of the underlying circle.
        n_wiggles (int): Specifies the number of oscillations in the distortion.
        deformation_amp (float): The amplitude of the deformation.
        embedding_dim (int): The number of dimensions in the embedding space.
        rot (torch.Tensor): A rotation matrix.

    Returns:
        Function: Function that maps angles to the embedding space with the specified deformation.

    Raises:
        NotImplementedError: If an unsupported distortion type is provided.
    """

    def immersion(angle):
        if deformation_type == "wiggles":
            amp = radius * (1 + deformation_amp * gs.cos(n_wiggles * angle))
        elif deformation_type == "bump":
            amp = radius * (
                    1
                    + deformation_amp * gs.exp(-5 * (angle - gs.pi / 2) ** 2)
                    + deformation_amp * gs.exp(-5 * (angle - 3 * gs.pi / 2) ** 2)
            )
        else:
            raise NotImplementedError(f"Unknown distortion: {deformation_type}")

        base_point = amp * gs.array([gs.cos(angle), gs.sin(angle)])
        base_point = gs.squeeze(base_point, axis=-1)
        if embedding_dim > 2:
            base_point = gs.concatenate([base_point, gs.zeros(embedding_dim - 2)])

        return gs.einsum("ij,j->i", rot, base_point)

    return immersion


def get_s2_low_immersion(radius, deformation_amp, embedding_dim, rot):
    """
    Returns the immersion of the manifold where the s2_low dataset concentrates. The immersion parameterizes a deformed sphere.

    Args:
        radius (float): The radius of the underlying sphere.
        deformation_amp (float): The amplitude of the deformation.
        embedding_dim (int): The number of dimensions in the embedding space.
        rot (torch.Tensor): A rotation matrix.

    Returns:
        Function: Function that maps angles to the embedding space with the specified deformation.

    Raises:
        NotImplementedError: If an unsupported distortion type is provided.
    """

    def spherical(theta, phi):
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)
        return gs.array([x, y, z])

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1 + deformation_amp * gs.exp(-5 * theta ** 2)
                + deformation_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        point = amplitude * spherical(theta, phi)
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def get_t2_low_immersion(major_radius, minor_radius, deformation_amp, embedding_dim, rot):
    """
    Returns the immersion of the manifold where the t2_low dataset concentrates. The immersion parameterizes a deformed torus.

    Args:
        major_radius (float): The major radius of the underlying torus.
        minor_radius (float): The minor radius of the underlying torus.
        deformation_amp (float): The amplitude of the deformation.
        embedding_dim (int): The number of dimensions in the embedding space.
        rot (torch.Tensor): A rotation matrix.

    Returns:
        Function: Function that maps angles to the embedding space with the specified deformation.

    Raises:
        NotImplementedError: If an unsupported distortion type is provided.
    """

    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1
                + deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
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


def get_s1_high(deformation_amp, embedding_dim, translation, rotation, deformation="peaks"):
    """
    Returns the immersion of the manifold where the s1_high dataset concentrates. The immersion parameterizes
    a deformed circle embedded in R^n with n <= 10.

    Args:
        deformation_amp (float): Amplitude of the deformation applied to the base circle.
        embedding_dim (int): Dimension of the embedding space.
        translation (torch.Tensor): Translation vector applied after the embedding.
        rotation (torch.Tensor): Rotation matrix applied after the embedding.
        deformation (str, optional): Type of deformation pattern ("peaks" or other).

    Returns:
        Function: A function mapping an angle to the deformed high-dimensional embedding.
    """

    def immersion(angle):
        if deformation == "peaks":
            x1 = gs.sin(angle)
            x2 = gs.cos(angle)
            x3 = gs.sin(0 * angle)
            x4 = deformation_amp * gs.cos(2 * angle)
            x5 = deformation_amp * gs.sin(3 * angle)
            x6 = deformation_amp * gs.cos(4 * angle)
            x7 = deformation_amp * gs.sin(5 * angle)
            x8 = deformation_amp * gs.cos(6 * angle)
            x9 = deformation_amp * gs.sin(7 * angle)
            x10 = deformation_amp * gs.cos(7 * angle)
        else:
            x1 = gs.sin(angle)
            x2 = gs.cos(angle)
            x3 = gs.sin(2 * angle)
            x4 = deformation_amp * gs.cos(2 * angle)
            x5 = deformation_amp * gs.sin(3 * angle)
            x6 = deformation_amp * gs.cos(3 * angle)
            x7 = deformation_amp * gs.sin(4 * angle)
            x8 = deformation_amp * gs.cos(4 * angle)
            x9 = deformation_amp * gs.sin(5 * angle)
            x10 = deformation_amp * gs.cos(5 * angle)

        point = gs.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])[:embedding_dim]
        point = gs.squeeze(point, axis=-1)
        point = rotate_translate(point, translation, rotation)
        return point

    return immersion


def get_s2_high_immersion(radius, deformation_amp, embedding_dim, translation, rotation):
    """
    Returns the immersion of the manifold where the s2_high dataset concentrates. The immersion parameterizes
    a deformed sphere embedded in a possibly higher-dimensional space.

    Args:
        radius (float): Radius of the underlying sphere.
        deformation_amp (float): Amplitude of the deformation applied to the base sphere.
        embedding_dim (int): Dimension of the embedding space.
        translation (torch.Tensor): Translation vector applied after the embedding.
        rotation (torch.Tensor): Rotation matrix applied after the embedding.

    Returns:
        Function: A function mapping spherical coordinates (theta, phi) to the deformed high-dimensional embedding.
    """

    def immersion(angle_pair):
        theta, phi = angle_pair

        amplitude = (
                1 + 0.5 * deformation_amp * (gs.exp(-5 * theta ** 2)
                                             + gs.exp(-5 * (theta - gs.pi) ** 2))
        )

        # Base S^2 coordinates in R^3
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)

        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        if embedding_dim > 5:
            for i in range(2):
                point[3 + i] = gs.sin((i + 1) * theta) * gs.cos((i + 2) * phi)

        point = rotate_translate(point, translation, rotation)
        return point

    return immersion


def get_t2_high_immersion(major_radius, minor_radius, embedding_dim, deformation_amp, translation, rotation):
    """
    Returns the immersion of the manifold where the t2_high dataset concentrates. The immersion parameterizes
    a deformed torus embedded in a possibly higher-dimensional space.

    Args:
        major_radius (float): Major radius of the underlying torus.
        minor_radius (float): Minor radius of the underlying torus.
        embedding_dim (int): Dimension of the embedding space.
        deformation_amp (float): Amplitude of the deformation.
        translation (torch.Tensor): Translation vector applied after the embedding.
        rotation (torch.Tensor): Rotation matrix applied after the embedding.

    Returns:
        Function: A function mapping toroidal coordinates (theta, phi) to the deformed high-dimensional embedding.
    """

    def immersion(angle_pair):
        theta, phi = angle_pair

        amplitude1 = (
                1 + 0.5 * (
                deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi) ** 2)
        )
        )
        amplitude2 = (
                deformation_amp * gs.exp(-2 * (phi - gs.pi / 2) ** 2) * gs.exp(-2 * (theta - gs.pi / 2) ** 2)
                + deformation_amp * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2) * gs.exp(-2 * (theta - 3 * gs.pi / 2) ** 2)
        )

        # Standard 3D torus coordinates
        x = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)

        point = amplitude1 * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)

        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        if embedding_dim > 7:
            point[3] = amplitude2
        for i in range(2):
            point[4 + i] = gs.sin((i + 1) * theta) * gs.cos((i + 2) * phi)

        point = rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_s1_low(rotation, n_points, radius, n_wiggles, deformation_amp,
                embedding_dim, noise_var, deformation_type="wiggles",
                random_seed=42):
    """
    Generate a dataset lying close to "wiggly" circle with noise.
    The dataset lies in a two-dimensional affine subspace of the potentially high-dimensional embedding space.

    Args:
        rotation (str): "random" for a random rotation.
        n_points (int): Number of sample points along the circle.
        radius (float): Radius of the base circle.
        n_wiggles (int): Number of oscillations in the deformation.
        deformation_amp (float): Amplitude of the deformation.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        noise_var (float): Variance of isotropic Gaussian noise.
        deformation_type (str, optional): Type of geodesic deformation.
            Options are "wiggles" or "bump". Defaults to "wiggles".
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with one column ``angles``
              containing intrinsic circle angles.

    Raises:
        NotImplementedError: If an unsupported deformation type is provided to
            the immersion constructor.
    """
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    immersion = get_s1_low_immersion(
        deformation_type=deformation_type,
        radius=radius,
        n_wiggles=n_wiggles,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        rot=rot
    )
    angles = gs.linspace(0, 2 * gs.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points,))
        data = data + radius * noise
    labels = pd.DataFrame({"angles": angles})
    return data, labels


def load_s2_low(rotation, n_points, radius, deformation_amp,
                embedding_dim, noise_var, random_seed=42):
    """
        Generate a dataset lying close to "onion-shaped" sphere with noise.
        The dataset lies in a three-dimensional affine subspace of the potentially high-dimensional embedding space.

    Args:
        rotation (str): "random" for a random rotation.
        n_points (int): Total number of points on the sphere. Must be a perfect square.
        radius (float): Radius of the base sphere.
        deformation_amp (float): Amplitude of the geodesic deformation.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        noise_var (float): Variance of isotropic Gaussian noise.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with columns ``thetas`` and
              ``phis`` for spherical coordinates of each sample.
    """
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    immersion = get_s2_low_immersion(radius, deformation_amp, embedding_dim, rot)
    sqrt_n_points = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(0.01, gs.pi - eps, sqrt_n_points)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return data, labels


def load_t2_low(rotation, n_points, major_radius, minor_radius,
                deformation_amp, embedding_dim, noise_var,
                random_seed=42):
    """
        Generate a dataset lying close to "squished" torus with noise.
        The dataset lies in a three-dimensional affine subspace of the potentially high-dimensional embedding space.

    Args:
        rotation (str): "random" for a random rotation.
        n_points (int): Total number of points on the torus. Must be a perfect square.
        major_radius (float): Major radius of the torus.
        minor_radius (float): Minor radius of the torus.
        deformation_amp (float): Amplitude of the geodesic deformation.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        noise_var (float): Variance of isotropic Gaussian noise.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with columns ``thetas`` and
              ``phis`` for intrinsic toroidal angles.
    """
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    immersion = get_t2_low_immersion(
        major_radius, minor_radius, deformation_amp, embedding_dim, rot
    )
    sqrt_n_points = int(gs.sqrt(n_points))
    thetas = gs.linspace(0, 2 * gs.pi, sqrt_n_points)
    phis = gs.linspace(0, 2 * gs.pi, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return data, labels


def load_s1_high(n_points, noise_var, deformation_amp, embedding_dim,
                 translation, rotation, random_seed=42):
    """
        Generate a dataset lying close to "scrunchy-like" circle with noise.
        The dataset lies in a nine-dimensional affine subspace of the embedding space R^10,
        with non-vanishing extrinsic curvature along nine dimensions.

    Args:
        n_points (int): Number of sample points along the circle.
        noise_var (float): Variance of isotropic Gaussian noise.
        deformation_amp (float): Amplitude of the deformation.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        translation (Union[str, torch.Tensor]): Either "random" to draw a random
            translation vector or a specific torch.Tensor of shape (embedding_dim,).
        rotation (Union[str, torch.Tensor]): Either "random" for a random rotation
            or a specific rotation matrix.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with one column ``angles`` for
              intrinsic circle angles.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)
    immersion = get_s1_high(
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        translation=trans,
        rotation=rot
    )
    angles = torch.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points,))
        data = data + noise
    labels = pd.DataFrame({"angles": angles})
    return data, labels


def load_s2_high(n_points, radius, noise_var, embedding_dim,
                 deformation_amp, translation, rotation,
                 random_seed=42):
    """
        Generate a dataset lying close to deformed sphere with noise.
        The dataset lies in a five-dimensional affine subspace of the embedding space R^10,
        with non-vanishing extrinsic curvature along five dimensions.

    Args:
        n_points (int): Total number of sample points on the sphere.
            Must be a perfect square.
        radius (float): Radius of the base sphere.
        noise_var (float): Variance of isotropic Gaussian noise.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        deformation_amp (float): Amplitude of the deformation.
        translation (Optional[Union[str, torch.Tensor]]): Either None, "random"
            for a random translation, or a fixed translation vector.
        rotation (Union[str, torch.Tensor]): Either "random" for a random SO(N)
            rotation or a specific rotation matrix.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with columns ``thetas`` and
              ``phis`` for spherical coordinates of each sample.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)
    rot = torch.eye(embedding_dim)
    trans = torch.zeros(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)

    immersion = get_s2_high_immersion(
        radius=radius,
        embedding_dim=embedding_dim,
        deformation_amp=deformation_amp,
        translation=trans,
        rotation=rot
    )
    sqrt_n_points = int(gs.sqrt(n_points))
    thetas = gs.arccos(np.linspace(0.99, -0.99, sqrt_n_points))
    phis = gs.linspace(0.01, 2 * np.pi - 0.01, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return data, labels


def load_t2_high(n_points, major_radius, minor_radius, noise_var,
                 embedding_dim, deformation_amp,
                 translation="random", rotation="random", random_seed=42):
    """
        Generate a dataset lying close to deformed torus with noise.
        The dataset lies in a five-dimensional affine subspace of the embedding space R^10,
        with non-vanishing extrinsic curvature along five dimensions.

    Args:
        n_points (int): Total number of sample points on the torus.
            Must be a perfect square.
        major_radius (float): Major radius of the torus.
        minor_radius (float): Minor radius of the torus.
        noise_var (float): Variance of isotropic Gaussian noise.
        embedding_dim (int): Dimension of the Euclidean embedding space.
        deformation_amp (float): Amplitude of the deformation.
        translation (Union[str, torch.Tensor], optional): Either "random" to
            generate a random translation vector or a specific vector. Defaults to "random".
        rotation (Union[str, torch.Tensor], optional): Either "random" to
            generate a random SO(N) rotation or a specific rotation matrix. Defaults to "random".
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor, pandas.DataFrame]:
            * data (torch.Tensor): Array of shape (n_points, embedding_dim)
              containing noisy embedded coordinates.
            * labels (pandas.DataFrame): DataFrame with columns ``thetas`` and
              ``phis`` for intrinsic toroidal angles.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)
    immersion = get_t2_high_immersion(
        major_radius=major_radius,
        minor_radius=minor_radius,
        embedding_dim=embedding_dim,
        deformation_amp=deformation_amp,
        translation=trans,
        rotation=rot
    )
    sqrt_n_points = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_n_points)
    angle_grid = torch.cartesian_prod(thetas, phis)
    data = torch.stack([immersion(pair) for pair in angle_grid])
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_n_points ** 2,))
        data = data + noise
    labels = pd.DataFrame({"thetas": angle_grid[:, 0], "phis": angle_grid[:, 1]})
    return data, labels

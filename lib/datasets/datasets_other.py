"""
Synthetic dataset generators for geometric manifolds.

This module provides high-dimensional dataset generators compatible with
Geomstats (PyTorch backend). It contains immersion functions and data
loaders for various manifolds (e.g., scrunchy curves, interlocked tori,
nested spheres).
"""

import os
import pandas as pd
import numpy as np
from skimage.measure import marching_cubes

# Set Geomstats backend before importing torch/geomstats
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import torch  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal  # noqa: E402
import geomstats.backend as gs  # noqa: E402
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402

from .utils import compute_frenet_frame, embedd, embedd_rotate_translate, rotate_translate
from .datasets_main import get_s1_high, get_s2_high_immersion, get_t2_high_immersion


def get_scrunchy_immersion(radius, n_wiggles, deformation_amp, embedding_dim, rot):
    """
    Returns the immersion of the manifold where the scrunchy dataset concentrates.
    The immersion parameterizes a deformed circle (S¹) embedded in R^n.

    Args:
        radius (float): Radius of the underlying circle.
        n_wiggles (int): Number of oscillations in the deformation.
        deformation_amp (float): Amplitude of the deformation along the z-axis.
        embedding_dim (int): Dimension of the embedding space.
        rot (torch.Tensor): Rotation matrix.

    Returns:
        Function: Function that maps an angle to the embedding space with the specified deformation.
    """

    def immersion(angle):
        x = radius * gs.cos(angle)
        y = radius * gs.sin(angle)
        z = deformation_amp * gs.cos(n_wiggles * angle)
        point = gs.squeeze(gs.array([x, y, z]), axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])
        return gs.einsum("ij,j->i", rot, point)

    return immersion


def load_scrunchy(rotation, n_points=1500, radius=1.0, n_wiggles=6,
                  deformation_amp=0.4, embedding_dim=10,
                  noise_var=0.01, random_seed=42):
    """
    Loads the scrunchy dataset.

    This dataset consists of points sampled along a deformed circle (S¹)
    embedded in R^n, where the deformation introduces oscillations
    along the z-axis.

    Args:
        rotation (str): "random" for a random rotation or any other value for the identity.
        n_points (int): Number of sampled points.
        radius (float): Radius of the underlying circle.
        n_wiggles (int): Number of oscillations in the deformation.
        deformation_amp (float): Amplitude of the deformation.
        embedding_dim (int): Dimension of the embedding space.
        noise_var (float): Variance of Gaussian noise added to the data.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            DataFrame of shape (n_points, embedding_dim) with the sampled data,
            and DataFrame of shape (n_points, 1) with the corresponding angles.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_scrunchy_immersion(
        radius=radius,
        n_wiggles=n_wiggles,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    angles = _get_equal_arc_length_angles(immersion, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points,))
        data = data + radius * noise

    return pd.DataFrame(data.numpy()), pd.DataFrame({"angles": angles.numpy()})


def generate_torus(n_points=5000, major_radius=5.0, minor_radius=1.0,
                   filled=False, noise_var=0.01, embedding_dim=3,
                   rotation=None, random_seed=42):
    """
    Returns a set of points sampled on or inside a torus embedded in R^n.

    Args:
        n_points (int): Number of sampled points.
        major_radius (float): Major radius of the torus (distance from center of tube to center of torus).
        minor_radius (float): Minor radius of the torus (radius of the tube).
        filled (bool): If True, sample uniformly inside the torus volume; if False, sample only on the surface.
        noise_var (float): Variance of Gaussian noise added to the sampled points.
        embedding_dim (int): Dimension of the embedding space.
        rotation (str or None): "random" for a random rotation or None for the identity.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            DataFrame of shape (n_points, embedding_dim) with sampled coordinates,
            and DataFrame with the corresponding angular parameters
            (theta, phi[, rho] depending on filled).
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if filled:
        theta = 2 * torch.pi * torch.rand(n_points)
        phi = 2 * torch.pi * torch.rand(n_points)
        rho = minor_radius * torch.sqrt(torch.rand(n_points))

        x = (major_radius + rho * torch.cos(phi)) * torch.cos(theta)
        y = (major_radius + rho * torch.cos(phi)) * torch.sin(theta)
        z = rho * torch.sin(phi)
        angles = torch.stack((theta, phi, rho), dim=1)
    else:
        theta = 2 * torch.pi * torch.rand(n_points)
        phi = 2 * torch.pi * torch.rand(n_points)

        x = (major_radius + minor_radius * torch.cos(phi)) * torch.cos(theta)
        y = (major_radius + minor_radius * torch.cos(phi)) * torch.sin(theta)
        z = minor_radius * torch.sin(phi)
        angles = torch.stack((theta, phi), dim=1)

    points = torch.stack((x, y, z), dim=1)

    # Apply rotation (no translation)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
        points = torch.stack([gs.einsum("ij,j->i", rot, p) for p in points])
    elif embedding_dim > 3:
        # If embedding_dim > 3 but no rotation requested, embed by zero-padding
        points = embedd(points, embedding_dim)

    # Add noise
    noise = noise_var * torch.randn_like(points) * minor_radius
    points = points + noise

    return pd.DataFrame(points.numpy()), pd.DataFrame(angles.numpy())


def load_wiggling_tube(n_phi, n_theta, minor_radius, noise_var,
                       wiggling_dim, embedding_dim, deformation_amp,
                       rotation="random", random_seed=42):
    """
    Loads the wiggling tube dataset.

    This dataset consists of a tubular neighborhood of a deformed S¹ curve
    embedded in R^n. The base curve is a high-dimensional S¹ with oscillations,
    and the tube is formed by circles around this curve using its Frenet frame.

    Args:
        n_phi (int): Number of sample points along the base curve.
        n_theta (int): Number of sample points along the circular cross-section.
        minor_radius (float): Radius of the tubular cross-section.
        noise_var (float): Variance of Gaussian noise added to the data.
        wiggling_dim (int): Dimension of the base curve embedding.
        embedding_dim (int): Dimension of the final embedding space.
        deformation_amp (float): Amplitude of the deformation of the base curve.
        rotation (str, optional): "random" for a random rotation or any other value for identity. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            DataFrame of shape (n_phi * n_theta, embedding_dim) with sampled coordinates,
            and DataFrame of shape (n_phi * n_theta, 2) containing (theta, phi) parameters.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    phis = torch.linspace(0, 2 * torch.pi, n_phi, requires_grad=True)
    thetas = torch.linspace(0, 2 * torch.pi, n_theta, requires_grad=True)

    # Base curve in wiggling_dim dimensions (no translation)
    base_rot = torch.eye(wiggling_dim)
    curve = get_s1_high(
        deformation_amp=deformation_amp,
        embedding_dim=wiggling_dim,
        translation=torch.zeros(wiggling_dim),
        rotation=base_rot,
    )

    data = []
    angles = []  # store corresponding (theta, phi) pairs

    for phi in phis:
        frame, _, _ = compute_frenet_frame(
            curve, phi, wiggling_dim,
            deformation_amp=deformation_amp,
            is_s1_high=True
        )
        e1 = frame[:, 1]
        e2 = frame[:, 2]
        center = curve(phi)

        for theta in thetas:
            offset = minor_radius * torch.cos(theta) * e1 + minor_radius * torch.sin(theta) * e2
            point = center + offset
            data.append(point)
            angles.append((theta.item(), phi.item()))

    data = torch.stack(data).detach()

    # Embed to higher dimension if needed and apply rotation
    if embedding_dim > wiggling_dim:
        rot = torch.eye(embedding_dim)
        if rotation == "random":
            rot = SpecialOrthogonal(n=embedding_dim).random_point()
        data = torch.stack([gs.einsum("ij,j->i", rot, p) for p in embedd(data, embedding_dim)])

    # Add Gaussian noise
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=minor_radius * noise_var * torch.eye(embedding_dim),
        ).sample((n_phi * n_theta,))
        data = data + noise

    # Sort by angles for consistency
    angles_np = np.array(angles)
    sort_idx = np.lexsort((angles_np[:, 1], angles_np[:, 0]))
    data = data[sort_idx]
    angles = angles_np[sort_idx]

    return pd.DataFrame(data.numpy()), pd.DataFrame(angles, columns=["theta", "phi"])


def load_interlocked_tori(n_points, major_radius, minor_radius, noise_var,
                          embedding_dim, deformation_amp,
                          rotation="random", random_seed=42):
    """
    Loads the interlocked tori dataset.

    This dataset consists of two tori embedded in R^n that are orthogonally
    interlocked. One torus lies in the xy-plane, and the second torus is rotated
    by 90 degrees around the x-axis.

    Args:
        n_points (int): Total number of sample points across both tori.
        major_radius (float): Major radius of each torus (distance from tube center to torus center).
        minor_radius (float): Minor radius of each torus (radius of the tube).
        noise_var (float): Variance of Gaussian noise added to the sampled points.
        embedding_dim (int): Dimension of the embedding space.
        deformation_amp (float): Amplitude of the deformation applied to each torus.
        rotation (str, optional): "random" for a random rotation or any other value for identity. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, Tuple[pd.Series, pd.DataFrame]]:
            DataFrame of shape (2 * n_points_per_torus, embedding_dim) with the sampled coordinates,
            and a tuple containing:
                - pd.Series of integers {0, 1} identifying the torus of each sample,
                - DataFrame of the corresponding (theta, phi) parameters.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Base torus immersion (no translation)
    base_rot = torch.eye(embedding_dim)
    immersion = get_t2_high_immersion(
        major_radius=major_radius,
        minor_radius=minor_radius,
        embedding_dim=embedding_dim,
        deformation_amp=deformation_amp,
        translation=torch.zeros(embedding_dim),
        rotation=base_rot,
    )

    # Grid of angles
    sqrt_npoints = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(eps, 2 * gs.pi - eps, sqrt_npoints)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_npoints)
    angle_grid = torch.cartesian_prod(thetas, phis)

    # First torus (xy-plane)
    torus1 = torch.stack([immersion(pair) for pair in angle_grid])

    # Second torus rotated 90 degrees around x-axis
    R_x90 = torch.eye(embedding_dim)
    R_x90[1, 1], R_x90[1, 2] = 0.0, -1.0
    R_x90[2, 1], R_x90[2, 2] = 1.0, 0.0
    torus2 = torch.stack([immersion(pair) for pair in angle_grid]) @ R_x90.T

    # Apply optional global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    torus1 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in torus1])
    torus2 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in torus2])

    # Add Gaussian noise
    if noise_var > 0:
        noise1 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_npoints ** 2,))
        noise2 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_npoints ** 2,))
        torus1 = torus1 + noise1
        torus2 = torus2 + noise2

    # Combine both tori
    data = torch.cat([torus1, torus2])
    angles = torch.cat([angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    entity_index = torch.tensor([0] * n_samples + [1] * n_samples, dtype=torch.long)

    return (
        pd.DataFrame(data.numpy()),
        (pd.Series(entity_index.numpy(), name="torus_index"),
         pd.DataFrame(angles.numpy(), columns=["theta", "phi"]))
    )


def load_interlocked_tubes(n_phi, n_theta, minor_radius, noise_var,
                           wiggling_dim, embedding_dim, deformation_amp,
                           rotation="random", random_seed=42):
    """
    Loads the interlocked tubes dataset.

    This dataset consists of two wiggling tubes embedded in R^n that are
    orthogonally interlocked. Each tube is a tubular neighborhood of a
    deformed S¹ curve, with the second tube rotated by 90 degrees.

    Args:
        n_phi (int): Number of sample points along the base curves.
        n_theta (int): Number of sample points along the circular cross-section.
        minor_radius (float): Radius of the tubular cross-section.
        noise_var (float): Variance of Gaussian noise added to the data.
        wiggling_dim (int): Dimension of the base curve embedding.
        embedding_dim (int): Dimension of the final embedding space.
        deformation_amp (float): Amplitude of the deformation of the base curves.
        rotation (str, optional): "random" for a random global rotation or
            any other value for identity. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, Tuple[pd.Series, pd.DataFrame]]:
            DataFrame of shape (2 * n_phi * n_theta, embedding_dim) with sampled coordinates,
            and a tuple containing:
                - pd.Series of integers {0, 1} identifying the tube of each sample,
                - DataFrame of the corresponding (theta, phi) parameters.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Generate two wiggling tubes independently
    tube1, angles1 = load_wiggling_tube(
        n_phi=n_phi,
        n_theta=n_theta,
        minor_radius=minor_radius,
        noise_var=noise_var,
        wiggling_dim=wiggling_dim,
        embedding_dim=embedding_dim,
        deformation_amp=deformation_amp,
        rotation="random",
        random_seed=random_seed
    )
    tube2, angles2 = load_wiggling_tube(
        n_phi=n_phi,
        n_theta=n_theta,
        minor_radius=minor_radius,
        noise_var=noise_var,
        wiggling_dim=wiggling_dim + 1,
        embedding_dim=embedding_dim,
        deformation_amp=deformation_amp,
        rotation="random",
        random_seed=random_seed
    )

    # Rotate the second tube by 90 degrees around the x-axis
    R_x90 = torch.eye(embedding_dim)
    R_x90[1, 1], R_x90[1, 2] = 0.0, -1.0
    R_x90[2, 1], R_x90[2, 2] = 1.0, 0.0
    tube2 = tube2 @ R_x90.T

    # Optional global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    tube1 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in tube1])
    tube2 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in tube2])

    n_points_per_tube = angles1.shape[0]

    # Add Gaussian noise
    if noise_var > 0:
        noise1 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points_per_tube,))
        noise2 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((n_points_per_tube,))
        tube1 = tube1 + noise1
        tube2 = tube2 + noise2

    # Combine tubes and labels
    data = torch.cat([tube1, tube2])
    angles = torch.cat([angles1, angles2])
    entity_index = torch.tensor(
        [0] * n_points_per_tube + [1] * n_points_per_tube,
        dtype=torch.long
    )

    return (
        pd.DataFrame(data.numpy()),
        (pd.Series(entity_index.numpy(), name="tube_index"),
         pd.DataFrame(angles.numpy(), columns=["theta", "phi"]))
    )


def generate_entangled_tori(n_points=5000, major_radius=5.0, minor_radius=1.0,
                            filled1=False, filled2=False,
                            noise_var=0.01, embedding_dim=3,
                            rotation="random", random_seed=42):
    """
    Generates a dataset of two entangled tori.

    This dataset consists of two tori embedded in R^n. One torus lies
    in the xy-plane, and the second torus is rotated by 90 degrees
    around the x-axis. Each torus can optionally be filled or represented
    only as a surface.

    Args:
        n_points (int): Total number of sample points across both tori.
        major_radius (float): Major radius of each torus.
        minor_radius (float): Minor radius of each torus.
        filled1 (bool): Whether to fill the first torus volume. Default is False.
        filled2 (bool): Whether to fill the second torus volume. Default is False.
        noise_var (float): Variance of Gaussian noise added to the data.
        embedding_dim (int): Dimension of the embedding space.
        rotation (str, optional): "random" for a random global rotation or
            any other value for identity. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - DataFrame of shape (n_points, embedding_dim) containing sampled coordinates.
            - Series of integers {0, 1} indicating the torus of each sample.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    n1 = n_points // 2
    n2 = n_points - n1

    # Generate individual tori
    torus1, _ = generate_torus(
        n1, major_radius, minor_radius,
        filled=filled1, noise_var=noise_var,
        embedding_dim=embedding_dim,
        rotation=None, random_seed=random_seed
    )
    torus2, _ = generate_torus(
        n2, major_radius, minor_radius,
        filled=filled2, noise_var=noise_var,
        embedding_dim=embedding_dim,
        rotation=None, random_seed=random_seed
    )

    # Rotate the second torus by 90 degrees around the x-axis
    R_x90 = torch.eye(embedding_dim)
    R_x90[1, 1], R_x90[1, 2] = 0.0, -1.0
    R_x90[2, 1], R_x90[2, 2] = 1.0, 0.0
    torus2 = torus2 @ R_x90.T

    # Optional global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    torus1 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in torus1])
    torus2 = torch.stack([gs.einsum("ij,j->i", rot, p) for p in torus2])

    # Combine both tori
    points = torch.cat([torus1, torus2], dim=0)

    labels = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long)
    ])

    return pd.DataFrame(points.numpy()), pd.Series(labels.numpy(), name="torus_index")


def load_n_torus(n_points=1000, n=2, radii=None, random_seed=42):
    """
    Loads a dataset of points sampled from an n-dimensional torus.

    The n-torus is represented as the product of n circles,
    each with a specified radius. Each point is parameterized by
    n independent angles, and mapped into R^{2n} as pairs of
    (r_i cos θ_i, r_i sin θ_i).

    Args:
        n_points (int): Number of points to sample from the torus.
        n (int): Intrinsic dimension of the torus (number of S¹ factors).
        radii (array-like of float, optional): Radii of each S¹ factor.
            If None, all radii are set to 1. Default is None.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, 2n) containing the sampled coordinates.
            - DataFrame of shape (n_points, n) containing the angular parameters θ_i.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Radii specification
    if radii is None:
        radii = np.ones(n)
    else:
        radii = np.asarray(radii, dtype=float)
        assert len(radii) == n, "Length of radii must match torus dimension n."

    # Sample n independent angles uniformly
    thetas = 2 * np.pi * np.random.rand(n_points, n)

    # Map each angle θ_i to (r_i cos θ_i, r_i sin θ_i)
    coords = []
    for i in range(n):
        ri = radii[i]
        theta_i = thetas[:, i]
        xi = ri * np.cos(theta_i)
        yi = ri * np.sin(theta_i)
        coords.append(xi)
        coords.append(yi)

    # Stack into shape (n_points, 2n)
    data = np.stack(coords, axis=1).T  # shape (2n, n_points)
    data = data.T  # shape (n_points, 2n)

    return (
        pd.DataFrame(data),
        pd.DataFrame(thetas, columns=[f"theta_{i + 1}" for i in range(n)])
    )


def _torus_implicit_field(x, y, z, major_radius, minor_radius):
    """
    Computes the implicit function value of a 3D torus.

    This function evaluates the standard implicit equation of a torus
    with major radius R and minor radius r in R^3:
        f(x, y, z) = (x² + y² + z² + R² - r²)² - 4 R² (x² + y²)

    Args:
        x (torch.Tensor): x-coordinates of the query points.
        y (torch.Tensor): y-coordinates of the query points.
        z (torch.Tensor): z-coordinates of the query points.
        major_radius (float): Major radius R of the torus (distance from center of tube to center of torus).
        minor_radius (float): Minor radius r of the torus (radius of the tube).

    Returns:
        torch.Tensor:
            Value of the implicit torus field at the given points.
            The torus surface is the zero level set of this function.
    """
    R, r = major_radius, minor_radius
    return (
            (x ** 2 + y ** 2 + z ** 2) ** 2
            - 2 * (R ** 2 + r ** 2) * (x ** 2 + y ** 2)
            + 2 * (R ** 2 - r ** 2) * z ** 2
            + (R ** 2 - r ** 2) ** 2
    )


def _genus3_field(x, y, z, n=3, major_radius=1.0, minor_radius=0.25):
    """
    Computes the implicit scalar field of a genus-3 surface.

    The genus-3 surface is constructed as the product of n small tori
    arranged evenly in a circle of radius 1.5 in the xy-plane. The
    implicit function is the product of the implicit functions of each
    constituent torus:
        f(x,y,z) = ∏_{k=1}^n f_torus(x - s_x[k], y - s_y[k], z; R, r)  − 10

    Args:
        x (torch.Tensor): x-coordinates of query points.
        y (torch.Tensor): y-coordinates of query points.
        z (torch.Tensor): z-coordinates of query points.
        n (int, optional): Number of tori composing the genus-3 surface.
            Default is 3.
        major_radius (float, optional): Major radius R of each torus.
            Default is 1.0.
        minor_radius (float, optional): Minor radius r of each torus.
            Default is 0.25.

    Returns:
        torch.Tensor:
            Value of the implicit genus-3 field at the given points.
            The genus-3 surface corresponds to the zero level set.
    """
    angles = torch.linspace(0, 2 * np.pi, n + 1, device=x.device)[:-1]
    shifts_x = 1.5 * torch.cos(angles)
    shifts_y = 1.5 * torch.sin(angles)

    f = torch.ones_like(x)
    for sx, sy in zip(shifts_x, shifts_y):
        xi = x - sx
        yi = y - sy
        f *= _torus_implicit_field(xi, yi, z, major_radius, minor_radius)

    return f - 10


def generate_genus3(n_points=5000, major_radius=1.0, minor_radius=0.25, noise_var=0.01, embedding_dim=3, rotation=None,
                    random_seed=42):
    """
    Generates a point cloud sampled from a genus-3 surface embedded in R^n.

    The genus-3 surface is constructed as the zero level set of the product
    of three small tori arranged in a circle in the xy-plane. Points are
    sampled from a dense marching-cubes surface reconstruction of the implicit
    field.

    Args:
        n_points (int, optional): Number of sampled points. Default is 5000.
        major_radius (float, optional): Major radius R of each torus. Default is 1.0.
        minor_radius (float, optional): Minor radius r of each torus. Default is 0.25.
        noise_var (float, optional): Variance of Gaussian noise added to the points. Default is 0.01.
        embedding_dim (int, optional): Dimension of the embedding space. Default is 3.
        rotation (torch.Tensor or str, optional): Rotation matrix or "random" to apply
            after embedding. Default is None (no rotation).
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - points (torch.Tensor): Sampled points of shape (n_points, embedding_dim).
            - labels (torch.Tensor): Dummy labels (all zeros) of shape (n_points,).
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Create dense grid
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(-3, 3, 100),
        torch.linspace(-3, 3, 100),
        torch.linspace(-0.5, 0.5, 50),
        indexing="ij"
    )
    vals = _genus3_field(grid_x, grid_y, grid_z, n=3,
                         major_radius=major_radius,
                         minor_radius=minor_radius)
    vals_np = vals.cpu().numpy()

    # Extract surface using marching cubes
    verts, faces, normals, _ = marching_cubes(vals_np, level=0)

    # Rescale vertices to world coordinates
    verts = torch.tensor(verts.copy(), dtype=torch.float32)
    verts[:, 0] = verts[:, 0] / 99 * 6 - 3
    verts[:, 1] = verts[:, 1] / 99 * 6 - 3
    verts[:, 2] = verts[:, 2] / 49 * 1 - 0.5

    # Sample points from vertices
    idx = torch.randint(0, verts.shape[0], (n_points,))
    points = verts[idx]

    # Optional rotation and embedding
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    points = torch.einsum("ij,nj->ni", rot, embedd(points, embedding_dim))

    # Add Gaussian noise
    if noise_var > 0:
        noise = noise_var * torch.randn_like(points) * minor_radius
        points = points + noise

    labels = torch.zeros(n_points, dtype=torch.long)
    return points, labels


def load_nested_spheres(
        n_points,
        major_radius,
        mid_radius,
        minor_radius,
        noise_var,
        embedding_dim,
        deformation_amp,
        rotation="random",
        random_seed=42
):
    """
    Generates three nested spherical point clouds (inner, middle, outer) with optional deformation
    and noise, embedded in R^n.

    Args:
        n_points (int): Total number of points to sample across all three spheres.
        major_radius (float): Radius of the outer sphere.
        mid_radius (float): Radius of the middle sphere.
        minor_radius (float): Radius of the inner sphere.
        noise_var (float): Variance of Gaussian noise added to the data.
        embedding_dim (int): Dimension of the embedding space.
        deformation_amp (float): Amplitude of the deformation applied to each sphere.
        rotation (torch.Tensor or str, optional): Rotation matrix to apply to each sphere or
            "random" to generate a random rotation. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (N, embedding_dim) with sampled coordinates.
            - DataFrame of shape (N, 3) with columns ["sphere_index","theta","phi"],
              where N = 3 * sqrt(n_points)^2.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Set rotations
    if rotation == "random":
        rot_inner = SpecialOrthogonal(n=embedding_dim).random_point()
        rot_mid = SpecialOrthogonal(n=embedding_dim).random_point()
        rot_outer = SpecialOrthogonal(n=embedding_dim).random_point()
    else:
        rot_inner = torch.eye(embedding_dim)
        rot_mid = torch.eye(embedding_dim)
        rot_outer = torch.eye(embedding_dim)

    # Immersions for each sphere (no translation)
    immersion_inner = get_s2_high_immersion(
        radius=minor_radius,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        translation=torch.zeros(embedding_dim),
        rotation=rot_inner,
    )
    immersion_mid = get_s2_high_immersion(
        radius=mid_radius,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        translation=torch.zeros(embedding_dim),
        rotation=rot_mid,
    )
    immersion_outer = get_s2_high_immersion(
        radius=major_radius,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        translation=torch.zeros(embedding_dim),
        rotation=rot_outer,
    )

    # Angular sampling
    sqrt_npoints = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.arccos(np.linspace(0.99, -0.99, sqrt_npoints))  # uniform on sphere
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_npoints)
    angle_grid = torch.cartesian_prod(thetas, phis)

    # Sample points on each sphere
    sphere_inner = torch.stack([immersion_inner(pair) for pair in angle_grid])
    sphere_mid = torch.stack([immersion_mid(pair) for pair in angle_grid])
    sphere_outer = torch.stack([immersion_outer(pair) for pair in angle_grid])

    # Concatenate data
    data = torch.cat([sphere_inner, sphere_mid, sphere_outer])
    angles = torch.cat([angle_grid, angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    sphere_index = torch.tensor(
        [0] * n_samples + [1] * n_samples + [2] * n_samples, dtype=torch.long
    )

    # Add Gaussian noise
    if noise_var > 0:
        noise1 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=minor_radius * noise_var * torch.eye(embedding_dim),
        ).sample((n_samples,))
        noise2 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=mid_radius * noise_var * torch.eye(embedding_dim),
        ).sample((n_samples,))
        noise3 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((n_samples,))
        noise = torch.cat([noise1, noise2, noise3])
        data = data + noise

    # Convert to pandas DataFrames
    df_data = pd.DataFrame(data.numpy())
    df_labels = pd.DataFrame({
        "sphere_index": sphere_index.numpy(),
        "theta": angles[:, 0].numpy(),
        "phi": angles[:, 1].numpy()
    })

    return df_data, df_labels


def get_sphere_high_dim_bump_immersion(
        radius,
        deformation_amp,
        bump_dim,
        bump_center,
        embedding_dim,
        rotation=None
):
    """
    Returns an immersion function for a high-dimensional sphere with localized bumps.

    This immersion maps spherical coordinates (theta, phi) to R^embedding_dim.
    The base surface is a 2-sphere of radius `radius` embedded in R^embedding_dim.
    Localized smooth bumps are added along specified embedding coordinates.

    Args:
        radius (float): Radius of the base 2-sphere.
        deformation_amp (float): Amplitude of the bump deformation.
        bump_dim (int or list[int]): Indices of the embedding coordinates
            along which bumps are added.
        bump_center (tuple[float, float] or list[tuple[float, float]]):
            Center angles (theta, phi) of the bumps. Must match the length of
            `bump_dim` if a list is provided.
        embedding_dim (int): Dimension of the embedding space.
        rotation (torch.Tensor or str, optional): Rotation matrix or "random"
            to apply a global rotation. Default is None (no rotation).

    Returns:
        function:
            A function `immersion(angle_pair)` that maps a pair (theta, phi)
            to a point in R^embedding_dim.
    """
    # Set rotation matrix
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    def immersion(angle_pair):
        theta, phi = angle_pair
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        # Embed base 3D sphere into higher dimension
        point = gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        point = embedd(point, embedding_dim)

        # Ensure bump_dims is a list
        bump_dims = bump_dim if isinstance(bump_dim, (list, tuple)) else [bump_dim]
        bump_centers = (
            [bump_center]
            if isinstance(bump_center, tuple)
               and len(bump_center) == 2
               and all(isinstance(x, (float, torch.Tensor)) for x in bump_center)
            else list(bump_center)
        )

        # Add smooth local bumps
        for dim, center in zip(bump_dims, bump_centers):
            theta_c, phi_c = center

            delta_theta = (theta - theta_c) % (2 * gs.pi)
            if delta_theta > gs.pi:
                delta_theta -= 2 * gs.pi
            delta_phi = (phi - phi_c) % (2 * gs.pi)
            if delta_phi > gs.pi:
                delta_phi -= 2 * gs.pi

            if abs(theta - theta_c) < 1 and abs(phi - phi_c) < 1:
                bump_theta = gs.exp(-1 / (1 - delta_theta ** 2))
                bump_phi = gs.exp(-1 / (1 - delta_phi ** 2))
                bump = radius * deformation_amp * bump_theta * bump_phi
            else:
                bump = 0

            point[dim] += bump

        # Apply global rotation
        point = gs.einsum("ij,j->i", rot, point)
        return point

    return immersion


def load_sphere_high_dim_bump(
        n_points,
        radius,
        noise_var,
        embedding_dim,
        deformation_amp,
        rotation="random",
        random_seed=42
):
    """
    Generates a high-dimensional sphere with smooth localized bumps.

    A 2-sphere of radius `radius` is embedded in R^embedding_dim and
    deformed by three smooth bumps centered at predefined angular locations.
    Optionally adds Gaussian noise and a global rotation.

    Args:
        n_points (int): Number of sample points on the sphere.
        radius (float): Radius of the base sphere.
        noise_var (float): Variance of Gaussian noise added to the samples.
        embedding_dim (int): Dimension of the embedding space.
        deformation_amp (float): Amplitude of the bump deformation.
        rotation (torch.Tensor or str, optional): Global rotation matrix or
            "random" to apply a random rotation. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, embedding_dim) with sampled coordinates.
            - DataFrame of shape (n_points, 2) with columns ["theta","phi"] for
              the spherical parameter values.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Bump specifications
    bump_dims = [embedding_dim - 3, embedding_dim - 2, embedding_dim - 1]
    bump_centers = [
        (torch.pi / 4, torch.pi / 2),
        (torch.pi / 4, 3 * torch.pi / 2),
        (torch.pi / 2, torch.pi / 2),
    ]

    # Base immersion with bumps (no translation)
    immersion = get_sphere_high_dim_bump_immersion(
        radius=radius,
        deformation_amp=deformation_amp,
        bump_dim=bump_dims,
        bump_center=bump_centers,
        embedding_dim=embedding_dim,
        rotation=None
    )

    # Sample angular grid for approximately uniform coverage
    sqrt_npoints = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.arccos(np.linspace(1 - eps, -1 + eps, sqrt_npoints))
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_npoints)
    angle_grid = torch.cartesian_prod(thetas, phis)

    # Evaluate immersion
    sphere = torch.stack([immersion(pair) for pair in angle_grid])

    # Add Gaussian noise
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((sphere.shape[0],))
        sphere = sphere + noise

    # Optional global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    data = torch.einsum("ij,nj->ni", rot, sphere)

    # Convert to pandas DataFrames
    df_data = pd.DataFrame(data.numpy())
    df_angles = pd.DataFrame({
        "theta": angle_grid[:, 0].numpy(),
        "phi": angle_grid[:, 1].numpy()
    })

    return df_data, df_angles


def load_nested_spheres_high_dim_bump(
        n_points,
        major_radius,
        mid_radius,
        minor_radius,
        noise_var,
        embedding_dim,
        deformation_amp,
        rotation="random",
        enclosing_sphere=False,
        random_seed=42
):
    """
    Generates three nested high-dimensional spheres with smooth localized bumps.

    Each sphere is a 2-sphere embedded in R^embedding_dim and deformed by a
    smooth bump along a specified embedding coordinate. Optionally adds Gaussian
    noise, applies a global rotation, and includes an additional enclosing sphere.

    Args:
        n_points (int): Total number of points to sample per nested sphere.
        major_radius (float): Radius of the outer sphere.
        mid_radius (float): Radius of the middle sphere.
        minor_radius (float): Radius of the inner sphere.
        noise_var (float): Variance of Gaussian noise added to all points.
        embedding_dim (int): Dimension of the embedding space.
        deformation_amp (float): Amplitude of the bump deformation.
        rotation (torch.Tensor or str, optional): Global rotation matrix or
            "random" to apply a random rotation. Default is "random".
        enclosing_sphere (bool, optional): If True, an additional enclosing
            hypersphere of radius 2*major_radius is generated. Default is False.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (N, embedding_dim) with sampled coordinates.
            - DataFrame of shape (N, 3) with columns ["sphere_index","theta","phi"].
              For enclosing sphere points, theta and phi contain dummy values.
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Bump specifications
    bump_dims = [embedding_dim - 3, embedding_dim - 2, embedding_dim - 1]
    bump_centers = [
        (torch.pi / 4, torch.pi / 2),
        (torch.pi / 4, 3 * torch.pi / 2),
        (torch.pi / 2, torch.pi / 2),
    ]

    # Base immersions (no translation, fixed initial rotation)
    immersion_inner = get_sphere_high_dim_bump_immersion(
        radius=minor_radius,
        deformation_amp=deformation_amp,
        bump_dim=bump_dims[0],
        bump_center=bump_centers[0],
        embedding_dim=embedding_dim,
        rotation=None
    )
    immersion_mid = get_sphere_high_dim_bump_immersion(
        radius=mid_radius,
        deformation_amp=deformation_amp,
        bump_dim=bump_dims[1],
        bump_center=bump_centers[1],
        embedding_dim=embedding_dim,
        rotation=None
    )
    immersion_outer = get_sphere_high_dim_bump_immersion(
        radius=major_radius,
        deformation_amp=deformation_amp,
        bump_dim=bump_dims[2],
        bump_center=bump_centers[2],
        embedding_dim=embedding_dim,
        rotation=None
    )

    # Sample angular grid
    sqrt_npoints = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.arccos(np.linspace(1 - eps, -1 + eps, sqrt_npoints))
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_npoints)
    angle_grid = torch.cartesian_prod(thetas, phis)

    # Generate nested spheres
    sphere_inner = torch.stack([immersion_inner(pair) for pair in angle_grid])
    sphere_mid = torch.stack([immersion_mid(pair) for pair in angle_grid])
    sphere_outer = torch.stack([immersion_outer(pair) for pair in angle_grid])

    data = torch.cat([sphere_inner, sphere_mid, sphere_outer])
    angles = torch.cat([angle_grid, angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    sphere_index = torch.tensor(
        [0] * n_samples + [1] * n_samples + [2] * n_samples,
        dtype=torch.long
    )

    # Add Gaussian noise
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((data.shape[0],))
        data = data + noise

    # Global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    data = torch.einsum("ij,nj->ni", rot, data)

    # Optional enclosing sphere
    if enclosing_sphere:
        n_extra = n_points * 2
        enclosing = torch.randn(n_extra, embedding_dim)
        enclosing = enclosing / enclosing.norm(dim=1, keepdim=True)
        enclosing = enclosing * (2 * major_radius)
        data = torch.cat([data, enclosing])
        enclosing_index = torch.full((n_extra,), 100, dtype=torch.long)
        dummy_angles = torch.full((n_extra, 2), 1.0)
        sphere_index = torch.cat([sphere_index, enclosing_index])
        angles = torch.cat([angles, dummy_angles])

    # Convert to pandas DataFrames
    df_data = pd.DataFrame(data.numpy())
    df_labels = pd.DataFrame({
        "sphere_index": sphere_index.numpy(),
        "theta": angles[:, 0].numpy(),
        "phi": angles[:, 1].numpy()
    })

    return df_data, df_labels


def get_clelia_immersion(r, c, embedding_dim, rotation=None):
    """
    Returns the immersion of the Clelia curve in high-dimensional space.

    The Clelia curve is a parametric curve on the 2-sphere defined by:
        x = r * sin(t) * cos(c * t)
        y = r * sin(t) * sin(c * t)
        z = r * cos(t)

    The 3D curve is embedded into R^embedding_dim and optionally rotated.

    Args:
        r (float): Radius of the base sphere.
        c (float): Frequency of the oscillation along the azimuthal direction.
        embedding_dim (int): Dimension of the embedding space.
        rotation (torch.Tensor or str, optional): Rotation matrix or
            "random" to apply a random rotation. Default is None (no rotation).

    Returns:
        Function:
            A function `immersion(angle)` that maps an angular parameter to a
            point in R^embedding_dim representing the Clelia curve.
    """
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    def immersion(angle):
        x = r * gs.sin(angle) * gs.cos(c * angle)
        y = r * gs.sin(angle) * gs.sin(c * angle)
        z = r * gs.cos(angle)

        point = gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        point = embedd(point, embedding_dim)
        point = gs.einsum("ij,j->i", rot, point)
        return point

    return immersion


def load_clelia_curve(
        n_points,
        r,
        c,
        noise_var,
        embedding_dim,
        rotation="random",
        random_seed=42
):
    """
    Generates samples of the Clelia curve embedded in a high-dimensional space.

    The Clelia curve is a parametric curve on the 2-sphere:
        x = r * sin(t) * cos(c * t)
        y = r * sin(t) * sin(c * t)
        z = r * cos(t)

    The 3D curve is embedded into R^embedding_dim and optionally rotated.
    Gaussian noise can be added to the samples.

    Args:
        n_points (int): Number of sample points along the curve.
        r (float): Radius of the base sphere.
        c (float): Frequency of the azimuthal oscillation.
        noise_var (float): Variance of the additive Gaussian noise.
        embedding_dim (int): Dimension of the embedding space.
        rotation (torch.Tensor or str, optional): Rotation matrix or "random"
            to apply a random rotation. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, embedding_dim) with sampled coordinates.
            - DataFrame of shape (n_points, 1) with the parameter values under column "angles".
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    # Immersion without translation
    immersion = get_clelia_immersion(r=r, c=c, embedding_dim=embedding_dim, rotation=rot)

    angles = gs.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    # Add Gaussian noise if specified
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((n_points,))
        data = data + noise

    return pd.DataFrame(data.numpy()), pd.DataFrame({"angles": angles.numpy()})


def get_8_curve_immersion(embedding_dim, rotation=None):
    """
    Returns the immersion of a figure-eight (∞) curve embedded in high-dimensional space.

    The 8-curve is defined in 3D by:
        x = sin(t)
        y = sin(2 t)
        z = 0.1 * sin(10 t)

    The curve is embedded into R^embedding_dim and optionally rotated.

    Args:
        embedding_dim (int): Dimension of the embedding space.
        rotation (torch.Tensor or str, optional): Rotation matrix or
            "random" to apply a random rotation. Default is None (no rotation).

    Returns:
        Function:
            A function `immersion(angle)` mapping an angular parameter to a
            point in R^embedding_dim representing the 8-curve.
    """
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    def immersion(angle):
        x = torch.sin(angle)
        y = torch.sin(2 * angle)
        z = 0.1 * torch.sin(10 * angle)

        point = torch.stack([x, y, z], dim=0)
        point = embedd(point, embedding_dim)
        point = torch.einsum("ij,j->i", rot, point)
        return point

    return immersion


def load_8_curve(
        n_points,
        noise_var,
        embedding_dim=3,
        rotation="random",
        random_seed=42
):
    """
    Generates samples of the figure-eight (∞) curve embedded in a high-dimensional space.

    The 8-curve is defined in 3D by:
        x = sin(t)
        y = sin(2 t)
        z = 0.1 * sin(10 t)

    The curve is embedded into R^embedding_dim and optionally rotated.
    Gaussian noise can be added to the samples.

    Args:
        n_points (int): Number of sample points along the curve.
        noise_var (float): Variance of the additive Gaussian noise.
        embedding_dim (int, optional): Dimension of the embedding space. Default is 3.
        rotation (torch.Tensor or str, optional): Rotation matrix or "random"
            to apply a random rotation. Default is "random".
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, embedding_dim) with sampled coordinates.
            - DataFrame of shape (n_points, 1) with the parameter values under column "angles".
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Determine global rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    # Immersion without translation
    immersion = get_8_curve_immersion(embedding_dim=embedding_dim, rotation=rot)

    # Parameter sampling
    angles = torch.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    # Add Gaussian noise if specified
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((n_points,))
        data = data + noise

    return pd.DataFrame(data.numpy()), pd.DataFrame({"angles": angles.numpy()})


def load_flower_scrunchy(
        rotation="random",
        n_points=1500,
        radius=1.0,
        n_wiggles=6,
        deformation_amp=0.4,
        embedding_dim=10,
        noise_var=0.01,
        random_seed=42
):
    """
    Generates samples of a high-dimensional flower-scrunchy curve.

    The curve is a deformed circle with additional radial oscillations:
        amp(t) = radius * (1 + 0.5 * deformation_amp * cos(3 t))
        z(t)   = deformation_amp * cos(n_wiggles * t)

    The 3D curve is embedded into R^embedding_dim and optionally rotated.
    Gaussian noise can be added to the samples.

    Args:
        rotation (torch.Tensor or str, optional): Rotation matrix or "random"
            to apply a random rotation. Default is "random".
        n_points (int, optional): Number of sample points. Default is 1500.
        radius (float, optional): Base circle radius. Default is 1.0.
        n_wiggles (int, optional): Number of oscillations along the z-direction. Default is 6.
        deformation_amp (float, optional): Amplitude of the deformation. Default is 0.4.
        embedding_dim (int, optional): Dimension of the embedding space. Default is 10.
        noise_var (float, optional): Variance of the additive Gaussian noise. Default is 0.01.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, embedding_dim) with sampled coordinates.
            - DataFrame of shape (n_points, 1) with the parameter values under column "angles".
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Determine rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    # Immersion without translation
    immersion = get_flower_scrunchy_immersion(
        radius=radius,
        n_wiggles=n_wiggles,
        deformation_amp=deformation_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    # Parameter sampling with approximately equal arc length spacing
    angles = _get_equal_arc_length_angles(immersion, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    # Add Gaussian noise if specified
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((n_points,))
        data = data + radius * noise

    # Convert to pandas DataFrames
    df_data = pd.DataFrame(data.numpy())
    df_angles = pd.DataFrame({"angles": angles.numpy()})

    return df_data, df_angles


def load_interlocking_rings_synthetic(
        rotation="random",
        n_points=1500,
        radius=1.0,
        embedding_dim=10,
        noise_var=0.01,
        random_seed=42
):
    """
    Generates samples of two interlocking rings (linked circles) embedded in a
    high-dimensional space.

    The dataset consists of two perpendicular circles of radius `radius` that
    intersect like two chain links. The 3D configuration is embedded into
    R^embedding_dim and optionally rotated. Gaussian noise can be added.

    Args:
        rotation (torch.Tensor or str, optional): Rotation matrix or "random"
            to apply a random rotation. Default is "random".
        n_points (int, optional): Total number of sample points. Default is 1500.
        radius (float, optional): Radius of each ring. Default is 1.0.
        embedding_dim (int, optional): Dimension of the embedding space. Default is 10.
        noise_var (float, optional): Variance of the additive Gaussian noise. Default is 0.01.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of shape (n_points, embedding_dim) with sampled coordinates.
            - DataFrame of shape (n_points, 1) with the parameter values under column "angles".
    """
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Determine rotation
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    # Immersion without translation
    immersion = get_interlocking_rings_immersion(
        radius=radius,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    # Parameter sampling
    angles = gs.linspace(0, 4 * gs.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    # Add Gaussian noise if specified
    if noise_var > 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim)
        ).sample((n_points,))
        data = data + radius * noise

    # Convert to pandas DataFrames
    df_data = pd.DataFrame(data.numpy())
    df_angles = pd.DataFrame({"angles": angles.numpy()})

    return df_data, df_angles


def _get_equal_arc_length_angles(
        immersion,
        n_points: int = 1500,
        oversample: int = 10000
) -> torch.Tensor:
    """
    Compute parameter values along a closed curve such that the sampled
    points are approximately equally spaced in arc length.

    This is achieved by:
        1. Densely sampling the curve using `oversample` parameter values.
        2. Computing the cumulative arc length along these samples.
        3. Interpolating to find the `n_points` parameters corresponding to
           equally spaced arc length fractions.

    Args:
        immersion (Callable[[torch.Tensor], torch.Tensor]):
            Function mapping an angle to a point on the curve.
        n_points (int, optional):
            Number of equally spaced points to return. Default is 1500.
        oversample (int, optional):
            Number of dense samples to approximate the arc length. Default is 10000.

    Returns:
        torch.Tensor:
            Parameter values of shape (n_points,) providing approximately
            equal arc length spacing along the curve, ranging from 0 to 2π.
    """
    # Dense sampling of the curve
    angles_fine = torch.linspace(0, 2 * torch.pi, oversample)
    points = torch.stack([immersion(a) for a in angles_fine])

    # Compute cumulative arc length
    dists = torch.norm(points[1:] - points[:-1], dim=1)
    arc_lengths = torch.cat([torch.zeros(1), torch.cumsum(dists, dim=0)])
    arc_lengths = arc_lengths / arc_lengths[-1]  # normalize to [0, 1]

    # Select n_points angles at equal arc length fractions
    desired = torch.linspace(0, 1, n_points)
    idxs = torch.searchsorted(arc_lengths, desired)
    idxs = torch.clamp(idxs, 0, oversample - 1)

    return angles_fine[idxs]

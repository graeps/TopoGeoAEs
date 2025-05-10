import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal


def generate_five_blobs(n_points_per_blob=1000, centers=None, std=0.2):
    if centers is None:
        centers = [
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([-1.0, -1.0, 1.0]),
            torch.tensor([1.0, -1.0, -1.0]),
            torch.tensor([-1.0, 1.0, -1.0]),
            torch.tensor([0.0, 0.0, 0.0])
        ]

    data = []
    labels = []

    for i, center in enumerate(centers):
        # Create multivariate normal distribution
        cov_matrix = std * torch.eye(3)
        distribution = MultivariateNormal(loc=center, covariance_matrix=cov_matrix)

        # Sample points
        blob_points = distribution.sample((n_points_per_blob,))

        data.append(blob_points)
        labels.append(torch.full((n_points_per_blob,), i))

    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)

    return data, labels


def generate_filled_torus(n_points=5000, R=3.0, r=1.0):
    # Generate random points inside a filled torus
    points = torch.zeros((n_points, 3))

    # Random angle around the ring
    theta = torch.linspace(0, 2 * np.pi, n_points)

    # Random radius inside the tube (using square root for uniform distribution in area)
    rho = torch.linspace(0, 1, n_points) * r

    # Random angle within the tube cross-section
    phi = torch.linspace(0, 2 * np.pi, n_points)

    # Convert to cartesian coordinates
    points[i, 0] = (R + rho * torch.cos(phi)) * torch.cos(theta)
    points[i, 1] = (R + rho * torch.cos(phi)) * torch.sin(theta)
    points[i, 2] = rho * torch.sin(phi)

    return points


def generate_nested_spheres(n_points=5000, radii=[1.0, 2.0, 3.0], thicknesses=[0.1, 0.1, 0.1]):
    num_spheres = len(radii)
    points_per_sphere = n_points // num_spheres

    all_points = []
    all_labels = []

    for i, (radius, thickness) in enumerate(zip(radii, thicknesses)):
        # Generate uniform points on a unit sphere
        x = torch.randn(points_per_sphere, 3)
        x_norm = x / torch.norm(x, dim=1, keepdim=True)

        # Add random radius to create a shell with thickness
        random_radii = radius + thickness * (torch.rand(points_per_sphere, 1) - 0.5)
        sphere_points = x_norm * random_radii

        all_points.append(sphere_points)
        all_labels.append(torch.full((points_per_sphere,), i))

    points = torch.cat(all_points, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return points, labels


def generate_entangled_tori(n_points=5000, R1=3.0, r1=1.0, R2=3.0, r2=1.0,
                            hollow1=False, hollow2=False, shell_thickness=0.2):
    points1 = torch.zeros((n_points, 3))
    points2 = torch.zeros((n_points, 3))

    # First torus along xz-plane, centered at origin
    for i in range(n_points):
        theta = 2 * np.pi * torch.rand(1)

        if hollow1:
            # For hollow torus, sample points near the surface
            rho = r1 - shell_thickness / 2 + shell_thickness * torch.rand(1)
        else:
            # For filled torus, sample points throughout the tube
            rho = torch.sqrt(torch.rand(1)) * r1

        phi = 2 * np.pi * torch.rand(1)

        points1[i, 0] = (R1 + rho * torch.cos(phi)) * torch.cos(theta)
        points1[i, 1] = (R1 + rho * torch.cos(phi)) * torch.sin(theta)
        points1[i, 2] = rho * torch.sin(phi)

    # Second torus along xy-plane but shifted and rotated to create entanglement
    offset = torch.tensor([0.0, 0.0, R2 / 2])

    for i in range(n_points):
        theta = 2 * np.pi * torch.rand(1)

        if hollow2:
            # For hollow torus, sample points near the surface
            rho = r2 - shell_thickness / 2 + shell_thickness * torch.rand(1)
        else:
            # For filled torus, sample points throughout the tube
            rho = torch.sqrt(torch.rand(1)) * r2

        phi = 2 * np.pi * torch.rand(1)

        # Create second torus with different orientation
        points2[i, 0] = (R2 + rho * torch.cos(phi)) * torch.cos(theta)
        points2[i, 1] = rho * torch.sin(phi)
        points2[i, 2] = (R2 + rho * torch.cos(phi)) * torch.sin(theta)

        # Apply offset to create entanglement
        points2[i] = points2[i] + offset

    # Combine the points and create labels
    all_points = torch.cat([points1, points2], dim=0)
    labels = torch.cat([
        torch.zeros(n_points),
        torch.ones(n_points)
    ])

    return all_points, labels


def generate_clelia_curve(n_points=5000, a=3.0, b=2.0, c=2.0):
    """
    Generate a Clelia curve in R^3

    The Clelia curve is parametrized by:
    x = a * sin(b*t) * cos(c*t)
    y = a * sin(b*t) * sin(c*t)
    z = a * cos(b*t)

    where the ratio b/c determines the curve's shape

    Parameters:
    -----------
    n_points : int
        Number of points on the curve
    a, b, c : float
        Parameters controlling the curve shape

    Returns:
    --------
    points : torch.Tensor
        Tensor of shape (n_points, 3) containing the points on the curve
    """
    t = torch.linspace(0, 2 * np.pi, n_points)

    x = a * torch.sin(b * t) * torch.cos(c * t)
    y = a * torch.sin(b * t) * torch.sin(c * t)
    z = a * torch.cos(b * t)

    points = torch.stack([x, y, z], dim=1)
    return points


def visualize_manifolds(manifolds_dict):
    """
    Visualize multiple manifolds in R^3

    Parameters:
    -----------
    manifolds_dict : dict
        Dictionary mapping manifold names to tuples of (points, labels)
    """
    n_manifolds = len(manifolds_dict)
    fig = plt.figure(figsize=(15, 5 * ((n_manifolds + 1) // 2)))

    for i, (name, data) in enumerate(manifolds_dict.items(), 1):
        ax = fig.add_subplot(((n_manifolds + 1) // 2), 2, i, projection='3d')

        if isinstance(data, tuple) and len(data) == 2:
            points, labels = data
            unique_labels = torch.unique(labels)

            for label in unique_labels:
                mask = (labels == label)
                ax.scatter(
                    points[mask, 0].numpy(),
                    points[mask, 1].numpy(),
                    points[mask, 2].numpy(),
                    alpha=0.6,
                    label=f'Component {int(label)}'
                )
            ax.legend()
        else:
            points = data
            ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                points[:, 2].numpy(),
                alpha=0.6
            )

        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # 1. Generate five blobs (contractible manifolds)
    blobs, blob_labels = generate_five_blobs(n_points_per_blob=500)

    # 2. Generate filled torus
    filled_torus = generate_filled_torus(n_points=2000)

    # 3. Generate nested spheres
    nested_spheres, sphere_labels = generate_nested_spheres(n_points=3000)

    # 4. Generate entangled tori (one filled, one hollow)
    entangled_tori, tori_labels = generate_entangled_tori(
        n_points=2000, hollow1=False, hollow2=True
    )

    # 5. Generate Clelia curve
    clelia_curve = generate_clelia_curve(n_points=1000)

    # Visualize all manifolds
    manifolds = {
        'Five Blobs': (blobs, blob_labels),
        'Filled Torus': filled_torus,
        'Nested Spheres': (nested_spheres, sphere_labels),
        'Entangled Tori': (entangled_tori, tori_labels),
        'Clelia Curve': clelia_curve
    }

    visualize_manifolds(manifolds)

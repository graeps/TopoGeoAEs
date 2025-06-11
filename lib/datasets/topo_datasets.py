import os
import torch
from skimage.measure import marching_cubes
import numpy as np
import matplotlib.pyplot as plt

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs  # noqa: E402
from geomstats._backend.pytorch.random import rand
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402
from torch.distributions.multivariate_normal import MultivariateNormal

from .utils import compute_frenet_frame


def generate_three_manifolds(manifold, n_points_per_manifold=1000, noise_var=0.2, embedding_dim=3, translations=None,
                             rotations=None):
    if translations is None:
        translations = [
            15 * torch.ones(embedding_dim),
            -15 * torch.ones(embedding_dim),
            torch.zeros(embedding_dim),
        ]

    if rotations is None:
        rotations = [
            SpecialOrthogonal(n=embedding_dim).random_point(),
            SpecialOrthogonal(n=embedding_dim).random_point(),
            SpecialOrthogonal(n=embedding_dim).random_point(),
        ]
    if manifold == "entangled_tori":
        points1, labels1 = generate_entangled_tori(n_points_per_manifold, filled1=True, filled2=True,
                                                   noise_var=noise_var,
                                                   embedding_dim=embedding_dim, translation=translations[0],
                                                   rotation=rotations[0])
        points2, labels2 = generate_entangled_tori(n_points_per_manifold, filled1=True, filled2=False,
                                                   noise_var=noise_var,
                                                   embedding_dim=embedding_dim, translation=translations[1],
                                                   rotation=rotations[1])
        points3, labels3 = generate_entangled_tori(n_points_per_manifold, filled1=True, filled2=False,
                                                   noise_var=noise_var,
                                                   embedding_dim=embedding_dim,
                                                   translation=translations[2], rotation=rotations[2])

    elif manifold == "sphere":
        points1, labels1 = generate_sphere(n_points_per_manifold, noise_var=noise_var, embedding_dim=embedding_dim,
                                           translation=translations[0], rotation=rotations[0])
        points2, labels2 = generate_sphere(n_points_per_manifold, noise_var=noise_var, embedding_dim=embedding_dim,
                                           translation=translations[1], rotation=rotations[1])
        points3, labels3 = generate_sphere(n_points_per_manifold, noise_var=noise_var, embedding_dim=embedding_dim,
                                           translation=translations[2], rotation=rotations[2])

    else:
        raise NotImplementedError(manifold)

    data = torch.cat((points1, points2, points3), dim=0)
    labels = torch.cat((labels1, labels2, labels3), dim=0)

    return data, labels


def get_torus_immersion(major_radius, minor_radius, embedding_dim, deformation_amp, translation, rotation):
    def immersion(angle_pair):
        theta, phi = angle_pair

        # Standard 3D torus coordinates
        x_coord = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y_coord = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z_coord = minor_radius * gs.sin(theta)

        point = torch.stack([x_coord, y_coord, z_coord], dim=0)
        point = _embedd(point, embedding_dim)

        if deformation_amp != 0.0:
            for i in range(3, embedding_dim):
                if i == embedding_dim - 1:
                    wiggle = 0
                else:
                    wiggle = deformation_amp * torch.cos(phi)
                point[i] = wiggle

        point = _rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_torus(n_points, major_radius, minor_radius, noise_var, embedding_dim, deformation_amp,
               translation="random",
               rotation="random", random_seed=42, ):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)

    immersion = get_torus_immersion(major_radius=major_radius, minor_radius=minor_radius, embedding_dim=embedding_dim,
                                    deformation_amp=deformation_amp,
                                    translation=trans, rotation=rot)

    sqrt_ntimes = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(eps, 2 * gs.pi - eps, sqrt_ntimes)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_ntimes)
    angle_grid = torch.cartesian_prod(thetas, phis)

    data = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        data = data + noise

    return data, angle_grid


def generate_torus(n_points=5000, major_radius=5.0, minor_radius=1.0, filled=False, noise_var=0.01, embedding_dim=3,
                   translation=None,
                   rotation=None, random_seed=42):
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
    points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)
    noise = noise_var * torch.randn_like(points) * minor_radius
    points += noise

    return points, angles


def load_wiggling_tube(n_phi, n_theta, minor_radius, noise_var, wiggling_dim, embedding_dim, deformation_amp,
                       rotation="random", random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    phis = torch.linspace(0, 2 * torch.pi, n_phi, requires_grad=True)
    thetas = torch.linspace(0, 2 * torch.pi, n_theta, requires_grad=True)

    rot = torch.eye(wiggling_dim)

    curve = get_scrunchy_dim_n(deformation_amp=deformation_amp, embedding_dim=wiggling_dim, translation=None,
                               rotation=rot)

    data = []
    angles = []  # To store corresponding (theta, phi) pairs for each point

    for phi in phis:
        frame, _, _ = compute_frenet_frame(curve, phi, wiggling_dim, deformation_amp=deformation_amp,
                                           is_scrunchy_dim_n=True)
        e1 = frame[:, 1]
        e2 = frame[:, 2]
        center = curve(phi)

        for theta in thetas:
            offset = minor_radius * torch.cos(theta) * e1 + minor_radius * torch.sin(theta) * e2
            point = center + offset
            data.append(point)
            angles.append((theta.item(), phi.item()))  # Store the angle values as tuples

    data = torch.stack(data).detach()

    if embedding_dim > wiggling_dim:
        rot = torch.eye(embedding_dim)
        if rotation == "random":
            rot = SpecialOrthogonal(n=embedding_dim).random_point()
        trans = torch.zeros(embedding_dim)

        data = _embedd_rotate_translate(point=data, embedding_dim=embedding_dim, translation=trans, rotation=rot)

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=minor_radius * noise_var * torch.eye(embedding_dim),
        ).sample((n_phi * n_theta,))
        data = data + noise

    angles_np = np.array(angles)  # shape: (n_phi * n_theta, 2)
    sort_idx = np.lexsort((angles_np[:, 1], angles_np[:, 0]))  # sort by theta, then by phi
    angles = angles_np[sort_idx]
    data = data[sort_idx]

    return data, torch.tensor(angles)


def load_interlocked_tori(n_points, major_radius, minor_radius, noise_var, embedding_dim,
                          deformation_amp, rotation, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    trans = torch.zeros(embedding_dim)

    immersion = get_torus_immersion(major_radius=major_radius, minor_radius=minor_radius, embedding_dim=embedding_dim,
                                    deformation_amp=deformation_amp,
                                    translation=trans, rotation=rot)

    R_x90 = torch.eye(embedding_dim)
    R_x90[1, 1], R_x90[1, 2] = 0.0, -1.0
    R_x90[2, 1], R_x90[2, 2] = 1.0, 0.0

    sqrt_ntimes = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.linspace(eps, 2 * gs.pi - eps, sqrt_ntimes)
    phis = gs.linspace(eps, 2 * gs.pi - eps, sqrt_ntimes)
    angle_grid = torch.cartesian_prod(thetas, phis)

    torus1 = torch.stack([immersion(pair) for pair in angle_grid])
    torus2 = torch.stack([immersion(pair) for pair in angle_grid])
    torus2 = torus2 @ R_x90.T
    translation = torch.zeros(embedding_dim)
    translation[0] = -major_radius
    torus2 += translation

    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    torus1 = torch.stack([gs.einsum("ij,j->i", rot, point) for point in torus1])
    torus2 = torch.stack([gs.einsum("ij,j->i", rot, point) for point in torus2])

    if noise_var != 0:
        noise1 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        noise2 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        torus1 = torus1 + noise1
        torus2 = torus2 + noise2

    data = torch.cat([torus1, torus2])
    angles = torch.cat([angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    entity_index = torch.tensor([0] * n_samples + [1] * n_samples, dtype=torch.long)
    labels = (entity_index, angles)

    return data, labels


def load_interlocked_tubes(n_phi, n_theta, minor_radius, noise_var, wiggling_dim, embedding_dim, deformation_amp,
                           rotation="random", random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(wiggling_dim)

    tube1, angles1 = load_wiggling_tube(n_phi=n_phi, n_theta=n_theta, minor_radius=minor_radius, noise_var=noise_var,
                                        wiggling_dim=wiggling_dim, embedding_dim=embedding_dim,
                                        deformation_amp=deformation_amp, rotation="random", random_seed=42)
    tube2, angles2 = load_wiggling_tube(n_phi=n_phi, n_theta=n_theta, minor_radius=minor_radius, noise_var=noise_var,
                                        wiggling_dim=wiggling_dim + 1, embedding_dim=embedding_dim,
                                        deformation_amp=deformation_amp, rotation="random", random_seed=42)

    R_x90 = torch.eye(embedding_dim)
    R_x90[1, 1], R_x90[1, 2] = 0.0, -1.0
    R_x90[2, 1], R_x90[2, 2] = 1.0, 0.0

    tube2 = tube2 @ R_x90.T
    translation = torch.zeros(embedding_dim)
    translation[0] = -1
    tube2 += translation

    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    tube1 = torch.stack([gs.einsum("ij,j->i", rot, point) for point in tube1])
    tube2 = torch.stack([gs.einsum("ij,j->i", rot, point) for point in tube2])

    n_points_per_tube = angles1.shape[0]

    if noise_var != 0:
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

    data = torch.cat([tube1, tube2])
    angles = torch.cat([angles1, angles2])
    entity_index = torch.tensor([0] * n_points_per_tube + [1] * n_points_per_tube, dtype=torch.long)
    labels = (entity_index, angles)

    return data, labels


def generate_entangled_tori(n_points=5000, major_radius=5.0, minor_radius=1.0, filled1=False, filled2=False,
                            noise_var=0.01, embedding_dim=3,
                            translation=None, rotation=None):
    n1 = n_points // 2
    n2 = n_points - n1

    torus1, _ = generate_torus(n1, major_radius, minor_radius, filled=filled1, noise_var=noise_var)
    torus2, _ = generate_torus(n2, major_radius, minor_radius, filled=filled2, noise_var=noise_var)

    R_x90 = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ])

    torus2 = torus2 @ R_x90.T
    torus2 += torch.tensor([-major_radius, 0.0, 0.0])
    points = torch.cat([torus1, torus2], dim=0)

    points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)

    labels = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long)
    ]).unsqueeze(dim=1)

    return points, labels


def load_n_torus(n_points=1000, n=2, radii=None):
    if radii is None:
        radii = np.ones(n)
    else:
        radii = np.asarray(radii)
        assert len(radii) == n, "Length of radii must match torus dimension n."

    thetas = 2 * np.pi * np.random.rand(n_points, n)

    # Map each angle θ_i to (r_i cos θ_i, r_i sin θ_i)
    data = []
    for i in range(n):
        ri = radii[i]
        theta_i = thetas[:, i]
        xi = ri * np.cos(theta_i)
        yi = ri * np.sin(theta_i)
        data.append(xi)
        data.append(yi)

    data = np.stack(data, axis=1).T  # shape (2n, n_points)
    return data.T  # shape (n_points, 2n)


def _torus_implicit_field(x, y, z, R, r):
    return (x ** 2 + y ** 2 + z ** 2) ** 2 - 2 * (R ** 2 + r ** 2) * (x ** 2 + y ** 2) + 2 * (
            R ** 2 - r ** 2) * z ** 2 + (R ** 2 - r ** 2) ** 2


def _genus3_field(x, y, z, n=3, R=1.0, r=0.25):
    angles = torch.linspace(0, 2 * np.pi, n + 1, device=x.device)[:-1]
    shifts_x = 1.5 * torch.cos(angles)
    shifts_y = 1.5 * torch.sin(angles)

    f = torch.ones_like(x)
    for sx, sy in zip(shifts_x, shifts_y):
        xi = x - sx
        yi = y - sy
        zi = z
        f *= _torus_implicit_field(xi, yi, zi, R, r)
    return f - 10


def generate_genus3(n_points=5000, R=1.0, r=0.25, noise_var=0.01, embedding_dim=3, translation=None, rotation=None):
    # Create dense grid
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(-3, 3, 100),
        torch.linspace(-3, 3, 100),
        torch.linspace(-0.5, 0.5, 50),
        indexing='ij'
    )
    vals = _genus3_field(grid_x, grid_y, grid_z, R=R, r=r)
    vals_np = vals.cpu().numpy()

    # Extract surface
    verts, faces, normals, _ = marching_cubes(vals_np, level=0)

    # Rescale vertices to world coords
    verts = torch.tensor(verts.copy())
    verts[:, 0] = verts[:, 0] / 99 * 6 - 3
    verts[:, 1] = verts[:, 1] / 99 * 6 - 3
    verts[:, 2] = verts[:, 2] / 49 * 1 - 0.5

    # Sample points
    idx = torch.randint(0, verts.shape[0], (n_points,))
    points = verts[idx]

    # Apply rotation/translation
    points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)

    # Add noise
    noise = noise_var * torch.randn_like(points) * r
    points += noise

    labels = torch.zeros(n_points)
    return points, labels


def generate_sphere(n_points, radius, filled=False, noise_var=0.01, embedding_dim=3, translation=None,
                    rotation=None, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if filled:
        u = torch.rand(n_points)
        v = torch.rand(n_points)
        w = torch.rand(n_points)

        theta = torch.acos(1 - 2 * u)
        phi = 2 * np.pi * v
        r = radius * w.pow(1 / 3)

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        angles = torch.stack((theta, phi, r), dim=1)

    else:
        theta = torch.acos(1 - 2 * torch.rand(n_points))
        phi = 2 * np.pi * torch.rand(n_points)

        x = radius * torch.sin(theta) * torch.cos(phi)
        y = radius * torch.sin(theta) * torch.sin(phi)
        z = radius * torch.cos(theta)

        angles = torch.stack((theta, phi), dim=1)

    points = torch.stack((x, y, z), dim=1)

    points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)

    noise = noise_var * torch.randn_like(points) * radius
    points += noise

    return points, angles


def get_sphere_immersion(radius, deformation_amp, embedding_dim, translation, rotation):
    def immersion(angle_pair):
        theta, phi = angle_pair
        amplitude = (
                1
                + deformation_amp * gs.exp(-5 * theta ** 2)
                + deformation_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = amplitude * gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        point = _embedd_rotate_translate(point=point, embedding_dim=embedding_dim, translation=translation,
                                         rotation=rotation)
        return point

    return immersion


def load_nested_spheres(n_points, major_radius, mid_radius, minor_radius, noise_var, embedding_dim, deformation_amp,
                        translation=None, rotation=None, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot_inner = SpecialOrthogonal(n=3).random_point()
    rot_mid = SpecialOrthogonal(n=3).random_point()
    rot_outer = SpecialOrthogonal(n=3).random_point()
    trans = torch.zeros(3)

    immersion_inner = get_sphere_immersion(radius=minor_radius, embedding_dim=3, deformation_amp=deformation_amp,
                                           translation=trans, rotation=rot_inner)
    immersion_mid = get_sphere_immersion(radius=mid_radius, embedding_dim=3, deformation_amp=deformation_amp,
                                         translation=trans, rotation=rot_mid)
    immersion_outer = get_sphere_immersion(radius=major_radius, embedding_dim=3, deformation_amp=deformation_amp,
                                           translation=trans, rotation=rot_outer)

    sqrt_ntimes = int(gs.sqrt(n_points))
    thetas = gs.arccos(
        np.linspace(0.99, -0.99, sqrt_ntimes))  # For more uniform distribution of sample points on sphere
    phis = gs.linspace(0, 2 * np.pi, sqrt_ntimes)
    angle_grid = torch.cartesian_prod(thetas, phis)

    sphere_inner = torch.stack([immersion_inner(pair) for pair in angle_grid])
    sphere_mid = torch.stack([immersion_mid(pair) for pair in angle_grid])
    sphere_outer = torch.stack([immersion_outer(pair) for pair in angle_grid])

    data = torch.cat([sphere_inner, sphere_mid, sphere_outer])
    angles = torch.cat([angle_grid, angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    sphere_index = torch.tensor([0] * n_samples + [1] * n_samples + [2] * n_samples, dtype=torch.long)
    labels = (sphere_index, angles)

    rot = torch.eye(n=embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    data = _embedd_rotate_translate(point=data, embedding_dim=embedding_dim, translation=trans, rotation=rot)

    if noise_var != 0:
        noise1 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=minor_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        noise2 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=mid_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        noise3 = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
        ).sample((sqrt_ntimes ** 2,))
        noise = torch.cat([noise1, noise2, noise3])
        data = data + noise

    return data, labels


def get_sphere_high_dim_bump_immersion(radius, deformation_amp, bump_dim, bump_center, embedding_dim,
                                       translation, rotation):
    def immersion(angle_pair):
        theta, phi = angle_pair
        theta_center, phi_center = bump_center

        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        point = _embedd(point, embedding_dim)

        delta_theta = (theta - theta_center) % (2 * torch.pi)
        if delta_theta > torch.pi:
            delta_theta -= 2 * torch.pi
        delta_phi = (phi - phi_center) % (2 * torch.pi)
        if delta_phi > torch.pi:
            delta_phi -= 2 * torch.pi

        if abs(theta - theta_center) < 1 and abs(phi - phi_center) < 1:
            bump_theta = torch.exp(-1 / (1 - delta_theta ** 2))
            bump_phi = torch.exp(-1 / (1 - delta_phi ** 2))
            bump = radius * deformation_amp * bump_theta * bump_phi
        else:
            bump = 0
        point[bump_dim] += bump

        point = _rotate_translate(point, translation, rotation)

        return point

    return immersion


def load_nested_spheres_high_dim_bump(n_points, major_radius, mid_radius, minor_radius, noise_var, embedding_dim,
                                      deformation_amp, rotation=None, translation=None, enclosing_sphere=False,
                                      random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(n=embedding_dim)
    trans = torch.zeros(embedding_dim)

    bump_dims = [embedding_dim - 3, embedding_dim - 2, embedding_dim - 1]
    bump_centers = [(torch.pi / 4, torch.pi / 2), (torch.pi / 4, 3 * torch.pi / 2), (torch.pi / 2, torch.pi / 2)]

    immersion_inner = get_sphere_high_dim_bump_immersion(radius=minor_radius, bump_dim=bump_dims[0],
                                                         bump_center=bump_centers[0],
                                                         embedding_dim=embedding_dim, deformation_amp=deformation_amp,
                                                         translation=trans, rotation=rot)
    immersion_mid = get_sphere_high_dim_bump_immersion(radius=mid_radius, bump_dim=bump_dims[1],
                                                       bump_center=bump_centers[1],
                                                       embedding_dim=embedding_dim, deformation_amp=deformation_amp,
                                                       translation=trans, rotation=rot)
    immersion_outer = get_sphere_high_dim_bump_immersion(radius=major_radius, bump_dim=bump_dims[2],
                                                         bump_center=bump_centers[2],
                                                         embedding_dim=embedding_dim, deformation_amp=deformation_amp,
                                                         translation=trans, rotation=rot)

    sqrt_ntimes = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.arccos(
        np.linspace(1 - eps, -1 + eps, sqrt_ntimes))  # For more uniform distribution of sample points on sphere
    phis = gs.linspace(eps, 2 * np.pi - eps, sqrt_ntimes)
    angle_grid = torch.cartesian_prod(thetas, phis)

    sphere_inner = torch.stack([immersion_inner(pair) for pair in angle_grid])
    sphere_mid = torch.stack([immersion_mid(pair) for pair in angle_grid])
    sphere_outer = torch.stack([immersion_outer(pair) for pair in angle_grid])

    data = torch.cat([sphere_inner, sphere_mid, sphere_outer])
    angles = torch.cat([angle_grid, angle_grid, angle_grid])
    n_samples = angle_grid.shape[0]
    sphere_index = torch.tensor([0] * n_samples + [1] * n_samples + [2] * n_samples, dtype=torch.long)
    labels = (sphere_index, angles)

    rot = torch.eye(n=embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    data = _rotate_translate(points=data, translation=trans, rotation=rot)

    if enclosing_sphere:
        # generate sphere S^(embedding_dim-1)
        n_samples = n_points * 2
        enclosing_sphere = torch.randn(n_samples, embedding_dim)
        enclosing_sphere = enclosing_sphere / enclosing_sphere.norm(dim=1, keepdim=True)
        enclosing_sphere = enclosing_sphere * (2 * major_radius)
        data = torch.cat([data, enclosing_sphere])
        sphere_index, angles = labels
        enclosing_sphere_index = torch.tensor([100] * n_samples, dtype=torch.long)
        dummy_angles = torch.full((n_samples, 2), 1.0)
        sphere_index = torch.cat([sphere_index, enclosing_sphere_index])
        angles = torch.cat([angles, dummy_angles])
        labels = (sphere_index, angles)

    return data, labels


def load_multi_dim_spheres(n_points, major_radius, mid_radius, minor_radius, noise_var, embedding_dim,
                           deformation_amp, rotation=None, translation=None, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    nested_spheres, labels = load_nested_spheres_high_dim_bump(n_points, major_radius, mid_radius, minor_radius,
                                                               noise_var, embedding_dim,
                                                               deformation_amp, rotation="random", translation=None,
                                                               random_seed=42)

    return data, labels


def get_clelia_immersion(r, c, embedding_dim, translation, rotation):
    def immersion(angles):
        x = r * gs.sin(angles) * gs.cos(c * angles)
        y = r * gs.sin(angles) * gs.sin(c * angles)
        z = r * gs.cos(angles)

        points = gs.array([x, y, z])
        points = gs.squeeze(points, axis=-1)
        points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)
        return points

    return immersion


def load_clelia_curve(n_points=5000, r=3.0, c=3.0, noise_var=0.01, embedding_dim=3, translation="random",
                      rotation="random", random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)

    immersion = get_clelia_immersion(r=r, c=c, embedding_dim=embedding_dim, translation=trans,
                                     rotation=rot)

    angles = gs.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])
    if noise_var != 0:
        noise = MultivariateNormal(loc=torch.zeros(embedding_dim),
                                   covariance_matrix=noise_var * torch.eye(embedding_dim), ).sample((n_points,))
        data = data + noise

    angles = angles.unsqueeze(dim=1)
    return data, angles


def get_8_curve_immersion(embedding_dim, translation, rotation):
    def immersion(angles):
        x = torch.sin(angles)
        y = torch.sin(2 * angles)
        z = 0.1 * torch.sin(10 * angles)

        points = torch.stack([x, y, z], dim=1)
        points = _embedd_rotate_translate(points, embedding_dim, translation, rotation)

        return points

    return immersion


def load_8_curve(n_points, noise_var, embedding_dim=3, translation="random", rotation="random", random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)
    immersion = get_8_curve_immersion(embedding_dim=embedding_dim, translation=trans, rotation=rot)

    angles = torch.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var != 0:
        noise = MultivariateNormal(loc=torch.zeros(embedding_dim),
                                   covariance_matrix=noise_var * torch.eye(embedding_dim), ).sample((n_points,))
        data = data + noise

    return data, angles


def get_scrunchy_dim_n(deformation_amp, embedding_dim, translation, rotation):
    def immersion(angle):
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

        # point = gs.array(terms)  # shape: [embedding_dim]
        point = gs.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])[:embedding_dim]
        point = gs.squeeze(point, axis=-1)
        point = _rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_scrunchy_dim_n(n_points, noise_var, deformation_amp, embedding_dim, translation, rotation, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    if translation == "random":
        trans = 100 * torch.rand(embedding_dim)
    immersion = get_scrunchy_dim_n(deformation_amp=deformation_amp, embedding_dim=embedding_dim, translation=trans,
                                   rotation=rot)

    angles = torch.linspace(0, 2 * torch.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    if noise_var != 0:
        noise = MultivariateNormal(loc=torch.zeros(embedding_dim),
                                   covariance_matrix=noise_var * torch.eye(embedding_dim), ).sample((n_points,))
        data = data + noise

    return data, angles


def _embedd_rotate_translate(point, embedding_dim, translation, rotation):
    point = _embedd(points=point, embedding_dim=embedding_dim)
    point = _rotate_translate(points=point, rotation=rotation, translation=translation)
    return point


def _embedd(points, embedding_dim):
    points = gs.array(points)
    if points.ndim == 1:
        pad_width = embedding_dim - points.shape[0]
        if pad_width > 0:
            points = gs.concatenate([points, gs.zeros(pad_width)])
    elif points.ndim == 2:
        pad_width = embedding_dim - points.shape[1]
        if pad_width > 0:
            zeros = gs.zeros((points.shape[0], pad_width))
            points = gs.concatenate([points, zeros], axis=1)
    else:
        raise ValueError("Input must be a 1D or 2D array.")
    return points


def _rotate_translate(points, translation, rotation):
    if points.ndim == 1:
        points = gs.einsum("ij,j->i", rotation, points)
        points = points + translation
    elif points.ndim == 2:
        points = gs.einsum("ij,nj->ni", rotation, points)
        points = points + translation
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")
    return points

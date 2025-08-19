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


def load_scrunchy(rotation,
                  n_points=1500,
                  radius=1.0,
                  n_wiggles=6,
                  deformation_amp=0.4,
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
        distortion_amp=deformation_amp,
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
        noisy_data = data + radius * noise
    else:
        noisy_data = data

    labels = angles.unsqueeze(dim=1)
    return noisy_data, labels

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


def get_flower_scrunchy_immersion(radius, n_wiggles, distortion_amp, embedding_dim, rot):
    def immersion(angle):
        amp = radius * (1 + distortion_amp / 2 * gs.cos(angle * 3))

        x = amp * gs.cos(angle)
        y = amp * gs.sin(angle)
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


def _get_equal_arc_length_angles(immersion, n_points=1500, oversample=10000):
    angles_fine = torch.linspace(0, 2 * torch.pi, oversample)
    points = torch.stack([immersion(a) for a in angles_fine])
    dists = torch.norm(points[1:] - points[:-1], dim=1)
    arc_lengths = torch.cat([torch.zeros(1), torch.cumsum(dists, dim=0)])
    arc_lengths = arc_lengths / arc_lengths[-1]

    desired = torch.linspace(0, 1, n_points)
    idxs = torch.searchsorted(arc_lengths, desired)
    idxs = torch.clamp(idxs, 0, oversample - 1)
    return angles_fine[idxs]



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
    trans = torch.zeros(wiggling_dim)

    curve = get_s1_high(deformation_amp=deformation_amp, embedding_dim=wiggling_dim, translation=trans,
                               rotation=rot)

    data = []
    angles = []  # To store corresponding (theta, phi) pairs for each point

    for phi in phis:
        frame, _, _ = compute_frenet_frame(curve, phi, wiggling_dim, deformation_amp=deformation_amp,
                                           is_s1_high=True)
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

    immersion = get_t2_high_immersion(major_radius=major_radius, minor_radius=minor_radius, embedding_dim=embedding_dim,
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


def load_nested_spheres(n_points, major_radius, mid_radius, minor_radius, noise_var, embedding_dim, deformation_amp,
                        translation, rotation, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if rotation == "random":
        rot_inner = SpecialOrthogonal(n=embedding_dim).random_point()
        rot_mid = SpecialOrthogonal(n=embedding_dim).random_point()
        rot_outer = SpecialOrthogonal(n=embedding_dim).random_point()
    else:
        rot_inner = torch.eye(embedding_dim)
        rot_mid = torch.eye(embedding_dim)
        rot_outer = torch.eye(embedding_dim)
    if translation is None:
        trans = torch.zeros(embedding_dim)
    else:
        trans = translation

    immersion_inner = get_s2_high_immersion(radius=minor_radius, embedding_dim=embedding_dim,
                                           deformation_amp=deformation_amp,
                                           translation=trans, rotation=rot_inner)
    immersion_mid = get_s2_high_immersion(radius=mid_radius, embedding_dim=embedding_dim,
                                         deformation_amp=deformation_amp,
                                         translation=trans, rotation=rot_mid)
    immersion_outer = get_s2_high_immersion(radius=major_radius, embedding_dim=embedding_dim,
                                           deformation_amp=deformation_amp,
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
        x = radius * gs.sin(theta) * gs.cos(phi)
        y = radius * gs.sin(theta) * gs.sin(phi)
        z = radius * gs.cos(theta)

        point = gs.array([x, y, z])
        point = gs.squeeze(point, axis=-1)
        point = _embedd(point, embedding_dim)

        # Ensure bump_dims is a list
        if isinstance(bump_dim, (list, tuple)):
            bump_dims = list(bump_dim)
        else:
            bump_dims = [bump_dim]

        if isinstance(bump_center, tuple) and len(bump_center) == 2 and all(
                isinstance(x, (float, torch.Tensor)) for x in bump_center):
            bump_centers = [bump_center]
        else:
            bump_centers = list(bump_center)

        for dim, center in zip(bump_dims, bump_centers):
            theta_center, phi_center = center

            delta_theta = (theta - theta_center) % (2 * gs.pi)
            if delta_theta > gs.pi:
                delta_theta -= 2 * gs.pi
            delta_phi = (phi - phi_center) % (2 * gs.pi)
            if delta_phi > gs.pi:
                delta_phi -= 2 * gs.pi

            if abs(theta - theta_center) < 1 and abs(phi - phi_center) < 1:
                bump_theta = gs.exp(-1 / (1 - delta_theta ** 2))
                bump_phi = gs.exp(-1 / (1 - delta_phi ** 2))
                bump = radius * deformation_amp * bump_theta * bump_phi
            else:
                bump = 0

            point[dim] += bump

        point = _rotate_translate(point, translation, rotation)
        return point

    return immersion


def load_sphere_high_dim_bump(n_points, radius, noise_var, embedding_dim,
                              deformation_amp, rotation=None, translation=None, random_seed=42):
    gs.random.seed(random_seed)
    torch.manual_seed(random_seed)

    rot = torch.eye(n=embedding_dim)
    trans = torch.zeros(embedding_dim)

    bump_dims = [embedding_dim - 3, embedding_dim - 2, embedding_dim - 1]
    bump_centers = [(torch.pi / 4, torch.pi / 2), (torch.pi / 4, 3 * torch.pi / 2), (torch.pi / 2, torch.pi / 2)]

    immersion = get_sphere_high_dim_bump_immersion(radius=radius, bump_dim=bump_dims,
                                                   bump_center=bump_centers,
                                                   embedding_dim=embedding_dim, deformation_amp=deformation_amp,
                                                   translation=trans, rotation=rot)

    sqrt_ntimes = int(gs.sqrt(n_points))
    eps = 1e-4
    thetas = gs.arccos(
        np.linspace(1 - eps, -1 + eps, sqrt_ntimes))  # For more uniform distribution of sample points on sphere
    phis = gs.linspace(eps, 2 * np.pi - eps, sqrt_ntimes)
    angle_grid = torch.cartesian_prod(thetas, phis)

    sphere = torch.stack([immersion(pair) for pair in angle_grid])

    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((sphere.shape[0],))
        sphere = sphere + noise

    rot = torch.eye(n=embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()
    trans = torch.zeros(embedding_dim)
    data = _rotate_translate(points=sphere, translation=trans, rotation=rot)

    return data, angle_grid


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
    if noise_var != 0:
        noise = MultivariateNormal(
            loc=torch.zeros(embedding_dim),
            covariance_matrix=noise_var * torch.eye(embedding_dim),
        ).sample((data.shape[0],))
        data = data + noise

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


def load_clelia_curve(n_points, r, c, noise_var, embedding_dim, translation,
                      rotation, random_seed=42):
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


def load_scrunchy(rotation,
                  n_points=1500,
                  radius=1.0,
                  n_wiggles=6,
                  deformation_amp=0.4,
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
        distortion_amp=deformation_amp,
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
        noisy_data = data + radius * noise
    else:
        noisy_data = data

    labels = angles.unsqueeze(dim=1)
    return noisy_data, labels


def load_flower_scrunchy(rotation,
                         n_points=1500,
                         radius=1.0,
                         n_wiggles=6,
                         deformation_amp=0.4,
                         embedding_dim=10,
                         noise_var=0.01, random_seed=42,
                         ):
    gs.random.seed(random_seed)
    rot = torch.eye(embedding_dim)
    if rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_flower_scrunchy_immersion(
        radius=radius,
        n_wiggles=n_wiggles,
        distortion_amp=deformation_amp,
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
        noisy_data = data + radius * noise
    else:
        noisy_data = data

    labels = angles.unsqueeze(dim=1)
    return noisy_data, labels


def load_interlocking_rings_synthetic(rotation,
                                      n_points=1500,
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
    angles = gs.linspace(0, 4 * gs.pi, n_points)
    data = torch.stack([immersion(angle) for angle in angles])

    noise = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=noise_var * torch.eye(embedding_dim),
    ).sample((n_points,))

    noisy_data = data + radius * noise
    labels = pd.DataFrame({"angles": angles})
    return noisy_data, labels


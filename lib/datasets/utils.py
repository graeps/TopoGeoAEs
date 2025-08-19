import torch
import os

# Set Geomstats backend before importing it
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs  # noqa: E402


def embedd_rotate_translate(point, embedding_dim, translation, rotation):
    point = embedd(points=point, embedding_dim=embedding_dim)
    point = rotate_translate(points=point, rotation=rotation, translation=translation)
    return point


def embedd(points, embedding_dim):
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


def rotate_translate(points, translation, rotation):
    if points.ndim == 1:
        points = gs.einsum("ij,j->i", rotation, points)
        points = points + translation
    elif points.ndim == 2:
        points = gs.einsum("ij,nj->ni", rotation, points)
        points = points + translation
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")
    return points


def compute_n_deriv(curve, t, order):
    r = curve(t)
    dim = r.shape[-1]
    derivatives = [r]

    for k in range(order):
        r = derivatives[-1]
        derivatives_at_t = []

        for i in range(dim):
            if k == 0:  # First derivative
                r_prime = torch.autograd.grad(r[i], t, create_graph=True)[0]
            else:  # Higher derivatives (reuse graph)
                r_prime = torch.autograd.grad(r[i], t, create_graph=True)[0]

            derivatives_at_t.append(r_prime)

        derivatives.append(torch.stack(derivatives_at_t, dim=-1))
    return derivatives


def gram_schmidt(V):
    """Orthonormalize a list of vectors using the Gram-Schmidt process."""
    n = V.shape[0]  # Number of vectors
    Q = torch.zeros_like(V)
    for i in range(n):
        v = V[i]
        for j in range(i):
            v -= torch.dot(Q[j], v) * Q[j]
        Q[i] = v / torch.norm(v)
    return Q


def compute_n_deriv_scrunchy(curve, t, order, deformation_amp):
    derivatives = [curve(t)]

    for k in range(1, order + 1):
        r = derivatives[-1]  # Get the last computed derivative
        dim = r.shape[-1]  # Get the dimension of the curve

        derivatives_at_t = []

        # Compute the derivative for each component of the curve
        for i in range(dim):
            n = (i // 2) + 1
            if i % 2 == 0:  # For sin(n * t)
                if k % 4 == 1:
                    r_prime = n ** k * torch.cos(n * t)
                elif k % 4 == 2:
                    r_prime = -n ** k * torch.sin(n * t)
                elif k % 4 == 3:
                    r_prime = -n ** k * torch.cos(n * t)
                else:
                    r_prime = n ** k * torch.sin(n * t)
            else:  # For cos(n * t)
                if k % 4 == 1:
                    r_prime = -n ** k * torch.sin(n * t)
                elif k % 4 == 2:
                    r_prime = -n ** k * torch.cos(n * t)
                elif k % 4 == 3:
                    r_prime = n ** k * torch.sin(n * t)
                else:
                    r_prime = n ** k * torch.cos(n * t)

            if i >= 2:
                derivatives_at_t.append(deformation_amp * r_prime)
            else:
                derivatives_at_t.append(r_prime)

        derivatives.append(torch.stack(derivatives_at_t, dim=-1))
    return derivatives


def compute_frenet_frame(curve, t, order, deformation_amp=None, is_s1_high=False):
    if is_s1_high:
        derivatives = compute_n_deriv_scrunchy(curve, t, order, deformation_amp)
    else:
        derivatives = compute_n_deriv(curve, t, order)

    # Stack the derivatives as rows to form a matrix
    V = torch.stack(derivatives[1:], dim=0)  # Skip the original curve value r(t) (first element)

    # Orthonormalize the derivatives to get the frame
    frame = gram_schmidt(V)

    # Compute the curvatures
    curvatures = []
    for i in range(order - 1):
        # Curvature κ_j = e_j' · e_(j+1)
        curvature = torch.dot(frame[i], frame[i + 1])
        curvatures.append(curvature)

    # Compute torsion, which is the largest curvature (last one)
    torsion = curvatures[-1] if curvatures else None

    return frame, curvatures, torsion


def generate_tube_from_curve(curve, tube_radius, n_phi, n_theta):
    phis = torch.linspace(0, 2 * torch.pi, n_phi, requires_grad=True)
    thetas = torch.linspace(0, 2 * torch.pi, n_theta, requires_grad=True)

    data = []
    for phi in phis:
        frame, _, _ = compute_frenet_frame(curve, phi, 10)
        e1 = frame[1]
        e2 = frame[2]
        center = curve(phi)
        for theta in thetas:
            offset = tube_radius * torch.cos(theta) * e1 + tube_radius * torch.sin(theta) * e2
            point = center + offset
            data.append(point)

    return torch.stack(data)

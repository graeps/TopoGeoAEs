import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def generate_toroidal_pointcloud(A, embed_dim, translation=False, num_points=80000, noise_std=0.00, device='cpu'):
    d = A.shape[0]  # Latent dimension
    A_inv_T_theta = torch.inverse(A).T

    # Step 1: Sample θ ∈ [0,1]^d uniformly
    theta = torch.rand(num_points, d, device=device)

    # Step 2: Compute standard torus embedding (4D)
    cos_theta = torch.cos(2 * torch.pi * theta @ A_inv_T_theta.T)
    sin_theta = torch.sin(2 * torch.pi * theta @ A_inv_T_theta.T)

    # Step 3: Apply shaping matrix A
    cos_shaped = cos_theta @ A.T  # [num_points, d]
    sin_shaped = sin_theta @ A.T  # [num_points, d]


    # Step 4: Form the shaped torus in 2d dimensions
    pointcloud = torch.cat([cos_shaped, sin_shaped], dim=1)  # [num_points, 2d]

    # Step 5: Add optional Gaussian noise
    if noise_std > 0:
        noise = noise_std * torch.randn_like(pointcloud)
        pointcloud += noise

    # Step 6: Embed into R^embed_dim (pad with zeros)
    extra_dims = torch.zeros(num_points, embed_dim - 4, device=device)
    pointcloud_embedded = torch.cat([pointcloud, extra_dims], dim=1)  # [num_points, embed_dim]

    # Step 7: Generate a random translation vector t ∈ R^10
    if translation:
        v_translation = torch.randn(embed_dim, device=device)
    else:
        v_translation = torch.zeros(embed_dim, device=device)

    # Step 8: Apply translation
    pointcloud_embedded = pointcloud_embedded + v_translation  # [num_points, embed_dim]

    return theta, pointcloud, pointcloud_embedded  # Return theta for validation


def load_shaped_torus(pointcloud, batch_size):
    # Wrap point cloud in dataset
    dataset = TensorDataset(pointcloud)

    # Split dataset
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

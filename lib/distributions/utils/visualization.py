import numpy as np
import torch
from matplotlib import pyplot as plt

from .. multivariate_generalized_von_mises import MGVonMises


def plot_mgvm_distr_2d(loc, scale, precision, num_samples=20000):
    grid_res = 1000

    loc_batch = loc.repeat(num_samples, 1)
    loc_batch2 = loc.repeat(grid_res ** 2, 1)

    scale_batch = scale.repeat(num_samples, 1)
    scale_batch2 = scale.repeat(grid_res ** 2, 1)

    precision_batch = precision.repeat(num_samples, 1, 1)
    precision_batch2 = precision.repeat(grid_res ** 2, 1, 1)

    # Instantiate the distribution
    mg_vm = MGVonMises(loc_batch, scale_batch, precision_batch)
    mg_vm_grid = MGVonMises(loc_batch2, scale_batch2, precision_batch2)

    # Generate samples
    samples = mg_vm.rsample(loc_batch.shape).cpu().numpy()
    samples_x, samples_y = samples[:, 0], samples[:, 1]

    # Define grid
    phi1s = torch.linspace(0, 2 * torch.pi, grid_res)
    phi2s = torch.linspace(0, 2 * torch.pi, grid_res)
    phi1, phi2 = torch.meshgrid(phi1s, phi2s, indexing='ij')
    phi = torch.stack((phi1, phi2), dim=-1)
    phi_reshaped = phi.reshape(-1, 2)

    # Compute unnormalized density
    unnorm_density = torch.exp(mg_vm_grid._log_unnormalized_prob(phi_reshaped))
    unnorm_density = unnorm_density.reshape(grid_res, grid_res)

    # Convert to numpy for plotting
    phi1_np = phi1.numpy()
    phi2_np = phi2.numpy()
    density_np = unnorm_density.squeeze().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot density function
    contour = ax.contour(phi1_np, phi2_np, density_np, levels=15, cmap="Dark2", alpha=0.8, norm="linear")
    plt.colorbar(contour, label="Density")

    # Overlay 2D histogram
    ax.hist2d(samples_x, samples_y, range=np.array([[0, 2 * np.pi], [0, 2 * np.pi]]), bins=(200, 200), cmap=plt.cm.jet)

    # Labels
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_title("2D Density Function & Sample Histogram")

    plt.show()


def plot_mgvm_distr_1d(loc, scale, precision, num_samples=20000):
    loc_batch = loc.repeat(num_samples, 1)
    scale_batch = scale.repeat(num_samples, 1)
    precision_batch = precision.repeat(num_samples, 1, 1)

    mg_vm = MGVonMises(loc_batch, scale_batch, precision_batch)

    samples = mg_vm.rsample(loc_batch.shape).squeeze().cpu().numpy()

    # Plot histogram of samples
    plt.figure(figsize=(8, 4))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')

    # Overlay expected von Mises density (approximate)
    phi_vals = torch.linspace(0, 2 * torch.pi, num_samples).unsqueeze(1)
    expected_density = torch.exp(mg_vm._log_unnormalized_prob(phi_vals)).squeeze().cpu().numpy()
    bin_width = (2 * torch.pi) / 50
    expected_density /= (expected_density.sum() * bin_width)

    plt.plot(phi_vals.cpu().numpy(), expected_density * num_samples / 50, 'r-', label="Expected Density")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Histogram of Samples vs. Expected Density")
    plt.show()

    # Check sample mean and variance
    sample_mean = samples.mean()
    sample_var = samples.var()

    print(f"Sample Mean: {sample_mean:.4f}, Expected Mean: {loc.item():.4f}")
    print(f"Sample Variance: {sample_var:.4f}")

from .visualization import show_training_history, plot_test_latents_on_torus, show_recon_mnist, \
    show_recon_mnist_ae, plot_latent_projections, \
    curvature_compute_plot_vm, curvature_compute_plot_euclidean, plot_data_latents_recon, \
    plot_empiric_curvature, get_vectors
from .debug import pass_single_batch
from .evaluation import compute_curvature_learned, compute_curvature_true, compute_curvature_true_latents, \
    compute_empiric_curvature, _compute_curvature, get_true_immersion, get_s2_synthetic_immersion, \
    get_s1_synthetic_immersion

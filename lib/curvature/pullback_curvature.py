from geomstats.geometry.pullback_metric import PullbackMetric


def _compute_curvature(z_grid, immersion, dim, embedding_dim):
    neural_manifold = NeuralManifoldIntrinsic(
        dim, embedding_dim, immersion, equip=False
    )
    neural_manifold.equip_with_metric(PullbackMetric)
    torch.unsqueeze(z_grid[0], dim=0)
    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion (Manifold Dim = 1)",
                         total=len(z_grid), leave=False):
            z = torch.unsqueeze(z, dim=0)
            curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
    else:
        curv = torch.full((len(z_grid), embedding_dim), torch.nan)
        for i, z in tqdm(enumerate(z_grid), desc="Computing curvature from immersion  (Manifold Dim > 1)",
                         total=len(z_grid), leave=False):
            try:
                curv[i, :] = neural_manifold.metric.mean_curvature_vector(z)
            except Exception as e:
                print(f"An error occurred for i={i}: {e}")
    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True).squeeze()

    # Apply quantile-based clipping to suppress numerical spikes
    valid_mask = ~torch.isnan(curv_norm)
    if valid_mask.any():
        q_low, q_high = torch.quantile(curv_norm[valid_mask], torch.tensor((0.01, 0.99)))
        curv_norm = torch.clamp(curv_norm, min=q_low.item(), max=q_high.item())
    curv_norm = torch.clamp(curv_norm, min=0.0)

    return curv, curv_norm
def compute_curvature_learned(config, model, latents=None, labels=None, n_grid_points=2000):
    if config.verbose:
        print("Computing learned curvature...")
    if config.model_type == 'EuclideanVAE':
        z_grid = latents
    elif config.model_type in {'VMFSphericalVAE', "VMFToroidalVAE", 'SphericalAE', 'ToroidalAE'}:
        if config.latent_dim == 2:
            anchore = (latents[0], latents[10])
            z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
            #z_grid = shift_z_grid(z_grid, anchore, config)
        elif config.latent_dim == 3:
            z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    elif config.model_type == 'VMToroidalVAE':
        z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    else:
        raise InvalidConfigError(f"Unknown model type: {config.model_type}")
    immersion = get_learned_immersion(model=model, config=config)
    if config.dataset_name in {"s1_low", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy", "s1_high"}:
        manifold_dim = 1
    elif config.dataset_name in {"s2_low", "t2_low", "s2_high", "sphere_high_dim", "t2_high", "wiggling_tube",
                                 "interlocked_tori", "nested_spheres"}:
        manifold_dim = 2
    else:
        raise NotImplementedError({"unknown dataset"})
    curv, curv_norm = _compute_curvature(z_grid=z_grid, immersion=immersion, dim=manifold_dim,
                                         embedding_dim=config.embedding_dim)
    return z_grid, labels, curv, curv_norm

def compute_curvature_true(config, labels=None, n_grid_points=2000, cache_dir="./curvature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    if config.dataset_name in {"s1_low", "s2_low", "t2_low", "scrunchy", "s1_high"}:
        deformation = config.deformation_amp
    else:
        deformation = config.deformation_amp
    name = f"{config.dataset_name}_{config.n_points_pullback_curv}_{deformation}.pt"
    cache_path = os.path.join(cache_dir, name)

    if os.path.exists(cache_path):
        if config.verbose:
            print(f"Loading cached curvature from {cache_path}")
        return torch.load(cache_path)

    if config.verbose:
        print("Computing true curvature...")

    if config.compute_emp_curv:
        angles = labels
    else:
        angles = get_z_grid(config, n_grid_points)

    if config.dataset_name in {"s1_low", "interlocking_rings_synthetic", "scrunchy", "clelia_curve", "8_curve",
                               "flower_scrunchy", "s1_high"}:
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, 1, config.embedding_dim)

    elif config.dataset_name in {"s2_low", "t2_low", "t2_high", "sphere_high_dim", "s2_high"}:
        immersion = get_true_immersion(config)
        curv, curv_norm = _compute_curvature(angles, immersion, 2, config.embedding_dim)

    elif config.dataset_name in {"nested_spheres", "nested_spheres_high_dim", "interlocked_tori", "interlocked_tubes"}:
        immersions = get_true_immersion(config)
        entity_indices, angles = labels[:, 0], labels[:, 1:]
        curv, curv_norm = [], []
        for i in entity_indices.unique(sorted=True):
            mask = (entity_indices == i)
            curv_i, curv_norm_i = _compute_curvature(
                angles[mask], immersions[int(i.item())], 2, 3
            )
            curv.append(curv_i)
            curv_norm.append(curv_norm_i)
        curv = torch.cat(curv)
        curv_norm = torch.cat(curv_norm)

    else:
        raise InvalidConfigError(f"Unknown dataset: {config.dataset_name}")

    torch.save((angles, curv, curv_norm), cache_path)
    return angles, curv, curv_norm


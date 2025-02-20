from ..utils.loss_functions import elbo


def pass_single_batch(model, train_loader, device="cpu"):
    model.train()

    batch = next(iter(train_loader))
    x = batch[0].to(device) # Assuming train_loader yields (data, labels)

    # Forward pass
    z, x_recon, posterior_params = model(x)
    mu, logvar = posterior_params

    loss, recon_loss, kl_loss = elbo(model.posterior_type, x, x_recon, posterior_params,
                                     model.latent_dim,
                                     device)

    # Backward pass
    loss.backward()

    # Capture gradient stats
    gradient_stats = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    outputs = {
        "input": x,
        "latent": z,
        "reconstruction": x_recon,
        "mu": mu,
        "logvar": logvar,
        "loss": loss.item(),
        "gradients": gradient_stats,
    }

    print("outputs")
    for name, grad in outputs["gradients"].items():
        print(f"{name}: Gradient mean={grad.mean().item()}, std={grad.std().item()}")

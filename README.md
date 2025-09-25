# Latent Topology and Geometry in Riemannian Autoencoders

This repository contains the code for my Diploma thesis *"Exploratory Analysis of Latent Topology and Geometry in
Riemannian Autoencoders"*.  
The thesis investigates the capacity of several autoencoder architectures to

1. learn representations of data that are **topologically and geometrically aligned** with the underlying data manifold,
   and
2. learn a **parametrization of the data manifold** by exploiting the decoder network.

In particular, the impact of **topological regularization** on the learned topology and geometry is analyzed.

![](.sphere_high_topo_euclidean.png)

The repository provides implementations of four classes of autoencoder architectures:

- **Euclidean Autoencoders (AEs)**
- **Euclidean Variational Autoencoders (VAEs)**
- **Manifold Autoencoders (Manifold-AEs)** with non-Euclidean latent spaces
- **Manifold Variational Autoencoders (Manifold-VAEs)** with non-Euclidean latent spaces

The `lib/` module contains the tools to train these models on synthetic datasets, visualize the learned latent
representations, and evaluate the quality of the recovered topology and geometry. Exemplary usage and the conducted experiments can be found in the
`notebooks/` directory.

## Dependencies and Related Work

This project builds on the following implementations:

- [Spherical VAEs](https://github.com/nicola-decao/s-vae-pytorch)
- [Toroidal VAEs, synthetic datasets, and pullback curvature estimation](https://github.com/geometric-intelligence/neurometry/tree/main/neurometry)
- [Topological autoencoders with persistence-based regularization](https://github.com/BorgwardtLab/topological-autoencoders)  

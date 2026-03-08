# Latent Topology and Geometry in some Riemannian Autoencoders

Code accompanying the Diploma thesis  
*"Exploratory Analysis of Latent Topology and Geometry in some Riemannian Autoencoders"*

The project investigates how different autoencoder architectures learn latent
representations that reflect the **topology and geometry of the underlying data manifold**.

In particular, the thesis studies whether autoencoders can

1. learn latent representations that are **topologically and geometrically aligned** with the data manifold
2. learn a meaningful **parametrization of the manifold through the decoder**, revealing information about the geometry of the underlying data manifold

It further investigates the **impact of topological regularization** on these properties.

<img src="sphere_high_topo_euclidean.png" width="70%">

---

### Overview

The repository contains implementations and experimental tools for analyzing
the latent topology and geometry of autoencoders.

The following model classes are implemented:

- **Euclidean Autoencoders (AE)**
- **Euclidean Variational Autoencoders (VAE)**
- **Manifold Autoencoders (Manifold-AE)** with non-Euclidean latent spaces
- **Manifold Variational Autoencoders (Manifold-VAE)** with non-Euclidean latent spaces

The code allows experiments on synthetic datasets with known topology, enabling
quantitative evaluation of how well the learned latent space reflects the true
structure of the data.

---

The `lib/` module provides utilities for

- training the different autoencoder architectures
- generating synthetic datasets with known topology
- estimating geometric quantities such as curvature
- analyzing the topology of latent representations
- visualizing latent embeddings and decoder maps

The `notebooks/` directory contains the experiments and visualizations used in the thesis.

---


### Installation

The project uses a [conda environment](https://www.anaconda.com/docs/getting-started/miniconda/main).

```bash
conda env create -f conda-env.yml
conda activate TopoGeoAEs
```

---

### Running Experiments

Experiments are provided as Jupyter notebooks in:

```
experiments/notebooks/
```

---

### Dependencies and Related Work

This project builds on the following implementations:

Spherical VAEs  
https://github.com/nicola-decao/s-vae-pytorch

Toroidal VAEs and geometric analysis tools  
https://github.com/geometric-intelligence/neurometry

Topological Autoencoders with persistence regularization  
https://github.com/BorgwardtLab/topological-autoencoders

---

### Results

The full thesis is available here:

`paper/thesis.pdf`

Parts of this work were accepted as an extended abstract and poster tracks at the NeurIPS 2025 **NeurReps** and **UniReps"" workshops

Extended abstracts:
`paper/unireps_extended_abstract.pdf`
`paper/neurreps_extended_abstract.pdf`

Posters:  
`paper/unireps_poster.pdf`
`paper/neurreps_poster.pdf`

---

### Citation

If you use this code, please cite the Diploma thesis:

```
@mastersthesis{samuelgraepler2025TopoGeoAEs,
  title={Exploratory Analysis of Latent Topology and Geometry in some Riemannian Autoencoders},
  author={Samuel Graepler},
  year={2025}
}

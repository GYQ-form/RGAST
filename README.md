# RGAST

RGAST: Relational Graph Attention Network for Spatial Transcriptome Analysis [[paper]](https://doi.org/10.1101/2024.08.09.607420)

This document will help you easily go through the RGAST model.

![fig1_00](https://github.com/GYQ-form/RGAST/assets/79566479/fe0655dc-2318-44e0-92bf-0aea3aad7163)

## Dependencies

The required Python packages and versions tested in our study are:

```
pytorch==2.4.1
scanpy==1.10.3
scikit-learn==1.5.2
pyg==2.6.1
scipy==1.14.1
numpy==2.0.1
pandas==2.2.3
```

## Installation

To install our package, run

```bash
git clone https://github.com/GYQ-form/RGAST.git
cd RGAST
pip install .
```

You can also clone the repo and install it in editable mode:

```bash
git clone https://github.com/GYQ-form/RGAST.git
cd RGAST
pip install -e .
```

## Usage

RGAST is a deep learning framework designed to infer multi-scale cell-cell communication (CCC) networks de novo from spatial transcriptomics (ST) data. RGAST integrates spatial proximity and transcriptional profiles using a relational graph attention mechanism. This approach allows RGAST to dynamically learn context-specific signaling patterns and reconstruct CCC networks without prior knowledge of ligand-receptor pairs, effectively capturing both local and global communication patterns. Besides, RGAST is also a versatile tool for many downstream ST analysis:

- spatial domain identification
- spatially variable gene (SVG) detection
- cell trajectory inference
- reveal intricate 3D spatial patterns across multiple sections of ST data

## Tutorial

We have prepared several basic tutorials  in https://github.com/GYQ-form/RGAST/tree/main/tutorial. You can quickly hands on RGAST by going through these tutorials. Model parameters trained in our study are also released in https://github.com/GYQ-form/RGAST/tree/main/model_path.

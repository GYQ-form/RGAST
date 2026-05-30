# RGAST
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18463983.svg)](https://doi.org/10.5281/zenodo.18463983)

RGAST: A Relational Graph Attention Network for Multi-Scale Cell-Cell Communication Inference from Spatial Transcriptomics [[paper]](https://doi.org/10.1101/2024.08.09.607420)

This document will help you easily go through the RGAST model.

![fig1](https://github.com/user-attachments/assets/585a8126-55a3-42a3-9e76-0c00597ccab4)


## Dependencies

The required Python packages and versions tested in our study are:

```
pytorch==2.8.0
scanpy==1.11.5
scikit-learn==1.7.2
pyg==2.7.0
scipy==1.17.0
numpy==2.3.0
pandas==2.3.3
```

## Installation

To install our package, run

```bash
git clone https://github.com/GYQ-form/RGAST.git
cd RGAST
pip install .
```

## Usage

RGAST is a deep learning framework designed to infer multi-scale cell-cell communication (CCC) networks de novo from spatial transcriptomics (ST) data. RGAST integrates spatial proximity and transcriptional profiles using a relational graph attention mechanism. This approach allows RGAST to dynamically learn context-specific signaling patterns and reconstruct CCC networks without prior knowledge of ligand-receptor pairs, effectively capturing both local and global communication patterns. Besides, RGAST is also a versatile tool for many downstream ST analysis:

- spatial domain identification
- spatially variable gene (SVG) detection
- cell trajectory inference
- reveal intricate 3D spatial patterns across multiple sections of ST data

## Tutorial

We have prepared several basic tutorials  in https://github.com/GYQ-form/RGAST/tree/main/tutorial. You can quickly hands on RGAST by going through these tutorials. 

## Analysis

To enhance the reproducibility of this study, we deposited all the custom code at [Zenodo repository](https://doi.org/10.5281/zenodo.18463983) for running RGAST used in the paper. A comprehensive README file has also been provided for easy using of these custom scripts.
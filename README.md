# stFormer: a foundation model for spatial transcriptomics

## 1. Introduction
stFormer incorporates ligand genes within the spatial niche into transformer encoder of single-cell transcriptomics, and outputs gene embeddings specific to the intracellular context and spatial niche. These gene representations can serve as input of various downstream applications, including cell clustering, cell type prediction, gene function prediction, and *in silico* perturbation analysis of ligand-receptor interaction.

![Framework Architecture](https://github.com/csh3/stFormer/blob/main/schematic_overview.png)

The model architecture is designed for ST data resolved at the single-cell level. We propose a biased cross-attention method to enable the model to do learning with single-cell resolution on low-resolution, whole-transcriptome Visium data, which is a widely available spatial resource. 

We assembled a pretraining corpus comprising ~4.1 million spatial samples from public human Visium datasets, spanning diverse tissues, development stages, and disease states.  After pretraining, stFormer is compatible with both single-cell and spot resolution ST data. 

## 2. Installation
You can install stFormer with the following command:

```
git clone https://github.com/csh3/stFormer.git
cd stFormer; pip install -e .
```

We recommend installing stFormer in a fresh python environment since we specify the installed versions of the python module dependencies.

## 3. Usage
The project consists of the following four folders.

* **stformer** folder holds source codes of the model.

* **datasets** folder holds jupyter notebooks for formatting ST data and formatted ST data used in downstream tasks. 

* **pretraining** folder holds pretrained model weights and the pretraining script. 

* **tasks** folder holds jupyter notebooks and gene lists for performing downstream tasks. It also holds the **scfoundation** folder, which contains files for scFoundation, a single-cell transcriptomics foundation model, for comparison.

The pretrained model weights in **pretraining**, formatted ST data in **datasets**, fine-tuned model weights in **tasks/fine-tuning_mse**, and output results in **tasks/figures** reported in reference [1] are deposited on the Zenodo data repository under record number 13895591.

## 4. Reference
[1] Cao, S. *et al*. stFormer: a foundation model for spatial transcriptomics. bioRxiv, 2024.2009.2027.615337 (2025). https://doi.org/10.1101/2024.09.27.615337




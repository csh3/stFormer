# stFormer

## 1. Introduction
stFormer is a foundation model for spatial transcriptomics (ST). It incorporates ligand genes within the spatial niche into Transformer encoder of single-cell transcriptomics, and outputs gene embeddings specific to the intracellular context and spatial niche. These gene representations can serve as input of diverse downstream applications, including cell clustering, receptor-dependent gene network inference, interacting ligand-receptor pair identification, and in silico perturbation analysis of ligand-receptor interaction.

![stFormer Architecture](https://github.com/csh3/stFormer/blob/main/schematic_overview.png)

The architecture of stFormer is designed for ST data resolved at the single-cell level. We propose a biased cross-attention method to extend the single-cell framework compatible with the spot-based Visium data, a widely used ST platform. 

The current version of stFormer was pretrained on a combined corpus of two Visium datasets respectively from human developmental intestine and myocardial infarction heart tissue. The pretraining data totally comprises about 0.58 million single-cell samples. stFormer is scalable to large pretraining corpus. 

## 2. Installation & dependencies
You can download the software package by the command:

```
$ git clone https://github.com/csh3/stFormer.git
```

or click the **Download Zip** button and decompress the code package.

The environment requires `Python3` with the following modules: 
`numpy`, `pandas`, `sklearn`, `torch`, `anndata`, `scanpy`, `flash_attn`

## 3. Usage
The project consists of the following five folders. Before excuting downstream tasks and pretraining, move the jupyter notebooks and python script to the main directory:

```
$ mv tasks/* .
$ mv pretraining/pretraining.py .
```

* **stformer** folder holds source codes of the model.

* **data** folder holds the ST data for pretraining and downstream tasks. A download link for data is provided in the file *download.txt*.

* **pretraining** folder holds the pretrained model weights and pretraining script. A download link for model weights is provided in the file *download.txt*.

* **scfoundation** folder holds files for the single-cell transcriptomics foundation model, scFoundation, for comparison.

* **tasks** folder holds the jupyter notebooks and gene lists for performing downstream tasks, as well as the output results. 

## 4. Reference
Cao, S. & Yuan, Y. stFormer: a foundation model for spatial transcriptomics. bioRxiv, 2024.2009.2027.615337 (2024). https://doi.org:10.1101/2024.09.27.615337




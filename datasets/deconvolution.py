import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import scanpy as sc
import numpy as np
import cell2location

#------------------------------ Get the arguments: dataset name and slide name ------------------------------
dataset = sys.argv[1]
slide = sys.argv[2]
print(f'Get the arguments dataset:{dataset} and slide:{slide}')

directory = f'{dataset}/{slide}'


#------------------------------ Preprocess the single-cell reference data ------------------------------
print("Preprocess the single-cell reference data")

if os.path.exists(f'{dataset}/sc_ref_index.txt'):
    sc_ref_lib = {}
    with open(f'{dataset}/sc_ref_index.txt', 'r') as f:
        for line in f:
            record = line.strip().split()
            sc_file = record[0]
            slides = record[1].split(',')
            for s in slides:
                sc_ref_lib[s] = sc_file
    sc_file = f'{dataset}/{sc_ref_lib[slide]}'
elif os.path.exists(f'{dataset}/sc.h5ad'):
    sc_file = f'{dataset}/sc.h5ad'
else:
    sc_file = f'{directory}/sc.h5ad'
adata_sc = sc.read(sc_file)
adata_sc.var['gene_symbols'] = adata_sc.var.index
adata_sc.var.set_index('gene_ids', drop=True, inplace=True)
adata_sc.obs['cell_type'] = adata_sc.obs['cell_type'].astype(str)

# filter marker genes
from cell2location.utils.filtering import filter_genes
selected = filter_genes(adata_sc, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
if 'Marker' in adata_sc.var.columns:
    markers = adata_sc.var_names[adata_sc.var['Marker']]
    selected = selected.union(markers)
adata_ref = adata_sc[:, selected].copy()
adata_ref.var['Marker'] = True

# prepare anndata for the regression model
if 'batch' in adata_ref.obs.columns:
    batch_key='batch'
elif 'sample' in adata_ref.obs.columns:
    batch_key='sample'
else:
    batch_key=None

technical_effects = []
if 'assay' in adata_ref.obs.columns:
    technical_effects.append('assay')
if 'donor' in adata_ref.obs.columns:
    technical_effects.append('donor')
if len(technical_effects) == 0:
    technical_effects = None

# obs keys: sample batch cell_type donor assay
cell2location.models.RegressionModel.setup_anndata(adata = adata_ref,
                        batch_key = batch_key,
                        labels_key = 'cell_type',
                        categorical_covariate_keys = technical_effects
                       )

# create the regression model
from cell2location.models import RegressionModel
mod = RegressionModel(adata_ref)

# train the regression model
mod.train(max_epochs=250, use_gpu=True)

# export the estimated cell abundance (summary of the posterior distribution).
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
)


#------------------------------ Preprocess the spatial data ------------------------------
print("Preprocess the spatial data")
# Load the spatial data
if os.path.exists(f'{directory}/feature_bc_matrix'):
    adata_vis = sc.read_10x_mtx(f'{directory}/feature_bc_matrix')
    adata_1 = sc.read_visium(directory, library_id=f'{dataset}-{slide}')
    if not (adata_1.obs_names == adata_vis.obs_names).all():
        print("The barcodes of mtx and h5 files are not consistent.")
        sys.exit(1)
    else:
        adata_vis.obs = adata_1.obs
        adata_vis.uns = adata_1.uns
        adata_vis.obsm = adata_1.obsm
else:
    adata_vis = sc.read_visium(directory, library_id=f'{dataset}-{slide}')
    if (adata_vis.var_names == adata_vis.var['gene_ids']).all():
        adata_ref1 = sc.read_visium('VISDP000002/VISDS000005', library_id='c')
        gene_index = adata_ref1.var_names.intersection(adata_vis.var_names).difference(adata_ref1.var_names[adata_ref1.var_names.duplicated()])
        adata_vis = adata_vis[:,gene_index]
        adata_vis.var['gene_ids'] = adata_ref1.var['gene_ids'].loc[adata_vis.var_names]
        adata_vis.write(f'{directory}/st.h5ad')

adata_vis.obs['sample'] = f'{dataset}-{slide}'
adata_vis.var['gene_symbols'] = adata_vis.var_names
adata_vis.var.set_index('gene_ids', drop=True, inplace=True)
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['gene_symbols']]
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]


#------------------------------ cell2location cell type decomposition ------------------------------
print("cell2location cell type decomposition")
# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']

# find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

# create cell2location model
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver,
    N_cells_per_location=30,
    detection_alpha=200
)

# train cell2location model
mod.train(max_epochs=30000,
          batch_size=None,
          train_size=1,
          use_gpu=True,
        )

# export the estimated cell abundance (summary of the posterior distribution).
adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
)
adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

# Compute expected expression per cell type
expected_dict = mod.module.model.compute_expected_per_cell_type(
    mod.samples["post_sample_q05"], mod.adata_manager
)

# Add to anndata layers
for i, n in enumerate(mod.factor_names_):
    adata_vis.layers[n] = expected_dict['mu'][i]


#------------------------------ Save the results ------------------------------
print('Save the final deconvolved ST data')
adata_vis.write(f"{directory}/deconv.h5ad")
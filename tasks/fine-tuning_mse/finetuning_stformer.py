# Copyright (c) 2025, Shenghao Cao & Ye Yuan. Shanghai Jiao Tong University, Shanghai 200240, China
# The classes and functions are modified from scGPT (https://github.com/bowang-lab/scGPT), copyright (c) 2022 suber

import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import Dict
import time
import warnings
warnings.filterwarnings('ignore')
import copy
import pickle

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from stformer import logger
from stformer.tokenizer import GeneVocab
from stformer.tokenizer import tokenize_and_pad_batch_2, random_mask_value
from stformer.model import TransformerModel
from stformer.loss import masked_mse_loss


class Tokenizer():
    def __init__(self, adata, tokenizer_dir, vocab, mask_ratio, mask_value, pad_value, pad_token):
        self.adata = adata
        self.tokenizer_dir = tokenizer_dir
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.load_data()
    
    def load_data(self):
        self.expression_matrix = self.adata.X
        self.niche_ligands_expression = self.adata.obsm['niche_ligands_expression']
        self.niche_composition = self.adata.obsm['niche_composition']

        gene_list_df = pd.read_csv(f'{self.tokenizer_dir}/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])
        self.gene_ids = np.array(self.vocab(gene_list), dtype=int)

        ligand_database = pd.read_csv(self.tokenizer_dir+'ligand_database.csv', header=0, index_col=0)
        ligand_symbol = ligand_database[ligand_database.sum(1)>1].index.values
        ligand_symbol = gene_list_df.loc[gene_list_df['gene_name'].isin(ligand_symbol), 'gene_name'].values
        self.ligand_ids = np.array(self.vocab(ligand_symbol.tolist())*25, dtype=int)

    def tokenize_data(self, k1, k2):
        expression_matrix = self.expression_matrix[k1:k2].A
        niche_ligands_expression = self.niche_ligands_expression[k1:k2].A
        niche_composition = self.niche_composition[k1:k2].A

        biases = np.zeros([niche_composition.shape[0], 24650])
        for k in range(biases.shape[0]):
            biases[k] = np.concatenate([[np.log(p)]*986 for p in niche_composition[k]])

        tokenized_data = tokenize_and_pad_batch_2(
            expression_matrix,
            niche_ligands_expression,
            biases,
            self.gene_ids,
            self.ligand_ids,
            pad_id = self.vocab[self.pad_token],
            pad_value = self.pad_value,
        )

        logger.info(
            f"tokenize sample number: {tokenized_data['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {tokenized_data['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {tokenized_data['niche_genes'].shape[1]}"
        )

        self.tokenized_data = tokenized_data

    def prepare_data(self):
        masked_values = random_mask_value(
            self.tokenized_data["center_values"],
            mask_ratio = self.mask_ratio,
            mask_value = self.mask_value,
            pad_value = self.pad_value,
        )

        logger.info(
            f"random masking ratio of masked values: "
            f"{(masked_values == self.mask_value).sum() / (masked_values - self.pad_value).count_nonzero() *100:2.2f}%"
        )

        self.data_pt = {
            "center_gene_ids": self.tokenized_data["center_genes"],
            "input_center_values": masked_values,
            "target_center_values": self.tokenized_data["center_values"],
            "niche_gene_ids": self.tokenized_data["niche_genes"],
            "input_niche_values": self.tokenized_data["niche_values"],
            "cross_attn_bias": self.tokenized_data["cross_attn_bias"],
        }
    
    def prepare_dataloader(self, batch_size):
        data_loader = DataLoader(
            dataset=SeqDataset(self.data_pt),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),
            pin_memory=True,
        )
        return data_loader


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["center_gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    

def train(model: nn.Module, loader: DataLoader, slice, mse_train, epoch) -> None:

    model.train()
    total_mse = 0.0
 
    start_time = time.time()

    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        center_gene_ids = batch_data["center_gene_ids"].to(device)
        input_center_values = batch_data["input_center_values"].to(device)
        target_center_values = batch_data["target_center_values"].to(device)
        niche_gene_ids = batch_data["niche_gene_ids"].to(device)
        input_niche_values = batch_data["input_niche_values"].to(device)
        cross_attn_bias = batch_data["cross_attn_bias"].to(device)

        encoder_src_key_padding_mask = niche_gene_ids.eq(vocab[pad_token])
        # encoder_src_key_padding_mask = torch.ones_like(niche_gene_ids, dtype=torch.bool).to(device)
        decoder_src_key_padding_mask = center_gene_ids.eq(vocab[pad_token])
        decoder_masked_positions = input_center_values.eq(mask_value)

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                    niche_gene_ids,
                    input_niche_values,
                    encoder_src_key_padding_mask,
                    center_gene_ids,
                    input_center_values,
                    decoder_src_key_padding_mask,
                    cross_attn_bias,
                )
    
            loss_mse = criterion(
                output_dict["mlm_output"], target_center_values, decoder_masked_positions
            )

        # model.zero_grad()
        scaler.scale(loss_mse).backward()
        if (batch + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()

        total_mse += loss_mse.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            sec_per_batch = (time.time() - start_time) / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"|epoch {epoch:2d} slice {slice:2d} | "
                f"{batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | sec/batch {sec_per_batch:5.1f} | "
                f"mse {cur_mse:5.5f} | "
            )
            mse_train.append(cur_mse)
            total_mse = 0
            start_time = time.time()



### Arguments
# model_checkpoint, adata_file, output_suffix, epoch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

### Load the pretraining model
logger.info("Load pretraining model")
if len(sys.argv) < 10:
    logger.info("Initialize pretraining model")

    embsize = 768
    d_hid = 3072
    nhead = 12
    nlayers = 6
    dropout = 0.1
    cell_emb_style = 'max-pool'

    logger.info("Loading scFoundation model ...")
    from tasks.scfoundation import load
    pretrainmodel, pretrainconfig = load.load_model_frommmf('../scfoundation/models/models.ckpt')

    model = TransformerModel(
        embsize,
        nhead,
        d_hid,
        nlayers,
        dropout = dropout,
        cell_emb_style = cell_emb_style,
        scfoundation_token_emb1 = copy.deepcopy(pretrainmodel.token_emb),
        scfoundation_token_emb2 = copy.deepcopy(pretrainmodel.token_emb),
        scfoundation_pos_emb1 = copy.deepcopy(pretrainmodel.pos_emb),
        scfoundation_pos_emb2 = copy.deepcopy(pretrainmodel.pos_emb),
    )

    del pretrainmodel

    model_file = sys.argv[1]
    pt_model = torch.load(f'../../pretraining/models/{model_file}', map_location='cpu')

    model_dict = model.state_dict()
    pretrained_dict = pt_model.state_dict()
    pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if 'cls_decoder' not in k and 'gcl_decoder' not in k
                # if k in model_dict and v.shape == model_dict[k].shape
    }
    # for k, v in pretrained_dict.items():
    #     logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    # for name, para in model.named_parameters():
    #     para.requires_grad = True
    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count)}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")


model = nn.DataParallel(model, device_ids = [3, 2, 0, 1])
device = torch.device("cuda:3")
model.to(device)
model.zero_grad()


### Set the training parameters
lr = 1e-4
amp = True
schedule_ratio = 0.9
schedule_interval = 1
log_interval = 20
batch_size = 4
accumulation_steps = 2

criterion = masked_mse_loss
optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=schedule_ratio
)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


### Set the data preparation parameters
pad_token = "<pad>"
pad_value = 103
mask_value = 102
mask_ratio = 0.15

tokenizer_dir = '../../stformer/tokenizer/'
vocab_file = tokenizer_dir + "scfoundation_gene_vocab.json"
vocab = GeneVocab.from_file(vocab_file)
vocab.append_token(pad_token)
vocab.set_default_index(vocab[pad_token])


### Load the fine-tuning data
adata_file = sys.argv[2]
adata = sc.read_h5ad(adata_file)

np.random.seed(0)
shuffled_indices = np.random.permutation(adata.n_obs)
adata = adata[shuffled_indices]

output_suffix = sys.argv[3]
logger.info(f"Fine-tuning {model_file} on {adata_file}.")
epochs = int(sys.argv[4])


### Finetuning the model on the dataset
tokenizer = Tokenizer(adata, tokenizer_dir, vocab, mask_ratio, mask_value, pad_value, pad_token)

sample_num = adata.shape[0]
mse_train = []
for epoch in range(epochs):
    for slice in range(int(sample_num/10000)+(1 if sample_num%10000 else 0)):
        tokenizer.tokenize_data(slice*10000, (slice+1)*10000)
        tokenizer.prepare_data()
        data_loader = tokenizer.prepare_dataloader(batch_size)

        train(model, data_loader, slice, mse_train, epoch)

# pickle.dump(mse_train, open(f'mse/{model_file[6:-5]}-{output_suffix}.pkl', 'wb'))
model_ckpt = model.module.to('cpu')
torch.save(model_ckpt, f'models/ft-{model_file[6:-5]}-{output_suffix}.ckpt')
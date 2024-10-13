# Copyright (c) 2024, Shenghao Cao & Ye Yuan. Shanghai Jiao Tong University, Shanghai 200240, China
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

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from stformer import logger
from stformer.tokenizer import GeneVocab
from stformer.tokenizer import tokenize_and_pad_batch, random_mask_value
from stformer.model import TransformerModel
from stformer.loss import masked_mse_loss


class SlideData():
    def __init__(self, data_path, slide, vocab, mask_ratio, mask_value, pad_value, pad_token):
        self.data_path = data_path
        self.slide = slide
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.load_data()
    
    def load_data(self):
        adata = sc.read_h5ad(f'{self.data_path}/{self.slide}/deconv.h5ad')

        scfoundation_gene_df = pd.read_csv(f'{tokenizer_dir}/scfoundation_gene_df.csv')
        scfoundation_gene_df.set_index('gene_ids', inplace=True)
        total_gene_num = adata.shape[1]
        adata = adata[:, adata.var_names.isin(scfoundation_gene_df.index)]
        adata.var['gene_name'] = scfoundation_gene_df.loc[adata.var_names, 'gene_symbols'].values
        seleted_gene_num = adata.shape[1]
        genes = adata.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        logger.info(
            f"match {seleted_gene_num}/{total_gene_num} genes "
            f"in vocabulary of size 19264."
        )

        ligand_database = pd.read_csv(tokenizer_dir+'ligand_database.csv', header=0, index_col=0)
        ligand_symbol = ligand_database[ligand_database.sum(1)>1].index.values
        ligand_ids = self.vocab(ligand_symbol.tolist())
        adata = adata[(adata[:,adata.var['gene_name'].isin(ligand_symbol)].X.sum(1)>0).A.T[0],:]

        celltype_proportion = adata.obsm['q05_cell_abundance_w_sf']
        celltype_proportion.rename(columns=lambda x: x[23:], inplace=True)
        celltype_proportion = celltype_proportion.div(celltype_proportion.sum(axis=1), axis=0)
        celltype_proportion[celltype_proportion < 0.05] = 0
        celltype_proportion = celltype_proportion.div(celltype_proportion.sum(axis=1), axis=0)

        for celltype in adata.layers.keys():
            adata.X = adata.layers[celltype]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.uns.pop('log1p')
            adata.layers[celltype] = adata.X

        self.adata = adata
        self.celltype_proportion = celltype_proportion
        self.gene_ids = gene_ids
        self.ligand_ids = ligand_ids
    
    def get_niche_samples(self):
        samples_expression = []
        samples_ctprop = []
        for i in range(self.adata.shape[0]):
            ct_prop = self.celltype_proportion.iloc[i][self.celltype_proportion.iloc[i]>0]

            niche_counts = np.concatenate([self.adata.layers[ct][i].A for ct in ct_prop.index])
            niche_counts[:,~np.isin(self.gene_ids, self.ligand_ids)] = 0

            niche_ctprop = ct_prop.values

            for ct in ct_prop.index:
                counts = self.adata.layers[ct][i].A
                samples_expression.append(np.concatenate([counts, niche_counts],axis=0))
                samples_ctprop.append(niche_ctprop)

        self.expression = samples_expression
        self.ctprop = samples_ctprop

    def tokenize_data(self):
        (   train_data,
            valid_data,
            train_ctprop,
            valid_ctprop,
        ) = train_test_split(
            self.expression, self.ctprop, test_size=0.1, shuffle=True
        )

        max_seq_len = np.max(np.count_nonzero(self.adata.X.A, axis=1))+2
        max_niche_cell_num = (self.celltype_proportion>0).sum(1).max()
        self.max_seq_len = max_seq_len
        self.max_niche_cell_num = max_niche_cell_num

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            train_ctprop,
            self.gene_ids,
            max_len = max_seq_len,
            max_niche_cell_num = max_niche_cell_num,
            vocab = self.vocab,
            pad_token = self.pad_token,
            pad_value = self.pad_value,
            append_cls = False,  # append <cls> token at the beginning
            include_zero_gene = False,
        )

        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            valid_ctprop,
            self.gene_ids,
            max_len = max_seq_len,
            max_niche_cell_num = max_niche_cell_num,
            vocab = self.vocab,
            pad_token = self.pad_token,
            pad_value = self.pad_value,
            append_cls = False,
            include_zero_gene = False,
        )

        logger.info(
            f"train set number of samples: {tokenized_train['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {tokenized_train['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {tokenized_train['niche_genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {tokenized_valid['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {tokenized_valid['niche_genes'].shape[1]}"
        )

        self.tokenized_train = tokenized_train
        self.tokenized_valid = tokenized_valid

    def prepare_data(self):
        masked_values_train = random_mask_value(
            self.tokenized_train["center_values"],
            mask_ratio = self.mask_ratio,
            mask_value = self.mask_value,
            pad_value = self.pad_value,
        )
        masked_values_valid = random_mask_value(
            self.tokenized_valid["center_values"],
            mask_ratio = self.mask_ratio,
            mask_value = self.mask_value,
            pad_value = self.pad_value,
        )
        logger.info(
            f"random masking ratio of masked values in train: "
            f"{(masked_values_train == self.mask_value).sum() / (masked_values_train - self.pad_value).count_nonzero() *100:2.2f}%"
            f"\n\t\t  random masking ratio of masked values in valid: "
            f"{(masked_values_valid == self.mask_value).sum() / (masked_values_valid - self.pad_value).count_nonzero() *100:2.2f}%"
        )

        train_data_pt = {
            "center_gene_ids": self.tokenized_train["center_genes"],
            "input_center_values": masked_values_train,
            "target_center_values": self.tokenized_train["center_values"],
            "niche_gene_ids": self.tokenized_train["niche_genes"],
            "input_niche_values": self.tokenized_train["niche_values"],
            "niche_feature_lens": self.tokenized_train["niche_feature_lens"],
            "cross_attn_bias": self.tokenized_train["cross_attn_bias"],
        }

        valid_data_pt = {
            "center_gene_ids": self.tokenized_valid["center_genes"],
            "input_center_values": masked_values_valid,
            "target_center_values": self.tokenized_valid["center_values"],
            "niche_gene_ids": self.tokenized_valid["niche_genes"],
            "input_niche_values": self.tokenized_valid["niche_values"],
            "niche_feature_lens": self.tokenized_valid["niche_feature_lens"],
            "cross_attn_bias": self.tokenized_valid["cross_attn_bias"],
        }

        return train_data_pt, valid_data_pt


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["center_gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


def train(model: nn.Module, loader: DataLoader, epoch, slide_num, slide, mse_train) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_mse = 0.0
 
    start_time = time.time()

    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        niche_feature_lens = batch_data["niche_feature_lens"].to(device)
        if niche_feature_lens.size(0)<7:
            continue
        center_gene_ids = batch_data["center_gene_ids"].to(device)
        input_center_values = batch_data["input_center_values"].to(device)
        target_center_values = batch_data["target_center_values"].to(device)
        niche_gene_ids = batch_data["niche_gene_ids"].to(device)
        input_niche_values = batch_data["input_niche_values"].to(device)
        cross_attn_bias = batch_data["cross_attn_bias"].to(device)

        encoder_src_key_padding_mask = niche_gene_ids.eq(vocab[pad_token])
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

        model.zero_grad()
        scaler.scale(loss_mse).backward()
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

        total_mse += loss_mse.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            sec_per_batch = (time.time() - start_time) / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
                f"{batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | sec/batch {sec_per_batch:5.1f} | "
                f"mse {cur_mse:5.5f} | "
            )
            mse_train.append(cur_mse)
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_num = 0
    
    with torch.no_grad():
        for batch_data in loader:
            niche_feature_lens = batch_data["niche_feature_lens"].to(device)
            if niche_feature_lens.size(0)<7:
                continue
            center_gene_ids = batch_data["center_gene_ids"].to(device)
            input_center_values = batch_data["input_center_values"].to(device)
            target_center_values = batch_data["target_center_values"].to(device)
            niche_gene_ids = batch_data["niche_gene_ids"].to(device)
            input_niche_values = batch_data["input_niche_values"].to(device)
            cross_attn_bias = batch_data["cross_attn_bias"].to(device)

            encoder_src_key_padding_mask = niche_gene_ids.eq(vocab[pad_token])
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
                loss = criterion(output_dict["mlm_output"], target_center_values, decoder_masked_positions)
            
            total_loss += loss.item() * decoder_masked_positions.sum().item()
            total_num += decoder_masked_positions.sum().item()
    
    return total_loss / total_num
        

def train_and_evaluate(model, train_data_pt, valid_data_pt, epoch, batch_size, slide_num, slide, mse_train, mse_valid):
    best_val_loss = float("inf")
    start_time = time.time()

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size,
        shuffle = False,
        drop_last = False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size,
        shuffle=False,
        drop_last=False,
    )

    train(model, train_loader, epoch, slide_num, slide, mse_train)

    val_loss = evaluate(model, valid_loader)
    mse_valid.append(val_loss)
        
    elapsed = time.time() - start_time
        
    logger.info("-" * 89)
    logger.info(
        f"| end of {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
        f"time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info(f"Best model with score {best_val_loss:5.4f}")


### Arguments
# dataset_name, model_checkpoint = sys.argv[1], sys.argv[2]

### Prepare the pretraining model
if len(sys.argv) < 3:
    logger.info("Initialize pretraining model")

    embsize = 768 #256
    d_hid = 3072 #1024
    nhead = 12 #4
    nlayers = 6 #12
    dropout = 0.1
    cell_emb_style = 'max-pool'

    logger.info("Loading scFoundation model ...")
    from scfoundation import load
    pretrainmodel, pretrainconfig = load.load_model_frommmf('scfoundation/models/models.ckpt')

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

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    # for name, para in model.named_parameters():
    #     para.requires_grad = True
    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

else:
    logger.info("Load pretraining model")
    model_file = sys.argv[2]
    model = torch.load(f'pretraining/{model_file}', map_location='cpu')
    # model = pickle.load(open(f'pretraining/{model_file}', 'rb'))

model = nn.DataParallel(model, device_ids = [3, 2, 1, 0])
device = torch.device("cuda:3")
model.to(device)


### Set the training parameters
lr = 1e-4
amp = True
schedule_ratio = 0.9
schedule_interval = 1
log_interval = 10
epochs = 3
batch_size = 8

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

tokenizer_dir = 'stformer/tokenizer/'
vocab_file = tokenizer_dir + "scfoundation_gene_vocab.json"
vocab = GeneVocab.from_file(vocab_file)
vocab.append_token(pad_token)
vocab.set_default_index(vocab[pad_token])


### Prepare the pretraining data
dataset = sys.argv[1]
data_path = f'data/{dataset}/'
slide_num = 0
for slide in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, slide)):
        slide_num += 1
logger.info(f"Train {epochs} epochs on dataset {dataset}: totally {slide_num} slides")


### Train the model on the dataset
mse_train = []
mse_valid = []
for epoch in range(1, epochs + 1):
    slide_num = 0
    for slide in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, slide)):
            slide_num += 1
            logger.info(f"Training epoch {epoch} on dataset {dataset} - slide {slide_num} {slide}")
            slideData = SlideData(data_path, slide, vocab, mask_ratio, mask_value, pad_value, pad_token)
            slideData.get_niche_samples()
            slideData.tokenize_data()
            # if slideData.max_niche_cell_num * slideData.max_seq_len < 30000:
            #     batch_size = 24
            # else:
            #     batch_size = 12
            train_data_pt, valid_data_pt = slideData.prepare_data()
            train_and_evaluate(model, train_data_pt, valid_data_pt, epoch, batch_size, slide_num, slide, mse_train, mse_valid)
            pickle.dump(mse_train, open(f'pretraining/mse/mse_train_{dataset}.pkl', 'wb'))
            pickle.dump(mse_valid, open(f'pretraining/mse/mse_valid_{dataset}.pkl', 'wb'))
    scheduler.step()

# pickle.dump(model, open(f'pretraining/model.ckpt', 'wb'))
model.to('cpu')
torch.save(model, f'pretraining/model.ckpt')
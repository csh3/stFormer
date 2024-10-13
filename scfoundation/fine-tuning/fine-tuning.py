# Copyright (c) 2024, Shenghao Cao & Ye Yuan. Shanghai Jiao Tong University, Shanghai 200240, China
# Some classes and functions are modified from scGPT (https://github.com/bowang-lab/scGPT), copyright (c) 2022 suber
# Some classes and functions are modified from scFoundation (https://github.com/biomap-research/scFoundation), copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import anndata
import scanpy as sc
import torch
from torch import nn
from scfoundation import load


class SlideData():
    def __init__(self, data_path, slide):
        self.data_path = data_path
        self.slide = slide
        self.load_data()

    def load_data(self):
        adata = sc.read_h5ad(f'{self.data_path}/{self.slide}/deconv.h5ad')
        
        scfoundation_gene_df = pd.read_csv(f'{tokenizer_dir}/scfoundation_gene_df.csv')
        scfoundation_gene_df.set_index('gene_ids', inplace=True)
        total_gene_num = adata.shape[1]
        adata = adata[:, adata.var_names.isin(scfoundation_gene_df.index)]
        adata.var['gene_name'] = scfoundation_gene_df.loc[adata.var_names, 'gene_symbols'].values
        seleted_gene_num = adata.shape[1]

        print(
            f"match {seleted_gene_num}/{total_gene_num} genes "
            f"in vocabulary of size 19264."
        )

        for celltype in adata.layers.keys():
            adata.X = adata.layers[celltype]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.uns.pop('log1p')
            adata.layers[celltype] = adata.X

        celltype_proportion = adata.obsm['q05_cell_abundance_w_sf']
        celltype_proportion.rename(columns=lambda x: x[23:], inplace=True)
        celltype_proportion = celltype_proportion.div(celltype_proportion.sum(axis=1), axis=0)
        celltype_proportion[celltype_proportion < 0.05] = 0
        celltype_proportion = celltype_proportion.div(celltype_proportion.sum(axis=1), axis=0)

        self.adata = adata
        self.celltype_proportion = celltype_proportion

    def get_sc_data(self):
        barcode_list = []
        gexpr_feature = []
        celltypes_labels = []
        for i in range(self.adata.shape[0]):
            barcode = self.adata.obs.index[i]
            ct_prop = self.celltype_proportion.iloc[i][self.celltype_proportion.iloc[i]>0]
            cell_num = 0
            for ct in ct_prop.index:
                celltypes_labels.append(ct)
                cell_num += 1
                barcode_list.append(f'{barcode}_{cell_num}')
                gexpr_feature.append(self.adata.layers[ct][i].A)
        gexpr_feature = np.concatenate(gexpr_feature)

        adata_sc = anndata.AnnData(X=gexpr_feature, obs=pd.DataFrame({'celltype': celltypes_labels}, index=barcode_list), var=pd.DataFrame({'gene_name': self.adata.var['gene_name'].values}, index=self.adata.var['gene_name']), obsm=None)
        self.adata = adata_sc

    def prepare_data(self, gexpr_feature, idx, col):
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)

        gene_list_df = pd.read_csv(f'{tokenizer_dir}/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])

        gexpr_feature, _ = load.main_gene_selection(gexpr_feature, gene_list)

        S = gexpr_feature.sum(1)
        T=S
        TS = np.concatenate([[np.log10(T)],[np.log10(S)]],axis=0).T
        data = np.concatenate([gexpr_feature,TS],axis=1)

        return data

    def prepare_train_and_valid_data(self):
        col = self.adata.var.gene_name.tolist()
        idx = self.adata.obs_names.tolist()
        gexpr_feature = self.adata.X

        (   gexpr_feature_train,
            gexpr_feature_valid,
            idx_train,
            idx_valid,
        ) = train_test_split(
            gexpr_feature, idx, test_size=0.1, shuffle=True
        )

        data_train = self.prepare_data(gexpr_feature_train, idx_train, col)
        data_valid = self.prepare_data(gexpr_feature_valid, idx_valid, col)

        return data_train, data_valid


class scFoundation(nn.Module):
    def __init__(
            self,
            scf_token_emb,
            scf_pos_emb,
            scf_encoder,
            scf_decoder,
            scf_decoder_embed,
            scf_norm,
            scf_to_final,
    ):
        super(scFoundation, self).__init__()

        # encoder
        self.token_emb = scf_token_emb
        self.pos_emb = scf_pos_emb

        # ## DEBUG
        self.encoder = scf_encoder

        ##### decoder
        self.decoder = scf_decoder
        self.decoder_embed = scf_decoder_embed
        self.norm = scf_norm
        self.to_final = scf_to_final
        # self.to_final = ExprDecoder(512)

    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels, decoder_data,
                decoder_position_gene_ids, decoder_data_padding_labels, **kwargs):

        # token and positional embedding
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight = 0)

        position_emb = self.pos_emb(encoder_position_gene_ids)
        x += position_emb
        x = self.encoder(x, padding_mask=padding_label)

        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data += position_emb

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        x = self.norm(x)
        # return x
        x = self.to_final(x)
        return x.squeeze(2) 

  
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(self, x):
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x)
        return pred_value
    

def mask_data(data, mask_ratio):
    data_raw = torch.from_numpy(data).float()
    data = data.copy()

    for i in range(len(data)):
        row = data[i]
        non_padding_idx = np.nonzero(row[:-2])[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = pretrainconfig['mask_token_id']
    data = torch.from_numpy(data).float()

    return data, data_raw


def getEncoderDecoderData(data, data_raw, config):
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)

    decoder_data = data
    decoder_data_padding = torch.full_like(data, False, device=data.device, dtype=torch.bool)
    decoder_position_gene_ids = data_gene_ids

    new_data_raw = data_raw
    
    encoder_data_labels = torch.logical_and(decoder_data != 0, decoder_data != config['mask_token_id'])
    encoder_data, encoder_data_padding = load.gatherData(decoder_data, encoder_data_labels, config['pad_token_id'])
    encoder_position_gene_ids, _ = load.gatherData(decoder_position_gene_ids, encoder_data_labels,config['pad_token_id'])
    
    data_mask_labels = decoder_data == config['mask_token_id']

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, data_mask_labels, decoder_position_gene_ids, new_data_raw


def train(model: nn.Module, data_train, batch_size, epoch, slide_num, slide, mse_train) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    train_data, train_data_raw = mask_data(data_train, mask_ratio)

    train_encoder_data, train_encoder_position_gene_ids, \
    train_encoder_data_padding, train_encoder_data_labels, \
    train_decoder_data, train_decoder_data_padding, \
    train_data_mask_labels, train_decoder_position_gene_ids, train_new_data_raw = getEncoderDecoderData(train_data, train_data_raw, pretrainconfig)

    num_batches = np.ceil(len(train_encoder_data)/batch_size).astype(int)
    for k in range(0, len(train_encoder_data), batch_size):
        batch = int(k/batch_size+1)
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(train_encoder_data[k:k+batch_size].to(device), train_encoder_data_padding[k:k+batch_size].to(device), 
                        train_encoder_position_gene_ids[k:k+batch_size].to(device), train_encoder_data_labels[k:k+batch_size].to(device),
                        train_decoder_data[k:k+batch_size].to(device),
                        train_decoder_position_gene_ids[k:k+batch_size].to(device), train_decoder_data_padding[k:k+batch_size].to(device))
            loss = ((prediction-train_new_data_raw[k:k+batch_size].to(device))[train_data_mask_labels[k:k+batch_size].to(device)]**2).mean()

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            cur_loss = total_loss / log_interval
            print(f"| {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
                f"{batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | loss {cur_loss:5.2f} | ")
            total_loss = 0.0
            mse_train.append(cur_loss)


def evaluate(model: nn.Module, data_valid, batch_size) -> None:
    model.eval()
    total_loss = 0.0
    total_num = 0

    valid_data, valid_data_raw = mask_data(data_valid, mask_ratio)

    valid_encoder_data, valid_encoder_position_gene_ids, \
    valid_encoder_data_padding, valid_encoder_data_labels, \
    valid_decoder_data, valid_decoder_data_padding, \
    valid_data_mask_labels, valid_decoder_position_gene_ids, valid_new_data_raw = getEncoderDecoderData(valid_data, valid_data_raw, pretrainconfig)
    
    with torch.no_grad():
        for k in range(0, len(valid_encoder_data), batch_size):
            with torch.cuda.amp.autocast(enabled=amp):
                prediction = model(valid_encoder_data[k:k+batch_size].to(device), valid_encoder_data_padding[k:k+batch_size].to(device), 
                            valid_encoder_position_gene_ids[k:k+batch_size].to(device), valid_encoder_data_labels[k:k+batch_size].to(device),
                            valid_decoder_data[k:k+batch_size].to(device),
                            valid_decoder_position_gene_ids[k:k+batch_size].to(device), valid_decoder_data_padding[k:k+batch_size].to(device))
                loss = ((prediction-valid_new_data_raw[k:k+batch_size].to(device))[valid_data_mask_labels[k:k+batch_size].to(device)]**2).mean()
            
            total_loss += loss.item() * valid_data_mask_labels[k:k+batch_size].sum().item()
            total_num += valid_data_mask_labels[k:k+batch_size].sum().item()
        
    return total_loss / total_num


def train_and_evaluate(model, data_train, data_valid, batch_size, epoch, slide_num, slide, mse_train, mse_valid):
    best_val_loss = float("inf")

    train(model, data_train, batch_size, epoch, slide_num, slide, mse_train)

    val_loss = evaluate(model, data_valid, batch_size)
    mse_valid.append(val_loss)
        
    print("-" * 89)
    print(f"| end of {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
        f"valid loss/mse {val_loss:5.4f}")
    print("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"Best model with score {best_val_loss:5.4f}")



### Prepare the pretraining model
print('Load the pretraining model')
pretrainmodel, pretrainconfig = load.load_model_frommmf('scfoundation/models/models.ckpt')
model = scFoundation(pretrainmodel.token_emb,
            pretrainmodel.pos_emb,
            pretrainmodel.encoder,
            pretrainmodel.decoder,
            pretrainmodel.decoder_embed,
            pretrainmodel.norm,
            pretrainmodel.to_final)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
# for name, para in model.named_parameters():
#     para.requires_grad = True
# for name, para in model.decoder.performer.net.layers[:5].named_parameters():
#     para.requires_grad = False
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
print(f"Total Pre freeze Params {(pre_freeze_param_count )}")
print(f"Total Post freeze Params {(post_freeze_param_count )}")

model = torch.nn.DataParallel(model, device_ids=[1, 3, 2, 0])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)


### Set the training parameters
mask_ratio = 0.15
lr=1e-4
amp=True
schedule_ratio=0.9
schedule_interval = 1
log_interval = 10
epochs = 3
batch_size = 8

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=schedule_ratio
)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


### Prepare the fine-tuning data
dataset = sys.argv[1]
data_path = f'data/{dataset}/'
tokenizer_dir = 'stformer/tokenizer/'
slide_num = 0
for slide in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, slide)):
        slide_num += 1
print(f"Fine-tuning {epochs} epochs on dataset {dataset}: totally {slide_num} slides")


### Fine-tuning the model on the dataset
mse_train = []
mse_valid = []
for epoch in range(1, epochs + 1):
    slide_num = 0
    for slide in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, slide)):
            slide_num += 1
            print(f"Fine-tuning epoch {epoch} on dataset {dataset} - slide {slide_num} {slide}")
            slideData = SlideData(data_path, slide)
            slideData.get_sc_data()
            data_train, data_valid = slideData.prepare_train_and_valid_data()
            train_and_evaluate(model, data_train, data_valid, batch_size, epoch, slide_num, slide, mse_train, mse_valid)
            pickle.dump(mse_train, open(f'scfoundation/fine-tuning/mse/mse_train_{dataset}.pkl', 'wb'))
            pickle.dump(mse_valid, open(f'scfoundation/fine-tuning/mse/mse_valid_{dataset}.pkl', 'wb'))
    scheduler.step()

model.to('cpu')
torch.save(model, f'scfoundation/fine-tuning/model_{dataset}.ckpt')
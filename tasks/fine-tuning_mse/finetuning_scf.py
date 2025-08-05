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
    def __init__(self, adata):
        self.adata = adata

    def prepare_data(self):
        gexpr_feature = self.adata.X.A
        S = gexpr_feature.sum(1)
        T=S
        TS = np.concatenate([[np.log10(T)],[np.log10(S)]],axis=0).T
        data = np.concatenate([gexpr_feature,TS],axis=1)

        return data


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


def train(model: nn.Module, data_train, batch_size, epoch, mse_train) -> None:
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
            print(f"| epoch {epoch:2d} | "
                f"{batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | loss {cur_loss:5.2f} | ")
            total_loss = 0.0
            mse_train.append(cur_loss)


### Prepare the pretraining model
print('Load the pretraining model')
pretrainmodel, pretrainconfig = load.load_model_frommmf('../scfoundation/models/models.ckpt')
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
model.zero_grad()


### Set the training parameters
mask_ratio = 0.15
lr=1e-4
amp=True
schedule_ratio=0.9
schedule_interval = 1
log_interval = 10
batch_size = 8

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=schedule_ratio
)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


### Load the fine-tuning data
adata_file = sys.argv[1]
adata = sc.read_h5ad(adata_file)

np.random.seed(0)
shuffled_indices = np.random.permutation(adata.n_obs)
adata = adata[shuffled_indices]

output_suffix = sys.argv[2]
print(f"Fine-tuning scFoundation on {adata_file}.")
epochs = int(sys.argv[3])

### Fine-tuning the model on the dataset
sample_num = adata.shape[0]
mse_train = []
for epoch in range(1, epochs + 1):
    slideData = SlideData(adata)
    data = slideData.prepare_data()
    for slice in range(int(sample_num/10000)+(1 if sample_num%10000 else 0)):
        train(model, data[slice*10000:(slice+1)*10000], batch_size, epoch, mse_train)

# pickle.dump(mse_train, open(f'mse/scf-{output_suffix}.pkl', 'wb'))
model_ckpt = model.module.to('cpu')
torch.save(model_ckpt, f'models/ft-scf-{output_suffix}.ckpt')
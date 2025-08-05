# Copyright (c) 2024, Shenghao Cao & Ye Yuan. Shanghai Jiao Tong University, Shanghai 200240, China
# Some classes and functions are modified from scGPT (https://github.com/bowang-lab/scGPT), copyright (c) 2022 suber
# The gene symbol and value encoding modules and initialization weights are from scFoundation (https://github.com/biomap-research/scFoundation), copyright 2023 BioMap (Beijing) Intelligence Technology Limited

from typing import Dict, Mapping, Optional, Any, Union, Callable
import itertools
import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, _get_seq_len, _detect_is_causal_mask
from flash_attn.bert_padding import pad_input
from stformer.flash_attention import FlashMHA

class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        do_cls: bool = False,
        do_gcl: bool = False,
        nlayers_cls: int = 3,
        n_cls: int = 2,
        nlayers_gcl: int = 3,
        n_gcl: int = 2,
        dropout: float = 0.1,
        cell_emb_style: str = "max-pool",
        pre_norm: bool = False,
        scfoundation_token_emb1: Any = None,
        scfoundation_token_emb2: Any = None,
        scfoundation_pos_emb1: Any = None,
        scfoundation_pos_emb2: Any = None,
    ):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.cell_emb_style = cell_emb_style
        self.pre_norm = pre_norm
        self.scfoundation_token_emb1 = scfoundation_token_emb1
        self.scfoundation_token_emb2 = scfoundation_token_emb2
        self.scfoundation_pos_emb1 = scfoundation_pos_emb1
        self.scfoundation_pos_emb2 = scfoundation_pos_emb2
        if cell_emb_style not in ["avg-pool", "max-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        decoder_layers = FlashTransformerDecoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=self.pre_norm,
        )
        self.transformer_decoder = BiasedTransformerDecoder(decoder_layers, nlayers)
        
        self.decoder = ExprDecoder(d_model)
        if do_cls:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_gcl:
            self.gcl_decoder = GclDecoder(d_model, n_gcl, nlayers=nlayers_gcl)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
    ) -> Tensor:
        src = self.scfoundation_pos_emb1(src)
        values = self.scfoundation_token_emb1(torch.unsqueeze(values, 2).float(), output_weight = 0)
        output = src + values
        return output  # (batch, seq_len, embsize)

    def _decode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        cross_attn_bias: Tensor,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        src = self.scfoundation_pos_emb2(src)
        values = self.scfoundation_token_emb2(torch.unsqueeze(values, 2).float(), output_weight = 0)
        tgt = src + values
        output = self.transformer_decoder(tgt, memory, cross_attn_bias, tgt_key_padding_mask=src_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        return output

    def _pad_information_of_split_input(self, encoder_feature_lens: Tensor, max_seqlen=None):
        if max_seqlen is None:
            max_seqlen = encoder_feature_lens.max().item()
        total_cell_num = encoder_feature_lens.size(0)
        key_padding_mask = torch.zeros((total_cell_num, max_seqlen), dtype=torch.bool, device=encoder_feature_lens.device)
        for i,val in enumerate(encoder_feature_lens):
            key_padding_mask[i, val:] = True
        indices = (~key_padding_mask.view(-1)).nonzero(as_tuple=True)[0]
        return indices, total_cell_num, max_seqlen, key_padding_mask

    def forward(
        self,
        encoder_src: Tensor,
        encoder_values: Tensor,
        encoder_src_key_padding_mask: Tensor,
        decoder_src: Tensor,
        decoder_values: Tensor,
        decoder_src_key_padding_mask: Tensor,
        cross_attn_bias: Tensor,
        CLS: bool = False,
        GCL: bool = False,
        output_gene_emb: bool = False,
    ) -> Mapping[str, Tensor]:
        
        memory = self._encode(encoder_src, encoder_values)
        decoder_output = self._decode(
            decoder_src, decoder_values, decoder_src_key_padding_mask, memory, encoder_src_key_padding_mask, cross_attn_bias
        )

        output = {}
        mlm_output = self.decoder(decoder_output)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)

        if self.cell_emb_style == 'max-pool':
            cell_emb = torch.cat([torch.max(decoder_output[k][~decoder_src_key_padding_mask[k]], dim=0)[0].unsqueeze(0) for k in range(decoder_output.size(0))])
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.cat([torch.mean(decoder_output[k][~decoder_src_key_padding_mask[k]], dim=0).unsqueeze(0) for k in range(decoder_output.size(0))])
        output["cell_emb"] = cell_emb

        if output_gene_emb:
            output["gene_emb"] = decoder_output
        
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if GCL:
            output["gcl_output"] = self.gcl_decoder(decoder_output)
        
        return output


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


#Flash attention decoder layer
class FlashTransformerDecoderLayer(nn.Module):
    r"""The class is modified from torch.nn.TransformerDecoderLayer to support the
    FlashAttention. It is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = FlashMHA(d_model, nhead, batch_first=batch_first, attention_dropout=dropout, 
                                attention_type='inner', bias=bias, **factory_kwargs)
        self.cross_attn = FlashMHA(d_model, nhead, batch_first=batch_first, attention_dropout=dropout,
                                attention_type='cross', bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cross_attn_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        
        # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
        tgt_key_padding_mask_ = ~tgt_key_padding_mask
        memory_key_padding_mask_ = ~memory_key_padding_mask
        
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_key_padding_mask_, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, cross_attn_bias, tgt_key_padding_mask_, memory_key_padding_mask_, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_key_padding_mask_, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, cross_attn_bias, tgt_key_padding_mask_, memory_key_padding_mask_, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, tgt_key_padding_mask, tgt_is_causal) -> Tensor:
        x = self.self_attn(x,tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, memory, attn_bias, tgt_key_padding_mask, memory_key_padding_mask, memory_is_causal) -> Tensor:
        output = torch.zeros_like(x, device=x.device)
        selected_index = ~((~memory_key_padding_mask).all(dim=1))
        
        x = x[selected_index]
        memory = memory[selected_index]
        attn_bias = attn_bias[selected_index]
        tgt_key_padding_mask = tgt_key_padding_mask[selected_index]
        memory_key_padding_mask = memory_key_padding_mask[selected_index]
       
        r = self.cross_attn(x, mem=memory, attn_bias=attn_bias, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, memory_is_causal=memory_is_causal)[0]
        
        output = output.type(r.dtype)
        output[selected_index] = r
        
        return self.dropout2(output)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


#Biased attention decoder
class BiasedTransformerDecoder(nn.Module):
    r"""The class is modified from torch.nn.TransformerDecoder to support the
    biased cross-attention. BiasedTransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the FlashTransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, cross_attn_bias: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, cross_attn_bias=cross_attn_bias, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

    
class ExprDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        return dict(pred=pred_value)


class ClsDecoder(nn.Module):
    """
    Decoder for cell classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
    

class GclDecoder(nn.Module):
    """
    Decoder for gene classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_gcl: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()

        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_gcl)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
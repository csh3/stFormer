### This file is modified from the original file in the pip module flash_attn(v1.0.5) 

import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange, repeat
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, attention_type='inner'):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.attention_type = attention_type

    def forward(self, qkv=None, q=None, kv=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """

        if self.attention_type == 'inner':
            assert qkv.dtype in [torch.float16, torch.bfloat16]
            assert qkv.is_cuda
    
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            nheads = qkv.shape[-2]
            x = rearrange(qkv, 'b s three h d -> b s (three h d)')
            x_unpad, indices, cu_seqlens, max_s = unpad_input(x, tgt_key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=tgt_is_causal
            )
            output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                        indices, batch_size, seqlen),
                            'b s (h d) -> b s h d', h=nheads)
        
        elif self.attention_type == 'inter':
            assert q.dtype in [torch.float16, torch.bfloat16]
            assert q.is_cuda
            assert kv.dtype in [torch.float16, torch.bfloat16]
            assert kv.is_cuda

            batch_size = q.shape[0]
            seqlen = q.shape[1]
            nheads = q.shape[-2]
            x = rearrange(q, 'b s h d -> b s (h d)')
            x_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(x, tgt_key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            y = rearrange(kv, 'b s two h d -> b s (two h d)')
            y_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(y, memory_key_padding_mask)
            y_unpad = rearrange(y_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_varlen_kvpacked_func(
                x_unpad, y_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=memory_is_causal
            )
            output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                        indices_q, batch_size, seqlen),
                            'b s (h d) -> b s h d', h=nheads)

        return output, None


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    The class is modified from flash_attn.modules.mha.CrossAttention
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, attn_bias, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            attn_bias: The tensor containing the attention bias to apply to the attention score. (B, Sk)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        scores = scores + rearrange(attn_bias, "b s -> b 1 1 s")
        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k), -np.inf, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            # causal mask needs to take into account the difference between seqlen_q and seqlen_k
            row_idx = rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = (
                seqlen_k
                if key_padding_mask is None
                else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            )
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -np.inf)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0, attention_type='inner',
                 device=None, dtype=None) -> None:
        self.batch_first = batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_type = attention_type

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        # assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        if attention_type == 'inner':
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
            self.attn = FlashAttention(attention_dropout=attention_dropout, attention_type=attention_type)

        elif attention_type == 'inter':
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
            self.attn = FlashAttention(attention_dropout=attention_dropout, attention_type=attention_type)

        elif attention_type == 'cross':
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
            self.attn = CrossAttention(attention_dropout=attention_dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, tgt, mem=None, attn_bias=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False):
        """tgt: (batch, seqlen_tgt, hidden_dim) (where hidden_dim = num heads * head dim)
        mem: (batch, seqlen_mem, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_bias: (batch, seqlen_mem)
        tgt_key_padding_mask: bool tensor of shape (batch, seqlen_tgt)
        memory_key_padding_mask: bool tensor of shape (batch, seqlen_mem)
        """
        if self.attention_type == 'inner':
            qkv = self.Wqkv(tgt)
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
            context, attn_weights = self.attn(qkv=qkv, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)

        elif self.attention_type == 'inter':
            q = self.Wq(tgt)
            q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
            kv = self.Wkv(mem)
            kv = rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.num_heads)
            context, attn_weights = self.attn(q=q, kv=kv, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, memory_is_causal=memory_is_causal)

        elif self.attention_type == 'cross':
            q = self.Wq(tgt)
            q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
            kv = self.Wkv(mem)
            kv = rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.num_heads)
            context, attn_weights = self.attn(q, kv, attn_bias, causal=memory_is_causal, key_padding_mask=memory_key_padding_mask)

        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
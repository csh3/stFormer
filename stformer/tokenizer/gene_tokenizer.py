# Copyright (c) 2024, Shenghao Cao & Ye Yuan. Shanghai Jiao Tong University, Shanghai 200240, China
# The script is modified from scGPT (https://github.com/bowang-lab/scGPT), copyright (c) 2022 suber

import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab

# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers import AutoTokenizer, BertTokenizer

from .. import logger


class GeneVocab(Vocab):
    """
    Vocabulary for genes.
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a gene list.

        Args:
            gene_list_or_vocab (List[str] or Vocab): List of gene names or a
                Vocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to "<pad>",
                if "<pad>" is in the vocabulary.
        """
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.vocab)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        Build a Vocab from an iterator. This function is modified from
        torchtext.vocab.build_vocab_from_iterator. The original function always
        splits tokens into characters, which is not what we want.

        Args:
            iterator (Iterable): Iterator used to build Vocab. Must yield list
                or iterator of tokens.
            min_freq (int): The minimum frequency needed to include a token in
                the vocabulary.
            specials (List[str]): Special symbols to add. The order of supplied
                tokens will be preserved.
            special_first (bool): Whether to add special tokens to the beginning

        Returns:
            torchtext.vocab.Vocab: A `Vocab` object
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        word_vocab = torch_vocab.vocab(ordered_dict, min_freq=min_freq)
        return word_vocab

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])


def get_default_gene_vocab() -> GeneVocab:
    """
    Get the default gene vocabulary, consisting of gene symbols and ids.
    """
    vocab_file = Path(__file__).parent / "default_gene_vocab.json"
    if not vocab_file.exists():
        logger.info(
            f"No existing default vocab, will build one and save to {vocab_file}"
        )
        return _build_default_gene_vocab(save_vocab_to=vocab_file)
    logger.info(f"Loading gene vocabulary from {vocab_file}")
    return GeneVocab.from_file(vocab_file)


def _build_default_gene_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> GeneVocab:
    """
    Build the default gene vocabulary from HGNC gene symbols.

    Args:
        download_source_to (str): Directory to download the source data.
        save_vocab_to (Path or str): Path to save the vocabulary. If None,
            the vocabulary will not be saved. Default to None.
    """
    gene_collection_file = (
        Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
    )
    if not gene_collection_file.exists():
        # download and save file from url
        url = (
            "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
            "col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
            "=on&order_by=gd_app_sym_sort&format=text&submit=submit"
        )
        import requests

        r = requests.get(url)
        gene_collection_file.write_text(r.text)

    logger.info(f"Building gene vocabulary from {gene_collection_file}")
    df = pd.read_csv(gene_collection_file, sep="\t")
    gene_list = df["Approved symbol"].dropna().unique().tolist()
    gene_vocab = GeneVocab(gene_list)  # no special tokens set in default vocab
    if save_vocab_to is not None:
        gene_vocab.save_json(Path(save_vocab_to))
    return gene_vocab


def tokenize_batch(
    data: np.ndarray,
    ctprop: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[List[Union[torch.Tensor, np.ndarray]]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data[0].shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data[0].shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data[0].shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data[0].shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        genes = []
        values = []
        mod_types = []

        for counts in row:
            if include_zero_gene:
                values.append(counts)
                genes.append(gene_ids)
                if mod_type is not None:
                    mod_types.append(mod_type)
            else:
                idx = np.nonzero(counts)[0]
                values.append(counts[idx])
                genes.append(gene_ids[idx])
                if mod_type is not None:
                    mod_types.append(mod_type[idx])
        
        ctp = ctprop[i]
        attn_bias = []
        for k in range(len(ctp)):
            attn_bias.append(torch.full((len(genes[k+1]),), np.log(ctp[k])))

        if append_cls:
            # each cell in one niche appends a cls token
            for j in range(len(genes)):
                genes[j] = np.insert(genes[j], 0, cls_id)
                values[j] = np.insert(values[j], 0, 0)
                if mod_type is not None:
                    mod_types[j] = np.insert(mod_types[j], 0, cls_id_mod_type)
        if return_pt:
            genes = [torch.from_numpy(k).long() for k in genes]
            values = [torch.from_numpy(k).float() for k in values]
            if mod_type is not None:
                mod_types = [torch.from_numpy(k).long() for k in mod_types]
    
        tokenized_data.append((genes, values, mod_types, attn_bias))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    max_niche_cell_num: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
):
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_niche_len = max_niche_cell_num*max_len
    max_center_gene_len = min(max_len, max(len(batch[i][0][0])  for i in range(len(batch))))
    max_niche_gene_len = min(max_niche_len, max(sum(len(batch[i][0][j]) for j in range(1,len(batch[i][0]))) for i in range(len(batch))))
    max_niche_cell_num = max(len(batch[i][0])-1 for i in range(len(batch)))

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]

    center_gene_ids_list = []
    center_values_list = []
    center_mod_types_list = []

    niche_gene_ids_list = []
    niche_values_list = []
    niche_mod_types_list = []
    feature_len_list = []

    attn_bias_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types, attn_bias = batch[i]
        center_gene_ids = gene_ids[0]
        center_values = values[0]
        if len(mod_types)>0:
            center_mod_types = mod_types[0]

        if len(center_gene_ids) > max_center_gene_len:
            if not cls_appended:
                idx = np.random.choice(len(center_gene_ids), max_center_gene_len, replace=False)
            else:
                idx = np.random.choice(len(center_gene_ids) - 1, max_center_gene_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            center_gene_ids = center_gene_ids[idx]
            center_values = center_values[idx]
            if len(mod_types)>0:
                center_mod_types = center_mod_types[idx]

        if len(center_gene_ids) < max_center_gene_len:
            center_gene_ids = torch.cat(
                    [
                        center_gene_ids,
                        torch.full(
                            (max_center_gene_len - len(center_gene_ids),), pad_id, dtype=center_gene_ids.dtype
                        ),
                    ]
                )
            center_values = torch.cat(
                    [
                        center_values,
                        torch.full((max_center_gene_len - len(center_values),), pad_value, dtype=center_values.dtype),
                    ]
                )
            if len(mod_types)>0:
                center_mod_types = torch.cat(
                        [
                            center_mod_types,
                            torch.full(
                                (max_center_gene_len - len(center_mod_types),), mod_pad_id, dtype=center_mod_types.dtype,),
                        ]
                    )
        
        center_gene_ids_list.append(center_gene_ids)
        center_values_list.append(center_values)
        if len(mod_types)>0:
            center_mod_types_list.append(center_mod_types)

        total_feature_len = sum([len(gene_ids[j]) for j in range(1,len(gene_ids))])
        if total_feature_len > max_niche_gene_len:
            # sample max_len genes
            for j in range(1,len(gene_ids)):
                cur_feature_len = len(gene_ids[j])
                sample_gene_num = int(cur_feature_len*max_niche_gene_len/total_feature_len)
                if not cls_appended:
                    idx = np.random.choice(len(gene_ids[j]), sample_gene_num, replace=False)
                else:
                    idx = np.random.choice(len(gene_ids[j]) - 1, sample_gene_num - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
                gene_ids[j] = gene_ids[j][idx]
                values[j] = values[j][idx]
                if len(mod_types)>0:
                    mod_types[j] = mod_types[j][idx]
        
        feature_len = torch.tensor([len(gene_ids[j]) for j in range(1,len(gene_ids))], dtype=torch.int64)
        if len(feature_len) < max_niche_cell_num:
            feature_len = torch.cat(
                [
                    feature_len,
                    torch.full(
                        (max_niche_cell_num - len(feature_len),),
                        0,
                        dtype=feature_len.dtype,
                    ),
                ]
            )

        if len(gene_ids[1:])>0:
            niche_gene_ids = torch.cat(gene_ids[1:])
            niche_values = torch.cat(values[1:])
            if len(mod_types)>0:
                niche_mod_types = torch.cat(mod_types[1:])
            attn_bias = torch.cat(attn_bias)
        else:
            niche_gene_ids = torch.tensor([], dtype=center_gene_ids.dtype)
            niche_values = torch.tensor([], dtype=center_values.dtype)
            if len(mod_types)>0:
                niche_mod_types = torch.tensor([], dtype=center_mod_types.dtype)
            attn_bias = torch.tensor([], dtype=center_values.dtype)

        if len(niche_gene_ids) < max_niche_gene_len:
            niche_gene_ids = torch.cat(
                    [
                        niche_gene_ids,
                        torch.full(
                            (max_niche_gene_len - len(niche_gene_ids),), pad_id, dtype=center_gene_ids.dtype
                        ),
                    ]
                )
            niche_values = torch.cat(
                    [
                        niche_values,
                        torch.full((max_niche_gene_len - len(niche_values),), pad_value, dtype=center_values.dtype),
                    ]
                )
            if len(mod_types)>0:
                niche_mod_types = torch.cat(
                        [
                            niche_mod_types,
                            torch.full(
                                (max_niche_gene_len - len(niche_mod_types),), mod_pad_id, dtype=center_mod_types.dtype,),
                        ]
                    )
            attn_bias = torch.cat(
                    [
                        attn_bias,
                        torch.full(
                            (max_niche_gene_len - len(attn_bias),), -np.inf, dtype=attn_bias.dtype
                        ),
                    ]
                )

        feature_len_list.append(feature_len)
        niche_gene_ids_list.append(niche_gene_ids)
        niche_values_list.append(niche_values)
        if len(mod_types)>0:
            niche_mod_types_list.append(niche_mod_types)
        attn_bias_list.append(attn_bias)

    batch_padded = {
        "center_genes": torch.stack(center_gene_ids_list, dim=0),
        "center_values": torch.stack(center_values_list, dim=0),
        "niche_genes": torch.stack(niche_gene_ids_list, dim=0),
        "niche_values": torch.stack(niche_values_list, dim=0),
        "niche_feature_lens": torch.stack(feature_len_list, dim=0),
        "cross_attn_bias": torch.stack(attn_bias_list, dim=0),
    }
    if len(center_mod_types_list)>0:
        batch_padded["center_mod_types"] = torch.stack(center_mod_types_list, dim=0)
    if len(niche_mod_types_list)>0:
        batch_padded["niche_mod_types"] = torch.stack(niche_mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    ctprop: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    max_niche_cell_num: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        ctprop,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        max_niche_cell_num,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0][0:]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code using only one model for
encoding both question and context
"""

import logging
from typing import Dict, List

import torch
from torch import Tensor as T
import numpy as np
from transformers.tokenization_bert import BertTokenizer

from dpr.models.biencoder import BiEncoder
from .hf_models import _add_special_tokens, get_optimizer, HFBertEncoder
from .hf_models import BertTensorizer as Tensorizer

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    ).to(cfg.device)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
            use_lamb=cfg.train.lamb,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg, biencoder)
    return tensorizer, biencoder, optimizer


def get_bert_tensorizer(cfg, biencoder):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    tokenizer = get_bert_tokenizer(
        pretrained_model_cfg, biencoder=biencoder, do_lower_case=cfg.do_lower_case
    )
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def get_bert_tokenizer(pretrained_cfg_name: str, biencoder: BiEncoder, do_lower_case: bool = True):
    """If needed, this tokenizer will be added one special token [QST] representing the question token"""
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )
    # Add [QST] token
    encoder_embeddings = biencoder.question_model.resize_token_embeddings()
    before = encoder_embeddings.weight.shape
    tokenizer.add_special_tokens({"additional_special_tokens": ["[QST]"]})

    with torch.no_grad():
        encoder_embeddings = biencoder.question_model.resize_token_embeddings(len(tokenizer))
        encoder_embeddings.weight[-1, :] = encoder_embeddings.weight[tokenizer.cls_token_id, :].detach().clone()  # intialize with [CLS] embedding
    assert biencoder.ctx_model.resize_token_embeddings().weight.shape[0] == encoder_embeddings.weight.shape[0], \
        "Context and question encoders are not the same!"
    logger.info(f"Added [QST] token: before: {tuple(before)}, after: {tuple(encoder_embeddings.weight.shape)}")

    return tokenizer


class BertTensorizer(Tensorizer):
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:  # title + passage
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:  # question
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
            # Change [CLS] to [QST]
            assert token_ids[0] == self.tokenizer.cls_token_id
            token_ids[0] = self.tokenizer.convert_tokens_to_ids("[QST]")

        token_ids = self.to_max_length(np.array(token_ids), apply_max_len=apply_max_len)

        return torch.from_numpy(token_ids)

    def concatenate_inputs(
        self,
        ids: Dict[str, List[int]],
        get_passage_offset: bool = False,
        to_max_length: bool = False,  # for backward compatibility, this is set to False by default
    ) -> T:
        """
        Simply concatenate inputs by adding [CLS] at the beginning and [SEP] at between and end.
        """
        # 3 mode: only question, only passage ("passage_title" + "passage") or all
        current_mode = set(ids.keys())
        allowed_modes = [{"question"}, {"passage_title", "passage"}, {"question", "passage_title", "passage"}]
        if current_mode not in allowed_modes:
            raise ValueError(f"Unexpected keys: {list(ids.keys())}")

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        qst_token = self.tokenizer.convert_tokens_to_ids("[QST]")

        if current_mode == {"question"}:
            assert not get_passage_offset
            token_ids =  np.concatenate([
                [qst_token],
                ids["question"],
                [sep_token],
            ])
        elif current_mode == {"passage_title", "passage"}:
            token_ids = np.concatenate([
                [cls_token],
                ids["passage_title"],
                [sep_token],
                ids["passage"],
                [sep_token],
            ])
            if get_passage_offset:
                passage_offset = 2 + len(ids["passage_title"])
        else:
            token_ids = np.concatenate([
                [qst_token],
                ids["question"],
                [sep_token],
                ids["passage_title"],
                [sep_token],
                ids["passage"],
                [sep_token],
            ])
            if get_passage_offset:
                passage_offset = 3 + len(ids["question"]) + len(ids["passage_title"])

        if to_max_length:
            token_ids = self.to_max_length(token_ids, apply_max_len=True)

        if get_passage_offset:
            return torch.from_numpy(token_ids), passage_offset
        return torch.from_numpy(token_ids)
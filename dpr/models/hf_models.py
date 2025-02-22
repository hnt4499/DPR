#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List, Dict, Set

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers.models.bert.tokenization_bert import BertTokenizer

from dpr.models.biencoder_retrievers.biencoder import BiEncoder
from dpr.models.extractive_readers.extractive_reader import Reader
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import load_state_dict_to_model


logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    biencoder = BiEncoder(question_encoder, ctx_encoder)
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )

    return BertTensorizer(tokenizer, sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    with_decay = {"params": [], "weight_decay": weight_decay}
    without_decay = {"params": [], "weight_decay": 0.0}
    count = 0

    for name, parameter in  model.named_parameters():
        if (not any(nd in name for nd in no_decay)) and parameter.requires_grad:
            with_decay["params"].append(parameter)
            count += parameter.numel()
        elif parameter.requires_grad:
            without_decay["params"].append(parameter)
            count += parameter.numel()

    optimizer_grouped_parameters = [with_decay, without_decay]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)

    logger.info(f"Initialized optimizer with {count} parameters: {optimizer}")
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim)
            if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(
            cfg_name if cfg_name else "bert-base-uncased",
            return_dict=False,
            **kwargs,
        )
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim,
            )
        else:
            return cls(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        # SIMPLIFIED: always select the first sequence output
        pooled_output = sequence_output[:, 0, :]

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

    def load_state(self, state_dict: dict):
        load_state_dict_to_model(self, state_dict)


class BertTensorizer(Tensorizer):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        max_length: int,
        pad_to_max: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def to_max_length(
        self,
        token_ids: np.ndarray,
        apply_max_len: bool = True,
    ) -> np.ndarray:
        """Pad or truncate to a specified maximum sequence length."""
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = np.concatenate([
                token_ids,
                [self.get_pad_id()] * (seq_len - len(token_ids)),
            ])
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return token_ids

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # Tokenizer automatic padding is explicitly disabled since its
        # inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        token_ids = self.to_max_length(
            np.array(token_ids),
            apply_max_len=apply_max_len,
        )

        return torch.from_numpy(token_ids)

    def concatenate_inputs(
        self,
        ids: Dict[str, List[int]],
        get_passage_offset: bool = False,
        to_max_length: bool = False,  # False by default for backward compat.
    ) -> T:
        """
        Simply concatenate inputs by adding [CLS] at the beginning and [SEP] at
        between and end.
        """
        # 3 mode: only question, only passage ("passage_title" + "passage") or
        # all
        current_mode = set(ids.keys())
        allowed_modes = [
            {"question"},
            {"passage_title", "passage"},
            {"question", "passage_title", "passage"},
        ]
        if current_mode not in allowed_modes:
            raise ValueError(f"Unexpected keys: {list(ids.keys())}")

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id

        if current_mode == {"question"}:
            assert not get_passage_offset
            token_ids = np.concatenate([
                [cls_token],
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
                [cls_token],
                ids["question"],
                [sep_token],
                ids["passage_title"],
                [sep_token],
                ids["passage"],
                [sep_token],
            ])
            if get_passage_offset:
                passage_offset = 3 + len(ids["question"]) + \
                    len(ids["passage_title"])

        token_ids = token_ids.astype("int64")
        if to_max_length:
            token_ids = self.to_max_length(token_ids, apply_max_len=True)

        if get_passage_offset:
            return torch.from_numpy(token_ids), passage_offset
        return torch.from_numpy(token_ids)

    def remove_padding(self, ids: T) -> T:
        if not (ids == self.get_pad_id()).any():
            return ids
        # Index of the first pad token
        first_pad_id = (ids == self.get_pad_id()).int().argmax()
        # Make sure all of tokens followed are pad tokens
        assert (ids[first_pad_id:] == self.get_pad_id()).all(), (
            f"All pad tokens ({self.get_pad_id()}) must be at the end of the "
            f"sequence: {ids}"
        )

        return ids[:first_pad_id]

    def unconcatenate_inputs(
        self,
        ids: T,
        components: Set[str],
    ) -> Dict[str, T]:

        # Remove padding
        ids = self.remove_padding(ids)
        if len(ids) == 0:  # full padding
            return None

        if components == {"question"}:
            return {"question": ids[1:-1]}

        elif components == {"passage_title", "passage"}:
            # Get all [SEP] indicies
            sep_ids: T = (
                ids == self.tokenizer.sep_token_id
            ).nonzero().squeeze()
            assert sep_ids.numel() == 2, (
                f"Expected 2 [SEP] ({self.tokenizer.sep_token_id}) tokens, "
                f"got {sep_ids.numel()} instead: {ids}"
            )

            passage_title_start = 1
            passage_title_end = sep_ids[0]
            passage_title_ids = ids[passage_title_start:passage_title_end]

            passage_start = passage_title_end + 1
            passage_end = -1
            passage_ids = ids[passage_start:passage_end]

            return {"passage_title": passage_title_ids, "passage": passage_ids}

        elif components == {"question", "passage_title", "passage"}:
            # Get all [SEP] indicies
            sep_ids: T = (
                ids == self.tokenizer.sep_token_id
            ).nonzero().squeeze()
            assert sep_ids.numel() == 3, (
                f"Expected 3 [SEP] ({self.tokenizer.sep_token_id}) tokens, "
                f"got {sep_ids.numel()} instead: {ids}"
            )

            question_start = 1
            question_end = sep_ids[0]
            question_ids = ids[question_start:question_end]

            passage_title_start = question_end + 1
            passage_title_end = sep_ids[1]
            passage_title_ids = ids[passage_title_start:passage_title_end]

            passage_start = passage_title_end + 1
            passage_end = -1
            passage_ids = ids[passage_start:passage_end]

            return {
                "question": question_ids,
                "passage_title": passage_title_ids,
                "passage": passage_ids,
            }
        else:
            raise ValueError(f"Invalid components: {components}")

    def tensor_to_text(
        self,
        tensor: T
    ) -> str:
        return self.tokenizer.decode(tensor)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        if tokens_tensor is None:
            return None
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]

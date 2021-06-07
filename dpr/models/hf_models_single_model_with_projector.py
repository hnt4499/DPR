#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Both question encoder and context encoder share the same base BERT encoder, with separate
projector head on top of it.
"""

import logging
from typing import List, Union

import torch.nn as nn
from transformers import BertModel

from dpr.models.biencoder import BiEncoder
from dpr.models.hf_models import get_optimizer, HFBertEncoder
from dpr.models.hf_models_single_model import get_bert_tensorizer

logger = logging.getLogger(__name__)


class BERTWithProjector(nn.Module):
    def __init__(self, base_encoder: BertModel, project_dims: Union[int, List[int]]):
        super(BERTWithProjector, self).__init__()
        self.base_encoder = base_encoder

        # Projector head
        if isinstance(project_dims, int):
            project_dims = [project_dims]

        prev_hidden_size = base_encoder.config.hidden_size
        self.projectors: List[nn.Module] = []

        for i, curr_hidden_size in enumerate(project_dims):
            linear_layer = nn.Linear(prev_hidden_size, curr_hidden_size)
            prev_hidden_size = curr_hidden_size
            self.projectors.append(linear_layer)

            # BatchNorm and ReLU
            if i != len(project_dims) - 1:
                self.projectors.append(nn.BatchNorm1d(num_features=prev_hidden_size))
                self.projectors.append(nn.ReLU())

        # Composite
        self.projectors = nn.Sequential(*self.projectors)

    def forward(self, *args, **kwargs):
        sequence_output, pooled_output, hidden_states = self.base_encoder(*args, **kwargs)
        del sequence_output, hidden_states  # these are not used in the whole package

        pooled_output = self.projectors(pooled_output)
        return None, pooled_output, None  # piped

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_encoder.resize_token_embeddings(*args, **kwargs)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    base_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    question_encoder = BERTWithProjector(base_encoder, project_dims=cfg.encoder.projection_dims)
    ctx_encoder = BERTWithProjector(base_encoder, project_dims=cfg.encoder.projection_dims)

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
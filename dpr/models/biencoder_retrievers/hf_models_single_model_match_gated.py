#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code using only one model for
encoding both question and context, with several layers on top of the encoder
; second stage: feed-forward, interactive matching)
"""

import logging

from .biencoder import MatchGated_BiEncoder
from .hf_models_single_model_match import get_optimizer, get_bert_tensorizer, HFBertEncoder

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

    biencoder = MatchGated_BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder, freeze_encoders=cfg.encoder.freeze_encoders,
    )

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
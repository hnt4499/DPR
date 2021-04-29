#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Reader model with inter-passage modeling.
"""

import logging


from.hf_models_inter_passage import get_optimizer, get_bert_tensorizer, HFBertEncoderWithNumLayers
from .reader import InterPassageReaderV2

logger = logging.getLogger(__name__)


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    num_layers = cfg.encoder.num_layers
    logger.info(f"Initializing InterPassageReaderV2 with number of layers for two components: {num_layers}.")

    encoder = HFBertEncoderWithNumLayers.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        num_hidden_layers=num_layers[0],
        **kwargs
    )

    inter_passage_encoder = HFBertEncoderWithNumLayers.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=0,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        num_hidden_layers=num_layers[1],
        **kwargs
    )

    reader = InterPassageReaderV2(
        encoder, inter_passage_encoder, bottleneck_size=cfg.encoder.bottleneck_size, hidden_size=None)

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
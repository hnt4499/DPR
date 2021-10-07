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

from transformers.models.bert.modeling_bert import BertModel, BertConfig

from ..hf_models import HFBertEncoder, get_optimizer, get_bert_tensorizer
from .extractive_reader import InterPassageReader


logger = logging.getLogger(__name__)


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    num_layers = cfg.encoder.num_layers
    logger.info(f"Initializing InterPassageReader with number of layers for two components: {num_layers}.")

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

    reader = InterPassageReader(encoder, inter_passage_encoder, hidden_size=None)

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


class HFBertEncoderWithNumLayers(HFBertEncoder):
    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        num_hidden_layers: int,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased",
                                         num_hidden_layers=num_hidden_layers, **kwargs)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim,
            )
        else:
            return cls(cfg, project_dim=projection_dim)
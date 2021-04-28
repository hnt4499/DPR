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
from functools import partial

import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel, BertConfig
from transformers.optimization import AdamW

from dpr.utils.model_utils import load_states_from_checkpoint
from .hf_models import HFBertEncoder, get_bert_tensorizer
from .reader import InterPassageReader

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
    
    # Load state dict for the main encoder
    if cfg.encoder.pretrained_encoder_file is not None:
        assert cfg.encoder.pretrained_file is None, "Ambiguous pretrained model"
        assert cfg.encoder.encoder_initialize_from in ["question_encoder", "ctx_encoder"]
        saved_state = load_states_from_checkpoint(cfg.encoder.pretrained_encoder_file).model_dict
        key_start = "question_model" if cfg.encoder.encoder_initialize_from == "question_encoder" else "ctx_model"
        
        new_saved_state = {}
        for key, param in saved_state.items():
            if not key.startswith(key_start):
                continue
            
            # Remove "xxx_model." part
            key = ".".join(key.split(".")[1:])

            # Truncate embeddings
            if "embeddings" in key and cfg.encoder.allow_embedding_size_mismatch:
                model_embeddings = [v for k, v in encoder.named_parameters() if k == key]
                assert len(model_embeddings) == 1
                model_embeddings = model_embeddings[0]

                assert len(param) >= len(model_embeddings)
                if len(param) > len(model_embeddings):
                    logger.info(f"Truncating pretrained embedding ('{key}') size from {len(param)} to {len(model_embeddings)} "
                                f"by simply selecting the first {len(model_embeddings)} embeddings.")
                    param = param[:len(model_embeddings)]
            
            new_saved_state[key] = param

        # Load to the model
        encoder.load_state(new_saved_state)

        # Freeze main encoder
        if cfg.encoder.encoder_freeze:
            for _, param in encoder.named_parameters():
                param.requires_grad = False
    
    elif cfg.encoder.encoder_freeze:
        raise ValueError("You should not freeze a model that is not trained on a QA task.")

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
            encoder_learning_rate=cfg.encoder.encoder_learning_rate,
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
            return HFBertEncoderWithNumLayers(cfg, project_dim=projection_dim)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    encoder_learning_rate: float = 1e-6,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    use_lamb: float = False,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    if encoder_learning_rate is None:
        encoder_learning_rate = learning_rate

    default_lr_group = {
        "with_decay": {"params": [], "weight_decay": weight_decay, "lr": learning_rate},
        "without_decay": {"params": [], "weight_decay": 0.0, "lr": learning_rate}
    }
    encoder_lr_group = {
        "with_decay": {"params": [], "weight_decay": weight_decay, "lr": encoder_learning_rate},
        "without_decay": {"params": [], "weight_decay": 0.0, "lr": encoder_learning_rate}
    }

    all_groups = {"default_lr": default_lr_group, "encoder_lr": encoder_lr_group}

    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count += parameter.numel()

        # Which learning rate
        if name.startswith("encoder."):
            key_lr = "encoder_lr"
        else:
            key_lr = "default_lr"

        # Whether to apply decay
        if (not any(nd in name for nd in no_decay)):
            key_decay = "with_decay"
        else:
            key_decay = "without_decay"
        
        all_groups[key_lr][key_decay]["params"].append(parameter)

    optimizer_grouped_parameters = [
        all_groups["default_lr"]["with_decay"],
        all_groups["default_lr"]["without_decay"],
        all_groups["encoder_lr"]["with_decay"],
        all_groups["encoder_lr"]["without_decay"],
    ]

    if use_lamb:
        from apex.optimizers.fused_lamb import FusedLAMB
        optimizer_init = partial(FusedLAMB, adam_w_mode=True)
    else:
        optimizer_init = AdamW
    optimizer = optimizer_init(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)

    logger.info(f"Initialized optimizer with {count} parameters: {optimizer}")
    return optimizer
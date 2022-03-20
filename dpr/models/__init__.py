#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
"Router"-like set of methods for component initialization with lazy imports
"""


import importlib


def import_submodule(path):
    """
    Import a submodule (e.g., a function inside a python file) using its full
    path.
    """
    (*module), sub_module = path.split(".")
    module = ".".join(module)
    module = importlib.import_module(module)
    return getattr(module, sub_module)


def init_comp(submodule_path, cfg, **kwargs):
    """
    Generic initializer.
    """
    initializer = import_submodule(submodule_path)
    return initializer(cfg=cfg, **kwargs)


def init_biencoder_components(encoder_type: str, cfg, **kwargs):
    BIENCODER_INITIALIZERS = {
        "hf_bert": "dpr.models.hf_models.get_bert_biencoder_components",
        "hf_bert_shared": \
            "dpr.models.hf_models_shared.get_bert_biencoder_components",
    }
    return init_comp(BIENCODER_INITIALIZERS[encoder_type], cfg=cfg, **kwargs)


def init_reader_components(reader_type: str, cfg, **kwargs):
    READER_INITIALIZERS = {
        "hf_bert": "dpr.models.hf_models.get_bert_reader_components",
    }
    return init_comp(READER_INITIALIZERS[reader_type], cfg=cfg, **kwargs)


def init_generative_reader_components(reader_type: str, cfg, **kwargs):
    GENERATIVE_READER_INITIALIZERS = {
        "fid_base": (
            "dpr.models.generative_readers.fid_base."
            "get_generative_reader_components"
        ),
        "fid_shared": (
            "dpr.models.generative_readers.fid_shared."
            "get_generative_reader_components"
        ),
    }
    return init_comp(
        GENERATIVE_READER_INITIALIZERS[reader_type], cfg=cfg, **kwargs)


def init_tenzorizer(encoder_type: str, cfg, **kwargs):
    TENSORIZER_INITIALIZERS = {
        # Extractive reader
        "hf_bert": "dpr.models.hf_models.get_bert_tensorizer",
        "hf_bert_shared": "dpr.models.hf_models_shared.get_bert_tensorizer",
        # Generative reader
        "fid_base": (
            "dpr.models.generative_readers.fid_base."
            "get_generative_tensorizer"
        ),
        "fid_shared": (
            "dpr.models.generative_readers.fid_shared."
            "get_generative_tensorizer"
        ),
    }
    return init_comp(TENSORIZER_INITIALIZERS[encoder_type], cfg=cfg, **kwargs)


def init_biencoder_loss(encoder_type: str, cfg, **kwargs):
    LOSS_INITIALIZERS = {
        "hf_bert": "dpr.models.biencoder_retrievers.biencoder.BiEncoderNllLoss",
        "hf_bert_shared": \
            "dpr.models.biencoder_retrievers.biencoder.BiEncoderNllLoss",
    }

    loss_kwargs = cfg.train.get("loss_kwargs", {})
    kwargs.update(loss_kwargs)
    return init_comp(LOSS_INITIALIZERS[encoder_type], cfg=cfg, **kwargs)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports
"""


def init_comp(initializers_dict, type, args=None, **kwargs):
    if type in initializers_dict:
        if args is None:
            return initializers_dict[type](**kwargs)
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError('unsupported model type: {}'.format(type))


"""------------------------------- BiEncoder --------------------------------"""


def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_biencoder_single_model(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_biencoder_single_model_match(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_match import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_pytext_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("pytext") is None:
        raise RuntimeError('Please install pytext lib')
    from .pytext_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_fairseq_roberta_biencoder(args, **kwargs):
    if importlib.util.find_spec("fairseq") is None:
        raise RuntimeError('Please install fairseq lib')
    from .fairseq_models import get_roberta_biencoder_components
    return get_roberta_biencoder_components(args, **kwargs)


BIENCODER_INITIALIZERS = {
    'hf_bert': init_hf_bert_biencoder,
    'hf_bert_single_model': init_hf_bert_biencoder_single_model,
    'hf_bert_single_model_match': init_hf_bert_biencoder_single_model_match,
    'pytext_bert': init_pytext_bert_biencoder,
    'fairseq_roberta': init_fairseq_roberta_biencoder,
}


def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


"""------------------------------- Reader -------------------------------"""


def init_hf_bert_reader(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_reader_components
    return get_bert_reader_components(args, **kwargs)


READER_INITIALIZERS = {
    'hf_bert': init_hf_bert_reader,
}


def init_reader_components(encoder_type: str, args, **kwargs):
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


"""------------------------------- Tensorizer -------------------------------"""


def init_hf_bert_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_tenzorizer_single_model(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_tenzorizer_single_model_match(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_match import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_roberta_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_roberta_tensorizer
    return get_roberta_tensorizer(args)


TENSORIZER_INITIALIZERS = {
    'hf_bert': init_hf_bert_tenzorizer,
    'hf_bert_single_model': init_hf_bert_tenzorizer_single_model,
    'hf_bert_single_model_match': init_hf_bert_tenzorizer_single_model_match,
    'hf_roberta': init_hf_roberta_tenzorizer,
    'pytext_bert': init_hf_bert_tenzorizer,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)


"""------------------------------- Loss -------------------------------"""


def init_hf_bert_loss(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_bert_loss_single_model(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_bert_loss_single_model_match(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import Match_BiEncoderNllLoss
    return Match_BiEncoderNllLoss(args)


def init_hf_roberta_loss(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


LOSS_INITIALIZERS = {
    'hf_bert': init_hf_bert_loss,
    'hf_bert_single_model': init_hf_bert_loss_single_model,
    'hf_bert_single_model_match': init_hf_bert_loss_single_model_match,
    'hf_roberta': init_hf_roberta_loss,
    'pytext_bert': init_hf_bert_loss,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_loss,  # using HF's code as of now
}


def init_loss(encoder_type: str, args, **kwargs):
    return init_comp(LOSS_INITIALIZERS, encoder_type, args, **kwargs)
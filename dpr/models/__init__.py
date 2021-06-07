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

def init_hf_bert_biencoder_single_model_match_gated(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_match_gated import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_biencoder_single_model_with_projector(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_with_projector import get_bert_biencoder_components
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
    'hf_bert_single_model_match_gated': init_hf_bert_biencoder_single_model_match_gated,
    'hf_bert_single_model_with_projector': init_hf_bert_biencoder_single_model_with_projector,

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


def init_hf_bert_inter_passage_reader(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_inter_passage import get_bert_reader_components
    return get_bert_reader_components(args, **kwargs)


def init_hf_bert_inter_passage_reader_from_retriever(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_inter_passage_from_retriever import get_bert_reader_components
    return get_bert_reader_components(args, **kwargs)


def init_hf_bert_inter_passage_reader_v2(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_inter_passage_v2 import get_bert_reader_components
    return get_bert_reader_components(args, **kwargs)


READER_INITIALIZERS = {
    'hf_bert': init_hf_bert_reader,
    'hf_bert_inter_passage': init_hf_bert_inter_passage_reader,
    'hf_bert_inter_passage_from_retriever': init_hf_bert_inter_passage_reader_from_retriever,
    'hf_bert_inter_passage_v2': init_hf_bert_inter_passage_reader_v2,
}


def init_reader_components(encoder_type: str, args, **kwargs):
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


"""------------------------------- One-for-all ------------------------------"""
def init_hf_bert_ofa_simple(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_simple import get_bert_one_for_all_components
    return get_bert_one_for_all_components(args, **kwargs)


def init_hf_bert_ofa_with_passage_scores(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_with_passage_scores import get_bert_one_for_all_components
    return get_bert_one_for_all_components(args, **kwargs)

def init_hf_bert_ofa_special_tokens(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_special_tokens import get_bert_one_for_all_components
    return get_bert_one_for_all_components(args, **kwargs)


OFA_INITIALIZERS = {
    'hf_bert_simple_ofa': init_hf_bert_ofa_simple,  # for backward compatibility
    'hf_bert_ofa_simple': init_hf_bert_ofa_simple,
    'hf_bert_ofa_with_passage_scores': init_hf_bert_ofa_with_passage_scores,
    'hf_bert_ofa_special_tokens': init_hf_bert_ofa_special_tokens,
}


def init_ofa_model(encoder_type: str, args, **kwargs):
    return init_comp(OFA_INITIALIZERS, encoder_type, args, **kwargs)


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


def init_hf_bert_tenzorizer_single_model_match_gated(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_match_gated import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_tenzorizer_single_model_with_projector(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_single_model_with_projector import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_ofa_simple_tensorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_simple import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_ofa_with_passage_scores_tensorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_with_passage_scores import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_bert_ofa_special_tokens_tensorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models_ofa_special_tokens import get_bert_tensorizer
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
    'hf_bert_single_model_match_gated': init_hf_bert_tenzorizer_single_model_match_gated,
    'hf_bert_single_model_with_projector': init_hf_bert_tenzorizer_single_model_with_projector,

    'hf_bert_ofa_simple': init_hf_bert_ofa_simple_tensorizer,
    'hf_bert_simple_ofa': init_hf_bert_ofa_simple_tensorizer,  # for backward compatibility
    'hf_bert_ofa_with_passage_scores': init_hf_bert_ofa_with_passage_scores_tensorizer,
    'hf_bert_ofa_special_tokens': init_hf_bert_ofa_special_tokens_tensorizer,

    'hf_roberta': init_hf_roberta_tenzorizer,
    'pytext_bert': init_hf_bert_tenzorizer,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)


"""------------------------------- Biencoder Loss -------------------------------"""


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


def init_hf_bert_loss_single_model_match_gated(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import MatchGated_BiEncoderNllLoss
    return MatchGated_BiEncoderNllLoss(args)


def init_hf_bert_loss_single_model_with_projector(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_bert_loss_ofa_simple(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_bert_loss_ofa_with_passage_scores(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_bert_loss_ofa_special_tokens(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


def init_hf_roberta_loss(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .biencoder import BiEncoderNllLoss
    return BiEncoderNllLoss(args)


LOSS_INITIALIZERS = {
    'hf_bert': init_hf_bert_loss,

    'hf_bert_single_model': init_hf_bert_loss_single_model,
    'hf_bert_single_model_match': init_hf_bert_loss_single_model_match,
    'hf_bert_single_model_match_gated': init_hf_bert_loss_single_model_match_gated,
    'hf_bert_single_model_with_projector': init_hf_bert_loss_single_model_with_projector,

    'hf_bert_ofa_simple': init_hf_bert_loss_ofa_simple,
    'hf_bert_simple_ofa': init_hf_bert_loss_ofa_simple,  # for backward compatibility
    'hf_bert_ofa_with_passage_scores': init_hf_bert_loss_ofa_with_passage_scores,
    'hf_bert_ofa_special_tokens': init_hf_bert_loss_ofa_special_tokens,

    'hf_roberta': init_hf_roberta_loss,
    'pytext_bert': init_hf_bert_loss,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_loss,  # using HF's code as of now
}


def init_loss(encoder_type: str, args, **kwargs):
    return init_comp(LOSS_INITIALIZERS, encoder_type, args, **kwargs)
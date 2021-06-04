from typing import List, Dict, Set

import torch
from torch import Tensor as T
import numpy as np

from dpr.models.biencoder import BiEncoder
from dpr.models.reader import Reader
from dpr.models.one_for_all_base import SimpleOneForAllModel
from dpr.models.hf_models import get_optimizer, HFBertEncoder

from dpr.models.hf_models_ofa_simple import do_ofa_fwd_pass  # same forward pass as `ofa_simple`
from dpr.models.hf_models_single_model import get_bert_tokenizer, _add_special_tokens
from dpr.models.hf_models_single_model import BertTensorizer as Tensorizer


def get_bert_one_for_all_components(cfg, inference_only: bool = False, **kwargs):
    """One-for-all model (i.e., single model for both retrieval and extractive question answering task) initialization."""
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0

    # Initialize base encoder
    base_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    # Initialize biencoder from a shared encoder
    biencoder = BiEncoder(
        question_model=base_encoder,
        ctx_model=base_encoder,
    ).to(cfg.device)

    biencoder_optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.biencoder_learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
            use_lamb=cfg.train.lamb,
        )
        if not inference_only
        else None
    )

    # Initialize reader model from a shared encoder
    reader = Reader(
        encoder=base_encoder,
        hidden_size=base_encoder.config.hidden_size,
    )

    reader_optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.reader_learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
            use_lamb=cfg.train.lamb,
        )
        if not inference_only
        else None
    )

    # Initialize tensorizer for one-for-all model
    tensorizer = get_bert_tensorizer(cfg, biencoder)

    # Initialize one-for-all model
    ofa_model = SimpleOneForAllModel(
        biencoder=biencoder,
        reader=reader,
        tensorizer=tensorizer,
    )

    return tensorizer, ofa_model, biencoder_optimizer, reader_optimizer, do_ofa_fwd_pass


def get_bert_tensorizer(cfg, biencoder):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    tokenizer = get_bert_tokenizer(
        pretrained_model_cfg, biencoder=biencoder, do_lower_case=cfg.do_lower_case
    )
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


class BertTensorizer(Tensorizer):
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
                [cls_token],
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
            # Get all [SEP] indices
            sep_ids: T = (ids == self.tokenizer.sep_token_id).nonzero().squeeze()
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
            # Get all [CLS] indices
            cls_ids: T = (ids == self.tokenizer.cls_token_id).nonzero().squeeze()
            assert cls_ids.numel() == 1, (
                f"Expected 1 [CLS] ({self.tokenizer.cls_token_id}) tokens, "
                f"got {cls_ids.numel()} instead: {ids}"
            )
            # Get all [SEP] indices
            sep_ids: T = (ids == self.tokenizer.sep_token_id).nonzero().squeeze()
            assert sep_ids.numel() == 2, (
                f"Expected 2 [SEP] ({self.tokenizer.sep_token_id}) tokens, "
                f"got {sep_ids.numel()} instead: {ids}"
            )

            question_start = 1
            question_end = cls_ids[0]
            question_ids = ids[question_start:question_end]

            passage_title_start = question_end + 1
            passage_title_end = sep_ids[0]
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
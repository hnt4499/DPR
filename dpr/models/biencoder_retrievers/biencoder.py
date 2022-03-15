#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.data_types import (
    BiEncoderSample,
    BiEncoderSampleTokenized,
    BiEncoderBatch,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState, load_state_dict_to_model
from dpr.utils.dist_utils import all_gather_list

logger = logging.getLogger(__name__)

# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos_q=0,
        representation_token_pos_c=0,
        positive_idxs: T = None,
        hard_negative_idxs: T = None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos_q,
        )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
            self.fix_ctx_encoder,
            representation_token_pos=representation_token_pos_c,
        )

        return q_pooled_out, ctx_pooled_out

    @classmethod
    def create_biencoder_input(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        General wrapper around different processing methods
        """
        if isinstance(samples[0], BiEncoderSample):
            return cls.create_biencoder_input_non_tokenized(
                samples=samples,
                tensorizer=tensorizer,
                insert_title=insert_title,
                num_hard_negatives=num_hard_negatives,
                num_other_negatives=num_other_negatives,
                shuffle=shuffle,
                shuffle_positives=shuffle_positives,
                hard_neg_fallback=hard_neg_fallback,
                query_token=query_token,
            )
        elif isinstance(samples[0], BiEncoderSampleTokenized):
            return cls.create_biencoder_input_tokenized(
                samples=samples,
                tensorizer=tensorizer,
                insert_title=insert_title,
                num_hard_negatives=num_hard_negatives,
                num_bm25_negatives=num_other_negatives,
                shuffle=shuffle,
                shuffle_positives=shuffle_positives,
                hard_neg_fallback=hard_neg_fallback,
                query_token=query_token,
            )
        else:
            raise NotImplementedError(
                f"Invalid sample type: {samples[0].__class__.__name__}")

    @classmethod
    def create_biencoder_input_non_tokenized(
        cls,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple using non-tokenized (i.e., raw text) data.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, title=ctx.title if (insert_title and ctx.title) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(
                        question, tensorizer, token_str=query_token
                    )
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(
                        tensorizer.text_to_tensor(" ".join([query_token, question]))
                    )
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            None,  # for backward compatibility
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    @classmethod
    def create_biencoder_input_tokenized(
        cls,
        samples: List[BiEncoderSampleTokenized],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_bm25_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple using tokenized data.
        :param samples: list of BiEncoderSampleTokenized-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives (densely retrieved) per question
        :param num_bm25_negatives: amount of BM25 negatives (sparsely retrieved) per question
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools. This is only effective for samples whose gold
            passage is available. In that case, the positive chosen is not necessarily the gold passage. Otherwise,
            the positive passages will be shuffled regardless of this parameter.
        :return: BiEncoderBatch tuple
        """
        question_tensors: List[T] = []
        ctx_ids: List[int] = []  # passage IDs
        ctx_tensors: List[T] = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        # Strict settings
        assert insert_title is True  # for now only allow `insert_title` to be True
        assert query_token is None

        for sample in samples:
            # Skip samples without positive passges (either gold or distant positives)
            if len(sample.positive_passages) == 0:
                continue
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if (shuffle and shuffle_positives) or (not sample.positive_passages[0].is_gold):
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            bm25_neg_ctxs = sample.bm25_negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question_ids = sample.query_ids

            if shuffle:
                random.shuffle(bm25_neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = bm25_neg_ctxs[0:num_hard_negatives]

            bm25_neg_ctxs = bm25_neg_ctxs[0:num_bm25_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + bm25_neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1 + len(bm25_neg_ctxs)
            hard_negatives_end_idx = len(all_ctxs)

            current_ctxs_len = len(ctx_tensors)

            # Context IDs
            ctx_id = [ctx.id for ctx in all_ctxs]
            ctx_ids.extend(ctx_id)

            # Context tensors
            sample_ctxs_tensors = [
                tensorizer.concatenate_inputs(
                    ids={"passage_title": list(ctx.title_ids), "passage": list(ctx.text_ids)},
                    get_passage_offset=False,
                    to_max_length=True,
                )
                for ctx in all_ctxs
            ]
            ctx_tensors.extend(sample_ctxs_tensors)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            question_tensors.append(
                tensorizer.concatenate_inputs(
                    ids={"question": question_ids},
                    get_passage_offset=False,
                    to_max_length=True
                )
            )

        ctx_ids = torch.tensor(ctx_ids, dtype=torch.int64)
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctx_ids,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        load_state_dict_to_model(self, saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def __init__(
        self,
        cfg,
        score_scaling: bool = False,  # http://arxiv.org/abs/2101.00408
    ):
        self.cfg = cfg
        self.score_scaling = score_scaling

    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        # Score scaling, as described in the paper:
        # Sachan, D. S., Patwary, M., Shoeybi, M., Kant, N., Ping, W., Hamilton, W. L., & Catanzaro, B. (2021).
        # End-to-End Training of Neural Retrievers for Open-Domain Question Answering.
        if self.score_scaling:
            hidden_size = q_vectors.shape[1]
            scores = scores / (hidden_size ** 0.5)

        return self.calc_given_score_matrix(scores, positive_idx_per_question, loss_scale=loss_scale)

    @staticmethod
    def calc_given_score_matrix(
        score_mat: T,
        positive_idx_per_question: list,
        loss_scale: float = None,
        reduction: str = "mean"
    ) -> Tuple[T, int]:
        """
        General utility to calculate NLL loss given the un-normalized score matrix of shape (n1, n2), where n1 is
        the number of questions and n2 is the number of passages.
        """
        softmax_scores = F.log_softmax(score_mat, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction=reduction,
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


class BiEncoderMultiSimilarityNllLoss(BiEncoderNllLoss):
    """
    BiEncoder multi-similarity NLL loss. In addition to computing the similarities between
    question and context vectors, we also compute the question-question and context-context
    similarities.

    Parameters
    ----------
    loss_scale_qq : float
        Loss scale value for question-question similarities.
    loss_scale_cc : float
        Loss scale value for context-context similarities.
    """
    def __init__(
        self,
        *args,
        loss_scale_qq: float = 1.0,
        loss_scale_cc: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_scale_qq = loss_scale_qq
        self.loss_scale_cc = loss_scale_cc

    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:

        scores_qc = self.get_scores(q_vectors, ctx_vectors)
        scores_qq = self.get_scores(q_vectors, q_vectors)
        scores_cc = self.get_scores(ctx_vectors, ctx_vectors)

        # Flatten if needed
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores_qc = scores_qc.view(q_num, -1)
            scores_qq = scores_qq.view(q_num, -1)
        if len(ctx_vectors.size()) > 1:
            ctx_num = ctx_vectors.size(0)
            scores_cc = scores_cc.view(ctx_num, -1)

        # Score scaling, as described in the paper:
        # Sachan, D. S., Patwary, M., Shoeybi, M., Kant, N., Ping, W., Hamilton, W. L., & Catanzaro, B. (2021).
        # End-to-End Training of Neural Retrievers for Open-Domain Question Answering.
        if self.score_scaling:
            hidden_size = q_vectors.shape[1]
            scores_qc = scores_qc / (hidden_size ** 0.5)
            scores_qq = scores_qq / (hidden_size ** 0.5)
            scores_cc = scores_cc / (hidden_size ** 0.5)

        return self.calc_given_score_matrix(
            [scores_qc, scores_qq, scores_cc],
            positive_idx_per_question,
            loss_scale=loss_scale,
        )

    def calc_given_score_matrix(
        self,
        score_mat: Tuple[T, T, T],
        positive_idx_per_question: list,
        loss_scale: float = None,
        reduction: str = "mean"
    ) -> Tuple[T, int]:
        """
        General utility to calculate NLL loss given the un-normalized score matrices, each
        of shape (n1, n2), where n1 is the number of questions and n2 is the number of
        passages in the batch.
        """
        scores_qc, scores_qq, scores_cc = score_mat
        device = scores_qc.device
        positive_idx_per_question = torch.tensor(positive_idx_per_question).to(device)

        def nll(input, target):
            softmax_scores = F.log_softmax(input, dim=1)
            loss = F.nll_loss(
                softmax_scores,
                target,
                reduction=reduction,
            )
            return loss

        # Main score: question - context similarities
        loss_qc = nll(scores_qc, positive_idx_per_question)
        _, max_idxs = torch.max(scores_qc, 1)
        correct_predictions_count = (max_idxs == positive_idx_per_question).sum()

        # Question-question similarities
        loss_qq = nll(
            scores_qq,
            torch.arange(end=len(scores_qc), dtype=torch.long, device=device),
        )
        # Context-context similarities
        loss_cc = nll(
            scores_cc,
            torch.arange(end=len(scores_cc), dtype=torch.long, device=device),
        )

        # Sum loss
        loss = loss_qc + self.loss_scale_qq * loss_qq + self.loss_scale_cc * loss_cc

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count


class Match_BiEncoder(BiEncoder):
    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        freeze_encoders: bool = False,  # useful for finetuning
    ):
        super(Match_BiEncoder, self).__init__(
            question_model, ctx_model, fix_q_encoder=fix_q_encoder, fix_ctx_encoder=fix_ctx_encoder
        )
        self.linear = nn.Linear(in_features=question_model.out_features * 4, out_features=1)  # linear projection

        assert freeze_encoders is not None
        if freeze_encoders:
            for name, parameter in question_model.named_parameters():
                parameter.requires_grad = False
            for name, parameter in ctx_model.named_parameters():
                parameter.requires_grad = False

    def forward(self, *args, is_matching=False, **kwargs) -> Tuple[T, T, T]:
        if not is_matching:
            return super(Match_BiEncoder, self).forward(*args, **kwargs)

        """Return an **interaction** matrix on top of the embeddings."""
        assert len(args) == 0 and len(kwargs) == 2
        q_pooled_out, ctx_pooled_out = kwargs["q_pooled_out"], kwargs["ctx_pooled_out"]
        # Shape
        q_pooled_out_r = q_pooled_out.unsqueeze(1).repeat(1, len(ctx_pooled_out), 1)  # (n1, n2, d)
        ctx_pooled_out_r = ctx_pooled_out.unsqueeze(0).repeat(len(q_pooled_out), 1, 1)  # (n1, n2, d)

        # Interact
        interaction_mul = q_pooled_out_r * ctx_pooled_out_r  # (n1, n2, d)
        interaction_diff = q_pooled_out_r - ctx_pooled_out_r  # (n1, n2, d)
        interaction_mat = torch.cat(
            [q_pooled_out_r, ctx_pooled_out_r, interaction_mul, interaction_diff], dim=2)  # (n1, n2, 4d)
        del q_pooled_out_r, ctx_pooled_out_r, interaction_mul, interaction_diff

        # Linear projection
        interaction_mat = self.linear(interaction_mat).squeeze(-1)  # (n1, n2)
        return interaction_mat


class Match_BiEncoderNllLoss(BiEncoderNllLoss):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        interaction_matrix: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list,
        loss_scale: Tuple[float, float]
    ) -> Tuple[T, int, int]:
        """Calculate loss for both metric learning and interaction layer."""

        if loss_scale is not None:
            assert len(loss_scale) == 2
        else:
            loss_scale = (None, None)

        # Calculate metric learning loss
        ml_loss, ml_correct_predictions_count = super(Match_BiEncoderNllLoss, self).calc(
            q_vectors,
            ctx_vectors,
            positive_idx_per_question,
            hard_negative_idx_per_question=hard_negative_idx_per_question,
            loss_scale=loss_scale[0]
        )  # "ml" stands for "metric learning"

        # Calculate interaction loss
        intr_loss, intr_correct_predictions_count = self.calc_given_score_matrix(
            interaction_matrix,
            positive_idx_per_question,
            loss_scale=loss_scale[1])

        # Total loss
        total_loss = ml_loss + intr_loss

        return total_loss, ml_correct_predictions_count, intr_correct_predictions_count


class MatchGated_BiEncoder(BiEncoder):
    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        freeze_encoders: bool = False,  # useful for finetuning
    ):
        super(MatchGated_BiEncoder, self).__init__(
            question_model, ctx_model, fix_q_encoder=fix_q_encoder, fix_ctx_encoder=fix_ctx_encoder
        )
        assert question_model.out_features == ctx_model.out_features
        embedding_size = question_model.out_features

        # format: linear_in_out
        self.linear_context_question = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.linear_question_question = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.linear_context_context = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.linear_question_context = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.sigmoid = nn.Sigmoid()

        assert freeze_encoders is not None
        if freeze_encoders:
            for name, parameter in question_model.named_parameters():
                parameter.requires_grad = False
            for name, parameter in ctx_model.named_parameters():
                parameter.requires_grad = False

    def forward(self, *args, is_matching=False, **kwargs) -> Tuple[T, T, T]:
        if not is_matching:
            return super(MatchGated_BiEncoder, self).forward(*args, **kwargs)

        """Return an **interaction** matrix on top of the embeddings."""
        assert len(args) == 0 and len(kwargs) == 2
        q_pooled_out, ctx_pooled_out = kwargs["q_pooled_out"], kwargs["ctx_pooled_out"]  # (n1, d), (n2, d)
        question = q_pooled_out.unsqueeze(1)  # (n1, 1, d)
        context = ctx_pooled_out.unsqueeze(0)  # (1, n2, d)
        del q_pooled_out, ctx_pooled_out

        # Question gate
        question_q = self.linear_question_question(question)  # (n1, 1, d)
        question_c = self.linear_context_question(context)  # (1, n2, d)
        question_gate = self.sigmoid(question_q + question_c)  # (n1, n2, d)
        del question_q, question_c

        # Context gate
        context_q = self.linear_question_context(question)  # (n1, 1, d)
        context_c = self.linear_context_context(context)  # (1, n2, d)
        context_gate = self.sigmoid(context_q + context_c)  # (n1, n2, d)
        del context_q, context_c

        # Apply question gate
        question = question * question_gate  # (n1, n2, d)
        del question_gate

        # Apply context gate
        context = context * context_gate  # (n1, n2, d)
        del context_gate

        # Calculate elementwise dot product
        interaction_mat = (question * context).sum(-1)  # (n1, n2)
        return interaction_mat


class MatchGated_BiEncoderNllLoss(Match_BiEncoderNllLoss):
    pass


class BiEncoderBarlowTwins(BiEncoder):
    """
    Biencoder with Barlow Twins loss function.

    Reference:
        Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins:
        Self-Supervised Learning via Redundancy Reduction.

    Adapted from:
        https://github.com/facebookresearch/barlowtwins/blob/e6f34a01c0cde6f05da6f431ef8a577b42e94e71/main.py#L207
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(BiEncoderBarlowTwins, self).__init__(*args, **kwargs)
        self.batchnorm_q = nn.BatchNorm1d(
            num_features=self.question_model.config.hidden_size,
            affine=False,
        )
        self.batchnorm_c = nn.BatchNorm1d(
            num_features=self.ctx_model.config.hidden_size,
            affine=False,
        )

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos_q=0,
        representation_token_pos_c=0,
        positive_idxs: T = None,
        hard_negative_idxs: T = None,
    ) -> Tuple[T, T]:
        """
        If `positive_idxs` and `hard_negative_idxs` are specified, meaning "training", we exclude
        negative passages from the input (since we don't need it).
        """
        if positive_idxs is not None and hard_negative_idxs is not None:
            assert len(positive_idxs) == len(question_ids)
            context_ids = context_ids[positive_idxs]
            ctx_segments = ctx_segments[positive_idxs]
            ctx_attn_mask = ctx_attn_mask[positive_idxs]

        q_pooled_out, ctx_pooled_out = super(BiEncoderBarlowTwins, self).forward(
            question_ids, question_segments, question_attn_mask,
            context_ids, ctx_segments, ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos_q=representation_token_pos_q,
            representation_token_pos_c=representation_token_pos_c,
        )

        # Apply batch normalization
        if q_pooled_out is not None:
            q_pooled_out = self.batchnorm_q(q_pooled_out)
        if ctx_pooled_out is not None:
            ctx_pooled_out = self.batchnorm_c(ctx_pooled_out)

        return q_pooled_out, ctx_pooled_out


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BiEncoderBarlowTwinsLoss(BiEncoderNllLoss):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Biencoder with Barlow Twins loss function.

        Reference:
            Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins:
            Self-Supervised Learning via Redundancy Reduction.

        Adapted from:
            https://github.com/facebookresearch/barlowtwins/blob/e6f34a01c0cde6f05da6f431ef8a577b42e94e71/main.py#L207

        If `q_vectors` and `ctx_vectors` have the same length, compute Barlow Twins loss
            function using a cross-correlation matrix.
        Otherwise, compute the loss as usual.
        """
        # Training: we don't need to compute negative passages' representations
        if len(q_vectors) == len(ctx_vectors):
            # Empirical cross-correlation matrix
            c = q_vectors.T @ ctx_vectors  # (H, H), where H is hidden size

            # Sum the cross-correlation matrix between all gpus
            batch_size = len(q_vectors) * self.cfg.distributed_world_size
            c.div_(batch_size)
            torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            # TODO: allow configuring lambda
            loss = on_diag + 5e-3 * off_diag
            return loss, torch.tensor([-1])

        else:
            scores = self.get_scores(q_vectors, ctx_vectors)

            if len(q_vectors.size()) > 1:
                q_num = q_vectors.size(0)
                scores = scores.view(q_num, -1)

            return self.calc_given_score_matrix(scores, positive_idx_per_question, loss_scale=loss_scale)


def _select_span_with_token(
    text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]"
) -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.extractive_readers.extractive_reader import pad_to_len

            query_tensor = pad_to_len(
                query_tensor, tensorizer.get_pad_id(), tensorizer.max_length
            )
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError(
                "[START_ENT] toke not found for Entity Linking sample query={}".format(
                    text
                )
            )
    else:
        return query_tensor


"""
Helper functions
"""


def gather(
    cfg,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
):
    """Helper function for `_calc*` functions to gather all needed data."""
    distributed_world_size = cfg.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    return global_q_vector, global_ctxs_vector, positive_idx_per_question, hard_negatives_per_question


def calc_loss(
    cfg,
    loss_function: BiEncoderNllLoss,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    # Gather data
    gathered_data = gather(cfg, local_q_vector, local_ctx_vectors, local_positive_idxs, local_hard_negatives_idxs)
    global_q_vector, global_ctxs_vector, positive_idx_per_question, hard_negatives_per_question = gathered_data

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    return loss, is_correct

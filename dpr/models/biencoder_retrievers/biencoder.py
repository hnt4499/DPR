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
    """
    Bi-Encoder model component. Encapsulates query/question and context/passage
    encoders.
    """

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
    ) -> Tuple[T, T, T]:

        if ids is None:
            return None, None, None
        return sub_model(ids, segments, attn_mask)


    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        positive_idxs: T = None,
        hard_negative_idxs: T = None,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _, q_pooled_out, _ = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
        )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _, ctx_pooled_out, _ = self.get_representation(
            ctx_encoder,
            context_ids,
            ctx_segments,
            ctx_attn_mask,
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
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple using non-tokenized
        (i.e., raw text) data.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text
            sequence
        :param insert_title: enables title insertion at the beginning of the
            context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken
            from samples' pools)
        :param num_other_negatives: amount of other negatives per question
            (taken from samples' pools)
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
                positive_ctx = positive_ctxs[
                    np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query

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
                    ctx.text,
                    title=ctx.title if (insert_title and ctx.title) else None,
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append([i for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx
                )
            ])
            question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat(
            [ctx.view(1, -1) for ctx in ctx_tensors],
            dim=0,
        )
        questions_tensor = torch.cat(
            [q.view(1, -1) for q in question_tensors],
            dim=0,
        )

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
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple using tokenized data.
        :param samples: list of BiEncoderSampleTokenized-s to create the batch
            for
        :param tensorizer: components to create model input tensors from a text
            sequence
        :param insert_title: enables title insertion at the beginning of the
            context sequences
        :param num_hard_negatives: amount of hard negatives (densely retrieved)
            per question
        :param num_bm25_negatives: amount of BM25 negatives (sparsely retrieved)
            per question
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools. This is only
            effective for samples whose gold passage is available. In that case,
            the positive chosen is not necessarily the gold passage. Otherwise,
            the positive passages will be shuffled regardless of this parameter.
        :return: BiEncoderBatch tuple
        """
        question_tensors: List[T] = []
        ctx_tensors: List[T] = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        # Strict settings
        assert insert_title is True  # only allow `insert_title` to be True

        for sample in samples:
            # Skip samples without positive passges (either gold or distant
            # positives)
            if len(sample.positive_passages) == 0:
                continue
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if (shuffle and shuffle_positives) or \
                    (not sample.positive_passages[0].is_gold):
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[
                    np.random.choice(len(positive_ctxs))]
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

            # Context tensors
            sample_ctxs_tensors = [
                tensorizer.concatenate_inputs(
                    ids={
                        "passage_title": list(ctx.title_ids),
                        "passage": list(ctx.text_ids),
                    },
                    get_passage_offset=False,
                    to_max_length=True,
                )
                for ctx in all_ctxs
            ]
            ctx_tensors.extend(sample_ctxs_tensors)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append([i for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx
                )
            ])
            question_tensors.append(
                tensorizer.concatenate_inputs(
                    ids={"question": question_ids},
                    get_passage_offset=False,
                    to_max_length=True,
                )
            )

        ctxs_tensor = torch.cat(
            [ctx.view(1, -1) for ctx in ctx_tensors],
            dim=0,
        ).to(torch.int64)
        questions_tensor = torch.cat(
            [q.view(1, -1) for q in question_tensors],
            dim=0,
        ).to(torch.int64)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
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
        Note that although hard_negative_idx_per_question in not currently in
        use, one can use it for the loss modifications. For example - weighted
        NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per
            batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        # Score scaling, as described in the paper:
        # Sachan, D. S., Patwary, M., Shoeybi, M., Kant, N., Ping, W., Hamilton,
        # W. L., & Catanzaro, B. (2021). End-to-End Training of Neural
        # Retrievers for Open-Domain Question Answering.
        if self.score_scaling:
            hidden_size = q_vectors.shape[1]
            scores = scores / (hidden_size ** 0.5)

        return self.calc_given_score_matrix(
            scores,
            positive_idx_per_question,
            loss_scale=loss_scale,
        )

    @staticmethod
    def calc_given_score_matrix(
        score_mat: T,
        positive_idx_per_question: list,
        loss_scale: float = None,
        reduction: str = "mean"
    ) -> Tuple[T, int]:
        """
        General utility to calculate NLL loss given the un-normalized score
        matrix of shape (n1, n2), where n1 is the number of questions and n2 is
        the number of passages.
        """
        softmax_scores = F.log_softmax(score_mat, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction=reduction,
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
            max_idxs == \
                torch.tensor(positive_idx_per_question).to(max_idxs.device)
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


"""
Helper functions
"""


def gather_biencoder_preds_helper(
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
            torch.empty_like(local_q_vector).cpu()
            .copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu()
            .copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=None,
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
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in positive_idx])
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
                    [[v + total_ctxs for v in l]
                     for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    return (
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
    )


def calc_loss_biencoder(
    cfg,
    loss_function: BiEncoderNllLoss,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP
    mode by exchanging the representations across all the nodes.
    """
    # Gather data
    gathered_data = gather_biencoder_preds_helper(
        cfg,
        local_q_vector, local_ctx_vectors,
        local_positive_idxs, local_hard_negatives_idxs,
    )
    global_q_vector, global_ctxs_vector, positive_idx_per_question, \
        hard_negatives_per_question = gathered_data

    # Calculate loss
    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    return loss, is_correct

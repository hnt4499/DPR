#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor
generator)
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as T
from torch.nn import CrossEntropyLoss

from dpr.data.reader_data import get_best_spans
from dpr.data.general_data_preprocess import TokenizedWikipediaPassages
from dpr.data.data_types import (
    ReaderSample,
    ReaderPassage,
    ReaderBatch,
    ReaderQuestionPredictions,
    SpanPrediction,
)
from dpr.utils.model_utils import (
    init_weights,
    CheckpointState,
    load_state_dict_to_model,
)
from dpr.utils.data_utils import Tensorizer


logger = logging.getLogger()


class Reader(nn.Module):

    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.qa_outputs, self.qa_classifier])

    def forward(
        self,
        input_ids: T,
        attention_mask: T,
        start_positions=None,
        end_positions=None,
        answer_mask=None,
        passage_scores: T = None,
        # Whether to normalize logits and passage scores before scaling
        do_softmax_before_score_scaling: bool = False,
    ):
        # Notations: N - number of questions in a batch, M - number of passages
        # per questions, L - sequence length
        N, M, L = input_ids.size()
        input_ids = input_ids.view(N * M, L)
        attention_mask = attention_mask.view(N * M, L)
        if passage_scores is not None:
            passage_scores = passage_scores.view(N * M, 1)

        start_logits, end_logits, relevance_logits = self._forward(
            input_ids,
            attention_mask,
            passage_scores=passage_scores,
            do_softmax_before_score_scaling=do_softmax_before_score_scaling,
        )
        if self.training:
            return compute_loss(
                start_positions, end_positions, answer_mask,
                start_logits, end_logits, relevance_logits,
                N, M,
            )

        return (
            start_logits.view(N, M, L),
            end_logits.view(N, M, L),
            relevance_logits.view(N, M),
        )

    def _forward(
        self,
        input_ids,
        attention_mask,
        passage_scores=None,
        do_softmax_before_score_scaling=False,
    ):
        # TODO: provide segment values
        sequence_output = self.encoder(
            input_ids, None, attention_mask
        )[0]  # (N * M, L, H)
        logits = self.qa_outputs(sequence_output)  # (N * M, L, 2)

        start_logits, end_logits = logits.split(
            1, dim=-1)  # (N * M, L, 1), (N * M, L, 1)
        start_logits = start_logits.squeeze(-1)  # (N * M, L)
        end_logits = end_logits.squeeze(-1)  # (N * M, L)

        rank_logits = self.qa_classifier(sequence_output[:, 0, :])  # (N * M, 1)

        # Retriever-reader interaction via passage scores
        if passage_scores is not None:

            if do_softmax_before_score_scaling:
                # `start_logits` and `end_logits` do not need to be normalized,
                # since they are passage-independent.
                passage_scores = F.softmax(passage_scores, dim=0)  # (N * M, 1)
                rank_logits = F.softmax(rank_logits, dim=0)  # (N * M, 1)

            start_logits = start_logits * passage_scores  # (N * M, L)
            end_logits = end_logits * passage_scores  # (N * M, L)
            rank_logits = rank_logits * passage_scores  # (N * M, 1)

        return start_logits, end_logits, rank_logits

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        load_state_dict_to_model(self, saved_state.model_dict, strict=strict)


def compute_loss(
    start_positions, end_positions, answer_mask,
    start_logits, end_logits, relevance_logits,
    N, M,
):
    start_positions = start_positions.view(
        N * M, -1)  # (N * M, K), where K = `cfg.train.max_n_answers`
    end_positions = end_positions.view(N * M, -1)  # (N * M, K)
    answer_mask = answer_mask.view(N * M, -1)  # (N * M, K)

    start_logits = start_logits.view(N * M, -1)  # (N * M, L)
    end_logits = end_logits.view(N * M, -1)  # (N * M, L)

    answer_mask = answer_mask.float().to(start_logits.device)  # (N * M, 1)

    ignored_index = start_logits.size(1)  # L
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # Compute switch loss; switch labels are all zero since positive passages
    # are always at the front of each sample during training (see
    # `_create_question_passages_tensors` function below)
    relevance_logits = relevance_logits.view(N, M)  # (N, M)
    switch_labels = torch.zeros(
        N, dtype=torch.long).to(start_logits.device)  # (N,)
    switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))  # scalar

    # compute span loss
    start_losses = [
        (loss_fct(start_logits, _start_positions) * _span_mask)
        for (_start_positions, _span_mask)  # (N * M,) and (N * M,)
        in zip(
            torch.unbind(start_positions, dim=1),
            torch.unbind(answer_mask, dim=1)
        )
    ]

    end_losses = [
        (loss_fct(end_logits, _end_positions) * _span_mask)
        for (_end_positions, _span_mask)  # (N * M,) and (N * M,)
        in zip(
            torch.unbind(end_positions, dim=1),
            torch.unbind(answer_mask, dim=1)
        )
    ]

    loss_tensor = (
        torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) +
        torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
    )  # (N * M, K)

    loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]  # (N, K)
    span_loss = _calc_mml(loss_tensor)

    return span_loss + switch_loss


def _calc_mml(loss_tensor):
    marginal_likelihood = torch.sum(
        torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float()),
        dim=1,
    )
    return -torch.sum(
        torch.log(
            marginal_likelihood + torch.ones(loss_tensor.size(0)).cuda()
            * (marginal_likelihood == 0).float()
        )
    )


def create_reader_input(
    wiki_data: TokenizedWikipediaPassages,
    tensorizer: Tensorizer,
    samples: List[ReaderSample],
    passages_per_question: int,
    max_length: int,
    max_n_answers: int,
    is_train: bool,
    shuffle: bool,
) -> ReaderBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s. This is
    compatible with `GeneralDataset`.
    :param wiki_data: all tokenized wikipedia passages
    :param tensorizer: initialized tensorizer (which contains the tokenizer)
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a
        batch
    :param max_length: max model input sequence length
    :param max_n_answers: max num of answers per single question
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: ReaderBatch instance
    """
    input_ids = []
    start_positions = []
    end_positions = []
    answers_masks = []

    empty_sequence = torch.Tensor().new_full(
        (max_length,),
        tensorizer.get_pad_id(),
        dtype=torch.long,
    )

    for sample in samples:
        if is_train:
            positive_ctxs = sample.gold_passages + sample.positive_passages
            negative_ctxs = sample.negative_passages
        else:
            positive_ctxs = []
            negative_ctxs = sample.positive_passages + sample.negative_passages
            # Need to re-sort samples based on their scores
            negative_ctxs = sorted(
                negative_ctxs, key=lambda x: x.score, reverse=True)
        question_token_ids = sample.question_token_ids

        sample_tensors = _create_question_passages_tensors(
            wiki_data,
            question_token_ids,
            tensorizer,
            positive_ctxs,
            negative_ctxs,
            passages_per_question,
            empty_sequence,
            max_n_answers,
            is_train,
            is_random=shuffle
        )

        if not sample_tensors:
            continue
        sample_input_ids, starts_tensor, ends_tensor, answer_mask \
            = sample_tensors

        input_ids.append(sample_input_ids)

        if is_train:
            start_positions.append(starts_tensor)
            end_positions.append(ends_tensor)
            answers_masks.append(answer_mask)

    input_ids = torch.cat(
        [ids.unsqueeze(0) for ids in input_ids],
        dim=0,
    )  # (N, M)

    if is_train:
        start_positions = torch.stack(start_positions, dim=0)
        end_positions = torch.stack(end_positions, dim=0)
        answers_masks = torch.stack(answers_masks, dim=0)

    return ReaderBatch(
        input_ids,
        start_positions,
        end_positions,
        answers_masks,
    )


def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    positive_a_spans = positives[idx].answers_spans
    return [
        span for span in positive_a_spans
        if (span[0] < max_len and span[1] < max_len)
    ]


def _get_positive_idx(
    positives: List[ReaderPassage],
    max_len: int,
    is_random: bool,
):
    # Select just one positive
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    if not _get_answer_spans(positive_idx, positives, max_len):
        # Question may be too long, find the first positive with at least one
        # valid span
        for i in range(len(positives)):
            if _get_answer_spans(i, positives, max_len):
                return i
        return None
    return positive_idx


def _create_question_passages_tensors(
    wiki_data: TokenizedWikipediaPassages,
    question_token_ids: np.ndarray,
    tensorizer: Tensorizer,
    positives: List[ReaderPassage],
    negatives: List[ReaderPassage],
    total_size: int,
    empty_ids: T,
    max_n_answers: int,
    is_train: bool,
    is_random: bool = True
):
    max_len = empty_ids.size(0)

    if is_train:
        # select just one positive
        positive_idx = _get_positive_idx(positives, max_len, is_random)
        if positive_idx is None:
            return None

        positive = positives[positive_idx]

        if getattr(positive, "sequence_ids", None) is None:
            # Load in passage tokens and title tokens
            positive.load_tokens(
                question_token_ids=question_token_ids,
                **wiki_data.get_tokenized_data(int(positive.id))
            )
            sequence_ids, passage_offset = tensorizer.concatenate_inputs({
                "question": positive.question_token_ids,
                "passage_title": positive.title_token_ids,
                "passage": positive.passage_token_ids,
            }, get_passage_offset=True)

            positive.sequence_ids = sequence_ids
            positive.passage_offset = passage_offset
            positive.answers_spans = [
                (start + passage_offset, end + passage_offset)
                for start, end in positive.answers_spans
            ]

        positive_a_spans = _get_answer_spans(
            positive_idx, positives, max_len)[0: max_n_answers]

        answer_starts = [span[0] for span in positive_a_spans]
        answer_ends = [span[1] for span in positive_a_spans]

        assert all(s < max_len for s in answer_starts)
        assert all(e < max_len for e in answer_ends)

        positive_input_ids = tensorizer.to_max_length(
            positive.sequence_ids.numpy(),
            apply_max_len=True,
        )
        positive_input_ids = torch.from_numpy(positive_input_ids)

        answer_starts_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_starts_tensor[0, 0:len(answer_starts)] = \
            torch.tensor(answer_starts)  # only first passage contains answer

        answer_ends_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_ends_tensor[0, 0:len(answer_ends)] = \
            torch.tensor(answer_ends)  # only first passage contains answer

        answer_mask = torch.zeros((total_size, max_n_answers), dtype=torch.long)
        answer_mask[0, 0:len(answer_starts)] = torch.tensor(
            [1 for _ in range(len(answer_starts))])

        positives_selected = [positive_input_ids]

    else:
        positives_selected = []
        answer_starts_tensor = None
        answer_ends_tensor = None
        answer_mask = None

    positives_num = len(positives_selected)
    negative_idxs = np.random.permutation(range(len(negatives))) if is_random \
        else range(len(negatives) - positives_num)

    negative_idxs = negative_idxs[:total_size - positives_num]
    negatives_selected = []

    for negative_idx in negative_idxs:
        negative = negatives[negative_idx]

        if getattr(negative, "sequence_ids", None) is None:
            # Load in passage tokens and title tokens
            negative.load_tokens(
                question_token_ids=question_token_ids,
                **wiki_data.get_tokenized_data(int(negative.id))
            )
            # Concatenate input tokens
            sequence_ids, passage_offset = tensorizer.concatenate_inputs({
                "question": negative.question_token_ids,
                "passage_title": negative.title_token_ids,
                "passage": negative.passage_token_ids,
            }, get_passage_offset=True)
            negative.sequence_ids = sequence_ids
            negative.passage_offset = passage_offset

        negative_input_ids = tensorizer.to_max_length(
            negative.sequence_ids.numpy(),
            apply_max_len=True,
        )
        negatives_selected.append(torch.from_numpy(negative_input_ids))

    while len(negatives_selected) < total_size - positives_num:
        negatives_selected.append(empty_ids.clone())

    input_ids = torch.stack(
        [t for t in positives_selected + negatives_selected],
        dim=0,
    ).to(torch.int64)

    return (
        input_ids,
        answer_starts_tensor,
        answer_ends_tensor,
        answer_mask,
    )


"""
Helper functions
"""


def get_best_prediction(
    max_answer_length: int,
    tensorizer,
    start_logits,
    end_logits,
    relevance_logits,
    samples_batch: List[ReaderSample],
    passage_thresholds: List[int] = None,
) -> List[ReaderQuestionPredictions]:

    questions_num, passages_per_question = relevance_logits.size()

    _, idxs = torch.sort(
        relevance_logits,
        dim=1,
        descending=True,
    )

    batch_results = []
    for q in range(questions_num):
        sample = samples_batch[q]

        # Need to re-sort samples based on their scores; see
        # `create_reader_input` function
        all_passages = sample.positive_passages + sample.negative_passages
        all_passages = sorted(all_passages, key=lambda x: x.score, reverse=True)

        non_empty_passages_num = len(all_passages)
        nbest: List[SpanPrediction] = []
        for p in range(passages_per_question):
            passage_idx = idxs[q, p].item()
            if (
                passage_idx >= non_empty_passages_num
            ):  # empty passage selected, skip
                continue
            reader_passage = all_passages[passage_idx]
            sequence_ids = reader_passage.sequence_ids
            sequence_len = sequence_ids.size(0)
            # Assuming question & title information is at the beginning of the
            # sequence
            passage_offset = reader_passage.passage_offset

            p_start_logits = start_logits[q, passage_idx].tolist()[
                passage_offset:sequence_len
            ]
            p_end_logits = end_logits[q, passage_idx].tolist()[
                passage_offset:sequence_len
            ]

            ctx_ids = sequence_ids.tolist()[passage_offset:]
            best_spans = get_best_spans(
                tensorizer,
                p_start_logits,
                p_end_logits,
                ctx_ids,
                max_answer_length,
                passage_idx,
                relevance_logits[q, passage_idx].item(),
                top_spans=10,
            )
            nbest.extend(best_spans)
            if len(nbest) > 0 and not passage_thresholds:
                break

        if passage_thresholds:
            passage_rank_matches = {}
            for n in passage_thresholds:
                # By this, it only selects
                curr_nbest = [pred for pred in nbest if pred.passage_index < n]
                passage_rank_matches[n] = curr_nbest[0]
            predictions = passage_rank_matches
        else:
            if len(nbest) == 0:
                predictions = {
                    passages_per_question: SpanPrediction("", -1, -1, -1, "")
                }
            else:
                predictions = {passages_per_question: nbest[0]}
        batch_results.append(
            ReaderQuestionPredictions(
                sample.question, predictions, sample.answers)
        )
    return batch_results

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for the Reader model related data processing tasks
"""

import collections
import logging

from typing import Tuple, List

from dpr.utils.data_utils import Tensorizer
from .general_data import DataPassage, DataSample, GeneralDataset, TokenizedWikipediaPassages

logger = logging.getLogger()


"""
-------------------------------- Dataset-related utilities --------------------------------
"""


class ReaderPassage(DataPassage):
    """For backward compatibility."""
    pass


class ReaderSample(DataSample):
    """For backward compatibility."""
    pass


class ExtractiveReaderDataset(GeneralDataset):
    def __init__(
        self,
        files: str,
        bm25_retrieval_file: str,
        wiki_data: TokenizedWikipediaPassages,
        is_train: bool,
        gold_passages_src: str,
        tensorizer: Tensorizer,
        run_preprocessing: bool,
        num_workers: int,
        debugging: bool = False,
    ):
        super(ExtractiveReaderDataset, self).__init__(
            mode="reader",
            files=files,
            bm25_retrieval_file=bm25_retrieval_file,
            wiki_data=wiki_data,
            is_train=is_train,
            gold_passages_src=gold_passages_src,
            tensorizer=tensorizer,
            run_preprocessing=run_preprocessing,
            num_workers=num_workers,
            debugging=debugging,
        )


SpanPrediction = collections.namedtuple(
    "SpanPrediction",
    [
        "prediction_text",
        "span_score",
        "relevance_score",
        "passage_index",
        "passage_token_ids",
    ],
)


"""
-------------------------------- Inference-related utilities --------------------------------
"""


def get_best_spans(
    tensorizer: Tensorizer,
    start_logits: List,
    end_logits: List,
    ctx_ids: List,
    max_answer_length: int,
    passage_idx: int,
    relevance_score: float,
    top_spans: int = 1,
) -> List[SpanPrediction]:
    """
    Finds the best answer span for the extractive Q&A model
    """
    scores = []
    for (i, s) in enumerate(start_logits):
        for (j, e) in enumerate(end_logits[i : i + max_answer_length]):
            scores.append(((i, i + j), s + e))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    chosen_span_intervals = []
    best_spans = []

    for (start_index, end_index), score in scores:
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= max_answer_length

        if any(
            [
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ]
        ):
            continue

        # extend bpe subtokens to full tokens
        start_index, end_index = _extend_span_to_full_words(
            tensorizer, ctx_ids, (start_index, end_index)
        )

        predicted_answer = tensorizer.to_string(ctx_ids[start_index : end_index + 1])
        best_spans.append(
            SpanPrediction(
                predicted_answer, score, relevance_score, passage_idx, ctx_ids
            )
        )
        chosen_span_intervals.append((start_index, end_index))

        if len(chosen_span_intervals) == top_spans:
            break
    return best_spans


def _extend_span_to_full_words(
    tensorizer: Tensorizer, tokens: List[int], span: Tuple[int, int]
) -> Tuple[int, int]:
    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and tensorizer.is_sub_word_id(tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and tensorizer.is_sub_word_id(tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index

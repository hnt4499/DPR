"""
Base, high-level classes for One-For-All models.
"""


import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor as T

from dpr.data.general_data import TokenizedWikipediaPassages
from dpr.data.data_types import (
    # Retriever (biencoder)
    BiEncoderSampleTokenized,
    BiEncoderBatch,
    BiEncoderPredictionBatch,
    BiEncoderDataConfig,
    BiEncoderTrainingConfig,
    # Reader
    ReaderSample,
    ReaderBatch,
    ReaderPredictionBatch,
    ReaderDataConfig,
    ReaderTrainingConfig,
)
from dpr.utils.data_utils import Tensorizer
from dpr.models.biencoder import BiEncoder
from dpr.models.reader import Reader, create_reader_input
from dpr.utils.model_utils import CheckpointStateOFA, load_state_dict_to_model


class SimpleOneForAllModel(nn.Module):
    def __init__(
        self,
        biencoder: BiEncoder,
        reader: Reader,
        tensorizer: Tensorizer,
    ):
        super(SimpleOneForAllModel, self).__init__()

        self.biencoder = biencoder
        self.reader = reader
        self.tensorizer = tensorizer

    def forward(
        self,
        mode: str,
        biencoder_batch: BiEncoderBatch,
        biencoder_config: BiEncoderTrainingConfig,
        reader_batch: ReaderBatch,
        reader_config: ReaderTrainingConfig,
    ) -> Union[
        BiEncoderPredictionBatch,
        ReaderPredictionBatch,
        Tuple[BiEncoderPredictionBatch, ReaderPredictionBatch],
    ]:
        assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"

        # Retriever (biencoder) forward pass
        if mode in ["retriever", "both"]:
            q_attn_mask = self.tensorizer.get_attn_mask(biencoder_batch.question_ids)
            ctx_attn_mask = self.tensorizer.get_attn_mask(biencoder_batch.context_ids)

            local_q_vector, local_ctx_vectors = self.biencoder(
                biencoder_batch.question_ids,
                biencoder_batch.question_segments,
                q_attn_mask,
                biencoder_batch.context_ids,
                biencoder_batch.ctx_segments,
                ctx_attn_mask,
                encoder_type=biencoder_config.encoder_type,
                representation_token_pos_q=biencoder_config.rep_positions_q,
                representation_token_pos_c=biencoder_config.rep_positions_c,
            )
            biencoder_pred_batch = BiEncoderPredictionBatch(
                question_vector=local_q_vector,
                context_vector=local_ctx_vectors
            )

        # Reader forward pass
        if mode in ["reader", "both"]:
            attn_mask = self.tensorizer.get_attn_mask(reader_batch.input_ids)
            reader_out = self.reader(
                reader_batch.input_ids,
                attn_mask,
                reader_batch.start_positions,
                reader_batch.end_positions,
                reader_batch.answers_mask,
                use_simple_loss=reader_config.use_simple_loss,
                average_loss=reader_config.average_loss,
                passage_scores=reader_batch.passage_scores,
            )
            if isinstance(reader_out, tuple):
                start_logits, end_logits, relevance_logits = reader_out
                loss = None
            else:
                start_logits, end_logits, relevance_logits = None, None, None
                loss = reader_out
            reader_pred_batch = ReaderPredictionBatch(
                total_loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                relevance_logits=relevance_logits,
            )

        if mode == "retriever":
            return biencoder_pred_batch
        elif mode == "reader":
            return reader_pred_batch
        else:
            return biencoder_pred_batch, reader_pred_batch

    def load_state(self, saved_state: CheckpointStateOFA):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]

        # Load
        load_state_dict_to_model(self, saved_state.model_dict)


def create_ofa_input(
    mode: str,
    wiki_data: TokenizedWikipediaPassages,
    tensorizer: Tensorizer,
    samples: List[Tuple[BiEncoderSampleTokenized, ReaderSample]],
    biencoder_config: BiEncoderDataConfig,
    reader_config: ReaderDataConfig,
) -> Union[
    BiEncoderBatch,
    List[ReaderBatch],
    Tuple[BiEncoderBatch, List[ReaderBatch]],
]:

    assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"
    retriever_samples, reader_samples = zip(*samples)

    # Retriever (bi-encoder)
    if mode in ["retriever", "both"]:
        biencoder_batch = BiEncoder.create_biencoder_input(
            samples=retriever_samples,
            tensorizer=tensorizer,
            insert_title=biencoder_config.insert_title,
            num_hard_negatives=biencoder_config.num_hard_negatives,
            num_other_negatives=biencoder_config.num_other_negatives,
            shuffle=biencoder_config.shuffle,
            shuffle_positives=biencoder_config.shuffle_positives,
            hard_neg_fallback=biencoder_config.hard_neg_fallback,
            query_token=biencoder_config.query_token,
        )

    # Reader
    if mode in ["reader", "both"]:
        num_samples = len(samples)
        num_sub_batches = reader_config.num_sub_batches
        assert num_sub_batches > 0

        sub_batch_size = math.ceil(num_samples / num_sub_batches)
        reader_batches: List[ReaderBatch] = []

        for batch_i in range(num_sub_batches):
            start = batch_i * sub_batch_size
            end = min(start + sub_batch_size, num_samples)
            if start >= end:
                break

            reader_batch = create_reader_input(
                wiki_data=wiki_data,
                tensorizer=tensorizer,
                samples=reader_samples[start:end],
                passages_per_question=reader_config.passages_per_question,
                max_length=reader_config.max_length,
                max_n_answers=reader_config.max_n_answers,
                is_train=reader_config.is_train,
                shuffle=reader_config.shuffle,
            )
            reader_batches.append(reader_batch)

    if mode == "retriever":
        return biencoder_batch
    elif mode == "reader":
        return reader_batches
    else:
        return biencoder_batch, reader_batches


def create_biencoder_input_from_reader_input(
    tensorizer: Tensorizer,
    reader_batch: ReaderBatch,
) -> BiEncoderBatch:

    input_ids = reader_batch.input_ids  # (N, M, L)
    question_ids: List[T] = []  # len N
    context_ids: List[T] = []  # len N * M

    for input_id_i in input_ids:
        for j, input_id in enumerate(input_id_i):
            ids = tensorizer.unconcatenate_inputs(
                input_id,
                components={"question", "passage_title", "passage"}
            )

            if ids is None:  # full padding
                context_ids.append(input_id)
                continue

            # Question
            question_id = tensorizer.concatenate_inputs(
                ids={"question": ids["question"].tolist()},
                get_passage_offset=False,
                to_max_length=True,
            )
            if j == 0:
                question_ids.append(question_id)
            else:
                assert (question_id == question_ids[-1]).all()

            # Passage
            passage_title = ids["passage_title"]
            passage = ids["passage"]
            context_ids.append(tensorizer.concatenate_inputs(
                ids={"passage_title": passage_title.tolist(), "passage": passage.tolist()},
                get_passage_offset=False,
                to_max_length=True,
            ))

    question_ids = torch.stack(question_ids)
    context_ids = torch.stack(context_ids)

    question_segments = torch.zeros_like(question_ids)
    context_segments = torch.zeros_like(context_ids)

    biencoder_batch = BiEncoderBatch(
        question_ids=question_ids,
        question_segments=question_segments,
        context_IDs=None,  # not used
        context_ids=context_ids,
        ctx_segments=context_segments,
        is_positive=None,  # not used
        hard_negatives=None,  # not used
        encoder_type=None,  # not used
    )
    return biencoder_batch
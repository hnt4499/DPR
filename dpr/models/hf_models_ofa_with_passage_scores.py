from typing import Tuple, List, Union

import torch
from torch import Tensor as T

from dpr.models.biencoder import BiEncoder
from dpr.models.reader import Reader
from dpr.models.one_for_all_base import SimpleOneForAllModel, create_biencoder_input_from_reader_input
from dpr.models.hf_models import get_optimizer, HFBertEncoder
from dpr.models.hf_models_single_model import get_bert_tensorizer
from dpr.data.data_types import (
    BiEncoderBatch,
    BiEncoderTrainingConfig,
    BiEncoderPredictionBatch,
    ReaderBatch,
    ReaderTrainingConfig,
    ReaderPredictionBatch,
    ForwardPassOutputsTrain,
)
from dpr.models.biencoder import calc_loss as calc_loss_biencoder
from dpr.models.reader import compute_loss as calc_loss_reader
from dpr.utils.model_utils import move_to_device


def do_ofa_fwd_pass(
    trainer,
    mode: str,
    backward: bool,  # whether to backward loss
    step: bool,  # whether to perform `optimizer.step()`
    biencoder_input: BiEncoderBatch,
    biencoder_config: BiEncoderTrainingConfig,
    reader_inputs: List[ReaderBatch],
    reader_config: ReaderTrainingConfig,
    inference_only: bool = False,
) -> Union[
    ForwardPassOutputsTrain,
    BiEncoderPredictionBatch,
    List[ReaderPredictionBatch],
    Tuple[BiEncoderPredictionBatch, List[ReaderPredictionBatch]]
]:

    """
    Note: if `inference_only` is set to True:
        1. No loss is computed.
        2. No backward pass is performed.
        3. All predictions are transformed to CPU to save memory.

    Note: biencoder_config is always required (even when `mode=="reader`).
    """

    assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"
    if inference_only:
        assert (not backward) and (not step) and (not trainer.model.training)
    biencoder_is_correct = None
    biencoder_input = None
    biencoder_preds = None
    reader_input_tot = None
    reader_preds_tot = None

    # Forward pass and backward pass for biencoder
    if mode in ["retriever", "both"]:
        biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), trainer.cfg.device))

        if trainer.model.training:
            biencoder_preds: BiEncoderPredictionBatch = trainer.model(
                mode="retriever",
                biencoder_batch=biencoder_input,
                biencoder_config=biencoder_config,
                reader_batch=None,
                reader_config=None,
            )

        else:
            with torch.no_grad():
                biencoder_preds: BiEncoderPredictionBatch = trainer.model(
                    mode="retriever",
                    biencoder_batch=biencoder_input,
                    biencoder_config=biencoder_config,
                    reader_batch=None,
                    reader_config=None,
                )

        if not inference_only:
            # Calculate biencoder loss
            biencoder_loss, biencoder_is_correct = calc_loss_biencoder(
                cfg=trainer.cfg,
                loss_function=trainer.biencoder_loss_function,
                local_q_vector=biencoder_preds.question_vector,
                local_ctx_vectors=biencoder_preds.context_vector,
                local_positive_idxs=biencoder_input.is_positive,
                local_hard_negatives_idxs=biencoder_input.hard_negatives,
                loss_scale=None,
            )
            biencoder_is_correct = biencoder_is_correct.sum().item()
            biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), "cpu"))

            # Re-calibrate loss
            if trainer.cfg.n_gpu > 1:
                biencoder_loss = biencoder_loss.mean()
            if trainer.cfg.train.gradient_accumulation_steps > 1:
                biencoder_loss = biencoder_loss / trainer.cfg.gradient_accumulation_steps

            if backward:
                assert trainer.model.training, "Model is not in training mode!"
                trainer.backward(
                    loss=biencoder_loss,
                    optimizer=trainer.biencoder_optimizer,
                    scheduler=trainer.biencoder_scheduler,
                    step=step,
                )

            else:
                biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), "cpu"))

            biencoder_preds = BiEncoderPredictionBatch(
                **move_to_device(biencoder_preds._asdict(), "cpu")
            )

    # Forward and backward pass for reader
    if mode in ["reader", "both"]:
        reader_total_loss = 0
        reader_input_tot: List[ReaderBatch] = []
        reader_preds_tot: List[ReaderPredictionBatch] = []

        for reader_input in reader_inputs:

            # First we need to forward all passages to the biencoder to get passage scores
            biencoder_input_i = create_biencoder_input_from_reader_input(
                tensorizer=trainer.tensorizer,
                reader_batch=reader_input,
            )
            biencoder_input_i = BiEncoderBatch(**move_to_device(biencoder_input_i._asdict(), trainer.cfg.device))
            # Forward
            if trainer.model.training:
                biencoder_preds_i: BiEncoderPredictionBatch = trainer.model(
                    mode="retriever",
                    biencoder_batch=biencoder_input_i,
                    biencoder_config=biencoder_config,
                    reader_batch=None,
                    reader_config=None,
                )
            else:
                with torch.no_grad():
                    biencoder_preds_i: BiEncoderPredictionBatch = trainer.model(
                    mode="retriever",
                    biencoder_batch=biencoder_input_i,
                    biencoder_config=biencoder_config,
                    reader_batch=None,
                    reader_config=None,
                )

            # Get passage scores
            question_vectors: T = biencoder_preds_i.question_vector  # (N, H)
            context_vectors: T = biencoder_preds_i.context_vector  # (N * M, H)
            N, H = question_vectors.size()

            question_vectors = question_vectors.view(N, 1, H)  # (N, 1, H)
            context_vectors = context_vectors.view(N, -1, H).permute(0, 2, 1)  # (N, H, M)

            if trainer.model.training:
                passage_scores = torch.matmul(question_vectors, context_vectors).squeeze(1)  # (N, M)
            else:
                with torch.no_grad():
                    passage_scores = torch.matmul(question_vectors, context_vectors).squeeze(1)  # (N, M)

            reader_input = ReaderBatch(**move_to_device(reader_input._asdict(), trainer.cfg.device))
            reader_input._replace(passage_scores=passage_scores)

            # Release memory
            del biencoder_input_i, biencoder_preds_i, question_vectors, context_vectors, passage_scores

            if trainer.model.training:
                reader_preds: ReaderPredictionBatch = trainer.model(
                    mode="reader",
                    biencoder_batch=None,
                    biencoder_config=None,
                    reader_batch=reader_input,
                    reader_config=reader_config,
                )

                reader_loss = reader_preds.total_loss / len(reader_inputs)  # scale by number of sub batches
                reader_total_loss += reader_loss

                # Re-calibrate loss
                if trainer.cfg.n_gpu > 1:
                    reader_loss = reader_loss.mean()
                if trainer.cfg.train.gradient_accumulation_steps > 1:
                    reader_loss = reader_loss / trainer.cfg.gradient_accumulation_steps

                if backward:
                    assert trainer.model.training, "Model is not in training mode!"
                    trainer.backward(
                        loss=reader_loss,
                        optimizer=trainer.reader_optimizer,
                        scheduler=trainer.reader_scheduler,
                        step=step,
                    )

            else:
                with torch.no_grad():
                    reader_preds: ReaderPredictionBatch = trainer.model(
                        mode="reader",
                        biencoder_batch=None,
                        biencoder_config=None,
                        reader_batch=reader_input,
                        reader_config=reader_config,
                    )

                if not inference_only:
                    questions_num, passages_per_question, _ = reader_input.input_ids.size()
                    reader_total_loss = calc_loss_reader(
                        start_positions=reader_input.start_positions,
                        end_positions=reader_input.end_positions,
                        answers_mask=reader_input.answers_mask,
                        start_logits=reader_preds.start_logits,
                        end_logits=reader_preds.end_logits,
                        relevance_logits=reader_preds.relevance_logits,
                        N=questions_num,
                        M=passages_per_question,
                        use_simple_loss=reader_config.use_simple_loss,
                        average=reader_config.average_loss,
                    )

            reader_input = ReaderBatch(**move_to_device(reader_input._asdict(), "cpu"))
            reader_input_tot.append(reader_input)

            reader_preds = ReaderPredictionBatch(
                **move_to_device(reader_preds._asdict(), "cpu")
            )
            reader_preds_tot.append(reader_preds)

    if inference_only:
        if mode == "retriever":
            return biencoder_preds
        elif mode == "reader":
            return reader_preds_tot
        else:
            return biencoder_preds, reader_preds_tot

    else:
        # Total loss; for now use 1:1 weights
        if mode == "retriever":
            loss = biencoder_loss
        elif mode == "reader":
            loss = reader_total_loss
        else:
            loss = biencoder_loss + reader_total_loss

        outputs = ForwardPassOutputsTrain(
            loss=loss,
            biencoder_is_correct=biencoder_is_correct,
            biencoder_input=biencoder_input,
            biencoder_preds=biencoder_preds,
            reader_input=reader_input_tot,
            reader_preds=reader_preds_tot,
        )
        return outputs


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
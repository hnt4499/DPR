from typing import Tuple, List

import torch

from dpr.models.biencoder import BiEncoder
from dpr.models.reader import Reader
from dpr.models.one_for_all_base import SimpleOneForAllModel
from dpr.models.hf_models import get_optimizer, HFBertEncoder
from dpr.models.hf_models_single_model import get_bert_tensorizer
from dpr.data.data_types import (
    BiEncoderBatch,
    BiEncoderTrainingConfig,
    BiEncoderPredictionBatch,
    ReaderBatch,
    ReaderTrainingConfig,
    ReaderPredictionBatch,
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
) -> Tuple[torch.Tensor, int]:

    assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"
    biencoder_is_correct = 0

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
        del biencoder_input  # release memory

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

    # Forward and backward pass for reader
    if mode in ["reader", "both"]:
        reader_total_loss = 0
        for reader_input in reader_inputs:
            reader_input = ReaderBatch(**move_to_device(reader_input._asdict(), trainer.cfg.device))

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

    # Total loss; for now use 1:1 weights
    if mode == "retriever":
        loss = biencoder_loss
    elif mode == "reader":
        loss = reader_total_loss
    else:
        loss = biencoder_loss + reader_total_loss

    return loss, biencoder_is_correct


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
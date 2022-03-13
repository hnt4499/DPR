#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Pipeline to train DPR One-for-all model. This script is largely edited from `train_dense_encoder.py`
and not cleaned up well (biencoder's artifacts might still be somewhere in the script).

This script also allows evaluating old models (which does not use general dataset scheme)
with using new dataset scheme.
"""

import logging
import math
import os
import random
import sys
import time
import yaml
from typing import Tuple, List
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np

from dpr.models import (
    init_ofa_model,
    init_biencoder_components,
    init_reader_components,
    init_loss,
)
from dpr.utils.legacy_utils import (
    load_states_from_checkpoint_legacy,
    convert_from_old_state_to_ofa,
)
from dpr.models.biencoder_retrievers.biencoder import BiEncoderNllLoss
from dpr.data.data_types import (
    BiEncoderBatch,
    BiEncoderDataConfig,
    BiEncoderPredictionBatch,
    BiEncoderTrainingConfig,
    ReaderBatch,
    ReaderDataConfig,
    ReaderSample,
    ReaderTrainingConfig,
    ReaderPredictionBatch,
    ReaderQuestionPredictions,
    ForwardPassOutputsTrain,
)
from dpr.utils.conf_utils import OneForAllDatasetsCfg
from dpr.data.general_data import GeneralDatasetScheme
from dpr.data.biencoder_data import (
    Dataset,
    DEFAULT_SELECTOR,
    RepStaticPosTokenSelector,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    ShardedDataIteratorClustering,
    MultiSetDataIterator,
)
from dpr.models.ofa.one_for_all_base import create_ofa_input, SimpleOneForAllModel
from dpr.models.ofa.hf_models_ofa_simple import do_ofa_fwd_pass as ofa_simple_fw_pass

from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.dist_utils import gather as _gather_reader
from dpr.utils.model_utils import (
    CheckpointState,
    setup_for_distributed_mode,
    get_schedule_linear,
    CheckpointStateOFA,
    get_model_file,
    get_model_obj,
)

from dpr.models.extractive_readers.extractive_reader import (
    get_best_prediction as _get_best_prediction_reader
)
from dpr.data.qa_validation import exact_match_score, f1_score


logger = logging.getLogger()
setup_logger(logger)


class OneForAllTrainer(object):
    """
    One-for-all training pipeline component. Can be used to initiate or resume training and validate the trained model
    one both retrieval and extractive question answering task.
    """

    def __init__(self, cfg: DictConfig):

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint_legacy(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

            if isinstance(saved_state, CheckpointStateOFA):
                self.mode = "normal"
                # Initialize everything
                gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
                tensorizer, model, biencoder_optimizer, reader_optimizer, forward_fn = init_ofa_model(
                    cfg.encoder.encoder_model_type, cfg, gradient_checkpointing=gradient_checkpointing,
                )

            else:
                # Only allowed during evaluation-only mode
                assert isinstance(saved_state, CheckpointState)
                assert cfg.train_datasets is None or len(cfg.train_datasets) == 0
                # Convert from old state to OFA state
                saved_state, self.mode = convert_from_old_state_to_ofa(saved_state)

                if self.mode == "biencoder":
                    # Sanity check
                    assert cfg.evaluate_retriever and (not cfg.evaluate_reader)
                    # Initialize everything
                    tensorizer, biencoder, _ = init_biencoder_components(
                        cfg.encoder.encoder_model_type, cfg, inference_only=True,
                    )
                    reader = None
                else:
                    # Sanity check
                    assert cfg.evaluate_reader and (not cfg.evaluate_retriever)
                    # Initialize everything
                    tensorizer, reader, _ = init_reader_components(
                        cfg.encoder.encoder_model_type, cfg, inference_only=True,
                    )
                    biencoder = None

                # Create a "fake" one-for-all model
                model = SimpleOneForAllModel(
                    biencoder=biencoder, reader=reader, tensorizer=tensorizer,
                )

                # Modify config
                cfg.ignore_checkpoint_optimizer = True
                cfg.ignore_checkpoint_offset = True
                cfg.gradient_checkpointing = False
                cfg.fp16 = False
                # Place holder for backward compatibility
                gradient_checkpointing = False
                biencoder_optimizer = None
                reader_optimizer = None
                forward_fn = ofa_simple_fw_pass  # always the simplest

        else:
            self.mode = "normal"
            # Initialize everything
            gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
            tensorizer, model, biencoder_optimizer, reader_optimizer, forward_fn = init_ofa_model(
                cfg.encoder.encoder_model_type, cfg, gradient_checkpointing=gradient_checkpointing,
            )

        model, (biencoder_optimizer, reader_optimizer) = setup_for_distributed_mode(
            model,
            [biencoder_optimizer, reader_optimizer],
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.forward_fn = forward_fn
        self.model = model
        self.cfg = cfg
        self.ds_cfg = OneForAllDatasetsCfg(cfg)
        self.biencoder_optimizer = biencoder_optimizer
        self.biencoder_scheduler_state = None
        self.reader_optimizer = reader_optimizer
        self.reader_scheduler_state = None
        self.clustering = cfg.biencoder.clustering
        if self.clustering:
            cfg.global_loss_buf_sz = 72000000  # this requires a lot of memory

        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.best_validation_result = None
        self.best_cp_name = None

        # Biencoder loss function (note that reader loss is automatically computed)
        self.biencoder_loss_function: BiEncoderNllLoss = init_loss(cfg.encoder.encoder_model_type, cfg)

        if saved_state:
            self._load_saved_state(saved_state)

        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):

        hydra_datasets = (
            self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        )
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names
            if is_train_set
            else self.ds_cfg.dev_datasets_names,
        )

        # randomized data loading to avoid file system congestion
        datasets_list: List[Dataset] = [ds for ds in hydra_datasets]
        rnd = random.Random(rank)
        rnd.shuffle(datasets_list)

        for dataset in datasets_list:
            if isinstance(dataset, GeneralDatasetScheme):
                dataset.load_data(wiki_data=self.ds_cfg.wiki_data, tensorizer=self.tensorizer)
            else:
                dataset.load_data()

        if is_train_set and self.clustering:
            sharded_iterators = [
                ShardedDataIteratorClustering(
                    self.cfg,
                    ds,
                    shard_id=self.shard_id,
                    num_shards=self.distributed_factor,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    shuffle_seed=shuffle_seed,
                    offset=offset,
                )
                for ds in hydra_datasets
            ]
        else:
            sharded_iterators = [
                ShardedDataIterator(
                    ds,
                    shard_id=self.shard_id,
                    num_shards=self.distributed_factor,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    shuffle_seed=shuffle_seed,
                    offset=offset,
                )
                for ds in hydra_datasets
            ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            rank=rank,
        )

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps

        reader_num_sub_batches = cfg.train.reader_num_sub_batches
        retriever_batch_size = cfg.train.batch_size
        reader_sub_batch_size = math.ceil(retriever_batch_size / reader_num_sub_batches)
        reader_num_sub_batches = math.ceil(retriever_batch_size / reader_sub_batch_size)

        if self.biencoder_scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading biencoder scheduler state %s", self.biencoder_scheduler_state)
            shift = int(self.biencoder_scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            self.biencoder_scheduler = get_schedule_linear(
                self.biencoder_optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )

            logger.info("Loading reader scheduler state %s", self.reader_scheduler_state)
            shift = int(self.reader_scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            self.reader_scheduler = get_schedule_linear(
                self.reader_optimizer,
                warmup_steps * reader_num_sub_batches,
                total_updates * reader_num_sub_batches,
                steps_shift=shift,
            )
        else:
            self.biencoder_scheduler = get_schedule_linear(
                self.biencoder_optimizer,
                warmup_steps,
                total_updates,
            )
            self.reader_scheduler = get_schedule_linear(
                self.reader_optimizer,
                warmup_steps * reader_num_sub_batches,
                total_updates * reader_num_sub_batches,
            )

        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        # TODO: allow setting metric to monitor
        # For now only monitoring average rank metric
        if epoch == cfg.biencoder.val_av_rank_start_epoch:
            self.best_validation_result = None

        if not cfg.dev_datasets:
            validation_loss = 0
        else:
            if epoch >= cfg.biencoder.val_av_rank_start_epoch:
                validation_loss = self.validate_biencoder_average_rank()
                self.validate_reader()
            else:
                validation_loss = self.validate_biencoder_nll()
                self.validate_reader()

        if save_cp:
            cp_name = self._save_checkpoint(epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_biencoder_nll(self) -> float:
        logger.info("Retriever NLL validation ...")
        cfg = self.cfg
        self.model.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        batches = 0
        dataset = 0

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)

            ds_cfg = self.ds_cfg.dev_datasets[dataset]

            # Biencoder data config
            biencoder_data_config = BiEncoderDataConfig(
                insert_title=True,
                num_hard_negatives=cfg.biencoder.hard_negatives,
                num_other_negatives=cfg.biencoder.other_negatives,
                shuffle=False,
                shuffle_positives=ds_cfg.shuffle_positives,
                hard_neg_fallback=True,
                query_token=ds_cfg.special_token,
            )

            # Prepare data
            biencoder_batch: BiEncoderBatch = create_ofa_input(
                mode="retriever",
                wiki_data=self.ds_cfg.wiki_data,
                tensorizer=self.tensorizer,
                samples=samples_batch,
                biencoder_config=biencoder_data_config,
                reader_config=None,
            )

            # Get the token to be used for representation selection
            rep_positions_q = ds_cfg.selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer, self.model,
            )
            rep_positions_c = ds_cfg.selector.get_positions(
                biencoder_batch.context_ids, self.tensorizer, self.model
            )

            # Biencoder training config
            biencoder_training_config = BiEncoderTrainingConfig(
                encoder_type=ds_cfg.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
            )

            with torch.no_grad():
                outputs: ForwardPassOutputsTrain = self.forward_fn(
                    trainer=self,
                    mode="retriever",
                    backward=False,
                    step=False,
                    biencoder_input=biencoder_batch,
                    biencoder_config=biencoder_training_config,
                    reader_inputs=None,
                    reader_config=None,
                )
            loss = outputs.loss
            correct_cnt = outputs.biencoder_is_correct

            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % cfg.train.log_batch_step == 0:
                logger.info(
                    "Retriever NLL: Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        to_log = (f"Retriever NLL Validation: loss = {total_loss:.4f} correct prediction ratio  "
                  f"{total_correct_predictions}/{total_samples} ~ {correct_ratio:.4f}")
        logger.info(to_log)

        return total_loss

    def validate_biencoder_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Retriever average rank validation ...")

        cfg = self.cfg
        self.model.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.biencoder.val_av_rank_bsz
        sim_score_f = self.biencoder_loss_function.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        dataset = 0
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            # samples += 1
            if (
                len(q_represenations)
                > cfg.biencoder.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            ds_cfg = self.ds_cfg.dev_datasets[dataset]

            # Biencoder data config
            biencoder_data_config = BiEncoderDataConfig(
                insert_title=True,
                num_hard_negatives=cfg.biencoder.hard_negatives,
                num_other_negatives=cfg.biencoder.other_negatives,
                shuffle=False,
                shuffle_positives=ds_cfg.shuffle_positives,
                hard_neg_fallback=True,
                query_token=ds_cfg.special_token,
            )

            # Prepare data
            biencoder_batch: BiEncoderBatch = create_ofa_input(
                mode="retriever",
                wiki_data=self.ds_cfg.wiki_data,
                tensorizer=self.tensorizer,
                samples=samples_batch,
                biencoder_config=biencoder_data_config,
                reader_config=None,
            )

            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_batch.context_ids.to(cfg.device)
            ctxs_segments = biencoder_batch.ctx_segments.to(cfg.device)
            bsz = ctxs_ids.size(0)

            # Get the token to be used for representation selection
            rep_positions_q = ds_cfg.selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer, self.model
            )
            rep_positions_c = ds_cfg.selector.get_positions(
                biencoder_batch.context_ids, self.tensorizer, self.model
            )

            # Biencoder training config
            biencoder_training_config = BiEncoderTrainingConfig(
                encoder_type=ds_cfg.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
            )

            # Split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_batch.question_ids.to(cfg.device), biencoder_batch.question_segments.to(cfg.device))
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # If we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                # Prepare input data
                biencoder_batch_j = BiEncoderBatch(
                    question_ids=q_ids,
                    question_segments=q_segments,
                    context_IDs=None,  # not used
                    context_ids=ctx_ids_batch,
                    ctx_segments=ctx_seg_batch,
                    is_positive=None,  # not used
                    hard_negatives=None,  # not used
                    encoder_type=None,  # not used
                )

                with torch.no_grad():
                    biencoder_preds: BiEncoderPredictionBatch = self.forward_fn(
                        trainer=self,
                        mode="retriever",
                        backward=False,
                        step=False,
                        biencoder_input=biencoder_batch_j,
                        biencoder_config=biencoder_training_config,
                        reader_inputs=None,
                        reader_config=None,
                        inference_only=True,
                    )
                    q_dense = biencoder_preds.question_vector
                    ctx_dense = biencoder_preds.context_vector

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_batch.is_positive
            positive_idx_per_question.extend(
                [total_ctxs + v for v in batch_positive_idxs]
            )

            logger.info(
                "Retriever Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                i,
                len(ctx_represenations),
                len(q_represenations),
            )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info(
            "Retriever Av.rank validation: total q_vectors size=%s", q_represenations.size()
        )
        logger.info(
            "Retriever Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
        )

        # Calculate cosine similarity scores
        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != cfg.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        logger.info(
            "Retriever Av.rank validation: average rank %s, total questions=%d", av_rank, q_num
        )

        return av_rank

    def validate_reader(self):
        logger.info("Reader validation ...")
        cfg = self.cfg
        self.model.eval()
        if self.dev_iterator is None:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )

        log_result_step = 1  # reader validation needs to be much more verbose
        all_results: List[ReaderQuestionPredictions] = []
        dataset = 0  # dataset index

        eval_top_docs = cfg.reader.eval_top_docs
        for i, samples_batch in enumerate(self.dev_iterator.iterate_ds_data()):

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            ds_cfg = self.ds_cfg.dev_datasets[dataset]

            # Reader data config
            reader_data_config = ReaderDataConfig(
                passages_per_question=cfg.reader.passages_per_question_predict,
                max_length=cfg.encoder.sequence_length,
                max_n_answers=cfg.reader.max_n_answers,
                is_train=False,
                shuffle=False,
                num_sub_batches=self.cfg.train.reader_num_sub_batches,
            )

            # Prepare data
            reader_batches: List[ReaderBatch] = create_ofa_input(
                mode="reader",
                wiki_data=self.ds_cfg.wiki_data,
                tensorizer=self.tensorizer,
                samples=samples_batch,
                biencoder_config=None,
                reader_config=reader_data_config,
            )
            # Reader training config
            reader_training_config = ReaderTrainingConfig(
                use_simple_loss=cfg.reader.use_simple_loss,
                average_loss=cfg.reader.average_loss,
                do_softmax_before_score_scaling=cfg.reader.do_softmax_before_score_scaling,
            )

            # Get the token to be used for representation selection
            # TODO: add support for other selectors than static selector
            assert isinstance(ds_cfg.selector, RepStaticPosTokenSelector)
            rep_positions_q = ds_cfg.selector.get_positions(
                None, self.tensorizer, self.model,
            )
            rep_positions_c = ds_cfg.selector.get_positions(
                None, self.tensorizer, self.model
            )

            # Biencoder training config
            biencoder_training_config = BiEncoderTrainingConfig(
                encoder_type=ds_cfg.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
            )

            # Do foward pass
            with torch.no_grad():
                reader_preds_batch: List[ReaderPredictionBatch] = self.forward_fn(
                    trainer=self,
                    mode="reader",
                    backward=False,
                    step=False,
                    biencoder_input=None,
                    biencoder_config=biencoder_training_config,
                    reader_inputs=reader_batches,
                    reader_config=reader_training_config,
                    inference_only=True,
                )

            start_logits, end_logits, relevance_logits = [], [], []
            for reader_preds in reader_preds_batch:
                start_logits.append(reader_preds.start_logits)
                end_logits.append(reader_preds.end_logits)
                relevance_logits.append(reader_preds.relevance_logits)

            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            relevance_logits = torch.cat(relevance_logits, dim=0)

            samples_batch: List[ReaderSample] = list(zip(*samples_batch))[1]
            batch_predictions = _get_best_prediction_reader(
                max_answer_length=self.cfg.reader.max_answer_length,
                tensorizer=self.tensorizer,
                start_logits=start_logits,
                end_logits=end_logits,
                relevance_logits=relevance_logits,
                samples_batch=samples_batch,
                passage_thresholds=eval_top_docs,
            )

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info("Reader eval step: %d ", i)

        ems = defaultdict(list)
        f1s = defaultdict(list)

        for q_predictions in all_results:
            gold_answers = q_predictions.gold_answers
            span_predictions = (
                q_predictions.predictions
            )  # {top docs threshold -> SpanPrediction()}
            for (n, span_prediction) in span_predictions.items():
                # Exact match
                em_hit = max(
                    [
                        exact_match_score(span_prediction.prediction_text, ga)
                        for ga in gold_answers
                    ]
                )
                ems[n].append(em_hit)

                # F1 score
                f1_hit = max(
                    [
                        f1_score(span_prediction.prediction_text, ga)
                        for ga in gold_answers
                    ]
                )
                f1s[n].append(f1_hit)

        # Sync between GPUs
        ems, f1s = _gather_reader(self.cfg, [ems, f1s])

        em = 0
        for n in sorted(ems[0].keys()):
            ems_n = sum([em[n] for em in ems], [])  # gather and concatenate
            em = np.mean(ems_n)

            if cfg.local_rank in [-1, 0]:
                logger.info("n=%d\tEM %.2f" % (n, em * 100))

        for n in sorted(f1s[0].keys()):
            f1s_n = sum([f1[n] for f1 in f1s], [])  # gather and concatenate
            f1 = np.mean(f1s_n)

            if cfg.local_rank in [-1, 0]:
                logger.info("n=%d\tF1 %.2f" % (n, f1 * 100))

        return em

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: bool,
    ):
        """Handling back-propagation."""
        if self.cfg.fp16:
            from apex import amp

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), self.cfg.train.max_grad_norm
                )
        else:
            loss.backward()
            if self.cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.max_grad_norm
                )

        if step:
            optimizer.step()
            scheduler.step()
            self.model.zero_grad()

    def _train_epoch(
        self,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        # General config
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        seed = cfg.seed

        self.model.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        dataset = 0
        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            ds_cfg = self.ds_cfg.train_datasets[dataset]

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            np.random.seed(seed + epoch + data_iteration)
            torch.manual_seed(seed + epoch + data_iteration)
            if cfg.n_gpu > 0:
                torch.cuda.manual_seed_all(seed + epoch + data_iteration)

            # Biencoder data config
            biencoder_data_config = BiEncoderDataConfig(
                insert_title=True,
                num_hard_negatives=cfg.biencoder.hard_negatives,
                num_other_negatives=cfg.biencoder.other_negatives,
                shuffle=True,
                shuffle_positives=ds_cfg.shuffle_positives,
                hard_neg_fallback=True,
                query_token=ds_cfg.special_token,
            )

            # Reader data config
            reader_data_config = ReaderDataConfig(
                passages_per_question=cfg.reader.passages_per_question,
                max_length=cfg.encoder.sequence_length,
                max_n_answers=cfg.reader.max_n_answers,
                is_train=True,
                shuffle=True,
                num_sub_batches=cfg.train.reader_num_sub_batches,
            )

            # Prepare data
            biencoder_batch, reader_batch = create_ofa_input(
                mode="both",
                wiki_data=self.ds_cfg.wiki_data,
                tensorizer=self.tensorizer,
                samples=samples_batch,
                biencoder_config=biencoder_data_config,
                reader_config=reader_data_config,
            )

            # Get the token to be used for representation selection
            selector = ds_cfg.selector if ds_cfg else DEFAULT_SELECTOR

            rep_positions_q = selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer, self.model,
            )
            rep_positions_c = selector.get_positions(
                biencoder_batch.context_ids, self.tensorizer, self.model,
            )

            # Biencoder training config
            biencoder_training_config = BiEncoderTrainingConfig(
                encoder_type=ds_cfg.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
            )

            # Reader training config
            reader_training_config = ReaderTrainingConfig(
                use_simple_loss=cfg.reader.use_simple_loss,
                average_loss=cfg.reader.average_loss,
                do_softmax_before_score_scaling=cfg.reader.do_softmax_before_score_scaling,
            )

            step = (i + 1) % self.cfg.train.gradient_accumulation_steps == 0  # whether to `step()` now
            outputs: ForwardPassOutputsTrain = self.forward_fn(
                trainer=self,
                mode="both",
                backward=True,
                step=step,
                biencoder_input=biencoder_batch,
                biencoder_config=biencoder_training_config,
                reader_inputs=reader_batch,
                reader_config=reader_training_config,
            )
            loss = outputs.loss
            correct_cnt = outputs.biencoder_is_correct

            # Record predictions if needed
            if self.clustering:
                iterator: ShardedDataIteratorClustering = train_data_iterator.iterables[dataset]
                iterator.record_predictions(epoch=epoch, model_outs=outputs)

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if i % log_result_step == 0:
                biencoder_lr = self.biencoder_optimizer.param_groups[0]["lr"]
                reader_lr = self.reader_optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, b_lr=%f, r_lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    biencoder_lr,
                    reader_lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(),
                )
                self.model.train()

        logger.info("Epoch finished on %d", cfg.local_rank)

        # If we just evaluate at the last iteration, we don't need to evaluate again
        if data_iteration % eval_step != 0:
            self.validate_and_save(epoch, data_iteration)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.model)
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointStateOFA(
            model_to_save.state_dict(),
            self.biencoder_optimizer.state_dict(),
            self.biencoder_scheduler.state_dict(),
            self.reader_optimizer.state_dict(),
            self.reader_scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointStateOFA):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if self.cfg.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.model)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state)

        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.biencoder_optimizer_dict:
                logger.info("Loading saved biencoder optimizer state ...")
                self.biencoder_optimizer.load_state_dict(saved_state.biencoder_optimizer_dict)

            if saved_state.biencoder_scheduler_dict:
                self.biencoder_scheduler_state = saved_state.biencoder_scheduler_dict

            if saved_state.reader_optimizer_dict:
                logger.info("Loading saved reader optimizer state ...")
                self.reader_optimizer.load_state_dict(saved_state.reader_optimizer_dict)

            if saved_state.reader_scheduler_dict:
                self.reader_scheduler_state = saved_state.reader_scheduler_dict


@hydra.main(config_path="conf", config_name="one_for_all_train_cfg")
def main(cfg: DictConfig):
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        # Save config
        with open("config.yaml", "w") as fout:
            yaml.dump(eval(str(cfg)), fout)

    trainer = OneForAllTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info("No train files are specified.")

        if cfg.evaluate_retriever:
            logger.info("Run 2 types of retriever validation for specified model file")
            trainer.validate_biencoder_nll()
            trainer.validate_biencoder_average_rank()

        if cfg.evaluate_reader:
            logger.info("Run reader validation for specified model file")
            trainer.validate_reader()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()

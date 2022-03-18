#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Minimal training code for dense encoder (retriever) with general data scheme.
Redundant, legacy logics like `ShardedDataIteratorClustering` has been
removed.
"""

import logging
import math
import os
import random
import sys
import time
import yaml
from typing import Tuple

import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf

# General data
from dpr.data.general_data import TokenizedWikipediaPassages
from dpr.data.biencoder_data import BiEncoderGeneralDataset

# Model related
from dpr.models import init_biencoder_components, init_loss
from dpr.models.biencoder_retrievers.biencoder import (
    BiEncoder,
    calc_loss,
)
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)

# Data related
from dpr.utils.data_utils import (
    Tensorizer,
    ShardedDataIterator,
    ShardedDataStreamIterator,
    MultiSetDataIterator,
)
from dpr.utils.dist_utils import all_gather_list

# Typing
from dpr.data.data_types import BiEncoderBatch
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)


logger = logging.getLogger()
setup_logger(logger)


class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, cfg: DictConfig):

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        gradient_checkpointing = getattr(cfg, "gradient_checkpointing", False)
        tensorizer, model, optimizer = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, gradient_checkpointing=gradient_checkpointing,
        )

        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.debugging = cfg.debugging
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.loss_function = init_loss(cfg.encoder.encoder_model_type, cfg)

        self.model_file = model_file
        if saved_state:
            self._load_saved_state(saved_state, resume=cfg.resume)

        self.dev_iterator = None
        self.wiki_data = None

    def get_data_iterator(
        self,
        path: str,
        iterator_class: str,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ):

        if self.wiki_data is None:
            self.wiki_data = TokenizedWikipediaPassages(data_file=self.cfg.wiki_psgs_tokenized)

        self.dataset = BiEncoderGeneralDataset(
            file=path,
            shuffle_positives=is_train_set,
            debugging=self.debugging,
            iterator_class=iterator_class,
            compress=self.cfg.compress and is_train_set,
        )

        self.dataset.load_data(wiki_data=self.wiki_data, tensorizer=self.tensorizer)

        if iterator_class == "ShardedDataIterator":
            iterator = ShardedDataIterator(
                self.dataset,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
        elif iterator_class == "ShardedDataStreamIterator":
            iterator = ShardedDataStreamIterator(
                self.dataset,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                offset=0,
            )
            if offset > 0:
                updates_per_epoch = iterator.max_iterations // \
                    self.cfg.train.gradient_accumulation_steps
                global_step = self.start_epoch * updates_per_epoch + self.start_batch
                iterator.set_offset(global_step)  # offset of this iterator is global step
        else:
            raise NotImplementedError

        return iterator

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train_files,
            cfg.train_iterator_class,
            cfg.train.batch_size,
            is_train_set=True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
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

        if self.scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(
                self.optimizer, warmup_steps, total_updates
            )

        # Eval before any training, but no checkpointing
        if self.model_file is not None and self.cfg.eval_first:
            logger.info("Evaluate loaded model before any training...")
            self.validate_and_save(
                self.start_epoch, iteration=None, scheduler=None, save_cp=False,
            )

        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, cfg.train.num_train_epochs):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int, scheduler, save_cp: bool = True):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = save_cp and cfg.local_rank in [-1, 0]

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        if epoch >= cfg.val_av_rank_start_epoch:
            validation_loss = self.validate_average_rank()
        else:
            validation_loss = self.validate_nll()

        if save_cp and epoch >= cfg.checkpoint_start_epoch:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)
        else:
            logger.info("Model is not saved.")

    def validate_nll(self) -> float:
        logger.info("NLL validation ...")
        cfg = self.cfg
        self.biencoder.eval()

        if self.dev_iterator is None:
            self.dev_iterator = self.get_data_iterator(
                cfg.dev_files,
                cfg.dev_iterator_class,
                cfg.train.dev_batch_size,
                is_train_set=False,
                shuffle=False,
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        batches = 0

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):

            logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)
            biencoder_input = BiEncoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                insert_title=True,
                num_hard_negatives=cfg.train.hard_negatives,
                num_other_negatives=cfg.train.other_negatives,
                shuffle=False,
                shuffle_positives=False,
                hard_neg_fallback=True,
                query_token=None,
            )

            # get the token to be used for representation selection
            rep_positions_q = self.dataset.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer, self.biencoder
            )
            rep_positions_c = self.dataset.selector.get_positions(
                biencoder_input.context_ids, self.tensorizer, self.biencoder
            )

            outp = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_input,
                self.tensorizer,
                self.loss_function,
                cfg,
                encoder_type=self.dataset.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
            )
            loss, correct_cnt = outp

            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % cfg.train.log_batch_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            f"NLL Validation: loss = {total_loss:.4f} correct prediction ratio  "
            f"{total_correct_predictions}/{total_samples} ~ {correct_ratio:.4f}"
        )

        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")

        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.dev_files,
                cfg.dev_iterator_class,
                cfg.train.dev_batch_size,
                is_train_set=False,
                shuffle=False,
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        sim_score_f = self.loss_function.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            biencoder_input = BiEncoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                insert_title=True,
                num_hard_negatives=cfg.train.val_av_rank_hard_neg,
                num_other_negatives=cfg.train.val_av_rank_other_neg,
                shuffle=False,
                shuffle_positives=False,
                hard_neg_fallback=True,
                query_token=None,
            )

            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids.to(cfg.device)
            ctxs_segments = biencoder_input.ctx_segments.to(cfg.device)
            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            rep_positions_q = self.dataset.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer, self.biencoder
            )
            rep_positions_c = self.dataset.selector.get_positions(
                biencoder_input.context_ids, self.tensorizer, self.biencoder
            )

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_input.question_ids.to(cfg.device), biencoder_input.question_segments.to(cfg.device))
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                q_attn_mask = q_attn_mask if q_ids is not None else q_attn_mask
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch).to(cfg.device)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=self.dataset.encoder_type,
                        representation_token_pos_q=rep_positions_q,
                        representation_token_pos_c=rep_positions_c,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend(
                [total_ctxs + v for v in batch_positive_idxs]
            )

            logger.info(
                "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                i,
                len(ctx_represenations),
                len(q_represenations),
            )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info(
            "Av.rank validation: total q_vectors size=%s", q_represenations.size()
        )
        logger.info(
            "Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
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
            "Av.rank validation: average rank %s, total questions=%d", av_rank, q_num
        )

        return av_rank

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        rolling_loss_step = cfg.train.train_rolling_loss_step
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(cfg.seed + epoch + data_iteration)

            biencoder_batch = BiEncoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                cfg.train.hard_negatives,
                cfg.train.other_negatives,
                shuffle=True,
                shuffle_positives=self.cfg.shuffle_positives,
                query_token=None,
            )

            # get the token to be used for representation selection
            rep_positions_q = self.dataset.selector.get_positions(
                biencoder_batch.question_ids, self.tensorizer, self.biencoder
            )
            rep_positions_c = self.dataset.selector.get_positions(
                biencoder_batch.context_ids, self.tensorizer, self.biencoder
            )

            outp = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                self.loss_function,
                cfg,
                encoder_type=self.dataset.encoder_type,
                rep_positions_q=rep_positions_q,
                rep_positions_c=rep_positions_c,
                loss_scale=None,
            )
            loss, correct_cnt = outp

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if cfg.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), cfg.train.max_grad_norm
                    )
            else:
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.biencoder.parameters(), cfg.train.max_grad_norm
                    )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % cfg.train.log_batch_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
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

            # Also log the first batch if resuming from a checkpoint to see
            # if the loss value is normal
            if self.model_file is not None and i == 0:
                logger.info(
                    f"Avg. loss per last 1 batches: {rolling_train_loss}",
                )

            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.biencoder.train()

        logger.info("Epoch finished on %d", cfg.local_rank)

        # If we just evaluate at the last iteration, we don't need to evaluate again
        if data_iteration % eval_step != 0:
            self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.output_dir, f"{cfg.checkpoint_file_name}.{epoch}.{offset}")
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState, resume: bool):
        # Load model weights
        logger.info("Loading model weights from saved state ...")
        model_to_load = get_model_obj(self.biencoder)
        model_to_load.load_state(saved_state)

        if resume:
            # offset is currently ignored since all checkpoints are made after full epochs
            epoch = saved_state.epoch
            offset = saved_state.offset
            if offset == 0:  # epoch has been completed
                epoch += 1
            logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

            self.start_epoch = epoch
            self.start_batch = offset

            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    loss_function,
    cfg,
    encoder_type: str,
    rep_positions_q=0,
    rep_positions_c=0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]:

    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos_q=rep_positions_q,
            representation_token_pos_c=rep_positions_c,
            # Only pass these during training
            positive_idxs=input.is_positive,
            hard_negative_idxs=input.hard_negatives,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type=encoder_type,
                representation_token_pos_q=rep_positions_q,
                representation_token_pos_c=rep_positions_c,
            )

    local_q_vector, local_ctx_vectors = model_out


    loss, is_correct = calc_loss(
        cfg,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.is_positive,
        input.hard_negatives,
        loss_scale=loss_scale,
    )
    is_correct = is_correct.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.gradient_accumulation_steps
    return loss, is_correct


@hydra.main(config_path="conf", config_name="biencoder_general_data_train_cfg")
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

    trainer = BiEncoderTrainer(cfg)
    trainer.run_train()


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

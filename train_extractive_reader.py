#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import json
import sys
import yaml

import hydra
import logging
import numpy as np
import os
import torch

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import List

from dpr.data.qa_validation import exact_match_score, f1_score
from dpr.data.data_types import ReaderBatch, ReaderQuestionPredictions
from dpr.data.general_data import ExtractiveReaderGeneralDataset
from dpr.data.general_data_preprocess import TokenizedWikipediaPassages
from dpr.models import init_reader_components
from dpr.models.extractive_readers.extractive_reader import (
    create_reader_input,
    get_best_prediction,
)
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    set_cfg_params_from_state,
    get_encoder_params_state_from_cfg,
    setup_logger,
    get_gpu_info,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    ShardedDataStreamIterator,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    setup_for_distributed_mode,
    get_model_obj,
)
from dpr.utils.dist_utils import gather


logger = logging.getLogger()
setup_logger(logger)


class ReaderTrainer(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        saved_state = None
        if cfg.model_file is not None:
            saved_state = load_states_from_checkpoint(cfg.model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        gradient_checkpointing = getattr(
            self.cfg, "gradient_checkpointing", False)
        tensorizer, reader, optimizer = init_reader_components(
            cfg.encoder.encoder_model_type, cfg,
            gradient_checkpointing=gradient_checkpointing,
        )

        reader, optimizer = setup_for_distributed_mode(
            reader,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.debugging = getattr(self.cfg, "debugging", False)
        self.wiki_data = None
        self.dev_iterator = None
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None

        self.model_file = cfg.model_file
        if saved_state:
            self._load_saved_state(saved_state, resume=cfg.resume)

    def get_data_iterator(
        self,
        path: str,
        iterator_class: str,
        batch_size: int,
        is_train: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ) -> ShardedDataIterator:

        if self.wiki_data is None:
            self.wiki_data = TokenizedWikipediaPassages(
                data_file=self.cfg.wiki_psgs_tokenized)

        dataset = ExtractiveReaderGeneralDataset(
            file=path,
            shuffle_positives=is_train,
            debugging=self.debugging,
            iterator_class=iterator_class,
            compress=self.cfg.compress and is_train,
        )
        dataset.load_data(wiki_data=self.wiki_data, tensorizer=self.tensorizer)

        if iterator_class == "ShardedDataIterator":
            iterator = ShardedDataIterator(
                dataset,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
        elif iterator_class == "ShardedDataStreamIterator":
            iterator = ShardedDataStreamIterator(
                dataset,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                offset=0,
            )
            if offset > 0:
                updates_per_epoch = iterator.max_iterations // \
                    self.cfg.train.gradient_accumulation_steps
                global_step = (
                    self.start_epoch * updates_per_epoch + self.start_batch)
                # Offset of this iterator is global step
                iterator.set_offset(global_step)
        else:
            raise NotImplementedError

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train_files,
            cfg.train_iterator_class,
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
        )

        logger.info(
            f"Total iterations per epoch={train_iterator.max_iterations}"
        )
        updates_per_epoch = (
            train_iterator.max_iterations //
            cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
            )
        else:
            scheduler = get_schedule_linear(
                self.optimizer, warmup_steps, total_updates)

        # Eval before any training, but no checkpointing
        if self.model_file is not None and self.cfg.eval_first:
            logger.info("Evaluate loaded model before any training...")
            self.validate_and_save(
                self.start_epoch, iteration=None, scheduler=None, save_cp=False,
            )

        eval_step = cfg.train.eval_step
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        global_step = self.start_epoch * updates_per_epoch + self.start_batch

        for epoch in range(self.start_epoch, cfg.train.num_train_epochs):
            logger.info("***** Epoch %d *****", epoch)
            global_step = self._train_epoch(
                scheduler, epoch, eval_step, train_iterator, global_step
            )

        if cfg.local_rank in [-1, 0]:
            logger.info(
                f"Training finished. Best validation checkpoint "
                f"{self.best_cp_name}",
            )

        return

    def validate_and_save(
        self,
        epoch: int,
        iteration: int,
        scheduler,
        save_cp: bool = True,
    ):
        cfg = self.cfg
        # in distributed DDP mode, save checkpoint for only one process
        save_cp = save_cp and cfg.local_rank in [-1, 0]
        reader_validation_score = self.validate()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if reader_validation_score > (self.best_validation_result or 0):
                self.best_validation_result = reader_validation_score
                self.best_cp_name = cp_name
                logger.info(
                    f"New Best validation checkpoint {cp_name} with validation "
                    f"score {reader_validation_score:.2f}"
                )

    def validate(self):
        logger.info("Validation ...")
        cfg = self.cfg
        self.reader.eval()
        if self.dev_iterator is None:
            self.dev_iterator = self.get_data_iterator(
                cfg.dev_files,
                cfg.dev_iterator_class,
                cfg.train.dev_batch_size,
                is_train=False,
                shuffle=False,
            )

        log_result_step = max(cfg.train.log_batch_step // 4, 1)
        all_results = []

        eval_top_docs = cfg.eval_top_docs
        for i, samples_batch in enumerate(self.dev_iterator.iterate_ds_data()):
            input = create_reader_input(
                self.wiki_data,
                self.tensorizer,
                samples_batch,
                cfg.passages_per_question_predict,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=False,
                shuffle=False,
            )

            input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            with torch.no_grad():
                start_logits, end_logits, relevance_logits = self.reader(
                    input.input_ids, attn_mask
                )

            batch_predictions = get_best_prediction(
                self.cfg.max_answer_length,
                self.tensorizer,
                start_logits,
                end_logits,
                relevance_logits,
                samples_batch,
                passage_thresholds=eval_top_docs,
            )

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info("Eval step: %d ", i)

            if self.debugging and i == 5:
                logger.info(
                    "Reached 5 iterations when debugging mode is on. "
                    "Early stopping..."
                )
                break

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
        ems, f1s = gather(self.cfg, [ems, f1s])

        em = 0
        for n in sorted(ems[0].keys()):
            ems_n = sum([em[n] for em in ems], [])  # gather and concatenate
            em = np.mean(ems_n)
            logger.info("n=%d\tEM %.2f" % (n, em * 100))

        f1 = 0
        for n in sorted(f1s[0].keys()):
            f1s_n = sum([f1[n] for f1 in f1s], [])  # gather and concatenate
            f1 = np.mean(f1s_n)
            logger.info("n=%d\tF1 %.2f" % (n, f1 * 100))

        if cfg.prediction_results_file:
            self._save_predictions(cfg.prediction_results_file, all_results)

        return em

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: ShardedDataIterator,
        global_step: int,
    ):
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step

        self.reader.train()
        epoch_batches = train_data_iterator.max_iterations

        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):

            data_iteration = train_data_iterator.get_iteration()

            # enables to resume to exactly same train state
            if cfg.fully_resumable:
                np.random.seed(cfg.seed + global_step)
                torch.manual_seed(cfg.seed + global_step)
                if cfg.n_gpu > 0:
                    torch.cuda.manual_seed_all(cfg.seed + global_step)

            input = create_reader_input(
                self.wiki_data,
                self.tensorizer,
                samples_batch,
                cfg.passages_per_question,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=True,
                shuffle=True,
            )

            loss = self._calc_loss(input)
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            max_grad_norm = cfg.train.max_grad_norm
            if cfg.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), max_grad_norm
                    )
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.reader.parameters(), max_grad_norm
                    )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.reader.zero_grad()
                global_step += 1

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, global_step=%d, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    global_step,
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    rolling_train_loss / rolling_loss_step,
                )
                rolling_train_loss = 0.0

            # Also log the first batch if resuming from a checkpoint to see
            # if the loss value is normal
            if self.model_file is not None and i == 0:
                logger.info(
                    f"Avg. loss per last 1 batches: {rolling_train_loss}",
                )

            if global_step % eval_step == 0:
                logger.info(
                    "Validation: Epoch: %d Step: %d/%d",
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.reader.train()

            if self.debugging and i == 5:
                logger.info(
                    "Reached 5 iterations when debugging mode is on. "
                    "Early stopping..."
                )
                break

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        return global_step

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.reader)
        cp = os.path.join(
            cfg.output_dir,
            cfg.checkpoint_file_name
            + "."
            + str(epoch)
            + ("." + str(offset) if offset > 0 else ""),
        )

        meta_params = get_encoder_params_state_from_cfg(cfg)

        state = CheckpointState(
            model_to_save.state_dict(),
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
        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info("Loading model weights from saved state ...")
            model_to_load.load_state(saved_state)

        if resume:
            # Set training statistics (epoch and batch where it is left off)
            epoch = saved_state.epoch
            offset = saved_state.offset
            if offset == 0:  # epoch has been completed
                epoch += 1
            logger.info(
                f"Loading checkpoint @ batch={offset} and epoch={epoch}"
            )
            self.start_epoch = epoch
            self.start_batch = offset

            # Load optimizer state
            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            # Get scheduler state
            self.scheduler_state = saved_state.scheduler_dict

    def _calc_loss(self, input: ReaderBatch) -> torch.Tensor:
        cfg = self.cfg
        input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

        loss = self.reader(
            input.input_ids,
            attn_mask,
            input.start_positions,
            input.end_positions,
            input.answers_mask,
        )

        if cfg.n_gpu > 1:
            loss = loss.mean()
        if cfg.train.gradient_accumulation_steps > 1:
            loss = loss / cfg.train.gradient_accumulation_steps

        return loss

    def _save_predictions(
        self, out_file: str, prediction_results: List[ReaderQuestionPredictions]
    ):
        logger.info("Saving prediction results to  %s", out_file)
        with open(out_file, "w", encoding="utf-8") as output:
            save_results = []
            for r in prediction_results:
                save_results.append(
                    {
                        "question": r.id,
                        "gold_answers": r.gold_answers,
                        "predictions": [
                            {
                                "top_k": top_k,
                                "prediction": {
                                    "text": span_pred.prediction_text,
                                    "score": span_pred.span_score,
                                    "relevance_score": \
                                        span_pred.relevance_score,
                                    "passage_idx": span_pred.passage_index,
                                    "passage": self.tensorizer.to_string(
                                        span_pred.passage_token_ids
                                    ),
                                },
                            }
                            for top_k, span_pred in r.predictions.items()
                        ],
                    }
                )
            output.write(json.dumps(save_results, indent=4) + "\n")


@hydra.main(config_path="conf", config_name="extractive_reader_train_cfg")
def main(cfg: DictConfig):

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)
    # For now only work with single-GPU and DDP mode
    get_gpu_info(rank=cfg.local_rank)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        # Save config
        with open("config.yaml", "w") as fout:
            yaml.dump(eval(str(cfg)), fout)

    trainer = ReaderTrainer(cfg)

    if cfg.train_files is not None:
        trainer.run_train()
    elif cfg.dev_files:
        logger.info("No train files are specified. Run validation.")
        trainer.validate()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are "
            "specified. Nothing to do."
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

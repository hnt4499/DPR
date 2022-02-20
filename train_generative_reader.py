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

from omegaconf import DictConfig, OmegaConf
from typing import List

from dpr.data.qa_validation import exact_match_score, f1_score
from dpr.data.data_types import (
    GenerativeReaderBatch,
    GenerativeReaderSample,
)
from dpr.data.general_data import TokenizedWikipediaPassages
from dpr.data.reader_data import (
    GenerativeReaderGeneralDataset,
)
from dpr.models import init_generative_reader_components
from dpr.models.generative_readers.fid_base import FiDT5
from dpr.models.generative_readers.generative_reader import create_generative_reader_input
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_generative_reader_params_state_from_cfg,
    set_generative_reader_cfg_params_from_state,
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
    get_model_file,
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

        model_file = get_model_file(self.cfg, self.cfg.checkpoint_file_name)
        if model_file is None:
            saved_state = None
        else:
            saved_state = load_states_from_checkpoint(model_file)
            set_generative_reader_cfg_params_from_state(saved_state.encoder_params, cfg)

        gradient_checkpointing = self.cfg.gradient_checkpointing
        tensorizer, reader, optimizer = init_generative_reader_components(
            cfg.encoder.encoder_model_type,
            cfg,
            num_passages=cfg.passages_per_question,
            device=cfg.device,
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
        self.reader: FiDT5 = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self._load_saved_state(saved_state)

        self.debugging = self.cfg.debugging
        self.wiki_data = None
        self.dev_iterator = None
        self.best_validation_result = None
        self.best_cp_name = None

    def _load_saved_state(self, saved_state: CheckpointState):
        if saved_state is None:
            self.scheduler_state = None
            self.start_epoch = 0
            self.start_batch = 0
            return

        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info(
            f"Loading checkpoint @ epoch={epoch} and local iteration={offset}"
        )
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load: FiDT5 = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info("Loading model weights from saved state ...")
            model_to_load.load_state(saved_state.model_dict)

        if saved_state.optimizer_dict:
            logger.info("Loading saved optimizer state ...")
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict

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
            self.wiki_data = TokenizedWikipediaPassages(data_file=self.cfg.wiki_psgs_tokenized)

        dataset = GenerativeReaderGeneralDataset(
            file=path,
            shuffle_positives=is_train,
            debugging=self.debugging,
            iterator_class=iterator_class,
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
            )
        else:
            raise NotImplementedError

        # apply deserialization hook to save memory
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

        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = cfg.train.warmup_steps

        # Model state is loaded from a checkpoint
        if self.scheduler_state:
            shift = int(self.scheduler_state["last_epoch"])
            logger.info(
                f"Loading scheduler state {self.scheduler_state} with step "
                f"shift={shift}"
            )
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        # New model
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

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
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

        return

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # in distributed DDP mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]
        reader_validation_score = self.validate()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if reader_validation_score > (self.best_validation_result or 0):
                self.best_validation_result = reader_validation_score
                self.best_cp_name = cp_name
                logger.info(f"New best validation checkpoint {cp_name} with validation score "
                            f"{reader_validation_score:.2f}")

    def validate(self):
        logger.info("Validation ...")
        cfg = self.cfg
        self.reader.eval()

        reader_model: FiDT5 = get_model_obj(self.reader)
        reader_model.set_num_passages(cfg.passages_per_question_predict)

        if self.dev_iterator is None:
            self.dev_iterator = self.get_data_iterator(
                cfg.dev_files,
                cfg.dev_iterator_class,
                cfg.train.dev_batch_size,
                is_train=False,
                shuffle=False,
            )

        log_result_step = cfg.train.log_batch_step // 4  # validation needs to be more verbose
        all_questions = []
        all_gold_answers = []
        all_predicted_answers = []

        for i, samples_batch in enumerate(self.dev_iterator.iterate_ds_data()):
            input = create_generative_reader_input(
                self.wiki_data,
                self.tensorizer,
                samples_batch,
                cfg.passages_per_question_predict,
                cfg.encoder.context_max_length,
                is_train=False,
                shuffle=False,
            )

            input = GenerativeReaderBatch(**move_to_device(input._asdict(), cfg.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            outputs = reader_model.generate(
                input_ids=input.input_ids,
                attention_mask=attn_mask,
                max_length=cfg.encoder.answer_max_length,
            )

            for out, sample in zip(outputs, samples_batch):
                sample: GenerativeReaderSample

                predicted_answer = self.tensorizer.decode(out)
                all_predicted_answers.append(predicted_answer)

                orig_question = self.tensorizer.decode(
                    sample.question_token_ids,
                    skip_special_tokens=True,
                )
                all_questions.append(orig_question)
                all_gold_answers.append(sample.answers)

            if i % log_result_step == 0:
                logger.info(
                    f"Eval step: {i}, question: {all_questions[-1]}, gold answers: "
                    f"{all_gold_answers[-1]}, generated answer: "
                    f"{all_predicted_answers[-1]} "
                )

            if self.debugging and i == 5:
                logger.info(
                    "Reached 10 iterations when debugging mode is on. "
                    "Early stopping..."
                )
                break

        ems = []
        f1s = []

        for gold_answers, predicted_answer in \
                zip(all_gold_answers, all_predicted_answers):
            # Exact match
            em_hit = max(
                [
                    exact_match_score(predicted_answer, ga)
                    for ga in gold_answers
                ]
            )
            ems.append(em_hit)

            # F1 score
            f1_hit = max(
                [
                    f1_score(predicted_answer, ga)
                    for ga in gold_answers
                ]
            )
            f1s.append(f1_hit)

        # Sync between GPUs
        ems, f1s = gather(self.cfg, [ems, f1s])

        em = np.mean(sum(ems, []))
        logger.info(f"EM {em * 100:.2f}")

        f1s = np.mean(sum(f1s, []))
        logger.info(f"F1 {f1s * 100:.2f}")

        if cfg.local_rank in [-1, 0] and cfg.prediction_results_file:
            self._save_predictions(
                cfg.prediction_results_file,
                all_questions,
                all_gold_answers,
                all_predicted_answers,
            )

        reader_model.set_num_passages(cfg.passages_per_question)
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

            input = create_generative_reader_input(
                self.wiki_data,
                self.tensorizer,
                samples_batch,
                cfg.passages_per_question,
                cfg.encoder.context_max_length,
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
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

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
                    "Reached 10 iterations when debugging mode is on. "
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

        meta_params = get_generative_reader_params_state_from_cfg(cfg)

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

    def _calc_loss(self, input: GenerativeReaderBatch) -> torch.Tensor:
        assert self.reader.training

        cfg = self.cfg
        input = GenerativeReaderBatch(**move_to_device(input._asdict(), cfg.device))
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)
        loss = self.reader(
            input_ids=input.input_ids,
            attention_mask=attn_mask,
            labels=input.answer_ids,
        )["loss"]

        if cfg.n_gpu > 1:
            loss = loss.mean()
        if cfg.train.gradient_accumulation_steps > 1:
            loss = loss / cfg.train.gradient_accumulation_steps

        return loss

    def _save_predictions(
        self,
        out_file: str,
        all_questions: List[str],
        all_gold_answers: List[List[str]],
        all_predicted_answers: List[str],
    ):
        logger.info("Saving prediction results to  %s", out_file)
        with open(out_file, "w", encoding="utf-8") as output:
            save_results = []
            for question, gold_answers, predicted_answer in \
                    zip(all_questions, all_gold_answers, all_predicted_answers):
                save_results.append(
                    {
                        "question": question,
                        "gold_answers": gold_answers,
                        "prediction": predicted_answer,
                    }
                )
            output.write(json.dumps(save_results, indent=4) + "\n")


@hydra.main(config_path="conf", config_name="generative_reader_train_cfg")
def main(cfg: DictConfig):

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)
    get_gpu_info(rank=cfg.local_rank)  # for now only work with single-GPU and DDP mode

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
            "Neither train_file or (model_file & dev_file) parameters are specified. "
            "Nothing to do."
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

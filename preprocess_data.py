#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import sys
import yaml

import hydra
import logging
import os

from omegaconf import DictConfig


from dpr.data.general_data import GeneralDataset, TokenizedWikipediaPassages
from dpr.models import init_tenzorizer
from dpr.options import setup_logger


logger = logging.getLogger()
setup_logger(logger)


class PreProcessor(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        tensorizer = init_tenzorizer(
            cfg.encoder.encoder_model_type, cfg,
        )

        self.tensorizer = tensorizer
        self.debugging = getattr(self.cfg, "debugging", False)
        self.wiki_data = None
        self.dev_iterator = None

    def get_data_iterator(
        self,
        path: str,
        is_train: bool,
    ) -> None:

        # Original, raw gold passages
        gold_passages_src = self.cfg.gold_passages_src
        if gold_passages_src:
            if not is_train:
                gold_passages_src = self.cfg.gold_passages_src_dev

        # Processed, 100-word split gold passages
        gold_passages_processed = (self.cfg.gold_passages_processed if is_train
                                   else self.cfg.gold_passages_processed_dev)

        if self.wiki_data is None:
            self.wiki_data = TokenizedWikipediaPassages(data_file=self.cfg.wiki_psgs_tokenized)

        bm25_retrieval_results = self.cfg.bm25_retrieval_results if is_train else None
        dataset = GeneralDataset(
            files=path,
            bm25_retrieval_file=bm25_retrieval_results,
            wiki_data=self.wiki_data,
            is_train=is_train,
            gold_passages_src=gold_passages_src,
            gold_passages_processed=gold_passages_processed,
            tensorizer=self.tensorizer,
            run_preprocessing=True,
            num_workers=self.cfg.num_workers,
            debugging=self.debugging,
            load_data=False,
            check_pre_tokenized_data=self.cfg.check_pre_tokenized_data
        )

        dataset.load_data()

    def load_data(self):
        # Load train data
        self.get_data_iterator(self.cfg.train_files, is_train=True)

        # Temporarily remove Wikipedia passages to avoid too much data being transferred in multiprocessing
        self.wiki_data.remove_data()

        # Load dev data
        self.get_data_iterator(self.cfg.dev_files, is_train=False)


@hydra.main(config_path="conf", config_name="preprocess_data")
def main(cfg: DictConfig):

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config
    with open("config.yaml", "w") as fout:
        yaml.dump(eval(str(cfg)), fout)

    processor = PreProcessor(cfg)
    processor.load_data()


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()

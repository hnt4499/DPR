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
import torch
torch.autograd.set_detect_anomaly(True)

from omegaconf import DictConfig


from dpr.data.general_data import TokenizedWikipediaPassages
from dpr.data.reader_data import ExtractiveReaderDataset
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

        gold_passages_src = self.cfg.gold_passages_src
        if gold_passages_src:
            if not is_train:
                gold_passages_src = self.cfg.gold_passages_src_dev

            assert os.path.exists(
                gold_passages_src
            ), "Please specify valid gold_passages_src/gold_passages_src_dev"

        if self.wiki_data is None:
            self.wiki_data = TokenizedWikipediaPassages(data_file=self.cfg.wiki_psgs_tokenized)

        bm25_retrieval_results = self.cfg.bm25_retrieval_results if is_train else None
        dataset = ExtractiveReaderDataset(
            path,
            bm25_retrieval_results,
            self.wiki_data,
            is_train,
            gold_passages_src,
            self.tensorizer,
            True,
            self.cfg.num_workers,
            debugging=self.debugging,
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
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()

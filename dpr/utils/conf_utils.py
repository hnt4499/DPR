import logging

import hydra
from omegaconf import DictConfig

from dpr.data.biencoder_data import GeneralDatasetScheme
from dpr.data.general_data import TokenizedWikipediaPassages


logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        datasets = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        if self.train_datasets_names:
            self.train_datasets = [
                hydra.utils.instantiate(datasets[ds_name])
                for ds_name in self.train_datasets_names
            ]
        else:
            self.train_datasets = []
        if cfg.dev_datasets:
            self.dev_datasets_names = cfg.dev_datasets
            logger.info("dev_datasets: %s", self.dev_datasets_names)
            self.dev_datasets = [
                hydra.utils.instantiate(datasets[ds_name])
                for ds_name in self.dev_datasets_names
            ]
        self.sampling_rates = cfg.train_sampling_rates

        # If any of the dataset is of general dataset scheme, we need to initialize
        # Wikipedia passages container
        all_datasets = self.train_datasets + self.dev_datasets
        if any(isinstance(dataset, GeneralDatasetScheme) for dataset in all_datasets):
            self.wiki_data = TokenizedWikipediaPassages(data_file=cfg.wiki_data)

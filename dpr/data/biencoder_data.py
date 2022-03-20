import os
import glob
import logging
from typing import List

import torch

from dpr.utils.data_utils import read_data_from_json_files
from dpr.data.data_types import BiEncoderPassage, BiEncoderSample


logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        shuffle_positives: bool = False,
        encoder_type: str = None,
    ):
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        return query


def get_dpr_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    else:
        # try to use data downloader
        from dpr.data.download_data import download

        return download(source_name)


class JsonQADataset(Dataset):
    def __init__(
        self,
        file: str,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
    ):
        super().__init__(
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
        )
        self.file = file
        self.data_files = []
        self.data = []
        self.normalize = normalize

        logger.info("Data files: %s", self.file)

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs =  json_sample["negative_ctxs"] \
            if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize \
                    else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [
            create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question

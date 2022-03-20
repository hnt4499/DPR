import csv
import logging
import collections
from typing import Dict

import torch

from dpr.data.data_types import BiEncoderPassage
from dpr.data.biencoder_data import (
    normalize_passage,
    normalize_question,
    get_dpr_files,
)


logger = logging.getLogger(__name__)
QASample = collections.namedtuple("QuerySample", ["query", "id", "answers"])


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined
        in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        if len(self.data_files) != 1:
            raise RuntimeError(
                f"RetrieverData source currently works with single files only. "
                f"Files specified: {self.data_files}"
            )
        self.file = self.data_files[0]


class CsvQASrc(RetrieverData):
    """
    Dataset used for retrieval (see `dense_retriever.py`).
    """
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
    ):
        super().__init__(file)
        self.data = None
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        return question

    def load_data(self):
        super().load_data()
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(
                    QASample(self._process_question(question), id, answers)
                )
        self.data = data


class CsvCtxSrc(RetrieverData):
    """
    Dataset used for reading Wikipedia context source (e.g.,
    `data.wikipedia_split.psgs_w100`)
    """
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])

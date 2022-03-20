"""
A general set of data utilities for both biencoder and reader models.
"""


import glob
import logging
from typing import Tuple, Union, Iterator

from dpr.data.biencoder_data import Dataset
from dpr.data.general_data_preprocess import (
    TokenizedWikipediaPassages,
    GeneralDatasetPreprocessor,
)
from dpr.utils.data_utils import Tensorizer
from dpr.data.data_types import (
    # Legacy
    DataPassage,
    # Retriever
    BiEncoderPassageTokenized,
    BiEncoderSampleTokenized,
    # Reader
    ReaderSample,
)


logger = logging.getLogger(__name__)



class GeneralDataset(Dataset):
    # Set to False to disable data post processing; a workaround in
    # `dpr.data.data_utils.ShardedDataStreamIterator` for resumability
    _data_post_processing = True

    def __init__(
        self,
        mode: str,
        file: str,
        shuffle_positives: bool = False,
        encoder_type: str = None,
        only_gold: bool = False,
        debugging: bool =  False,
        iterator_class: str = "ShardedDataIterator",
        compress: bool = False,
    ):
        """One-for-all dataset using general dataset scheme. This dataset can
        be used for retriever (by setting `mode=="retriever"`), reader
        (by setting `mode=="reader"`) for both (by setting `mode=="both"`).
        For now this data is implemented under `biencoder_data` for some
        backward compatibility reasons.

        :param file: either path to a single dataset file (*.json) or a glob
            pattern to preprocessed pickle (*.pkl) files.
        :param only_gold: whether to keep only samples whose gold passage is
            available. Useful for retriever dev set, since previously all
            retriever data have gold passages. Data discrepancy could result
            in wrong selection of the best model during evaluation.
        :else: see `dpr.data.GeneralDataset`.
        """
        super(GeneralDataset, self).__init__(
            shuffle_positives=shuffle_positives,
            encoder_type=encoder_type,
        )

        # TODO: try normalizing questions and passages
        assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"
        self.mode = mode
        self.normalize = False
        self.only_gold = only_gold

        # Data should already be pre-processed
        pickle_files = glob.glob(
            file.replace(".json", "") +
            (".preprocessed.*.json" if compress else ".*.pkl")
        )
        assert len(pickle_files) > 0, "Data should be already processed"

        # Initialize general dataset
        self.dataset = GeneralDatasetPreprocessor(
            files=file,
            bm25_retrieval_file=None,
            wiki_data=None,
            is_train=None,
            gold_passages_src=None,
            gold_passages_processed=None,
            tensorizer=None,
            run_preprocessing=True,
            num_workers=None,
            debugging=debugging,
            load_data=True,
            iterator_class=iterator_class,
            compress=compress,
        )

    def load_data(
        self,
        wiki_data: TokenizedWikipediaPassages,
        tensorizer: Tensorizer,
    ):
        self.wiki_data = wiki_data
        self.wiki_data.load_data()
        self.dataset.load_data()

        # Remove those whose gold passage info is not available
        if self.only_gold:
            orig_len = len(self.dataset)
            logger.info(
                "Removing samples whose gold passage info is not available."
            )
            self.dataset.data = [
                sample for sample in self.dataset.data
                if len(sample.gold_passages) > 0
            ]
            logger.info(
                f"Number of samples: before filtering: {orig_len}, after "
                f"filtering: {len(self.dataset)}"
            )

    def _process_query(self, query: str):
        # We don't use this function
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def _get_item_from_reader_sample(
        self,
        reader_sample: ReaderSample,
    ) -> Union[
        BiEncoderSampleTokenized,  # `mode=="retriever"`
        ReaderSample,  # `mode=="reader`; `ReaderSample` is `DataSample`
        Tuple[BiEncoderSampleTokenized, ReaderSample],  # `mode=="both"`
    ]:
        # Reader sample is without any further pre-processing
        if self.mode == "reader" or not GeneralDataset._data_post_processing:
            return reader_sample

        # Retriever sample needs further pre-processing for backward
        # compatibility
        retriever_sample = BiEncoderSampleTokenized()
        retriever_sample.query_ids = reader_sample.question_token_ids

        positive_ctxs = reader_sample.gold_passages + \
            reader_sample.positive_passages

        # TODO: allow other kinds of positives and negatives, such as distantly
        # positives
        hard_negative_ctxs = reader_sample.negative_passages
        bm25_negative_ctxs = reader_sample.bm25_negative_passages

        def create_passage(ctx: DataPassage):
            # Load passage tokens and title tokens first
            tokens = self.wiki_data.get_tokenized_data(int(ctx.id))
            ctx.load_tokens(**tokens)

            return BiEncoderPassageTokenized(
                id=ctx.id,
                is_gold=ctx.is_gold,
                text_ids=ctx.passage_token_ids,
                title_ids=ctx.title_token_ids,
            )

        retriever_sample.positive_passages = [
            create_passage(ctx) for ctx in positive_ctxs]
        retriever_sample.hard_negative_passages = [
            create_passage(ctx) for ctx in hard_negative_ctxs]
        retriever_sample.bm25_negative_passages = [
            create_passage(ctx) for ctx in bm25_negative_ctxs]

        if self.mode == "retriever":
            return retriever_sample
        return retriever_sample, reader_sample

    def __getitem__(self, index) -> Union[
        BiEncoderSampleTokenized,  # `mode=="retriever"`
        ReaderSample,  # `mode=="reader`; `ReaderSample` is `DataSample`
        Tuple[BiEncoderSampleTokenized, ReaderSample]  # `mode=="both"`
    ]:
        return self._get_item_from_reader_sample(self.dataset[index])

    def __iter__(self) -> Iterator[Union[
        BiEncoderSampleTokenized,  # `mode=="retriever"`
        ReaderSample,  # `mode=="reader`; `ReaderSample` is `DataSample`
        Tuple[BiEncoderSampleTokenized, ReaderSample]  # `mode=="both"`
    ]]:
        for reader_sample in self.dataset:
            yield self._get_item_from_reader_sample(reader_sample)


class BiEncoderGeneralDataset(GeneralDataset):
    def __init__(self, **kwargs):
        super(BiEncoderGeneralDataset, self).__init__(
            mode="retriever",
            **kwargs,
        )


class ExtractiveReaderGeneralDataset(GeneralDataset):
    def __init__(self, **kwargs):
        super(ExtractiveReaderGeneralDataset, self).__init__(
            mode="reader",
            **kwargs,
        )


class GenerativeReaderGeneralDataset(GeneralDataset):
    def __init__(self, **kwargs):
        super(GenerativeReaderGeneralDataset, self).__init__(
            mode="reader",
            **kwargs,
        )

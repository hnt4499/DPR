"""
A general set of data utilities for both biencoder and reader models.
"""

import os
import json
import glob
import math
import pickle
import random
import logging
import collections
import multiprocessing
from functools import partial
from typing import List, Tuple, Dict, Optional, Iterable

import torch
from tqdm import tqdm
import numpy as np

from dpr.data.data_types import DataPassage, DataSample
from dpr.utils.data_utils import (
    Tensorizer,
    read_serialized_data_from_files,
    DataPassageCompressor,
)
from dpr.data.answers_processing import get_expanded_answer
from dpr.utils.dist_utils import is_main_process, broadcast_object
from dpr.utils.misc_utils import count_lines_text_file

logger = logging.getLogger()


class TokenizedWikipediaPassages(object):
    """
    Container for all tokenized wikipedia passages.
    """
    def __init__(self, data_file):
        self.data_files = glob.glob(data_file)
        logger.info(f"Data files: {self.data_files}")

        # Read data
        self.data = None
        self.id2idx = None

    def load_data(self):
        if self.data is None:
            logger.info("Reading **entire** tokenized passages into memory.")
            self.data = []
            self.id2idx = {}  # index mapping

            for shard_id, data_file in enumerate(self.data_files):
                logger.info(f"Reading file {data_file}")
                with open(data_file, "rb") as fin:
                    shard_data = pickle.load(fin)
                    self.data.append(shard_data)

                    # Index mapping
                    for idx, passage_id in enumerate(shard_data["id"]):
                        assert passage_id not in self.id2idx
                        self.id2idx[passage_id] = (shard_id, idx)

            logger.info(f"Done. Number of passages: {sum(len(shard_data['id']) for shard_data in self.data)}")

    def remove_data(self):
        self.data = None

    def get_tokenized_data(self, id):
        if self.data is None:
            self.load_data()

        shard_id, idx = self.id2idx[id]
        shard_data = self.data[shard_id]
        # Retrieve data
        text: np.ndarray = shard_data["text"][idx]
        text_length = shard_data["text_length"][idx]
        text = text[:text_length]  # remove padding

        title: np.ndarray = shard_data["title"][idx]
        title_length = shard_data["title_length"][idx]
        title = title[:title_length]  # remove padding

        return {"passage_token_ids": text, "title_token_ids": title}


class GeneralDataset(torch.utils.data.Dataset):
    """
    General-purpose dataset for both retriever (biencoder) and reader. Input data is expected to be output of a
    trained retriever. This serves as a preprocessor as well as a container, to be wrapper by other classes.
    """
    def __init__(
        self,
        files: str,
        bm25_retrieval_file: str,
        wiki_data: TokenizedWikipediaPassages,
        is_train: bool,
        gold_passages_src: str,
        gold_passages_processed: str,
        tensorizer: Tensorizer,
        run_preprocessing: bool,
        num_workers: int,
        debugging: bool = False,
        load_data: bool = True,
        check_pre_tokenized_data: bool = True,
        cfg: "PreprocessingCfg" = None,
        compress: bool = False,
        iterator_class: str = "ShardedDataIterator",
    ):
        """Initialize general dataset.

        :param mode: one of ["retriever", "reader"]. Dataset mode.
        :param files: either path to a single dataset file (*.json) or a glob pattern to preprocessed pickle (*.pkl) files.
        :param bm25_retrieval_file: path to the pre-processed BM25 retrieval results.
        :param wiki_data: pre-tokenized wikipedia passages.
        :param is_train: whether this dataset is training set or evaluation set. Different preprocess settings will be
            applied to different types of training sets.
        :param gold_passages_src: path to the gold passage file.
        :param tensorizer: initialized tensorizer.
        :param run_preprocessing: whether to perform preprocessing. Useful for DDP mode, where only one process should do
            preprocessing while other processes wait for its results.
        :param num_workers: number of workers for the preprocessing step.
        :param load_data: whether to load pre-processes data into memory. Disable this if you want to pre-process data only
        :param check_pre_tokenized_data: whether to check if the pre-tokenized context data is the same as original
            context passage after tokenized.
        :param cfg: configs to overwrite default preprocessing configs. Can be partially set (i.e., some key-value pairs
            may be set and some may not.)
        :param compress: whether to compress the outputs and save it as a *.json file.
        :param iterator_class: data iterator class. Useful when reading a stream of data.
        """
        self.files = files
        self.bm25_retrieval_file = bm25_retrieval_file
        self.wiki_data = wiki_data
        self.data: List[DataSample] = None
        self.is_train = is_train
        self.gold_passages_src = gold_passages_src
        self.gold_passages_processed = gold_passages_processed
        self.tensorizer = tensorizer
        self.run_preprocessing = run_preprocessing
        self.num_workers = num_workers
        self.debugging = debugging
        self.load_data_ = load_data
        self.check_pre_tokenized_data = check_pre_tokenized_data
        self.cfg = cfg
        self.compress = compress

        # Iterator class
        allowed_iterator_class = ["ShardedDataIterator", "ShardedDataStreamIterator"]
        if iterator_class not in allowed_iterator_class:
            raise ValueError(
                f"Invalid iterator class: {iterator_class}. Allowed classes are "
                f"{allowed_iterator_class}"
            )
        self.iterator_class = iterator_class

    def __getitem__(self, index: int) -> DataSample:
        if self.iterator_class != "ShardedDataIterator":
            raise NotImplementedError
        return self.data[index]

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        """
        Logic for compressed data files for memory efficiency. To be used
        with `ShardedDataStreamIterator`.
        """
        if self.iterator_class != "ShardedDataStreamIterator":
            raise NotImplementedError

        # Hard code logic: always shuffle data files for randomness
        if is_main_process():
            indices = list(range(len(self.preprocessed_data_files)))
            random.shuffle(indices)
            broadcast_object(indices)
        else:
            indices = broadcast_object()
        preprocessed_data_files = [self.preprocessed_data_files[i] for i in indices]
        num_samples_per_file = [self.num_samples_per_file[i] for i in indices]

        # Read data
        for path, num_samples in zip(preprocessed_data_files, num_samples_per_file):
            logger.info(
                f"[{self.__class__.__name__}] Start reading compressed data from {path} consisting of "
                f"{num_samples} lines."
            )
            with open(path, "r") as fin:
                for line in fin:
                    datapoint = json.loads(line.strip())
                    yield DataPassageCompressor.convert_from_json(datapoint)

    def load_data(self):
        if self.data is None:
            data_files = glob.glob(self.files)
            if len(data_files) == 0:
                raise RuntimeError("No data files found")
            preprocessed_data_files = self._get_preprocessed_files(data_files)

            if self.load_data_:
                if self.debugging:
                    logger.info("Debugging mode is on. Restricting to at most 2 data files.")
                    preprocessed_data_files = preprocessed_data_files[:2]

                logger.info(f"Reading data files: {preprocessed_data_files}")

                # Compressed
                if self.iterator_class == "ShardedDataIterator":
                    self.data = read_serialized_data_from_files(preprocessed_data_files)
                    self.num_samples = len(self.data)
                elif self.iterator_class == "ShardedDataStreamIterator":
                    self.preprocessed_data_files = preprocessed_data_files
                    self.num_samples_per_file = [
                        count_lines_text_file(path) for path in preprocessed_data_files
                    ]
                    self.num_samples = sum(self.num_samples_per_file)
                else:
                    raise NotImplementedError

    def _get_preprocessed_files(
        self,
        data_files: List,
    ):
        """
        Get preprocessed data files, if exist, or generate new ones.
        """

        serialized_files = [file for file in data_files if file.endswith(".pkl")]
        if len(serialized_files) > 0:
            return serialized_files
        assert len(data_files) == 1, "Only 1 source file pre-processing is supported."

        # Data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace(".json", "")
            out_file_prefix = os.path.join(dir_path, base_name)

            files = glob.glob(out_file_prefix + "*.pkl")
            files_2 = glob.glob(out_file_prefix + ".preprocessed.*.json")  # for compressed data
            return files + files_2, out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if len(serialized_files) > 0:
            logger.info("Found preprocessed files. %s", serialized_files)
            return serialized_files

        # Preprocess data files as needed
        logger.info(
            "Data are not preprocessed for reader training. Start pre-processing ..."
        )

        # Start pre-processing and save results
        if self.run_preprocessing:
            # Temporarily disable auto-padding to save disk space usage of serialized files
            self.tensorizer.set_pad_to_max(False)
            serialized_files = preprocess_retriever_results(
                self.is_train,
                data_files[0],
                self.bm25_retrieval_file,
                self.wiki_data,
                out_file_prefix,
                self.gold_passages_src,
                self.gold_passages_processed,
                self.tensorizer,
                self.check_pre_tokenized_data,
                num_workers=self.num_workers,
                debugging=self.debugging,
                cfg=self.cfg,
                compress=self.compress,
            )
            self.tensorizer.set_pad_to_max(True)

            # Sync with other processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            # Sync with other processes
            torch.distributed.barrier()
            serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        return serialized_files


class GeneralDatasetScheme(object):
    """General dataset marker."""
    def load_data(self, wiki_data: TokenizedWikipediaPassages, tensorizer: Tensorizer):
        raise NotImplementedError


def preprocess_retriever_results(
    is_train_set: bool,
    input_file: str,
    bm25_retrieval_file: str,
    wiki_data: TokenizedWikipediaPassages,
    out_file_prefix: str,
    gold_passages_file: str,
    gold_passages_processed_file: str,
    tensorizer: Tensorizer,
    check_pre_tokenized_data: bool = True,
    num_workers: int = 8,
    double_check: bool = True,
    debugging: bool = False,
    cfg: "PreprocessingCfg" = None,
    compress: bool = False,
) -> List[str]:
    """
    Preprocess the dense retriever outputs (or any compatible file format) into the general retriever/reader input data and
    serializes them into a set of files.
    Conversion splits the input data into multiple chunks and processes them in parallel. Each chunk results are stored
    in a separate file with name out_file_prefix.{number}.pkl
    :param is_train_set: if the data should be processed for a train set (with applicable settings)
    :param input_file: path to a json file with data to convert
    :param bm25_retrieval_file: path to the pre-processed BM25 retrieval results
    :param wiki_data: pre-tokenized wikipedia passages.
    :param out_file_prefix: output path prefix.
    :param gold_passages_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param gold_passages_processed_file: path to the preprocessed gold passages pickle file. Unlike `gold_passages_file` which
        contains original gold passages, this file should contain processed, matched, 100-word split passages that match
        with the original gold passages.
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param check_pre_tokenized_data: whether to check if the pre-tokenized context data is the same as original
        context passage after tokenized.
    :param num_workers: the number of parallel processes for conversion
    :param double_check: double check whether the pre-tokenized tokens are correct
    :param cfg: configs to overwrite default preprocessing configs.
    :param compress: whether to compress the outputs and save it as a *.json file.
    :return: path to serialized, preprocessed pickle files
    """
    # Load dense retriever results
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.loads("".join(f.readlines()))
        num_samples_orig = len(samples)
        if debugging:
            samples = samples[:1000]
    logger.info(
        "Loaded %d questions + retrieval results from %s", len(samples), input_file
    )

    # Load BM25 (sparse) retriever results
    if bm25_retrieval_file is not None:
        with open(bm25_retrieval_file, "rb") as f:
            bm25_samples = pickle.load(f)
            assert len(bm25_samples) == num_samples_orig
            if debugging:
                bm25_samples = bm25_samples[:1000]
        logger.info(
        "Loaded %d BM25 retrieval results from %s", len(bm25_samples), bm25_retrieval_file
    )
    else:
        bm25_samples = [tuple() for _ in range(len(samples))]

    ds_size = len(samples)
    step = max(math.ceil(ds_size / num_workers), 1)
    chunks = [
        (j, samples[i : i + step], bm25_samples[i: i + step])
        for j, i in enumerate(range(0, ds_size, step))
    ]
    logger.info("Split data into %d chunks", len(chunks))

    # We only keep first 1000 passage texts of each chunk for double checking
    logger.info("Releasing memory...")
    c = 1000 if double_check else 0
    for _, samples, _ in chunks:
        if c >= len(samples):
            continue
        for sample in samples[c:]:
            for ctx in sample["ctxs"]:
                del ctx["title"]
                del ctx["text"]

    logger.info(f"Initializing {num_workers} workers")
    workers = multiprocessing.Pool(num_workers)

    processed = 0
    _parse_batch = partial(
        _preprocess_samples_by_chunk,
        wiki_data=wiki_data,
        out_file_prefix=out_file_prefix,
        gold_passages_file=gold_passages_file,
        gold_passages_processed_file=gold_passages_processed_file,
        tensorizer=tensorizer,
        is_train_set=is_train_set,
        check_pre_tokenized_data=check_pre_tokenized_data,
        cfg=cfg,
        compress=compress,
    )
    serialized_files = []
    for file_name in workers.map(_parse_batch, chunks):
        processed += 1
        serialized_files.append(file_name)
        logger.info("Chunks processed %d, data saved to %s", processed, file_name)
    logger.info("Preprocessed data stored in %s", serialized_files)

    return serialized_files


def _preprocess_samples_by_chunk(
    samples: Tuple,
    wiki_data: TokenizedWikipediaPassages,
    out_file_prefix: str,
    gold_passages_file: str,
    gold_passages_processed_file: str,
    tensorizer: Tensorizer,
    is_train_set: bool,
    check_pre_tokenized_data: bool,
    cfg: "PreprocessingCfg" = None,
    compress: bool = False,
) -> str:
    chunk_id, samples, bm25_samples = samples
    logger.info("Start batch %d", len(samples))
    iterator = _preprocess_retriever_data(
        samples,
        bm25_samples,
        wiki_data,
        gold_passages_file,
        gold_passages_processed_file,
        tensorizer,
        is_train_set=is_train_set,
        check_pre_tokenized_data=check_pre_tokenized_data,
        cfg=cfg,
    )

    # Gather results
    results = []
    for r in tqdm(iterator):
        r.on_serialize()
        results.append(r)

    out_file = out_file_prefix + "." + str(chunk_id) + (".json" if compress else ".pkl")
    # Compress
    if compress:
        logger.info(f"Compressing and writing results to {out_file}")
        with open(out_file, "w") as fout:
            for results_i in tqdm(results):
                results_i = DataPassageCompressor.convert_to_json(results_i)
                fout.write(json.dumps(results_i) + "\n")
    # Old way
    else:
        with open(out_file, mode="wb") as f:
            logger.info("Serialize %d results to %s", len(results), out_file)
            pickle.dump(results, f)

    return out_file


# Configuration for reader model passage selection
PreprocessingCfg = collections.namedtuple(
    "PreprocessingCfg",
    [
        "skip_no_positives",
        "include_gold_passage",
        "gold_page_only_positives",
        "expand_answers",
        "max_bm25_positives",
        "max_bm25_negatives",
        # Previously every passage has a marker `has_answer` denoting whether it contains the
        # answer string or not (apparently generated by some distant supervision heuristic). If setting
        # `recheck_negatives` to True, passages whose `has_answer` is False are rechecked by newly implemented
        # heuristic to update its `has_answer` attribute. By this, previously-negative passages can become
        # positive (in fact, there is a substantial amount of such passages). It might be the reason the model
        # performance degrades. Hence, this is recommended to be set to False.
        "recheck_negatives",
        # Whether passages containing answer string but not from gold page should be discarded or considered
        # negatives. This option is only applicable when `gold_page_only_positives` is set to True.
        # Note that instead of discarding such passages, it will instead be stored in `distantly_positive_passages`.
        "should_negatives_contain_answer",
        # In retriever (NOT reader) data preparing step, question (but NOT passages) are normalized
        "normalize_questions",
        # Set True for generative reader (FiD), where positive or negative don't matter
        "skip_finding_answer_spans",
        # If answers cannot be found in the passage that has been marked as having answers, do lowercase for both the
        # passage and answers and try to find them again; newly added to support generative readers (e.g., FiD)
        # This happens when `dense_retrieve.py` uses uncased BERT, which marks a passage as having answer spans
        # although their cases (upper or lower cases) might not match.
        "try_lower_case_if_answers_not_found",
    ],
)

DEFAULT_PREPROCESSING_CFG_TRAIN = PreprocessingCfg(
    skip_no_positives=True,
    include_gold_passage=False,  # whether to include the gold passage itself
    gold_page_only_positives=True,  # whether positive passages should only be from the gold passages
    expand_answers=True,  # expand the set of answers; see `answers_processing.py`
    max_bm25_positives=10,
    max_bm25_negatives=30,
    recheck_negatives=False,  # see description above
    should_negatives_contain_answer=False,  # see description above
    normalize_questions=True,  # see description above
    skip_finding_answer_spans=False,
    try_lower_case_if_answers_not_found=False,
)

"""
Data for the generative reader (FiD) has been processed using
the config below. Change it to `DEFAULT_PREPROCESSING_CFG_TRAIN`
to reproduce
"""
DEFAULT_PREPROCESSING_CFG_TRAIN_GENERATIVE_READER = PreprocessingCfg(
    skip_no_positives=False,
    include_gold_passage=False,  # whether to include the gold passage itself
    gold_page_only_positives=False,  # whether positive passages should only be from the gold passages
    expand_answers=True,  # expand the set of answers; see `answers_processing.py`
    max_bm25_positives=10,
    max_bm25_negatives=30,
    recheck_negatives=False,  # see description above
    should_negatives_contain_answer=False,  # see description above
    normalize_questions=True,  # see description above
    skip_finding_answer_spans=True,
    try_lower_case_if_answers_not_found=False,
)


def _preprocess_retriever_data(
    samples: List[Dict],
    bm25_samples: List[Tuple[Tuple[int, float]]],
    wiki_data: TokenizedWikipediaPassages,
    gold_info_file: Optional[str],
    gold_info_processed_file: str,
    tensorizer: Tensorizer,
    cfg: PreprocessingCfg = None,
    is_train_set: bool = True,
    check_pre_tokenized_data: bool = True,
) -> Iterable[DataSample]:
    """
    Converts retriever results into general retriever/reader training data.
    :param samples: samples from the retriever's json file results
    :param bm25_samples: bm25 retrieval results; list of tuples of tuples of (passage_id, score), where passages of each
        sample are already sorted by their scores
    :param gold_info_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param gold_info_processed_file: path to the preprocessed gold passages pickle file. Unlike `gold_passages_file`
        which contains original gold passages, this file should contain processed, matched, 100-word split passages that
        match with the original gold passages.
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param cfg: PreprocessingCfg object with positive and negative passage selection parameters
    :param is_train_set: if the data should be processed as a train set
    :return: iterable of DataSample objects which can be consumed by the reader model
    """
    # Overwrite default configs
    if cfg is None:
        cfg = {}
    elif isinstance(cfg, PreprocessingCfg):
        cfg = cfg._asdict()
    default_cfg = DEFAULT_PREPROCESSING_CFG_TRAIN._asdict()
    default_cfg.update(cfg)
    cfg = PreprocessingCfg(**default_cfg)

    gold_passage_map, canonical_questions = (
        _get_gold_ctx_dict(gold_info_file) if gold_info_file is not None else ({}, {})
    )
    processed_gold_passage_map = (
        _get_processed_gold_ctx_dict(gold_info_processed_file) if gold_info_processed_file is not None else {}
    )

    number_no_positive_samples = 0
    number_samples_from_gold = 0
    number_samples_with_gold = 0
    assert len(samples) == len(bm25_samples)

    for sample, bm25_sample in zip(samples, bm25_samples):
        # Refer to `_get_gold_ctx_dict` for why we need to distinguish between two types of questions
        # Here `processed_question` refer to tokenized questions, where `question` refer to
        # canonical questions.
        processed_question = sample["question"]
        if processed_question in canonical_questions:
            question = canonical_questions[processed_question]
        else:
            question = processed_question

        question_token_ids: np.ndarray = tensorizer.text_to_tensor(
            normalize_question(question) if cfg.normalize_questions else question,
            add_special_tokens=False,
        ).numpy()

        orig_answers = sample["answers"]
        if cfg.expand_answers:
            expanded_answers = [get_expanded_answer(answer) for answer in orig_answers]
        else:
            expanded_answers = []
        all_answers = orig_answers + sum(expanded_answers, [])

        passages = _select_passages(
            wiki_data,
            sample,
            bm25_sample,
            question,
            processed_question,
            question_token_ids,
            orig_answers,
            expanded_answers,
            all_answers,
            tensorizer,
            gold_passage_map,
            processed_gold_passage_map,
            cfg,
            is_train_set,
            check_pre_tokenized_data,
        )
        gold_passages = passages[0]
        positive_passages, negative_passages, distantly_positive_passages = passages[1:4]
        bm25_positive_passages, bm25_negative_passages, bm25_distantly_positive_passages = passages[4:]

        if is_train_set and len(positive_passages) == 0:
            number_no_positive_samples += 1
            if cfg.skip_no_positives:
                continue

        if any(ctx for ctx in positive_passages if ctx.is_from_gold):
            number_samples_from_gold += 1

        if len(gold_passages) > 0:
            number_samples_with_gold += 1

        yield DataSample(
            question,
            question_token_ids=question_token_ids,
            answers=all_answers,
            orig_answers=orig_answers,
            expanded_answers=expanded_answers,
            # Gold
            gold_passages=gold_passages,
            # Dense
            positive_passages=positive_passages,
            distantly_positive_passages=distantly_positive_passages,
            negative_passages=negative_passages,
            # Sparse
            bm25_positive_passages=bm25_positive_passages,
            bm25_distantly_positive_passages=bm25_distantly_positive_passages,
            bm25_negative_passages=bm25_negative_passages,
        )

    logger.info(f"Number of samples whose at least one positive passage is "
                f"from the same article as the gold passage: {number_samples_from_gold}")
    logger.info(f"Number of samples whose gold passage is available: {number_samples_with_gold}")
    logger.info(f"Number of samples with no positive passages: {number_no_positive_samples}")


def _select_passages(
    wiki_data: TokenizedWikipediaPassages,
    sample: Dict,
    bm25_sample: Tuple[Tuple[int, float]],
    question: str,
    processed_question: str,
    question_token_ids: np.ndarray,
    answers: List[str],
    expanded_answers: List[List[str]],
    all_answers: List[str],
    tensorizer: Tensorizer,
    gold_passage_map: Dict[str, DataPassage],
    processed_gold_passage_map: Dict[str, DataPassage],
    cfg: PreprocessingCfg,
    is_train_set: bool,
    check_pre_tokenized_data: bool,
) -> Tuple[
    List[DataPassage], List[DataPassage], List[DataPassage], List[DataPassage],
    List[DataPassage], List[DataPassage], List[DataPassage]
]:
    """
    Select and process valid passages for training/evaluation.
    """
    # Tokenize answers
    answers_token_ids: List[np.ndarray] = [
        tensorizer.text_to_tensor(a, add_special_tokens=False).numpy()
        for a in all_answers
    ]

    # Gold context; we want to cover more gold passages, that's why we are matching both
    # `processed_question` and `question` (canonical question).
    if question in processed_gold_passage_map or processed_question in processed_gold_passage_map:
        if question in processed_gold_passage_map:
            gold_ctx = processed_gold_passage_map[question]
        else:
            gold_ctx = processed_gold_passage_map[processed_question]

        gold_ctx = _load_tokens_into_ctx(
            gold_ctx, question_token_ids, wiki_data, tensorizer, check_pre_tokenized_data,
        )  # load question, passage title and passage tokens into the context object
        gold_ctx = _find_answer_spans(
            tensorizer, gold_ctx, question, all_answers, answers_token_ids,
            warn_if_no_answer=True, raise_if_no_answer=False,
            warn_if_has_answer=False, raise_if_has_answer=False,
            recheck_negatives=False,
            skip_finding_answer_spans=cfg.skip_finding_answer_spans,
            try_lower_case_if_answers_not_found=cfg.try_lower_case_if_answers_not_found,
        )  # find answer spans for all passages
        if gold_ctx.has_answer:
            gold_ctxs = [gold_ctx]
        else:
            gold_ctxs = []
    else:
        gold_ctxs = []

    # Densely retrieved contexts
    ctxs = [DataPassage(is_from_bm25=False, **ctx) for ctx in sample["ctxs"]]
    ctxs = [
        _load_tokens_into_ctx(ctx, question_token_ids, wiki_data, tensorizer, check_pre_tokenized_data)
        for ctx in ctxs
    ]  # load question, passage title and passage tokens into the context object
    # Find answer spans for all passages
    ctxs: List[DataPassage] = [
        _find_answer_spans(
            tensorizer, ctx, question, all_answers, answers_token_ids,
            warn_if_no_answer=ctx.has_answer,  # warn if originally it contains answer string
            warn_if_has_answer=(not ctx.has_answer),  # warn if originally it does NOT contain answer string
            recheck_negatives=cfg.recheck_negatives,
            skip_finding_answer_spans=cfg.skip_finding_answer_spans,
            try_lower_case_if_answers_not_found=cfg.try_lower_case_if_answers_not_found,
        )
        for ctx in ctxs
    ]

    # Sparsely retrieved contexts (BM25)
    bm25_ctxs = [
        DataPassage(id=passage_id, score=score, is_from_bm25=True)
        for passage_id, score in bm25_sample
    ]
    bm25_ctxs = [
        _load_tokens_into_ctx(ctx, question_token_ids, wiki_data, tensorizer, check_pre_tokenized_data)
        for ctx in bm25_ctxs
    ]  # load question, passage title and passage tokens into the context object
    # Find answer spans for all passages
    bm25_ctxs: List[DataPassage] = [
        _find_answer_spans(
            tensorizer, ctx, question, all_answers, answers_token_ids,
            warn_if_no_answer=False, warn_if_has_answer=False,
            recheck_negatives=True,  # `has_answer` of any BM25 passage is None
            skip_finding_answer_spans=cfg.skip_finding_answer_spans,
            try_lower_case_if_answers_not_found=cfg.try_lower_case_if_answers_not_found,
        )
        for ctx in bm25_ctxs
    ]

    # Filter positives and negatives using distant supervision
    positive_samples = list(filter(lambda ctx: ctx.has_answer, ctxs))
    distantly_positive_samples: List[DataPassage] = []
    negative_samples = list(filter(lambda ctx: not ctx.has_answer, ctxs))
    bm25_positive_samples = list(filter(lambda ctx: ctx.has_answer, bm25_ctxs))
    bm25_distantly_positive_samples: List[DataPassage] = []
    bm25_negative_samples = list(filter(lambda ctx: not ctx.has_answer, bm25_ctxs))

    # Filter unwanted positive passages if training
    if is_train_set:

        # Check whether each positive is from the same page (article) as
        # the corresponding gold positive passages
        for positives in [positive_samples, bm25_positive_samples]:
            for ctx in positives:
                _is_from_gold_wiki_page(
                    gold_passage_map,
                    ctx,
                    tensorizer.tensor_to_text(torch.from_numpy(ctx.title_token_ids)),
                    question,
                    tensorizer,
                )

        # Get positives that are from gold positive passages
        if cfg.gold_page_only_positives:
            selected_positive_ctxs: List[DataPassage] = []
            selected_negative_ctxs: List[DataPassage] = negative_samples
            selected_bm25_positive_ctxs: List[DataPassage] = []
            selected_bm25_negative_ctxs: List[DataPassage] = bm25_negative_samples

            for positives, selected_positives, selected_negatives, distantly_positives in [
                (positive_samples, selected_positive_ctxs, selected_negative_ctxs, distantly_positive_samples),
                (bm25_positive_samples, selected_bm25_positive_ctxs, selected_bm25_negative_ctxs, bm25_distantly_positive_samples)
            ]:
                for ctx in positives:
                    if ctx.is_from_gold:
                        selected_positives.append(ctx)
                    else:  # if it has answer but does not come from gold passage
                        if cfg.should_negatives_contain_answer:
                            selected_negatives.append(ctx)
                        else:
                            distantly_positives.append(ctx)
        else:
            selected_positive_ctxs = positive_samples
            selected_negative_ctxs = negative_samples
            selected_bm25_positive_ctxs = bm25_positive_samples
            selected_bm25_negative_ctxs = bm25_negative_samples

        # Fallback to positive ctx not from gold passages
        if len(selected_positive_ctxs) == 0:
            selected_positive_ctxs = positive_samples
        if len(selected_bm25_positive_ctxs) == 0:
            selected_bm25_positive_ctxs = bm25_positive_samples

        # Optionally include gold passage itself if it is still not in the positives list
        if cfg.include_gold_passage:
            if question in gold_passage_map:
                gold_passage = gold_passage_map[question]
                gold_passage.is_gold = True
                gold_passage.has_answer = True  # assuming it has answer

                gold_passage = _find_answer_spans(
                    tensorizer, gold_passage, question, all_answers, answers_token_ids,
                    warn_if_no_answer=False, warn_if_has_answer=False,
                    recheck_negatives=True,
                    skip_finding_answer_spans=cfg.skip_finding_answer_spans,
                    try_lower_case_if_answers_not_found=cfg.try_lower_case_if_answers_not_found,
                )  # warn below

                if not gold_passage.has_answer:
                    logger.warning(
                        "No answer found in GOLD passage: passage='%s', question='%s', answers=%s, expanded_answers=%s",
                        gold_passage.passage_text,
                        question,
                        answers,
                        expanded_answers,
                    )
                selected_positive_ctxs.append(gold_passage)  # append anyway, since we need this for retriever (not reader)
            else:
                logger.warning(f"Question '{question}' has no gold positive")

    else:
        # NOTE: See `create_reader_input` function in `reader.py` to see how
        # positive and negative samples are merged (keeping their original order)
        selected_positive_ctxs = positive_samples
        selected_negative_ctxs = negative_samples
        selected_bm25_positive_ctxs = bm25_positive_samples
        selected_bm25_negative_ctxs = bm25_negative_samples

    # Restrict number of BM25 passages
    selected_bm25_positive_ctxs = selected_bm25_positive_ctxs[:cfg.max_bm25_positives]
    selected_bm25_negative_ctxs = selected_bm25_negative_ctxs[:cfg.max_bm25_negatives]

    return (
        gold_ctxs,
        selected_positive_ctxs, selected_negative_ctxs, distantly_positive_samples,
        selected_bm25_positive_ctxs, selected_bm25_negative_ctxs, bm25_distantly_positive_samples,
    )


def _load_tokens_into_ctx(
    ctx: DataPassage,
    question_token_ids: np.ndarray,
    wiki_data: TokenizedWikipediaPassages,
    tensorizer: Tensorizer,
    check_pre_tokenized_data: bool = True,
) -> DataPassage:
    tokens = wiki_data.get_tokenized_data(int(ctx.id))

    # Double check if needed
    if ctx.passage_text is not None:
        orig_passage_ids = tensorizer.text_to_tensor(
            ctx.passage_text, add_special_tokens=False,
        ).numpy()
        if check_pre_tokenized_data and (len(orig_passage_ids) != len(tokens["passage_token_ids"]) or \
                not (orig_passage_ids == tokens["passage_token_ids"]).all()):
            raise ValueError(
                f"Passage token mismatch: id: {ctx.id}, orig: {orig_passage_ids}, "
                f"pre-processed: {tokens['passage_token_ids']}. If the sequence lengths are different,"
                f" this might be because the maximum length of the tokenizer is set differently during "
                f"pre-processing and training."
            )

        orig_title_ids = tensorizer.text_to_tensor(
            ctx.title, add_special_tokens=False,
        ).numpy()
        if check_pre_tokenized_data and (len(orig_title_ids) != len(tokens["title_token_ids"]) or \
                not (orig_title_ids == tokens["title_token_ids"]).all()):
            raise ValueError(
                f"Passage title token mismatch: id: {ctx.id}, orig: {orig_title_ids}, "
                f"pre-processed: {tokens['title_token_ids']}. If the sequence lengths are different,"
                f" this might be because the maximum length of the tokenizer is set differently during "
                f"pre-processing and training."
            )

    ctx.load_tokens(
        question_token_ids=question_token_ids, **tokens
    )  # load question, passage and passage title tokens

    # Remove redundant data
    ctx.on_serialize(remove_tokens=False)

    return ctx


def _find_answer_spans(
    tensorizer: Tensorizer,
    ctx: DataPassage,
    question: str,
    answers : List[str],
    answers_token_ids: List[List[int]],
    warn_if_no_answer: bool = False,
    raise_if_no_answer: bool = False,
    warn_if_has_answer: bool = False,
    raise_if_has_answer: bool = False,
    recheck_negatives: bool = False,
    skip_finding_answer_spans: bool = False,  # for generative reader (FiD), where positive or negative don't matter
    try_lower_case_if_answers_not_found: bool = False,
) -> DataPassage:

    if skip_finding_answer_spans:
        return ctx

    if (not recheck_negatives) and (not ctx.has_answer):
        return ctx

    answer_spans = [
        _find_answer_positions(ctx.passage_token_ids, answers_token_ids[i])
        for i in range(len(answers))
    ]

    # flatten spans list
    answer_spans = [item for sublist in answer_spans for item in sublist]
    answers_spans = list(filter(None, answer_spans))

    # Lowercase both the passage and answers if the passage originally contains answers
    if len(answers_spans) == 0 and ctx.has_answer and try_lower_case_if_answers_not_found:
        passage_text_lowercased = tensorizer.tensor_to_text(torch.from_numpy(ctx.passage_token_ids)).lower()
        passage_token_ids_lowercased = tensorizer.text_to_tensor(
            text=passage_text_lowercased.lower(), add_special_tokens=False)
        answers_token_ids_lowercased = [
            tensorizer.text_to_tensor(answer.lower(), add_special_tokens=False)
            for answer in answers
        ]
        answers_spans = [
            _find_answer_positions(passage_token_ids_lowercased, answer_token_ids_lowercased)
            for answer_token_ids_lowercased in answers_token_ids_lowercased
        ]
        answers_spans = list(filter(None, sum(answers_spans, [])))

    if len(answers_spans) == 0 and (warn_if_no_answer or raise_if_no_answer):
        passage_text = tensorizer.tensor_to_text(torch.from_numpy(ctx.passage_token_ids))
        passage_title = tensorizer.tensor_to_text(torch.from_numpy(ctx.title_token_ids))
        message = (
            f"No answer found in passage id={ctx.id} text={passage_text}, title={passage_title}, "
            f"answers={answers}, question={question}"
        )

        if raise_if_no_answer:
            raise ValueError(message)
        else:
            logger.warning(message)

    if len(answers_spans) > 0 and (warn_if_has_answer or raise_if_has_answer):
        passage_text = tensorizer.tensor_to_text(torch.from_numpy(ctx.passage_token_ids))
        passage_title = tensorizer.tensor_to_text(torch.from_numpy(ctx.title_token_ids))
        message = (
            f"Answer FOUND in passage id={ctx.id} text={passage_text}, title={passage_title}, "
            f"answers={answers}, question={question}"
        )

        if raise_if_has_answer:
            raise ValueError(message)
        else:
            logger.warning(message)

    ctx.answers_spans = answers_spans
    ctx.has_answer = bool(answers_spans)

    return ctx


def _find_answer_positions(ctx_ids: np.ndarray, answer: np.ndarray) -> List[Tuple[int, int]]:
    c_len = len(ctx_ids)
    a_len = len(answer)
    answer_occurences = []
    for i in range(0, c_len - a_len + 1):
        if (answer == ctx_ids[i : i + a_len]).all():
            answer_occurences.append((i, i + a_len - 1))
    return answer_occurences


def _get_gold_ctx_dict(file: str) -> Tuple[Dict[str, DataPassage], Dict[str, str]]:
    gold_passage_infos = (
        {}
    )  # question|question_tokens -> DataPassage (with title and gold ctx)

    # original NQ dataset has 2 forms of same question - original, and tokenized.
    # Tokenized form is not fully consisted with the original question if tokenized by some encoder tokenizers
    # Specifically, this is the case for the BERT tokenizer.
    # Depending of which form was used for retriever training and results generation, it may be useful to convert
    # all questions to the canonical original representation.
    original_questions = {}  # question from tokens -> original question (NQ only)

    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)["data"]

    for sample in data:
        question = sample["question"]
        question_from_tokens = (
            sample["question_tokens"] if "question_tokens" in sample else question
        )
        original_questions[question_from_tokens] = question
        title = sample["title"].lower()
        context = sample["context"]  # Note: This one is cased
        rp = DataPassage(sample["example_id"], text=context, title=title)
        if question in gold_passage_infos:
            logger.info("Duplicate question %s", question)
            rp_exist = gold_passage_infos[question]
            logger.info(
                "Duplicate question gold info: title new =%s | old title=%s",
                title,
                rp_exist.title,
            )
            logger.info("Duplicate question gold info: new ctx =%s ", context)
            logger.info(
                "Duplicate question gold info: old ctx =%s ", rp_exist.passage_text
            )

        gold_passage_infos[question] = rp
        gold_passage_infos[question_from_tokens] = rp
    return gold_passage_infos, original_questions


def _get_processed_gold_ctx_dict(file: str) -> Dict[str, DataPassage]:
    gold_passage_infos: Dict[str, DataPassage] = (
        {}
    )

    with open(file, "rb") as f:
        logger.info("Reading file %s" % file)
        data: List[Tuple[str, int]] = pickle.load(f)

    for question, gold_passage_id in data:
        if question in gold_passage_infos:
            logger.info(f"Duplicate gold info: question: '{question}', gold passage IDs: "
                        f"{gold_passage_infos[question]} and {gold_passage_id}.")
            if gold_passage_infos[question] != gold_passage_id:
                logger.warning(
                    f"Each question should have only one unique gold passage; found at least 2: "
                    f"{gold_passage_infos[question]} and {gold_passage_id}. Keeping only the first appearance."
                )

        else:
            passage = DataPassage(
                id=gold_passage_id,
                score=1000,  # follow DPR's original data convention
                has_answer=True,
                is_from_bm25=False,
            )
            passage.is_gold = True
            gold_passage_infos[question] = passage

    return gold_passage_infos


def _is_from_gold_wiki_page(
    gold_passage_map: Dict[str, DataPassage],
    passage: DataPassage,
    passage_title: str,
    question: str,
    tensorizer: Tensorizer,
):
    gold_info = gold_passage_map.get(question, None)
    if gold_info is not None:
        if gold_info.title is None:
            gold_title = tensorizer.tensor_to_text(gold_info.title_token_ids)
        else:
            gold_title = gold_info.title
        is_from_gold_wiki_page = passage_title.lower() == gold_title.lower()
    else:
        is_from_gold_wiki_page = False

    passage.is_from_gold = is_from_gold_wiki_page
    return is_from_gold_wiki_page


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question
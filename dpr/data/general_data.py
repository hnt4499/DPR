"""
A general set of data utilities for both biencoder and reader models.
"""

import os
import json
import glob
import math
import pickle
import logging
import collections
import multiprocessing
from functools import partial
from typing import List, Tuple, Dict, Optional, Iterable

import torch
from torch import Tensor as T
from tqdm import tqdm

from dpr.utils.data_utils import Tensorizer, read_serialized_data_from_files
from .answers_processing import get_expanded_answer

logger = logging.getLogger()


class DataPassage(object):
    """
    Container to collect and cache all Q&A passages related attributes before generating the retriever/reader input
    """

    def __init__(
        self,
        id=None,
        text: str = None,
        title: str = None,
        score=None,
        has_answer: bool = None,
    ):

        self.id = id  # passage ID
        self.is_gold = False  # whether this is exactly a gold passage
        self.is_from_gold = False  # whether this is from the gold passage (or same article as of the gold passage)
        # string passage representations
        self.passage_text = text
        self.title = title
        self.score = score
        self.has_answer = has_answer
        self.answers_spans = None

        # Token ids
        self.question_token_ids = None
        self.title_token_ids = None
        self.passage_token_ids = None

        # For backward compatibility
        self.sequence_ids = None
        self.passage_offset = None

    def on_serialize(self):
        self.passage_text = None
        self.title = None

        self.question_token_ids = self.question_token_ids.numpy()
        self.title_token_ids = self.title_token_ids.numpy()
        self.passage_token_ids = self.passage_token_ids.numpy()

    def on_deserialize(self):
        # Do nothing
        pass


class DataSample(object):
    """
    Container to collect all Q&A passages data per single question
    """

    def __init__(
        self,
        question: str,
        answers: List[str],  # all answers
        orig_answers: List[str],
        expanded_answers: List[List[str]],
        positive_passages: List[DataPassage] = [],
        negative_passages: List[DataPassage] = [],
    ):
        self.question = question
        self.answers = answers  # all answers (including expanded set of answers)
        self.orig_answers = orig_answers  # original set of answers
        self.expanded_answers = expanded_answers  # expanded set of answers using heuristics
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages

    def on_serialize(self):
        for passage in self.positive_passages + self.negative_passages:
            passage.on_serialize()

    def on_deserialize(self):
        for passage in self.positive_passages + self.negative_passages:
            passage.on_deserialize()


class GeneralDataset(torch.utils.data.Dataset):
    """
    General-purpose dataset for both retriever (biencoder) and reader. Input data is expected to be output of a
    trained retriever.
    """
    def __init__(
        self,
        mode: str,
        files: str,
        is_train: bool,
        gold_passages_src: str,
        tensorizer: Tensorizer,
        run_preprocessing: bool,
        num_workers: int,
    ):
        """Initialize general dataset.

        :param mode: one of ["retriever", "reader"]. Dataset mode.
        :param files: either path to a single dataset file (*.json) or a glob pattern to preprocessed pickle (*.pkl) files.
        :param is_train: whether this dataset is training set or evaluation set. Different preprocess settings will be
            applied to different types of training sets.
        :param gold_passages_src: path to the gold passage file.
        :param tensorizer: initialized tensorizer.
        :param run_preprocessing: whether to perform preprocessing. Useful for DDP mode, where only one process should do
            preprocessing while other processes wait for its results.
        :param num_workers: number of workers for the preprocessing step.
        """
        self.mode = mode  # unused for now
        self.files = files
        self.data = []
        self.is_train = is_train
        self.gold_passages_src = gold_passages_src
        self.tensorizer = tensorizer
        self.run_preprocessing = run_preprocessing
        self.num_workers = num_workers

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data(
        self,
        debugging: bool = False,
    ):
        data_files = glob.glob(self.files)
        if len(data_files) == 0:
            raise RuntimeError("No data files found")
        preprocessed_data_files = self._get_preprocessed_files(data_files)

        if debugging:
            logger.info("Debugging mode is on. Restricting to at most 2 data files.")
            preprocessed_data_files = preprocessed_data_files[:2]

        logger.info(f"Reading data files: {preprocessed_data_files}")
        self.data = read_serialized_data_from_files(preprocessed_data_files)

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
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

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
                out_file_prefix,
                self.gold_passages_src,
                self.tensorizer,
                num_workers=self.num_workers,
            )
            self.tensorizer.set_pad_to_max(True)

            # Sync with other processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            # Sync with other processes
            torch.distributed.barrier()
            serialized_files = _find_cached_files(data_files[0])
        return serialized_files


def preprocess_retriever_results(
    is_train_set: bool,
    input_file: str,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    num_workers: int = 8,
) -> List[str]:
    """
    Preprocess the dense retriever outputs (or any compatible file format) into the general retriever/reader input data and
    serializes them into a set of files.
    Conversion splits the input data into multiple chunks and processes them in parallel. Each chunk results are stored
    in a separate file with name out_file_prefix.{number}.pkl
    :param is_train_set: if the data should be processed for a train set (with applicable settings)
    :param input_file: path to a json file with data to convert
    :param out_file_prefix: output path prefix.
    :param gold_passages_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param num_workers: the number of parallel processes for conversion
    :return: path to serialized, preprocessed pickle files
    """
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.loads("".join(f.readlines()))
    logger.info(
        "Loaded %d questions + retrieval results from %s", len(samples), input_file
    )
    workers = multiprocessing.Pool(num_workers)
    ds_size = len(samples)
    step = max(math.ceil(ds_size / num_workers), 1)
    chunks = [samples[i : i + step] for i in range(0, ds_size, step)]
    chunks = [(i, chunks[i]) for i in range(len(chunks))]

    logger.info("Split data into %d chunks", len(chunks))

    processed = 0
    _parse_batch = partial(
        _preprocess_samples_by_chunk,
        out_file_prefix=out_file_prefix,
        gold_passages_file=gold_passages_file,
        tensorizer=tensorizer,
        is_train_set=is_train_set,
    )
    serialized_files = []
    for file_name in workers.map(_parse_batch, chunks):
        processed += 1
        serialized_files.append(file_name)
        logger.info("Chunks processed %d, data saved to %s", processed, file_name)
    logger.info("Preprocessed data stored in %s", serialized_files)

    return serialized_files


def _preprocess_samples_by_chunk(
    samples: List,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    is_train_set: bool,
) -> str:
    chunk_id, samples = samples
    logger.info("Start batch %d", len(samples))
    iterator = _preprocess_retriever_data(
        samples,
        gold_passages_file,
        tensorizer,
        is_train_set=is_train_set,
    )

    # Gather results
    results = []
    for r in tqdm(iterator):
        r.on_serialize()
        results.append(r)

    out_file = out_file_prefix + "." + str(chunk_id) + ".pkl"
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
    ],
)

DEFAULT_PREPROCESSING_CFG_TRAIN = PreprocessingCfg(
    skip_no_positives=True,
    include_gold_passage=False,  # whether to include the gold passage itself
    gold_page_only_positives=True,  # whether positive passages should only be from the gold passages
    expand_answers=True,  # expand the set of answers; see `answers_processing.py`
)


def _preprocess_retriever_data(
    samples: List[Dict],
    gold_info_file: Optional[str],
    tensorizer: Tensorizer,
    cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    is_train_set: bool = True,
) -> Iterable[DataSample]:
    """
    Converts retriever results into general retriever/reader training data.
    :param samples: samples from the retriever's json file results
    :param gold_info_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param cfg: PreprocessingCfg object with positive and negative passage selection parameters
    :param is_train_set: if the data should be processed as a train set
    :return: iterable of DataSample objects which can be consumed by the reader model
    """
    gold_passage_map, canonical_questions = (
        _get_gold_ctx_dict(gold_info_file) if gold_info_file is not None else ({}, {})
    )

    no_positive_passages = 0
    positives_from_gold = 0

    def tokenize_all_texts(sample: DataPassage, question: str):
        """Tokenizer all texts within a DataPassage, including question, passage title and passage text"""
        if sample.question_token_ids is None:
            sample.question_token_ids = tensorizer.text_to_tensor(
                question, add_special_tokens=False,
            )

        if sample.title_token_ids is None:
            sample.title_token_ids = tensorizer.text_to_tensor(
                sample.title, add_special_tokens=False,
            )

        if sample.passage_token_ids is None:
            sample.passage_token_ids = tensorizer.text_to_tensor(
                sample.passage_text, add_special_tokens=False,
            )

        return sample

    for sample in samples:
        question = sample["question"]

        orig_answers = sample["answers"]
        if cfg.expand_answers:
            expanded_answers = [get_expanded_answer(answer) for answer in orig_answers]
        else:
            expanded_answers = []
        all_answers = orig_answers + sum(expanded_answers, [])

        if question in canonical_questions:
            question = canonical_questions[question]

        positive_passages, negative_passages = _select_passages(
            sample,
            question,
            orig_answers,
            expanded_answers,
            all_answers,
            tensorizer,
            gold_passage_map,
            cfg.gold_page_only_positives,
            cfg.include_gold_passage,
            is_train_set,
        )
        # create concatenated sequence ids for each passage and adjust answer spans
        positive_passages = [
            tokenize_all_texts(sample_, question) for sample_ in positive_passages
        ]
        negative_passages = [
            tokenize_all_texts(sample_, question) for sample_ in negative_passages
        ]

        if is_train_set and len(positive_passages) == 0:
            no_positive_passages += 1
            if cfg.skip_no_positives:
                continue

        if next(iter(ctx for ctx in positive_passages if ctx.score == -1), None):
            positives_from_gold += 1

        yield DataSample(
            question,
            answers=all_answers,
            orig_answers=orig_answers,
            expanded_answers=expanded_answers,
            positive_passages=positive_passages,
            negative_passages=negative_passages,
        )

    logger.info("no positive passages samples: %d", no_positive_passages)
    logger.info("positive passages from gold samples: %d", positives_from_gold)


def _select_passages(
    sample: Dict,
    question: str,
    answers: List[str],
    expanded_answers: List[List[str]],
    all_answers: List[str],
    tensorizer: Tensorizer,
    gold_passage_map: Dict[str, DataPassage],
    gold_page_only_positives: bool,
    include_gold_passage: bool,
    is_train_set: bool,
) -> Tuple[List[DataPassage], List[DataPassage]]:
    """
    Select and process valid passages for training/evaluation.
    """

    ctxs = [DataPassage(**ctx) for ctx in sample["ctxs"]]
    answers_token_ids = [
        tensorizer.text_to_tensor(a, add_special_tokens=False) for a in all_answers
    ]

    # Find answer spans for all passages
    ctxs: List[DataPassage] = [
        _find_answer_spans(
            tensorizer, ctx, question, all_answers, answers_token_ids,
            warn_if_no_answer=ctx.has_answer,  # warn if originally it contains answer string
        )
        for ctx in ctxs
    ]

    # Filter positives and negatives using distant supervision
    positive_samples = list(filter(lambda ctx: ctx.has_answer, ctxs))
    negative_samples = list(filter(lambda ctx: not ctx.has_answer, ctxs))

    # Filter unwanted positive passages if training
    if is_train_set:
        selected_positive_ctxs: List[DataPassage] = []
        # Get positives that are from gold positive passages
        for ctx in positive_samples:
            if gold_page_only_positives and _is_from_gold_wiki_page(gold_passage_map, ctx, ctx.title, question):
                selected_positive_ctxs.append(ctx)

        # Fallback to positive ctx not from gold passages
        if len(selected_positive_ctxs) == 0:
            selected_positive_ctxs = positive_samples

        # Optionally include gold passage itself if it is still not in the positives list
        if include_gold_passage:
            if question in gold_passage_map:
                gold_passage = gold_passage_map[question]
                gold_passage.is_gold = True
                gold_passage.has_answer = True  # assuming it has answer

                gold_passage = _find_answer_spans(
                    tensorizer, gold_passage, question, all_answers, answers_token_ids,
                    warn_if_no_answer=False
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
        selected_positive_ctxs = positive_samples

    return selected_positive_ctxs, negative_samples


def _find_answer_spans(
    tensorizer: Tensorizer,
    ctx: DataPassage,
    question: str,
    answers : List[str],
    answers_token_ids: List[List[int]],
    warn_if_no_answer: bool = False,
):
    if ctx.has_answer:
        if ctx.passage_token_ids is None:
            ctx.passage_token_ids = tensorizer.text_to_tensor(
                ctx.passage_text, add_special_tokens=False
            )

        answer_spans = [
            _find_answer_positions(ctx.passage_token_ids, answers_token_ids[i])
            for i in range(len(answers))
        ]

        # flatten spans list
        answer_spans = [item for sublist in answer_spans for item in sublist]
        answers_spans = list(filter(None, answer_spans))
        ctx.answers_spans = answers_spans

        if len(answers_spans) == 0 and warn_if_no_answer:
            logger.warning(
                "No answer found in passage id=%s text=%s, answers=%s, question=%s",
                ctx.id,
                ctx.passage_text,
                answers,
                question,
            )

        ctx.has_answer = bool(answers_spans)

    return ctx


def _find_answer_positions(ctx_ids: T, answer: T) -> List[Tuple[int, int]]:
    c_len = ctx_ids.size(0)
    a_len = answer.size(0)
    answer_occurences = []
    for i in range(0, c_len - a_len + 1):
        if (answer == ctx_ids[i : i + a_len]).all():
            answer_occurences.append((i, i + a_len - 1))
    return answer_occurences


def _get_gold_ctx_dict(file: str) -> Tuple[Dict[str, DataPassage], Dict[str, str]]:
    gold_passage_infos = (
        {}
    )  # question|question_tokens -> ReaderPassage (with title and gold ctx)

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


def _is_from_gold_wiki_page(
    gold_passage_map: Dict[str, DataPassage], passage: DataPassage, passage_title: str, question: str
):
    gold_info = gold_passage_map.get(question, None)
    if gold_info is not None:
        is_from_gold_wiki_page = passage_title.lower() == gold_info.title.lower()
    else:
        is_from_gold_wiki_page = False

    passage.is_from_gold = is_from_gold_wiki_page
    return is_from_gold_wiki_page
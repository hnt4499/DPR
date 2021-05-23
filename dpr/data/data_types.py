"""
Definition of all data types used in training.
"""

import collections
from typing import List

import numpy as np
from torch import Tensor as T


"""
General
"""

class DataPassage(object):
    """
    Container to collect and cache all Q&A passages related attributes before generating the retriever/reader input
    """

    def __init__(
        self,
        id: str = None,
        text: str = None,
        title: str = None,
        score: float = None,
        has_answer: bool = None,
        is_from_bm25: bool = False,
    ):

        self.id = int(id)  # passage ID
        self.is_gold = False  # whether this is exactly a gold passage
        self.is_from_gold = False  # whether this is from the gold passage (or same article as of the gold passage)
        self.is_from_bm25 = is_from_bm25

        # String passage representations; used for double checking only
        self.passage_text = text
        self.title = title

        # Other information
        self.score = float(score) if score is not None else score
        self.has_answer = has_answer
        self.answers_spans = None

        # Token ids
        self.question_token_ids = None
        self.title_token_ids = None
        self.passage_token_ids = None

        # For backward compatibility
        self.sequence_ids = None
        self.passage_offset = None

    def load_tokens(
        self,
        question_token_ids: np.ndarray = None,
        title_token_ids: np.ndarray = None,
        passage_token_ids: np.ndarray = None
    ):
        """
        All these arrays are expected to be numpy array.
        """
        if question_token_ids is not None:
            self.question_token_ids = question_token_ids.copy()
        if title_token_ids is not None:
            self.title_token_ids = title_token_ids.copy()
        if passage_token_ids is not None:
            self.passage_token_ids = passage_token_ids.copy()

    def on_serialize(self, remove_tokens=True):
        self.passage_text = None
        self.title = None

        # Remove tokens
        if remove_tokens:
            self.title_token_ids = None
            self.passage_token_ids = None
            self.question_token_ids = None

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
        question_token_ids: np.ndarray,
        answers: List[str],  # all answers
        orig_answers: List[str],
        expanded_answers: List[List[str]],
        # Gold
        gold_passages: List[DataPassage] = [],
        # Dense
        positive_passages: List[DataPassage] = [],
        distantly_positive_passages: List[DataPassage] = [],
        negative_passages: List[DataPassage] = [],
        # Sparse
        bm25_positive_passages: List[DataPassage] = [],
        bm25_distantly_positive_passages: List[DataPassage] = [],
        bm25_negative_passages: List[DataPassage] = [],
    ):
        self.question = question
        self.question_token_ids = question_token_ids
        self.answers = answers  # all answers (including expanded set of answers)
        self.orig_answers = orig_answers  # original set of answers
        self.expanded_answers = expanded_answers  # expanded set of answers using heuristics

        # Gold
        self.gold_passages = gold_passages

        # Dense
        self.positive_passages = positive_passages
        self.distantly_positive_passages = distantly_positive_passages
        self.negative_passages = negative_passages

        # Sparse
        self.bm25_positive_passages = bm25_positive_passages
        self.bm25_distantly_positive_passages = bm25_distantly_positive_passages
        self.bm25_negative_passages = bm25_negative_passages

        self.container = [
            # Gold
            self.gold_passages,
            # Dense
            self.positive_passages,
            self.distantly_positive_passages,
            self.negative_passages,
            # Sparse
            self.bm25_positive_passages,
            self.bm25_distantly_positive_passages,
            self.bm25_negative_passages,
        ]

    def on_serialize(self):
        self.question = None  # remove text
        for passages in self.container:
            for passage in passages:
                passage.on_serialize()

    def on_deserialize(self):
        for passages in self.container:
            for passage in passages:
                passage.on_deserialize()


"""
Retriever (BiEncoder)
"""


# Non-tokenized
BiEncoderPassage = collections.namedtuple(
    "BiEncoderPassage",
    [
        "text",
        "title",
    ]
)

class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


# Tokenized
BiEncoderPassageTokenized = collections.namedtuple(
    "BiEncoderPassageTokenized",
    [
        "id",
        "is_gold",
        "text_ids",
        "title_ids",
    ]
)
class BiEncoderSampleTokenized(object):
    query_ids: np.ndarray
    positive_passages: List[BiEncoderPassageTokenized]
    hard_negative_passages: List[BiEncoderPassageTokenized]
    bm25_negative_passages: List[BiEncoderPassageTokenized]

# Training batch
BiEncoderBatch = collections.namedtuple(
    "BiEncoderBatch",
    [
        "question_ids",
        "question_segments",
        "context_IDs",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)


"""
Reader
"""

class ReaderPassage(DataPassage):
    """For backward compatibility."""
    pass


class ReaderSample(DataSample):
    """For backward compatibility."""
    pass

ReaderBatch = collections.namedtuple(
    'ReaderBatch',
    [
        'context_IDs',
        'input_ids',
        'start_positions',
        'end_positions',
        'answers_mask',
        'passage_scores',
    ],
    defaults=[None],  # default `passage_scores` to None
)

SpanPrediction = collections.namedtuple(
    "SpanPrediction",
    [
        "prediction_text",
        "span_score",
        "relevance_score",
        "passage_index",
        "passage_token_ids",
    ],
)


"""
Configs
For now for One-For-All models only
"""


# During input preparing
BiEncoderDataConfig = collections.namedtuple(
    "BiEncoderDataConfig",
    [
        "insert_title",
        "num_hard_negatives",
        "num_other_negatives",
        "shuffle",
        "shuffle_positives",
        "hard_neg_fallback",
        "query_token",
    ],
)

# During forward pass
BiEncoderTrainingConfig = collections.namedtuple(
    "BiEncoderTrainingConfig",
    [
        "encoder_type",
        "rep_positions_q",
        "rep_positions_c",
    ],
)


BiEncoderPredictionBatch = collections.namedtuple(
    "BiEncoderPredictionBatch",
    [
        "question_vector",
        "context_vector",
    ],
)


# During input preparing
ReaderDataConfig = collections.namedtuple(
    "ReaderDataConfig",
    [
        "passages_per_question",
        "max_length",  # encoder maximum sequence length, e.g., 256 or 350
        "max_n_answers",  # maximum number of answer spans to marginalize per each sample
        "is_train",
        "shuffle",
        "num_sub_batches",
    ],
)

# During forward pass
ReaderTrainingConfig = collections.namedtuple(
    "ReaderTrainingConfig",
    [
        "use_simple_loss",
        "average_loss",
        "do_softmax_before_score_scaling",
    ],
)

ReaderPredictionBatch = collections.namedtuple(
    "ReaderPredictionBatch",
    [
        # During training
        "total_loss",
        # During evaluation
        "start_logits",
        "end_logits",
        "relevance_logits",
    ],
)

ReaderQuestionPredictions = collections.namedtuple(
    "ReaderQuestionPredictions",
    [
        "id",
        "predictions",
        "gold_answers",
    ]
)
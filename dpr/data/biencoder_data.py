import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple, Union

import hydra
import jsonlines
import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor as T

from dpr.data.tables import Table
from dpr.utils.data_utils import read_data_from_json_files, Tensorizer
from dpr.data.data_types import (
    # Non-tokenized
    BiEncoderPassage,
    BiEncoderSample,
    # Tokenized
    BiEncoderPassageTokenized,
    BiEncoderSampleTokenized,
    # Else
    DataPassage,
    DataSample,
    ReaderSample,
)
from dpr.data.general_data import TokenizedWikipediaPassages, GeneralDataset, GeneralDatasetScheme, _find_answer_positions

logger = logging.getLogger(__name__)


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer, model: torch.nn.Module = None):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer, model: torch.nn.Module = None):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer, model: torch.nn.Module = None):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


class RandomTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position  # in case during inference

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer, model: torch.nn.Module):
        attention_masks = tenzorizer.get_attn_mask(input_ids)
        rep_positions = []

        for attention_mask in attention_masks:
            if model.training:
                input_length = (attention_mask != 0).sum()
                rep_position = random.randint(0, input_length - 1)
                rep_positions.append(rep_position)
            else:
                # Fall back to default
                rep_positions.append(self.static_position)
        rep_positions = torch.tensor(rep_positions, dtype=torch.int8).unsqueeze(-1).repeat(1, 2)
        return rep_positions


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        if selector:
            logger.info(f"Initializing selector with config:\n{selector}")
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.sample_by_cat = False

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index) -> BiEncoderSample:
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

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
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        ctx_boundary_aug: int = 0,
        ctx_min_len: int = 50,
        sample_by_cat: bool = False,
        category_mapping_path: str = None,
        sampled_idxs_path: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = []
        self.data = []
        self.normalize = normalize
        self.ctx_boundary_aug = ctx_boundary_aug  # context span boundary augmentation
        self.ctx_min_len = ctx_min_len  # used in conjunction with `ctx_boundary_aug`
        self.sample_by_cat = sample_by_cat  # used to fire a signal for `data_utils.ShardedDataIterator`
        self.sampled_idxs_path = sampled_idxs_path  # used when `sample_by_cat` is True
        self.category_mapping_path = category_mapping_path  # used when `sample_by_cat` is True and `sampled_idxs_path` is None

        logger.info("Data files: %s", self.file)

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def _boundary_aug(self, text):
        if self.ctx_boundary_aug <= 0:
            return text
        text = text.split()
        left_aug = random.randint(0, self.ctx_boundary_aug)
        right_aug = random.randint(0, self.ctx_boundary_aug)
        if (len(text) - left_aug - right_aug) < self.ctx_min_len:  # fall back to orignal text
            return " ".join(text)
        return " ".join(text[left_aug:len(text) - right_aug])

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        answers = json_sample["answers"]
        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        # Context boundary augmentation
        if self.ctx_boundary_aug > 0:
            # Positives
            extreme_hard_negative_ctxs = []  # "extremely" hard negative contexts
            new_positive_ctxs = []
            for ctx in positive_ctxs:
                new_ctx = self._boundary_aug(ctx["text"])  # do boundary augmentation
                new_ctx = {"text": new_ctx, "title": ctx.get("title", None)}
                # check if it results in another positive context
                if any(answer.lower() in new_ctx["text"].lower() for answer in answers):
                    new_positive_ctxs.append(new_ctx)
                else:
                    new_positive_ctxs.append(ctx)
                    extreme_hard_negative_ctxs.append(new_ctx)
            positive_ctxs = new_positive_ctxs

            # Negatives
            new_negative_ctxs = []
            for ctx in negative_ctxs:
                new_ctx = self._boundary_aug(ctx["text"])
                new_ctx = {"text": new_ctx, "title": ctx.get("title", None)}
                new_negative_ctxs.append(new_ctx)
            negative_ctxs = new_negative_ctxs

            new_hard_negative_ctxs = []
            for ctx in hard_negative_ctxs:
                new_ctx = self._boundary_aug(ctx["text"])
                new_ctx = {"text": new_ctx, "title": ctx.get("title", None)}
                new_hard_negative_ctxs.append(new_ctx)
            hard_negative_ctxs = new_hard_negative_ctxs

            # Make "extremely" hard negative contexts more likely to be chosen
            to_add = 1 if len(hard_negative_ctxs) == 0 else len(hard_negative_ctxs)
            hard_negative_ctxs.extend(extreme_hard_negative_ctxs * to_add)

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)

    def get_qas(self) -> Tuple[List[str], List[str]]:
        return [s["question"] for s in self.data], [s["answers"] for s in self.data]

    def get_qas_range(
        self, start_idx: int, end_idx: int
    ) -> Tuple[List[str], List[str]]:
        return (
            [s["question"] for s in self.data[start_idx:end_idx]],
            [s["answers"] for s in self.data[start_idx:end_idx]],
        )


class JsonQADatasetWithAllPassages(JsonQADataset):
    """Scalable dataset with a dataframe of all (necessary) passages"""
    def __init__(
        self,
        file: str,
        all_passages_file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        ctx_boundary_aug: int = 0,
        ctx_aug_prob: float = 0.0,
        sample_by_cat: bool = False,
        category_mapping_path: str = None,
        sampled_idxs_path: str = None,
    ):

        super(JsonQADatasetWithAllPassages, self).__init__(
            file=file,
            selector=selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            normalize=normalize,
            query_special_suffix=query_special_suffix,
            ctx_boundary_aug=ctx_boundary_aug,
            ctx_min_len=50,  # not used
            sample_by_cat=sample_by_cat,
            category_mapping_path=category_mapping_path,
            sampled_idxs_path=sampled_idxs_path,
        )
        self.all_passages_file = all_passages_file
        self.ctx_aug_prob = ctx_aug_prob

    def load_data(self, datasets: List[torch.utils.data.Dataset], tensorizer: Tensorizer):
        super(JsonQADatasetWithAllPassages, self).load_data()
        self.tensorizer = tensorizer

        # Retrieve all passages if possible
        all_passages = None
        for dataset in datasets:
            if hasattr(dataset, "all_passages"):
                if all_passages is None:
                    all_passages = dataset.all_passages
                else:
                    assert all_passages == dataset.all_passages

        # If not possible, read it in
        if all_passages is None:
            logger.info(f"Reading all passages at {os.path.realpath(self.all_passages_file)}")
            self.all_passages = pd.read_csv(self.all_passages_file, index_col=0)
        else:
            self.all_passages = all_passages

    def _boundary_aug(self, index: int, answers: List[str]):
        def tokenize(t):
            """Shorcut for tokenizer"""
            tokenized_ids = self.tensorizer.tokenizer.encode(
                t,
                add_special_tokens=False,
                max_length=10000,
                pad_to_max_length=False,
                truncation=True,
            )
            return np.array(tokenized_ids)

        def id_to_token(id):
            """Convert a single token ID to token string"""
            return self.tensorizer.tokenizer.convert_ids_to_tokens(int(id))

        data = self.all_passages.loc[index]
        text, title = data[["text", "title"]]

        if self.ctx_boundary_aug <= 0 or random.random() < self.ctx_aug_prob:
            if title in ["nan", "NaN"] or title is np.nan:
                title = None
            return text, text, title, None

        text_ids = tokenize(text)
        answers_ids = [tokenize(answer) for answer in answers]

        # Previous passage
        length_to_augment = [0]  # 0 means no augmentation
        if index - 1 in self.all_passages.index and self.all_passages.loc[index - 1, "title"] == title:
            prev_passage = self.all_passages.loc[index - 1, "text"]
            prev_passage_ids = tokenize(prev_passage)

            length_to_augment.extend(range(-self.ctx_boundary_aug, 0))

        # Next passage
        if index + 1 in self.all_passages.index and self.all_passages.loc[index + 1, "title"] == title:
            next_passage = self.all_passages.loc[index + 1, "text"]
            next_passage_ids = tokenize(next_passage)

            length_to_augment.extend(range(1, self.ctx_boundary_aug + 1))

        length_to_augment = random.choice(length_to_augment)

        if length_to_augment < 0:  # augment with previous passage
            # Shrink until we find a token that is not a subword
            left_aug_idx = len(prev_passage_ids) + length_to_augment
            while 0 < left_aug_idx < len(prev_passage_ids) - 1:
                token_str: str = id_to_token(prev_passage_ids[left_aug_idx])
                if token_str.startswith("##") or token_str.startswith(" ##"):
                    left_aug_idx += 1
                else:
                    break

            right_aug_idx = len(text_ids) + length_to_augment
            while 0 < right_aug_idx < len(text_ids) - 1:
                token_str: str = id_to_token(text_ids[right_aug_idx + 1])
                if token_str.startswith("##") or token_str.startswith(" ##"):
                    right_aug_idx -= 1
                else:
                    break

            aug_ids = np.concatenate([
                prev_passage_ids[left_aug_idx:],
                text_ids[:right_aug_idx],
            ])

        elif length_to_augment > 0:  # augment with next passage
            # Shrink until we find a token that is not a subword
            left_aug_idx = length_to_augment
            while 0 < left_aug_idx < len(text_ids) - 1:
                token_str: str = id_to_token(text_ids[left_aug_idx])
                if token_str.startswith("##") or token_str.startswith(" ##"):
                    left_aug_idx += 1
                else:
                    break

            right_aug_idx = length_to_augment
            while 0 < right_aug_idx < len(next_passage_ids) - 1:
                token_str: str = id_to_token(next_passage_ids[right_aug_idx + 1])
                if token_str.startswith("##") or token_str.startswith(" ##"):
                    right_aug_idx -= 1
                else:
                    break

            aug_ids = np.concatenate([
                text_ids[left_aug_idx:],
                next_passage_ids[:right_aug_idx],
            ])

        else:
            aug_ids = None

        if aug_ids is None:
            aug_text = text
            answers_spans = None
        else:
            # Re-parse answers
            answers_spans = [
                _find_answer_positions(aug_ids, answer_ids)
                for answer_ids in answers_ids
            ]
            answers_spans = sum(answers_spans, [])  # flatten
            answers_spans = list(filter(None, answers_spans))  # remove invalid entries
            aug_text = self.tensorizer.tokenizer.decode(aug_ids)

        if title in ["nan", "NaN"] or title is np.nan:
            title = None

        return text, aug_text, title, answers_spans

    def __getitem__(self, index) -> BiEncoderSample:
        self.tensorizer.set_pad_to_max(False)

        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = self._process_query(json_sample["question"])

        answers = json_sample["answers"]
        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        # Context boundary augmentation
        # Positives
        new_positive_ctxs = []
        for ctx in positive_ctxs:
            passage_id = int(ctx["id"])
            ctx, aug_ctx, title, answers_spans = self._boundary_aug(
                passage_id, answers)  # do boundary augmentation
            orig_ctx = {"text": ctx, "title": title}
            aug_ctx = {"text": aug_ctx, "title": title}

            # Check if it results in another positive context
            if answers_spans is not None and len(answers_spans) > 0:
                new_positive_ctxs.append(aug_ctx)
            else:
                new_positive_ctxs.append(orig_ctx)
        positive_ctxs = new_positive_ctxs

        # Negatives
        new_negative_ctxs = []
        for ctx in negative_ctxs:
            passage_id = int(ctx["id"])
            ctx, aug_ctx, title, answers_spans = self._boundary_aug(
                passage_id, answers)
            orig_ctx = {"text": ctx, "title": title}
            aug_ctx = {"text": aug_ctx, "title": title}

            # Check if it results in another positive context
            if answers_spans is not None and len(answers_spans) > 0:
                new_negative_ctxs.append(orig_ctx)
            else:
                new_negative_ctxs.append(aug_ctx)
        negative_ctxs = new_negative_ctxs

        new_hard_negative_ctxs = []
        for ctx in hard_negative_ctxs:
            passage_id = int(ctx["id"])
            ctx, aug_ctx, title, answers_spans = self._boundary_aug(
                passage_id, answers)
            orig_ctx = {"text": ctx, "title": title}
            aug_ctx = {"text": aug_ctx, "title": title}

            # Check if it results in another positive context
            if answers_spans is not None and len(answers_spans) > 0:
                new_hard_negative_ctxs.append(orig_ctx)
            else:
                new_hard_negative_ctxs.append(aug_ctx)
        hard_negative_ctxs = new_hard_negative_ctxs

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage(
                normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
                ctx["title"],
            )

        r.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        r.negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        r.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]

        self.tensorizer.set_pad_to_max(True)
        return r


class OneForAllDataset(Dataset, GeneralDatasetScheme):
    def __init__(
        self,
        mode: str,
        file: str,
        selector: DictConfig = None,
        shuffle_positives: bool = False,
        special_token: str = None,
        encoder_type: str = None,
        only_gold: bool = False,
        debugging: bool =  False,
    ):
        """One-for-all dataset using general dataset scheme. This dataset can be used for retriever (by setting `mode=="retriever"`),
        reader (by setting `mode=="reader"`) for both (by setting `mode=="both"`).
        For now this data is implemented under `biencoder_data` for some backward compatibility reasons.

        :param file: either path to a single dataset file (*.json) or a glob pattern to preprocessed pickle (*.pkl) files.
        :only_gold: whether to keep only samples whose gold passage is available. Useful for retriever dev set, since previously
            all retriever data have gold passages. Data discrepancy could result in wrong selection of the best model during evaluation.
        """
        super(OneForAllDataset, self).__init__(
            selector,
            special_token=special_token,
            shuffle_positives=shuffle_positives,
            encoder_type=encoder_type,
            query_special_suffix=None,
        )

        # TODO: try normalizing questions and passages
        assert mode in ["retriever", "reader", "both"], f"Invalid mode: {mode}"
        self.mode = mode
        self.normalize = False
        self.only_gold = only_gold

        # Data should already be pre-processed
        pickle_files = file.replace(".json", ".*.pkl")
        pickle_files = glob.glob(pickle_files)
        assert len(pickle_files) > 0, "Data should be already processed"

        # Initialize general dataset
        self.dataset = GeneralDataset(
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
        )

    def load_data(self, wiki_data: TokenizedWikipediaPassages, tensorizer: Tensorizer):
        self.wiki_data = wiki_data
        self.wiki_data.load_data()
        self.dataset.load_data()

        # Remove those whose gold passage info is not available
        if self.only_gold:
            orig_len = len(self.dataset)
            logger.info("Removing samples whose gold passage info is not available.")
            self.dataset.data = [sample for sample in self.dataset.data if len(sample.gold_passages) > 0]
            logger.info(f"Number of samples: before filtering: {orig_len}, after filtering: {len(self.dataset)}")

    def _process_query(self, query: str):
        # We don't use this function
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Union[
        BiEncoderSampleTokenized,  # `mode=="retriever"`
        ReaderSample,  # `mode=="reader`; note that `ReaderSample` is basically `DataSample`
        Tuple[BiEncoderSampleTokenized, ReaderSample]  # `mode=="both"`
    ]:
        # Reader sample is without any further pre-processing
        reader_sample = self.dataset[index]
        if self.mode == "reader":
            return reader_sample

        # Retriever sample needs further pre-processing for backward compatibility
        retriever_sample = BiEncoderSampleTokenized()
        retriever_sample.query_ids = reader_sample.question_token_ids

        positive_ctxs = reader_sample.gold_passages + reader_sample.positive_passages

        # TODO: allow other kinds of positives and negatives, such as distantly positives
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

        retriever_sample.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        retriever_sample.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        retriever_sample.bm25_negative_passages = [create_passage(ctx) for ctx in bm25_negative_ctxs]

        if self.mode == "retriever":
            return retriever_sample
        return retriever_sample, reader_sample


class BiEncoderGeneralDataset(OneForAllDataset):
    def __init__(self, **kwargs):
        super(BiEncoderGeneralDataset, self).__init__(mode="retriever", **kwargs)


class BiEncoderGeneralDatasetWithDataAugmentation(BiEncoderGeneralDataset):
    def __init__(
        self,
        ctx_aug_boundary: int = 15,  # maximum number of tokens to augment for each side
        ctx_aug_prob: float = 0.3,  # augmentation probability
        **kwargs,
    ):
        super(BiEncoderGeneralDatasetWithDataAugmentation, self).__init__(**kwargs)
        self.ctx_aug_boundary = ctx_aug_boundary
        self.ctx_aug_prob = ctx_aug_prob

    def load_data(self, wiki_data: TokenizedWikipediaPassages, tensorizer: Tensorizer):
        super(BiEncoderGeneralDatasetWithDataAugmentation, self).load_data(
            wiki_data, tensorizer)
        self.tensorizer = tensorizer

    def _boundary_aug(self, ctx: DataPassage, answers_ids: List[np.ndarray]) -> DataPassage:
        if self.ctx_aug_boundary <= 0:
            return ctx
        ctx_id = int(ctx.id)
        ctx_passage = ctx.passage_token_ids
        ctx_title = ctx.title_token_ids

        lengths_to_augment = [0]  # 0 means no augmentation
        # Previous passage
        try:
            prev_ctx = self.wiki_data.get_tokenized_data(ctx_id - 1)
            prev_passage = prev_ctx["passage_token_ids"]
            prev_title = prev_ctx["title_token_ids"]

            if len(ctx_title) == len(prev_title) and (ctx_title == prev_title).all():
                lengths_to_augment.extend(range(-self.ctx_aug_boundary, 0))
        except (IndexError, KeyError):  # there is no such index in the database
            pass

        # Next passage
        try:
            next_ctx = self.wiki_data.get_tokenized_data(ctx_id + 1)
            next_passage = next_ctx["passage_token_ids"]
            next_title = next_ctx["title_token_ids"]

            if len(ctx_title) == len(next_title) and (ctx_title == next_title).all():
                lengths_to_augment.extend(range(1, self.ctx_aug_boundary + 1))
        except (IndexError, KeyError):  # there is no such index in the database
            pass

        length_to_augment = random.choice(lengths_to_augment)
        original_length = len(ctx_passage)

        # Do augmentation: only do augmentation when both the previous and next passages
        # are from the same article as the current passage
        if len(lengths_to_augment) != (2 * self.ctx_aug_boundary + 1) or length_to_augment == 0:
            aug_ids = None
        elif length_to_augment < 0:  # augment with previous passage
            aug_ids = np.concatenate([
                prev_passage[length_to_augment:],
                ctx_passage[:length_to_augment]
            ])
            assert len(aug_ids) == original_length
        elif length_to_augment > 0:  # augment with next passage
            aug_ids = np.concatenate([
                ctx_passage[length_to_augment:],
                next_passage[:length_to_augment],
            ])
            assert len(aug_ids) == original_length

        if aug_ids is not None:
            # Re-parse answers
            answers_spans = [
                _find_answer_positions(aug_ids, answer_ids)
                for answer_ids in answers_ids
            ]
            answers_spans = sum(answers_spans, [])  # flatten
            answers_spans = list(filter(None, answers_spans))  # remove invalid entries

            # Create new passage object and update augmented data
            new_ctx = DataPassage(id=ctx.id)
            new_ctx.is_gold = ctx.is_gold
            new_ctx.title_token_ids = ctx_title.copy()
            new_ctx.passage_token_ids = aug_ids.copy()
            new_ctx.answers_spans = answers_spans

            ctx = new_ctx

        return ctx

    def __getitem__(self, index) -> BiEncoderSampleTokenized:
        self.tensorizer.set_pad_to_max(False)

        sample: DataSample = self.dataset[index]
        answers = sample.answers
        answers_ids = [
            self.tensorizer.tokenizer.encode(
                answer,
                add_special_tokens=False,
                max_length=10000,
                pad_to_max_length=False,
                truncation=True,
            )
            for answer in answers
        ]
        answers_ids = [np.array(answer_ids) for answer_ids in answers_ids]

        retriever_sample = BiEncoderSampleTokenized()
        retriever_sample.query_ids = sample.question_token_ids

        positive_ctxs = sample.gold_passages + sample.positive_passages

        # TODO: allow other kinds of positives and negatives, such as distantly positives
        hard_negative_ctxs = sample.negative_passages
        bm25_negative_ctxs = sample.bm25_negative_passages

        # Load tokens
        for ctx in positive_ctxs + hard_negative_ctxs + bm25_negative_ctxs:
            tokens = self.wiki_data.get_tokenized_data(int(ctx.id))
            ctx.load_tokens(**tokens)

        # Context boundary augmentation
        if self.ctx_aug_boundary > 0 and random.random() < self.ctx_aug_prob:
            # Positives
            new_positive_ctxs = []

            for ctx in positive_ctxs:
                # Do augmentation
                aug_ctx = self._boundary_aug(ctx, answers_ids)
                # Check if it results in another positive context
                if aug_ctx.answers_spans is not None and len(aug_ctx.answers_spans) > 0:
                    new_positive_ctxs.append(aug_ctx)
                else:
                    new_positive_ctxs.append(ctx)
            positive_ctxs = new_positive_ctxs

            # Hard negatives
            new_hard_negative_ctxs = []
            for ctx in hard_negative_ctxs:
                # Do augmentation
                aug_ctx = self._boundary_aug(ctx, answers_ids)
                # Check if it results in another positive context
                if aug_ctx.answers_spans is not None and len(aug_ctx.answers_spans) > 0:
                    new_hard_negative_ctxs.append(ctx)
                else:
                    new_hard_negative_ctxs.append(aug_ctx)
            hard_negative_ctxs = new_hard_negative_ctxs

            # BM25 hard negatives
            new_bm25_negative_ctxs = []
            for ctx in bm25_negative_ctxs:
                # Do augmentation
                aug_ctx = self._boundary_aug(ctx, answers_ids)
                # Check if it results in another positive context
                if aug_ctx.answers_spans is not None and len(aug_ctx.answers_spans) > 0:
                    new_bm25_negative_ctxs.append(ctx)
                else:
                    new_bm25_negative_ctxs.append(aug_ctx)
            bm25_negative_ctxs = new_bm25_negative_ctxs

        def create_passage(ctx: DataPassage):
            return BiEncoderPassageTokenized(
                id=ctx.id,
                is_gold=ctx.is_gold,
                text_ids=ctx.passage_token_ids,
                title_ids=ctx.title_token_ids,
            )

        retriever_sample.positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        retriever_sample.hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        retriever_sample.bm25_negative_passages = [create_passage(ctx) for ctx in bm25_negative_ctxs]

        self.tensorizer.set_pad_to_max(True)
        return retriever_sample


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


class Cell:
    def __init__(self):
        self.value_tokens: List[str] = []
        self.type: str = ""
        self.nested_tables: List[Table] = []

    def __str__(self):
        return " ".join(self.value_tokens)

    def to_dpr_json(self, cell_idx: int):
        r = {"col": cell_idx}
        r["value"] = str(self)
        return r


class Row:
    def __init__(self):
        self.cells: List[Cell] = []

    def __str__(self):
        return "| ".join([str(c) for c in self.cells])

    def visit(self, tokens_function, row_idx: int):
        for i, c in enumerate(self.cells):
            if c.value_tokens:
                tokens_function(c.value_tokens, row_idx, i)

    def to_dpr_json(self, row_idx: int):
        r = {"row": row_idx}
        r["columns"] = [c.to_dpr_json(i) for i, c in enumerate(self.cells)]
        return r


class Table(object):
    def __init__(self, caption=""):
        self.caption = caption
        self.body: List[Row] = []
        self.key = None
        self.gold_match = False

    def __str__(self):
        table_str = "<T>: {}\n".format(self.caption)
        table_str += " rows:\n"
        for i, r in enumerate(self.body):
            table_str += " row #{}: {}\n".format(i, str(r))

        return table_str

    def get_key(self) -> str:
        if not self.key:
            self.key = str(self)
        return self.key

    def visit(self, tokens_function, include_caption: bool = False) -> bool:
        if include_caption:
            tokens_function(self.caption, -1, -1)
        for i, r in enumerate(self.body):
            r.visit(tokens_function, i)

    def to_dpr_json(self):
        r = {
            "caption": self.caption,
            "rows": [r.to_dpr_json(i) for i, r in enumerate(self.body)],
        }
        if self.gold_match:
            r["gold_match"] = 1
        return r


class NQTableParser(object):
    def __init__(self, tokens, is_html_mask, title):
        self.tokens = tokens
        self.is_html_mask = is_html_mask
        self.max_idx = len(self.tokens)
        self.all_tables = []

        self.current_table: Table = None
        self.tables_stack = collections.deque()
        self.title = title

    def parse(self) -> List[Table]:
        self.all_tables = []
        self.tables_stack = collections.deque()

        for i in range(self.max_idx):

            t = self.tokens[i]

            if not self.is_html_mask[i]:
                # cell content
                self._on_content(t)
                continue

            if "<Table" in t:
                self._on_table_start()
            elif t == "</Table>":
                self._on_table_end()
            elif "<Tr" in t:
                self._onRowStart()
            elif t == "</Tr>":
                self._onRowEnd()
            elif "<Td" in t or "<Th" in t:
                self._onCellStart()
            elif t in ["</Td>", "</Th>"]:
                self._on_cell_end()

        return self.all_tables

    def _on_table_start(self):
        caption = self.title
        parent_table = self.current_table
        if parent_table:
            self.tables_stack.append(parent_table)

            caption = parent_table.caption
            if parent_table.body and parent_table.body[-1].cells:
                current_cell = self.current_table.body[-1].cells[-1]
                caption += " | " + " ".join(current_cell.value_tokens)

        t = Table()
        t.caption = caption
        self.current_table = t
        self.all_tables.append(t)

    def _on_table_end(self):
        t = self.current_table
        if t:
            if self.tables_stack:  # t is a nested table
                self.current_table = self.tables_stack.pop()
                if self.current_table.body:
                    current_cell = self.current_table.body[-1].cells[-1]
                    current_cell.nested_tables.append(t)
        else:
            logger.error("table end without table object")

    def _onRowStart(self):
        self.current_table.body.append(Row())

    def _onRowEnd(self):
        pass

    def _onCellStart(self):
        current_row = self.current_table.body[-1]
        current_row.cells.append(Cell())

    def _on_cell_end(self):
        pass

    def _on_content(self, token):
        if self.current_table.body:
            current_row = self.current_table.body[-1]
            current_cell = current_row.cells[-1]
            current_cell.value_tokens.append(token)
        else:  # tokens outside of row/cells. Just append to the table caption.
            self.current_table.caption += " " + token


def read_nq_tables_jsonl(path: str) -> Dict[str, Table]:
    tables_with_issues = 0
    single_row_tables = 0
    nested_tables = 0
    regular_tables = 0
    total_tables = 0
    total_rows = 0
    tables_dict = {}

    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            tokens = jline["tokens"]

            if "( hide ) This section has multiple issues" in " ".join(tokens):
                tables_with_issues += 1
                continue

            mask = jline["html_mask"]
            # page_url = jline["doc_url"]
            title = jline["title"]
            p = NQTableParser(tokens, mask, title)
            tables = p.parse()

            # table = parse_table(tokens, mask)

            nested_tables += len(tables[1:])

            for t in tables:
                total_tables += 1

                # calc amount of non empty rows
                non_empty_rows = sum(
                    [
                        1
                        for r in t.body
                        if r.cells and any([True for c in r.cells if c.value_tokens])
                    ]
                )

                if non_empty_rows <= 1:
                    single_row_tables += 1
                else:
                    regular_tables += 1
                    total_rows += len(t.body)

                    if t.get_key() not in tables_dict:
                        tables_dict[t.get_key()] = t

            if len(tables_dict) % 1000 == 0:
                logger.info("tables_dict %d", len(tables_dict))

    logger.info("regular tables %d", regular_tables)
    logger.info("tables_with_issues %d", tables_with_issues)
    logger.info("single_row_tables %d", single_row_tables)
    logger.info("nested_tables %d", nested_tables)
    return tables_dict


def get_table_string_for_answer_check(table: Table):  # this doesn't use caption
    table_text = ""
    for r in table.body:
        table_text += " . ".join([" ".join(c.value_tokens) for c in r.cells])
    table_text += " . "
    return table_text


class JsonLTablesQADataset(Dataset):
    def __init__(
        self,
        file: str,
        is_train_set: bool,
        selector: DictConfig = None,
        shuffle_positives: bool = False,
        max_negatives: int = 1,
        seed: int = 0,
        max_len=100,
        split_type: str = "type1",
    ):
        super().__init__(selector, shuffle_positives=shuffle_positives)
        self.data_files = glob.glob(file)
        self.data = []
        self.is_train_set = is_train_set
        self.max_negatives = max_negatives
        self.rnd = random.Random(seed)
        self.max_len = max_len
        self.linearize_func = JsonLTablesQADataset.get_lin_func(split_type)

    def load_data(self):
        data = []
        for path in self.data_files:
            with jsonlines.open(path, mode="r") as jsonl_reader:
                data += [jline for jline in jsonl_reader]

        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample["question"]
        positive_ctxs = json_sample["positive_ctxs"]
        hard_negative_ctxs = json_sample["hard_negative_ctxs"]

        if self.shuffle_positives:
            self.rnd.shuffle(positive_ctxs)

        if self.is_train_set:
            self.rnd.shuffle(hard_negative_ctxs)
        positive_ctxs = positive_ctxs[0:1]
        hard_negative_ctxs = hard_negative_ctxs[0 : self.max_negatives]

        r.positive_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, True), ctx["caption"])
            for ctx in positive_ctxs
        ]
        r.negative_passages = []
        r.hard_negative_passages = [
            BiEncoderPassage(self.linearize_func(self, ctx, False), ctx["caption"])
            for ctx in hard_negative_ctxs
        ]
        return r

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_lin_func(cls, split_type: str):
        f = {
            "type1": JsonLTablesQADataset._linearize_table,
        }
        return f[split_type]

    @classmethod
    def split_table(cls, t: dict, max_length: int):
        rows = t["rows"]
        header = None
        header_len = 0
        start_row = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                header = row_lin
                header_len += row_len
                start_row = i
                break

        chunks = []
        current_rows = [header]
        current_len = header_len

        for i in range(start_row + 1, len(rows)):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                current_rows.append(row_lin)
                current_len += row_len
            if current_len >= max_length:
                # linearize chunk
                linearized_str = "\n".join(current_rows) + "\n"
                chunks.append(linearized_str)
                current_rows = [header]
                current_len = header_len

        if len(current_rows) > 1:
            linearized_str = "\n".join(current_rows) + "\n"
            chunks.append(linearized_str)
        return chunks

    def _linearize_table(self, t: dict, is_positive: bool) -> str:
        rows = t["rows"]
        selected_rows = set()
        rows_linearized = []
        total_words_len = 0

        # get the first non empty row as the "header"
        for i, r in enumerate(rows):
            row_lin, row_len = JsonLTablesQADataset._linearize_row(r)
            if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                selected_rows.add(i)
                rows_linearized.append(row_lin)
                total_words_len += row_len
                break

        # split to chunks
        if is_positive:
            row_idx_with_answers = [ap[0] for ap in t["answer_pos"]]

            if self.shuffle_positives:
                self.rnd.shuffle(row_idx_with_answers)
            for i in row_idx_with_answers:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
                    selected_rows.add(i)
                    rows_linearized.append(row_lin)
                    total_words_len += row_len
                if total_words_len >= self.max_len:
                    break

        if total_words_len < self.max_len:  # append random rows

            if self.is_train_set:
                rows_indexes = np.random.permutation(range(len(rows)))
            else:
                rows_indexes = [*range(len(rows))]

            for i in rows_indexes:
                if i not in selected_rows:
                    row_lin, row_len = JsonLTablesQADataset._linearize_row(rows[i])
                    if len(row_lin) > 1:  # TODO: change to checking cell value tokens
                        selected_rows.add(i)
                        rows_linearized.append(row_lin)
                        total_words_len += row_len
                    if total_words_len >= self.max_len:
                        break

        linearized_str = ""
        for r in rows_linearized:
            linearized_str += r + "\n"

        return linearized_str

    @classmethod
    def _linearize_row(cls, row: dict) -> Tuple[str, int]:
        cell_values = [c["value"] for c in row["columns"]]
        total_words = sum(len(c.split(" ")) for c in cell_values)
        return ", ".join([c["value"] for c in row["columns"]]), total_words


def split_tables_to_chunks(
    tables_dict: Dict[str, Table], max_table_len: int, split_type: str = "type1"
) -> List[Tuple[int, str, str, int]]:
    tables_as_dicts = [t.to_dpr_json() for k, t in tables_dict.items()]
    chunks = []
    chunk_id = 0
    for i, t in enumerate(tables_as_dicts):
        # TODO: support other types
        assert split_type == "type1"
        table_chunks = JsonLTablesQADataset.split_table(t, max_table_len)
        title = t["caption"]
        for c in table_chunks:
            # chunk id , text, title, external_id
            chunks.append((chunk_id, c, title, i))
            chunk_id += 1
        if i % 1000 == 0:
            logger.info("Splitted %d tables to %d chunks", i, len(chunks))
    return chunks

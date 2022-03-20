#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""

import json
import math
import pickle
import random
import logging
import itertools
from typing import List, Iterator, Callable, Dict, Set

import numpy as np
import torch
from torch import Tensor as T

from dpr.data.data_types import DataPassage, DataSample


logger = logging.getLogger()


class DataPassageCompressor:
    """
    Compress / decompress DataSample / DataPassage objects.
    """

    @staticmethod
    def convert_to_json(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        elif isinstance(obj, (tuple, list)):
            # Convert
            list_of_subobjs = [
                DataPassageCompressor.convert_to_json(i) for i in obj]

            # Compress
            if len(obj) > 0 and isinstance(obj[0], (DataPassage, DataSample)):
                keys = [tuple(i.keys()) for i in list_of_subobjs]
                assert len(set(keys)) == 1  # all obj must be of the same type
                compressed_obj = {key: [] for key in keys[0]}
                for subobj in list_of_subobjs:
                    for key, value in subobj.items():
                        compressed_obj[key].append(value)
                assert "compressed" not in compressed_obj
                compressed_obj["compressed"] = {
                    "num_objects": len(obj),
                    "keys": keys[0],
                }  # marker
                return compressed_obj
            else:
                return type(obj)(list_of_subobjs)

        elif isinstance(obj, dict):
            return {
                key: DataPassageCompressor.convert_to_json(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, (DataPassage, DataSample)):
            t = obj.__class__.__name__
            obj_dict = obj.__dict__.copy()

            # Remove container; this attribute can be reconstructed later
            if isinstance(obj, DataSample):
                obj_dict.pop("container")

            # Add marker
            assert "type" not in obj_dict
            obj_dict["type"] = t
            return DataPassageCompressor.convert_to_json(obj_dict)

        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise NotImplementedError(f"Unsupported object type: {type(obj)}")

    @staticmethod
    def convert_from_json(obj):
        if isinstance(obj, (int, float, str, bool, type(None), np.ndarray)):
            return obj

        elif isinstance(obj, (tuple, list)):
            # np array
            if len(obj) > 0 and all(isinstance(i, (int, float)) for i in obj):
                return np.array(obj)
            else:
                list_of_subobjs = [
                    DataPassageCompressor.convert_from_json(i) for i in obj]
                return type(obj)(list_of_subobjs)

        elif isinstance(obj, dict):
            # De-compress
            if "compressed" in obj:
                compressed_info = obj.pop("compressed")
                num_objects = compressed_info["num_objects"]
                keys = compressed_info["keys"]
                list_of_subobjs = []

                for i in range(num_objects):
                    subobj = {key: obj[key][i] for key in keys}
                    subobj = DataPassageCompressor.convert_from_json(subobj)
                    list_of_subobjs.append(subobj)
                return list_of_subobjs

            elif "type" in obj:
                obj_type = obj.pop("type")
                obj = {
                    key: DataPassageCompressor.convert_from_json(value)
                    for key, value in obj.items()
                }

                # DataPassage requires a special kind of initialization
                if obj_type == "DataPassage":
                    obj_ = DataPassage(id=-1)
                    for key, value in obj.items():
                        setattr(obj_, key, value)
                    obj = obj_
                # DataSample requires an additional `container` attribute
                elif obj_type == "DataSample":
                    obj = DataSample(**obj)
                    obj.container = [
                        obj.gold_passages,
                        obj.positive_passages,
                        obj.distantly_positive_passages,
                        obj.negative_passages,
                        obj.bm25_positive_passages,
                        obj.bm25_distantly_positive_passages,
                        obj.bm25_negative_passages,
                    ]
                return obj

            else:
                return {
                    key: DataPassageCompressor.convert_from_json(value)
                    for key, value in obj.items()
                }

        else:
            raise NotImplementedError(f"Unsupported object type: {type(obj)}")


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for path in paths:
        logger.info("Reading file %s", path)
        with open(path, "rb") as reader:
            data = pickle.load(reader)
        results.extend(data)
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for path in paths:
        logger.info("Reading file %s" % path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.extend(data)
    logger.info("Total data size: {}".format(len(results)))
    return results


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every
    node should handle its own part of the data.
    Instead of cutting data shards by their min size, it sets the amount of
    iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations
    (might be useful for DP mode).
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
    ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(
            self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            f"samples_per_shard={samples_per_shard}, "
            f"shard_start_idx={self.shard_start_idx}, "
            f"shard_end_idx={self.shard_end_idx}, "
            f"max_iterations={self.max_iterations}"
        )

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        return self.max_iterations

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for i in range(len(self.data)):
            visitor_func(self.data[i])

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.data)))
        if self.shuffle:
            # To be able to resume, same shuffling should be used when starting
            # from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    # TODO: merge with iterate_ds_sampled_data
    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        # If resuming iteration somewhere in the middle of epoch, one needs to
        # adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(
            self.iteration * self.batch_size,
            len(shard_indices),
            self.batch_size,
        ):
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0:self.batch_size - len(items)])
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        # Some shards may done iterating while the others are at the last batch.
        # Just return the first batch
        while self.iteration < max_iterations:
            logger.debug("Fulfilling non complete shard=".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0 : self.batch_size]
            items = [self.data[idx] for idx in items_idxs]
            yield items

        logger.info(
            "Finished iterating, iteration={}, shard={}".format(
                self.iteration, self.shard_id
            )
        )
        # reset the iteration status
        self.iteration = 0

    def iterate_ds_sampled_data(
        self, num_iterations: int, epoch: int = 0
    ) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        logger.info(
            "Finished iterating, iteration={}, shard={}".format(
                self.iteration, self.shard_id
            )
        )
        # TODO: reset the iteration status?
        self.iteration = 0

    def get_dataset(self) -> torch.utils.data.Dataset:
        return self.data


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility
    methods.
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional
    # title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ) -> T:
        raise NotImplementedError

    def to_max_length(
        self,
        token_ids: np.ndarray,
        apply_max_len: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def concatenate_inputs(
        self,
        ids: Dict[str, List[int]],
        get_passage_offset: bool = False,
        to_max_length: bool = False,
    ) -> T:
        """
        Concatenate inputs for either retriever or reader model.
        """
        raise NotImplementedError

    def unconcatenate_inputs(
        self,
        ids: T,
        components: Set[str],
    ) -> Dict[str, T]:
        """
        Split concatenated input to its components. (inputs and outputs are
        reversed of `concatenate_inputs`)
        """
        raise NotImplementedError

    def tensor_to_text(
        self,
        tensor: T
    ) -> str:
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class ShardedDataStreamIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every
    node should handle its own part of the data stream. This class aims at huge
    data stream that cannot be fit into memory at the beginning.
    """

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        offset: int = 0,
    ):

        self.data = data
        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)
        self.batch_size = batch_size
        self.iteration = 0
        self.offset = offset
        self.apply_func = None  # to be set later

        samples_per_shard = math.ceil(len(self.data) / self.shards_num)
        self.max_iterations = math.ceil(samples_per_shard / batch_size)

        logger.info(
            f"[{ShardedDataStreamIterator}] "
            f"samples_per_shard={samples_per_shard},"
            f" max_iterations={self.max_iterations}",
        )

    def set_offset(self, offset: int):
        """Lazy offset setter"""
        assert self.offset == 0
        self.offset = offset

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        assert self.apply_func is None, "`.apply()` should only be called once"
        self.apply_func = visitor_func

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        assert self.iteration == 0
        data_iterator = iter(self.data)

        # If offset > 0, virtually iterate over data for resumability
        if self.offset > 0:
            logger.info(
                f"Found offset={self.offset}. Virtually iterating over data "
                f"for as many iterations for resumability..."
            )

            # A workaround to make resuming much faster
            from dpr.data.general_data_preprocess import \
                GeneralDatasetPreprocessor
            GeneralDatasetPreprocessor._data_post_processing = False
            from dpr.data.general_data import GeneralDataset
            GeneralDataset._data_post_processing = False

            for _ in range(self.offset):
                self.iteration += 1
                try:
                    for _ in range(self.batch_size * self.shards_num):
                        next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.data)
                    self.iteration = 0

            self.offset = 0  # remove offset
            GeneralDatasetPreprocessor._data_post_processing = True
            GeneralDataset._data_post_processing = True

        while True:
            self.iteration += 1
            items = []

            try:
                # Construct batch
                for _ in range(self.batch_size):
                    for _ in range(self.shard_id):
                        next(data_iterator)
                    item = next(data_iterator)
                    if self.apply_func is not None:
                        self.apply_func(item)
                    items.append(item)
                    for _ in range(self.shard_id + 1, self.shards_num):
                        next(data_iterator)
                yield items

            except StopIteration:
                if len(items) > 0:
                    yield items

                logger.info(
                    f"Finished iterating, iteration={self.iteration}, "
                    f"shard={self.shard_id}"
                )
                # reset the iteration status
                self.iteration = 0
                return

    def get_dataset(self) -> torch.utils.data.Dataset:
        return self.data

    def get_max_iterations(self) -> int:
        return self.max_iterations

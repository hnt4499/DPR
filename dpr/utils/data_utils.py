#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import os
import json
import logging
import pickle
import random
import time

import itertools
import math

import numpy as np
import torch
from torch import Tensor as T
from typing import List, Iterator, Callable, Tuple, Dict

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results = data
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
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

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
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
        for sample in self.data:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.data)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    # TODO: merge with iterate_ds_sampled_data
    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(
            self.iteration * self.batch_size, len(shard_indices), self.batch_size
        ):
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
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


class ShardedDataIteratorWithCategories(ShardedDataIterator):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data, and where passages of the same batch should have the same category.
    Note that for now it is designed for `JsonQADataset` dataset. It might or might not work with other datasets.
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
        category_mapping: dict = None,  # for debugging purposes
    ):
        super(ShardedDataIteratorWithCategories, self).__init__(
            data,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
            strict_batch_size=strict_batch_size
        )

        # Read sampled indices
        self.sampled_idxs_path = data.sampled_idxs_path
        if self.sampled_idxs_path is not None:
            logger.info(f"Sampled indices found at {os.path.realpath(self.sampled_idxs_path)}")

        # Read category mapping
        self.category_mapping_path = data.category_mapping_path
        if category_mapping is not None:
            self.category_mapping = category_mapping
        elif self.sampled_idxs_path is None:
            logger.info(f"Reading category mapping at {os.path.realpath(self.category_mapping_path)}....")
            with open(self.category_mapping_path, "r") as fin:
                self.category_mapping = json.load(fin)

        # Build a mapping that maps an index to wiki index and vice versa
        # Note that we always take the first positive passage for each sample
        if self.sampled_idxs_path is None:
            logger.info("Building wiki indexing...")
            self.wiki_mapping = {"idx2wiki": {}, "wiki2idx": {}}
            for idx in range(len(self.data)):
                key = "passage_id" if "passage_id" in self.data.data[idx]["positive_ctxs"][0] else "id"
                wiki_idx = int(self.data.data[idx]["positive_ctxs"][0][key])

                self.wiki_mapping["idx2wiki"][idx] = wiki_idx
                if wiki_idx not in self.wiki_mapping["wiki2idx"]:
                    self.wiki_mapping["wiki2idx"][wiki_idx] = []
                self.wiki_mapping["wiki2idx"][wiki_idx].append(idx)  # one passage can map to multiple indices

    def get_shard_indices(
        self, epoch: int,
        return_all_indices: bool=False,  # for pre-generating indices only
    ):
        indices = list(range(len(self.data)))

        if self.shuffle:
            if self.sampled_idxs_path is not None:
                with open(self.sampled_idxs_path.format(epoch), "r") as fin:
                    indices = json.load(fin)
            else:
                # Sample if not exist
                new_indices = [-1] * len(self.data)
                not_sampled = np.ones(shape=(len(self.data),), dtype="int8")  # whether an i-th sample is not sampled
                epoch_rnd = random.Random(self.shuffle_seed + epoch)  # to be resumable and sync
                np_epoch_rnd = np.random.RandomState(seed=self.shuffle_seed + epoch)  # to be resumable and sync

                samples_per_shard = math.ceil(len(self.data) / self.shards_num)
                batches_per_shard = math.ceil(samples_per_shard / self.batch_size)

                logger.info("Sampling indicies such that samples in the same batch have the same category. This may take a while...")
                start = time.time()

                # Log
                log_interval = 50
                curr_interval = 0
                total_intervals = batches_per_shard * self.shards_num

                # Iterate from batch to batch first
                for batch_i in range(batches_per_shard):
                    # Iterate from shard to shard later to ensure equality
                    for shard_i in range(self.shards_num):
                        batch_start = (shard_i * samples_per_shard + batch_i * self.batch_size)
                        batch_end = min(batch_start + self.batch_size, (shard_i + 1) * samples_per_shard, len(self.data))

                        # First sample a random sample
                        not_sampled_idxs = not_sampled.nonzero()[0]  # sample indices that are not sampled
                        first_sample = np_epoch_rnd.choice(not_sampled_idxs)
                        new_indices[batch_start] = int(first_sample)
                        not_sampled[first_sample] = 0

                        # Sample samples whose category matches that of the first sample in the batch
                        first_sample_wiki_idx = str(self.wiki_mapping["idx2wiki"][first_sample])  # weirdly key in json file is str instead of int
                        for idx in range(batch_start + 1, batch_end):

                            # Sample a random level
                            level = epoch_rnd.choice(["l1", "l2", "l3"])
                            cat_i = self.category_mapping[level]["id2cls"][first_sample_wiki_idx]  # category of the first sample

                            # Get samples with the same level category as the first sample in the batch
                            samples_with_same_cat_wiki_idx = self.category_mapping[level]["cls2ids"][str(cat_i)]
                            samples_with_same_cat_idx = []
                            for wiki_idx in samples_with_same_cat_wiki_idx:
                                if wiki_idx in self.wiki_mapping["wiki2idx"]:
                                    idxs = self.wiki_mapping["wiki2idx"][wiki_idx]  # candidate indices
                                    idxs = [i for i in idxs if not_sampled[i] == 1]  # keep those not sampled
                                    samples_with_same_cat_idx.extend(idxs)

                            to_sample = samples_with_same_cat_idx
                            if len(to_sample) == 0:  # if there is no sample with the same cat, sample randomly one from the non-sampled pool
                                to_sample = not_sampled.nonzero()[0]
                            # Sample a random sample from a list of candidate samples
                            sample_i = np_epoch_rnd.choice(to_sample)
                            new_indices[idx] = int(sample_i)
                            not_sampled[sample_i] = 0

                        # Log
                        if curr_interval % log_interval == 0:
                            logger.info(f"Sampling: {curr_interval}/{total_intervals}")
                        curr_interval += 1

                indices = new_indices
                time_elapsed = time.time() - start
                logger.info(f"Sampling took {time_elapsed:.2f}s")

        # Check
        assert sorted(indices) == list(range(len(self.data)))

        if not return_all_indices:
            indices = indices[self.shard_start_idx : self.shard_end_idx]
        return indices


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        sampling_rates: List = [],
        rank: int = 0,
    ):
        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info("rank=%d; Multi set data sizes %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            self.max_its_pr_ds = [
                int(ds.max_iterations_num() * sampling_rates[i])
                for i, ds in enumerate(datasets)
            ]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)
        logger.info(
            "rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds
        )
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        logger.info("rank=%d; Iteration start", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set: %s",
            self.rank,
            [it.get_iteration() for it in self.iterables],
        )

        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            iterators.append(
                self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch)
            )

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        logger.info(
            "rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices)
        )
        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning(
                    "rank=%d; Next item in the source %s is None", self.rank, source_idx
                )

        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        logger.info(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def get_dataset(self, ds_id: int) -> torch.utils.data.Dataset:
        return self.iterables[ds_id].get_dataset()

    def get_datasets(self) -> List[torch.utils.data.Dataset]:
        return [it.get_dataset() for it in self.iterables]


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError
    
    def concatenate_inputs(
        self,
        ids: Dict[str, List[int]],
        get_passage_offset: bool = False,
    ) -> T:
        """
        Concatenate inputs for either retriever or reader model.
        """
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

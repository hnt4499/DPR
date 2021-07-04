# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import random
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor as T
from torch.utils.checkpoint import checkpoint
import numpy as np

from transformers.modeling_t5 import (
    T5ForConditionalGeneration,
    T5Model,
    T5Config,
    T5Stack,
    T5Block,
)
from transformers import T5Tokenizer

from ..data.general_data import TokenizedWikipediaPassages
from ..data.data_types import (
    GenerativeReaderPassage,
    GenerativeReaderSample,
    GenerativeReaderBatch,
)


logger = logging.getLogger(__name__)


class FiDT5(T5ForConditionalGeneration):
    def __init__(
        self,
        config,
        num_passages: int,
        device: str,
        gradient_checkpointing: bool = True,
    ):
        super(FiDT5, self).__init__(config)
        self.num_passages = num_passages
        self._device = device
        self.gradient_checkpointing = gradient_checkpointing

    @classmethod
    def init_model(
        cls,
        cfg_name: str,
        num_passages: int,
        device: str,
        dropout: float = 0.1,
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
        **kwargs,
    ) -> T5Model:
        """
        Main interface to initialize a FiD model.

        Parameters
        ----------
        num_passages : int
            Number of top-k passages for the reader to consider (i.e., read) for
            each example.
        gradient_checkpointing : bool
            Whether to enable gradient checkpointing.
        """
        cfg = T5Config.from_pretrained(cfg_name, **kwargs)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            model = cls.from_pretrained(
                cfg_name,
                config=cfg,
                num_passages=num_passages,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            model = cls(
                cfg,
                num_passages=num_passages,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
            )

        # Wrap after initialized
        model.lazy_wrap()
        return model

    def set_num_passages(self, num_passages: int):
        self.num_passages = num_passages
        self.encoder.set_num_passages(num_passages)

    @property
    def is_wrapped(self):
        return isinstance(self.encoder, FiDT5Encoder)

    def lazy_wrap(self):
        """
        Wraps only the encoder such that:
        1. Input tensors are properly handled specific for FiD model (i.e., each
        top-k passage is processed independently).
        2. Gradient checkpointing is integrated into appropriate modules, since
        HF doesn't provide this functionality for T5 yet.
        """
        if self.is_wrapped:
            raise RuntimeError("Lazy wrapping should only be called once.")

        # Wrap encoder with gradient checkpointing and custom forward pass logic
        self.encoder = FiDT5Encoder(
            self.encoder,
            num_passages=self.num_passages,
            device=self._device,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # Wrap decoder with gradient checkpointing
        self.decoder = wrap_gradient_checkpointing(
            self.decoder,
            device=self._device,
            stack_level=True,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def forward(self, input_ids: T = None, attention_mask: T = None, **kwargs):
        """
        We need to resize the input ids and attention mask as (B, N * L) instead
        of (B * N, L) here because the T5 forward method uses the input tensors'
        shape to infer dimensions used in the decoder.
        Note that in EncoderWrapper, the input ids and attention mask are re-resized
        as (B * N, L).
        """
        if input_ids is not None and input_ids.ndim == 3:
            assert input_ids.shape[1] == self.num_passages
            assert attention_mask.shape[1] == self.num_passages

        if input_ids is not None:
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super(FiDT5, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def generate(self, input_ids: T, attention_mask: T, max_length: int):
        assert input_ids.ndim == 3 and input_ids.shape[1] == self.num_passages
        assert attention_mask.ndim == 3 and attention_mask.shape[1] == self.num_passages

        # We need to resize the inputs here, as the generate method expect 2D tensors
        return super(FiDT5, self).generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )


class FiDT5Encoder(nn.Module):
    """
    Wrapper for T5 encoder to obtain a Fusion-in-Decoder model.

    Wraps only the encoder such that:
    1. Input tensors are properly handled specific for FiD model (i.e., each
    top-k passage is processed independently).
    2. Gradient checkpointing is integrated into appropriate modules, since
    HF doesn't provide this functionality for T5 yet.

    Parameters
    ----------
    num_passages : int
        Number of top-k passages for the reader to consider (i.e., read) for
        each example.
    gradient_checkpointing : bool
        Whether to enable gradient checkpointing.
    """
    def __init__(
        self,
        encoder: T5Stack,
        num_passages: int,
        device: str,
        gradient_checkpointing: bool = True,
    ):
        super(FiDT5Encoder, self).__init__()
        self.num_passages = num_passages
        self.encoder = encoder

        # Wrap each block in the T5 encoder with gradient checkpointing
        wrap_gradient_checkpointing(
            encoder,
            device=device,
            stack_level=False,
            gradient_checkpointing=gradient_checkpointing,
        )

    def set_num_passages(self, num_passages: int):
        self.num_passages = num_passages

    def forward(self, input_ids: T, attention_mask: T, **kwargs):
        """
        Forward pass logic, which takes care of processing input passages
        independently. Because of the forward pass of `FiDT5`, the inputs
        (B, N * L) (to comply with T5). We hence need to reshape it the
        other way round (i.e., (B * N, L)), so that each passage is
        processed independently.
        """
        # Sanity check
        batch_size, total_length = input_ids.shape
        assert (total_length % self.num_passages) == 0
        passage_length = total_length // self.num_passages

        input_ids = input_ids.view(-1, passage_length)
        attention_mask = attention_mask.view(-1, passage_length)

        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (
            outputs[0].view(batch_size, self.num_passages * passage_length, -1),
        ) + outputs[1:]
        return outputs


def _flatten(inputs, flattened_inputs: list, structure: list):
    """
    Helper for the `flatten` function.
    """
    if inputs is None or isinstance(inputs, T):
        flattened_inputs.append(inputs)
        structure.append(-1)

    elif isinstance(inputs, (list, tuple)):
        sub_structure = []
        for input in inputs:
            _flatten(input, flattened_inputs, sub_structure)
        structure.append(tuple(sub_structure))

    else:
        raise TypeError(f"Unrecognized input type: {type(inputs)}")


def flatten(inputs):
    """"
    Recursively traverse the inputs and flatten it into a list of tensors.
    Also returns a list of structed objects that helps to recover the
    original inputs.

    Parameters
    ----------
    inputs
        Input to flatten. Recursively speaking, sub-object of this
        `inputs` (including `inputs` itself) must be either a None,
        a tensor or a tuple/list of these types.

    Returns
    -------
        A tuple of (flattened_inputs, structure), where the structure
        object could be used to recover the original structured inputs.
    """
    flattened_inputs = []
    structure = []
    _flatten(inputs, flattened_inputs, structure)
    return tuple(flattened_inputs), structure[0]


def _unflatten(inputs, unflattened_inputs: list, structure, index: int = 0):
    """
    Helper for the `unflatten` function.
    """
    if isinstance(structure, int):
        unflattened_inputs.append(inputs[index])
        return index + 1
    elif isinstance(structure, (list, tuple)):
        unflattened_input = []

        for sub_structure in structure:
            index = _unflatten(inputs, unflattened_input, sub_structure, index)
        unflattened_inputs.append(tuple(unflattened_input))
        return index
    else:
        raise TypeError(f"Unrecognized length object type: {type(structure)}")


def unflatten(inputs, structure: list):
    """
    Recover the flatten inputs obtained from `flatten` function.
    """
    unflatten_inputs = []
    _unflatten(inputs, unflatten_inputs, structure)
    return unflatten_inputs[0]


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing. This is needed since checkpointing requires output to be
    a tensor or a tuple of tensors.
    """
    def __init__(
        self,
        module: T5Block,
        device: str,
        gradient_checkpointing: bool = True,
    ):
        super(CheckpointWrapper, self).__init__()
        self.module = module
        self._device = device
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, *args, **kwargs):
        """
        Forward pass with gradient checkpointing.
        """
        if self.gradient_checkpointing and self.training:

            def get_empty_tensor():
                """Helper function to get empty tensor with gradients"""
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=self._device,
                    requires_grad=True,
                )
                return empty

            structure = []  # workaround

            def custom_forward(*_args):
                output: List[Union[T, Tuple[T]]] = self.module(*_args, **kwargs)

                # Flatten output
                output, structure_ = flatten(output)
                structure.append(structure_)

                # Replace None with a trainable tensor
                output = [
                    output_i if output_i is not None else get_empty_tensor()
                    for output_i in output
                ]
                return tuple(output)

            output: List[T] = checkpoint(
                custom_forward,
                *args,
            )

            # Unwrap outputs
            output = [
                output_i if output_i.numel() > 0 else None
                for output_i in list(output)
            ]
            output = unflatten(output, structure[0])

        else:
            output = self.module(*args, **kwargs)
        return output


def wrap_gradient_checkpointing(
    module: T5Stack,
    device: str,
    stack_level: bool,
    gradient_checkpointing: bool = True,
):
    """
    Wrap a T5 block with gradient checkpointing. This function works with both
    T5 encoder and decoder.

    Parameters
    ----------
    stack_level : bool
        Whether to checkpoint at stack level or block level. If true, this
        function returns wrapped version of the stack. Otherwise, the
        wrapping is done in-place.
    """
    if stack_level:
        wrapped_module = CheckpointWrapper(
            module=module,
            device=device,
            gradient_checkpointing=gradient_checkpointing,
        )
        return wrapped_module
    else:
        wrapped_block = []
        for sub_module in module.block:
            wrapped_mod = CheckpointWrapper(
                module=sub_module,
                device=device,
                gradient_checkpointing=gradient_checkpointing
            )
            wrapped_block.append(wrapped_mod)
        module.block = nn.ModuleList(wrapped_block)


class FiDTensorizer:
    """
    Fusion-in-Decoder "Tensorizer".

    Note that in T5 there is no "special" tokens. Therefore, setting
    `add_special_tokens` won't have effect.
    """
    def __init__(
        self,
        cfg_name: str,
        context_max_length: int = 350,
        answer_max_length: int = 20,
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(cfg_name)
        self.context_max_length = context_max_length
        self.answer_max_length = answer_max_length

    def encode(
        self,
        string: str,
        max_length: int = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ):
        """
        Encode an abitrary string into ids.

        Parameters
        ----------
        max_length : int
            Max length of the encoded sequence. If not specified,
            `context_max_length` will be used instead.
        """
        if max_length is None and (padding or truncation):
            max_length = self.context_max_length

        ids = self.tokenizer.encode(
            string,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

        assert len(ids) == 1
        return ids[0]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    def to_tensor(self, *args: Union[np.ndarray, T]) -> T:
        """
        Convert input array(s) to torch tensor(s).
        """
        new_objs = []
        for obj in args:
            assert isinstance(obj, (np.ndarray, T))
            if isinstance(obj, np.ndarray):
                obj = torch.from_numpy(obj)
            new_objs.append(obj)

        if len(args) == 1:
            new_objs = new_objs[0]
        return new_objs

    def to_max_length(
        self,
        ids: Union[np.ndarray, T],
        max_length: int = None,
    ) -> T:
        """
        Pad or truncate an encoded ids to max length

        Parameters
        ----------
        max_length : int
            Max length of the encoded sequence. If not specified,
            `context_max_length` will be used instead.
        """
        ids = self.to_tensor(ids)
        if max_length is None:
            max_length = self.context_max_length

        if len(ids) > max_length:
            ids = ids[:max_length]
        elif len(ids) < max_length:
            to_pad = [self.pad_token_id] * (max_length - len(ids))
            ids = torch.cat([
                ids,
                torch.tensor(to_pad).to(ids),
            ], dim=0)

        assert len(ids) == max_length
        return ids

    def encode_answer(self, answer: str) -> T:
        """
        Encoder answer (target) string into token ids.
        """
        answer_ids = self.encode(
            answer + " </s>",
            max_length=self.answer_max_length,
        )
        return answer_ids

    def concatenate_answer_ids(self, answer_ids: Union[np.ndarray, T]) -> T:
        """
        Concatenate answer (target) ids with appropriate special tokens/prompts.
        """
        answer_ids = self.to_tensor(answer_ids)
        answer_ids = torch.cat([
            answer_ids,
            self.encode(" </s>", padding=False, truncation=False),
        ])
        answer_ids = self.to_max_length(answer_ids, max_length=self.answer_max_length)
        return answer_ids

    def encode_question_and_passage_pair(
        self,
        question: str,
        passage_title: str,
        passage: str,
    ) -> T:
        """
        Encode a question-passage_title-passage pair into token ids.
        """
        concat_str = f"question: {question}, title: {passage_title}, context: {passage}"
        concat_ids = self.encode(
            concat_str,
            max_length=self.context_max_length,
        )
        return concat_ids

    def concatenate_question_and_passage_ids_pair(
        self,
        question: Union[np.ndarray, T],
        passage_title: Union[np.ndarray, T],
        passage: Union[np.ndarray, T],
    ) -> T:
        """
        Concatenate a question-passage_title-passage ids pair with appropriate special
        tokens/prompts.
        """
        question, passage_title, passage = self.to_tensor(
            question, passage_title, passage
        )
        concat_ids = torch.cat([
            self.encode("question: ", padding=False, truncation=False),
            question,
            self.encode("title: ", padding=False, truncation=False),
            passage_title,
            self.encode("context: ", padding=False, truncation=False),
            passage,
        ])
        concat_ids = self.to_max_length(concat_ids, max_length=self.context_max_length)
        return concat_ids

    def get_attn_mask(self, tokens_tensor: T) -> T:
        if tokens_tensor is None:
            return None
        return tokens_tensor != self.pad_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    # Below are series of functions that are defined for backward compatibility
    # only. These functions should **only** be called in the preprocessing step.
    def set_pad_to_max(self, pad: bool):
        pass

    def text_to_tensor(self, text, add_special_tokens: bool = False):
        # Note that there is no "special tokens" for T5
        return self.encode(text, padding=False, truncation=True)

    def tensor_to_text(self, tensor, skip_special_tokens: bool = True):
        return self.decode(tensor, skip_special_tokens)


def create_generative_reader_input(
    wiki_data: TokenizedWikipediaPassages,
    tensorizer: FiDTensorizer,
    samples: List[GenerativeReaderSample],
    passages_per_question: int,
    max_length: int,
    is_train: bool,
    shuffle: bool,
) -> GenerativeReaderBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s. This is compatible with `GeneralDataset`.
    :param wiki_data: all tokenized wikipedia passages
    :param tensorizer: initialized tensorizer (which contains the tokenizer)
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a batch
    :param max_length: max model input sequence length
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: ReaderBatch instance
    """
    context_IDs = []
    input_ids = []
    answer_ids = []

    empty_sequence = torch.Tensor().new_full(
        (max_length,), tensorizer.pad_token_id, dtype=torch.long
    )

    for sample in samples:
        # Get answer from one of the positive passage; for training only
        if is_train:
            sample_answer = random.choice(sample.answers)
            sample_answer_ids = tensorizer.encode_answer(sample_answer)
        else:
            sample_answer_ids = None
        # Question IDs
        question_token_ids = sample.question_token_ids

        # Prepare contexts
        ctxs = sample.positive_passages + sample.negative_passages
        if is_train and shuffle:
            # Shuffle contexts when training
            random.shuffle(ctxs)
        else:
            # Sort contexts when testing, since we are only allowed to
            # use top-k retrieved samples
            ctxs = sorted(ctxs, key=lambda x: x.score, reverse=True)

        sample_tensors = _create_question_passages_tensors(
            wiki_data,
            question_token_ids,
            tensorizer,
            ctxs,
            passages_per_question,
            empty_sequence,
        )
        context_ID, sample_input_ids = sample_tensors

        context_IDs.append(context_ID)
        input_ids.append(sample_input_ids)
        answer_ids.append(sample_answer_ids)

    context_IDs = torch.stack(context_IDs, dim=0)  # (N, M)
    input_ids = torch.stack(input_ids, dim=0)  # (N, M)
    if is_train:
        answer_ids = torch.stack(answer_ids, dim=0)  # (N, M_short)

    assert len(context_IDs) == len(input_ids) == len(answer_ids)
    return GenerativeReaderBatch(context_IDs, input_ids, answer_ids)


def _create_question_passages_tensors(
    wiki_data: TokenizedWikipediaPassages,
    question_token_ids: np.ndarray,
    tensorizer: FiDTensorizer,
    ctxs: List[GenerativeReaderPassage],
    total_size: int,
    empty_ids: T,
):
    context_IDs: List[int] = []
    context_selected = []

    for context in ctxs[:total_size]:

        if getattr(context, "sequence_ids", None) is None:
            # Load in passage tokens and title tokens
            context.load_tokens(
                question_token_ids=question_token_ids,
                **wiki_data.get_tokenized_data(int(context.id))
            )
            # Concatenate input tokens
            sequence_ids = tensorizer.concatenate_question_and_passage_ids_pair(
                question=context.question_token_ids,
                passage_title=context.title_token_ids,
                passage=context.passage_token_ids,
            )
            context.sequence_ids = sequence_ids

        context_IDs.append(context.id)
        context_selected.append(context.sequence_ids)

    while len(context_selected) < total_size:
        context_IDs.append(-1)
        context_selected.append(empty_ids.clone())

    context_IDs = torch.tensor(context_IDs, dtype=torch.int64)
    input_ids = torch.stack(context_selected, dim=0)
    assert len(context_IDs) == len(input_ids)

    return context_IDs, input_ids
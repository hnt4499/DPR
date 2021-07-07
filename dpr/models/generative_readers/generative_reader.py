# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import copy
import inspect
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
    T5LayerNorm,
    T5LayerFF,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
)
from transformers import T5Tokenizer

from ...data.general_data import TokenizedWikipediaPassages
from ...data.data_types import (
    GenerativeReaderPassage,
    GenerativeReaderSample,
    GenerativeReaderBatch,
)


logger = logging.getLogger(__name__)


class CustomT5Block(T5Block):
    """
    Custom T5 block that allows the use of pre-initialized T5 layers.
    """
    def __init__(
        self,
        config,
        has_relative_attention_bias: bool = False,
        layers: nn.ModuleList = None,
    ):
        super(T5Block, self).__init__()  # note that call stack
        self.config = config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        if layers is None:
            self.layer = nn.ModuleList()
            self.layer.append(T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias
            ))
            if self.is_decoder:
                self.layer.append(T5LayerCrossAttention(
                    config,
                    has_relative_attention_bias=has_relative_attention_bias
                ))

            self.layer.append(T5LayerFF(config))
        else:
            self.layer = layers

    def to_encoder(self):
        """
        Return a new object that shares parameters with the current object,
        and with the cross attention layer removed. This function should
        only be called from a T5 block of a decoder (NOT encoder).
        """
        assert self.is_decoder, (
            "This function must be called from a block of a decoder"
        )

        layers = nn.ModuleList([
            self.layer[0],
            self.layer[2],
        ])  # remove the second layer
        config = copy.deepcopy(self.config)
        config.is_decoder = False

        return CustomT5Block(
            config,
            self.has_relative_attention_bias,
            layers=layers,
        )


class CustomT5Stack(T5Stack):
    """
    Custom T5 stack that allows the use of pre-initialized T5Blocks.
    """
    def __init__(
        self,
        config,
        embed_tokens: nn.Embedding = None,
        block: nn.ModuleList = None,
    ):
        super(T5Stack, self).__init__(config)  # note the call stack

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if block is None:
            self.block = nn.ModuleList(
                [CustomT5Block(config, has_relative_attention_bias=bool(i == 0))
                 for i in range(config.num_layers)]
            )
        else:
            self.block = block

        self.final_layer_norm = T5LayerNorm(  # we don't need to share this layer
            config.d_model,
            eps=config.layer_norm_epsilon,
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()


class FiDT5(T5ForConditionalGeneration):
    def __init__(
        self,
        config,
        num_passages: int,
        device: str,
        gradient_checkpointing: bool = True,
        share_encoder_decoder: bool = False,
    ):
        super(T5ForConditionalGeneration, self).__init__(config)  # note that call stack
        self.model_dim = config.d_model
        self.num_passages = num_passages
        self._device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.share_encoder_decoder = share_encoder_decoder

        # Embedding look up
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize the decoder first as it has more layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = CustomT5Stack(decoder_config, self.shared, block=None)

        # Initialize the encoder, with the option of parameter sharing
        if share_encoder_decoder:
            logger.info(
                f"[{self.__class__.__name__}] Sharing the weights of encoder "
                f"and decoder architectures"
            )
            block = self.decoder.block

            # Need to convert this block to be compatible with encoder module
            new_block = nn.ModuleList()
            for block_i in block:
                block_i: CustomT5Block
                block_i = block_i.to_encoder()
                new_block.append(block_i)
            block = new_block

        else:
            logger.info(
                f"[{self.__class__.__name__}] Weights of the encoder and decoder "
                f"are not shared!"
            )
            block = None

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = CustomT5Stack(encoder_config, self.shared, block=block)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()

    @classmethod
    def init_model(
        cls,
        cfg_name: str,
        num_passages: int,
        device: str,
        dropout: float = 0.1,
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
        share_encoder_decoder: bool = False,
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
        share_encoder_decoder : bool
            Whether to share the weights of the encoder and decoder components
            of the underlying T5 model.
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
                share_encoder_decoder=share_encoder_decoder,
            )
        else:
            model = cls(
                cfg,
                num_passages=num_passages,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
                share_encoder_decoder=share_encoder_decoder,
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
            stack_level=False,
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
        Forward pass logic, which takes care of processing input passages independently.
        Because of the forward pass of `FiDT5`, the inputs was resized to (B, N * L)
        (to comply with T5). We hence need to reshape it the other way round
        (i.e., (B * N, L)), so that each passage is processed independently.
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


def _flatten(inputs, flattened_inputs: list, structure: list, only_tensor: bool):
    """
    Helper for the `flatten` function.
    """
    if isinstance(inputs, (list, tuple)):
        sub_structure = []
        for input in inputs:
            _flatten(input, flattened_inputs, sub_structure, only_tensor)
        structure.append(tuple(sub_structure))

    elif (only_tensor and (inputs is None or isinstance(inputs, T))) or (not only_tensor):
        flattened_inputs.append(inputs)
        structure.append(-1)

    else:
        raise TypeError(f"Unrecognized input type: {type(inputs)}")


def flatten(inputs, only_tensor: bool):
    """"
    Recursively traverse the inputs and flatten it into a list of tensors.
    Also returns a list of structured objects that helps to recover the
    original inputs.

    Parameters
    ----------
    inputs
        Input to flatten. Recursively speaking, sub-object of this
        `inputs` (including `inputs` itself) should be either a None,
        a tensor or a tuple/list of these types. See `only_tensor`.
    only_tensor : bool
        If True, recursively check if every "leaf" object is either None
        or a torch tensor. Raise an error if an object of another type
        is found. This is useful for checking for outputs of `forward`
        functions.

    Returns
    -------
        A tuple of (flattened_inputs, structure), where the `structure`
        object could be used to recover the original structured inputs.
    """
    flattened_inputs = []
    structure = []
    _flatten(inputs, flattened_inputs, structure, only_tensor)
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


def convert_kwargs_to_args(func, args, kwargs, exclude_self: bool):
    """
    Inspect the input function `func` and convert positional arguments
    (`args`) and keyword arguments (`kwargs`) to a tuple of positional arguments
    (only `args`), while filling missing keyword arguments with their default values.

    Example:
    >>> def f(x, y=None, z=1, t=False):
    >>>     pass
    >>> convert_kwargs_to_args(f)  # raise TypeError, since `x` is a required argument
    >>> convert_kwargs_to_args(f, 1)  # (1, None, 1, False)
    >>> convert_kwargs_to_args(f, 1, t=True, y=[1, 2, 3])  # (1, [1, 2, 3], 1, True)

    Parameters
    ----------
    exclude_self : bool
        Whether to exclude `self` from the list of arguments. Useful for methods (not
        functions).
    """
    # Get function specs
    func_spec = inspect.getfullargspec(func)
    func_args = func_spec.args[1:] if exclude_self else func_spec.args
    func_defaults = func_spec.defaults  # kwargs' defaults
    func_defaults = [] if func_defaults is None else list(func_defaults)

    # Sanity check
    if len(args) < len(func_args) - len(func_defaults):
        num_missing = len(func_args) - len(func_defaults) - len(args)
        args_missing = func_args[len(args):len(args) + num_missing]
        args_missing = [
            arg_missing for arg_missing in args_missing
            if arg_missing not in kwargs
        ]

        if len(args_missing) > 0:
            raise TypeError(
                f"Missing {num_missing} required positional argument: {args_missing}"
            )

    # Process args and kwargs, merge them into args
    all_args = list(args).copy()
    # Collect all positional arguments that have been fed as keyword arguments
    while len(all_args) < len(func_args) - len(func_defaults):
        arg_name = func_args[len(all_args)]
        assert arg_name in kwargs  # this has been checked in the previous code
        value = kwargs.pop(arg_name)
        all_args.append(value)

    # Collect all keyword arguments in sequential order, filling missing values with default
    # values
    num_remain_args = len(func_args) - len(all_args)
    remain_args = func_args[-num_remain_args:]
    remain_defaults = func_defaults[-num_remain_args:]

    for remain_arg, remain_default in zip(remain_args, remain_defaults):
        if remain_arg in kwargs:
            value = kwargs.pop(remain_arg)
            all_args.append(value)
        else:
            all_args.append(remain_default)

    assert len(kwargs) == 0
    return tuple(all_args)


def is_tensor(object):
    """
    Recursively check whether the input object is a tensor or a tuple / list of tensor.
    Note that if one of the (sub-)objects is a tuple / list containing some tensors and
    some non-tensors, this function will return False, considering that object as a
    non-tensor object.
    """
    if isinstance(object, T):
        return True
    elif isinstance(object, (list, tuple)):
        is_tensors = [is_tensor(o) for o in object]
        return all(is_tensors)
    else:
        return False


def wrap_function(func, args, exclude_self: bool, _func=None):
    """
    This function pre-fills all non-tensor arguments with their provided values
    and return a new function (wrapped over the original function) that only
    accepts tensor arguments.

    Parameters
    ----------
    args : tuple
        Tuple of processed positional arguments returned by `convert_kwargs_to_args`.
    exclude_self : bool
        Whether to exclude `self` from the list of arguments. Useful for methods (not
        functions).
    _func
        Function to be wrapped over. Normal usage is that: `func` is the forward
        function and `_func` is the `__call__` function. If not provided, use
        `func` instead.

    Returns
    -------
    A tuple of (wrapped function, args), where args contains only tensor objects.
    Calling `func(args)` would return the desired results.
    """
    # Get function specs
    func_spec = inspect.getfullargspec(func)
    func_args = func_spec.args[1:] if exclude_self else func_spec.args
    if len(func_args) != len(args):
        raise ValueError(
            f"Arguments mismatch. Got {len(args)} arguments, while the function "
            f"requires {len(func_args)} arguments. Make sure to check "
            f"`exclude_self` option and that `args` is processed using the "
            f"`convert_kwargs_to_args` function."
        )

    # Process args and kwargs
    processed_arg_values = []  # tensor container
    processed_arg_names = []  # contains argument name of corresponding `processed_args`
    processed_kwargs = {}

    for arg_value, arg_name in zip(args, func_args):
        if is_tensor(arg_value):
            processed_arg_values.append(arg_value)
            processed_arg_names.append(arg_name)
        else:
            processed_kwargs[arg_name] = arg_value

    # Wrap function
    _func = _func if _func is not None else func
    def wrapper(*args):
        kwargs = processed_kwargs.copy()
        args_with_names = dict(zip(processed_arg_names, args))
        kwargs.update(args_with_names)
        return _func(**kwargs)

    return wrapper, processed_arg_values


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing. This is needed since checkpointing requires output to be
    a tensor or a tuple of tensors.
    """
    def __init__(
        self,
        module: CustomT5Block,
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
            # Convert all arguments into positional arguments; here we assume
            # that the function of interest is `self.module.forward`
            args = convert_kwargs_to_args(
                self.module.forward,
                args=args,
                kwargs=kwargs,
                exclude_self=True,
            )
            # Pre-fill the function with arguments that are not tensors
            wrapped_func, args = wrap_function(
                self.module.forward,
                args=args,
                exclude_self=True,
                _func=self.module.__call__,  # wrap over this function instead of `forward`
            )

            # Flatten the arguments so that all Tensors are present at the root level
            args, input_structure = flatten(args, only_tensor=False)
            output_structure = []  # workaround to get output structure from inside `custom_forward`

            def get_empty_tensor():
                """Helper function to get empty tensor with gradients"""
                empty = torch.tensor(
                    [],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=True,
                )
                return empty

            def custom_forward(dummy, *_args):
                # Unflatten the inputs
                _args = unflatten(_args, input_structure)

                # Forward
                output: List[Union[T, Tuple[T]]] = wrapped_func(*_args)

                # Flatten output
                output, output_structure_ = flatten(output, only_tensor=True)
                output_structure.append(output_structure_)

                # Replace None with a tensor that requires grad
                output = [
                    output_i if output_i is not None else get_empty_tensor()
                    for output_i in output
                ]
                return tuple(output)

            output: List[T] = checkpoint(
                custom_forward,
                get_empty_tensor(),  # need at least one tensor that requires grad
                *args,
            )

            # Unwrap outputs
            output = [
                output_i if output_i.numel() > 0 else None
                for output_i in list(output)
            ]
            output = unflatten(output, output_structure[0])

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
        return module


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
        answer_ids = self.to_max_length(answer_ids, max_length=self.answer_max_length)
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
        concat_ids = self.to_max_length(concat_ids, max_length=self.context_max_length)
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
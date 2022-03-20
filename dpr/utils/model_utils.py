#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import collections
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location


logger = logging.getLogger()


CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)


def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
    gradient_checkpointing: bool = False,
) -> Tuple[
    nn.Module,
    Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]
]:
    model.to(device)
    # Gradient checkpointing AND DDP
    if fp16 or (gradient_checkpointing and local_rank != -1):
        try:
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use fp16 training."
            )

        fp16_opt_level = fp16_opt_level if fp16 else "O0"
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        if gradient_checkpointing:
            model = apex.parallel.DistributedDataParallel(
                model,
                delay_allreduce=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device if device else local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

    return model, optimizer


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):

    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step)
            / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def init_weights(modules: List):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(
        model_file,
        map_location=lambda s, l: default_restore_location(s, "cpu")
    )
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)


def load_state_dict_to_model(
    model: nn.Module,
    state_dict: dict,
    strict: bool = False,
):
    model_keys = set(model.state_dict().keys())
    pretrained_model_keys = set(state_dict.keys())

    # Check validity
    keys_redundant = pretrained_model_keys - model_keys
    if len(keys_redundant) > 0:
        to_log = (
            f"Redundant keys detected in the pre-trained state dict: "
            f"{list(keys_redundant)}."
        )
        if strict:
            raise KeyError(to_log)
        else:
            logger.warn(to_log)

    keys_missing = model_keys - pretrained_model_keys
    if len(keys_missing) > 0:
        to_log = (
            f"Missing keys detected in the pretrained state dict: "
            f"{list(keys_missing)}."
        )
        if strict:
            raise KeyError(to_log)
        else:
            logger.warn(to_log)

    # Remove redundant params
    for key_redundant in keys_redundant:
        del state_dict[key_redundant]

    if len(state_dict) == 0:
        raise ValueError("No weight to load.")

    # Get parameters count
    param_count = 0
    for _, param in state_dict.items():
        param_count += param.numel()
    logger.info(f"Loading {param_count} parameters to the model...")

    # Load
    model.load_state_dict(state_dict, strict=strict)

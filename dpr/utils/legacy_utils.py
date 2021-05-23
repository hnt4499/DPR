import logging
from typing import Union, Tuple

import torch
from torch.serialization import default_restore_location

from .model_utils import CheckpointState, CheckpointStateOFA


logger = logging.getLogger()


def load_states_from_checkpoint_legacy(model_file: str) -> Union[CheckpointState, CheckpointStateOFA]:
    logger.info("Reading saved model from %s and auto infer its type", model_file)
    state_dict = torch.load(
        model_file, map_location=lambda s, l: default_restore_location(s, "cpu")
    )
    logger.info("model_state_dict keys %s", state_dict.keys())

    # Use simple heuristic
    if "optimizer_dict" in state_dict:
        return CheckpointState(**state_dict)
    else:
        return CheckpointStateOFA(**state_dict)


def convert_from_old_state_to_ofa(saved_state: CheckpointState) -> Tuple[CheckpointState, str]:
    """
    Convert a model dictionary from old format (either biencoder or reader) to new format
    (one-for-all model).
    """
    # Simple heuristic
    if any(k.startswith("question_model") for k in saved_state.model_dict.keys()):
        model_type = "biencoder"
    else:
        model_type = "reader"

    # Modify key
    new_model_dict = {}
    for key, value in saved_state.model_dict.items():
        new_model_dict[f"{model_type}.{key}"] = value

    saved_state = saved_state._replace(model_dict=new_model_dict)
    return saved_state, model_type

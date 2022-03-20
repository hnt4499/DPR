import os
import logging
from typing import List

from hydra import initialize, compose


logger = logging.getLogger(__name__)


def initialize_hydra_config(config_dir, config_name, overrides: List[str] = []):
    """
    Initialize hydra config for debugging.
    """
    # Convert `config_dir` to a relative path to the current utils file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.relpath(os.path.abspath(config_dir), start=curr_dir)
    # Initialize
    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

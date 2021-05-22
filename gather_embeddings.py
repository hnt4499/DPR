#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Small snippet to gather generated embeddings
"""

import os
import sys
from pathlib import Path
from shutil import move

import hydra

from omegaconf import DictConfig



@hydra.main(config_path="conf", config_name="gather_embeddings")
def main(cfg: DictConfig):
    os.makedirs(cfg.dst_dir, exist_ok=True)

    for src_path in Path(cfg.src_dir).rglob(cfg.pattern):
        _, filename = os.path.split(src_path)
        dst_path = os.path.join(cfg.dst_dir, filename)
        move(src_path, dst_path)
        print(f"Moved {os.path.relpath(src_path)} to {os.path.relpath(dst_path)}")


if __name__ == "__main__":
    hydra_formatted_args = []
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    main()

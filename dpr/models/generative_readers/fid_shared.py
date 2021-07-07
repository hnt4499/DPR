"""
FiD with shared encoder-decoder architecture to minimize memory footprint.
"""


from .fid_base import (
    get_generative_reader_components as base_fid_reader,
    get_generative_tensorizer,
)


__all__ = ["get_generative_reader_components", "get_generative_tensorizer"]


def get_generative_reader_components(*args, **kwargs):
    return base_fid_reader(*args, share_encoder_decoder=True, **kwargs)
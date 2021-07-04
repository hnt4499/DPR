from .generative_reader import FiDT5, FiDTensorizer
from .hf_models import get_optimizer


def get_generative_reader_components(
    cfg,
    num_passages: int,
    device: str,
    inference_only: bool = False,
    gradient_checkpointing: bool = True,
    **kwargs,
):
    dropout = getattr(cfg.encoder, "dropout", 0.0)
    reader = FiDT5.init_model(
        cfg.encoder.pretrained_model_cfg,
        num_passages=num_passages,
        device=device,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_generative_tensorizer(cfg)
    return tensorizer, reader, optimizer


def get_generative_tensorizer(cfg):
    context_max_length = cfg.encoder.context_max_length
    answer_max_length = cfg.encoder.answer_max_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    return FiDTensorizer(
        cfg_name=pretrained_model_cfg,
        context_max_length=context_max_length,
        answer_max_length=answer_max_length,
    )
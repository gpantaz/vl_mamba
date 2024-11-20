from typing import NamedTuple

import torch
from loguru import logger
from pytorch_lightning import LightningModule


class ParametersCounter(NamedTuple):
    """Counter for parameters."""

    total: int
    trainable: int
    non_trainable: int


def count_model_parameters(model: LightningModule) -> ParametersCounter:
    """Count paramters within the model."""
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    non_trainable = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )

    return ParametersCounter(
        trainable=trainable,
        non_trainable=non_trainable,
        total=trainable + non_trainable,
    )


def compute_trainable_params(model: torch.nn.Module) -> None:
    """Compute trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([p.numel() for p in model_parameters])
    logger.info(
        f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params"
    )

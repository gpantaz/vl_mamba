from typing import Any

import torch
from safetensors.torch import load_model
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file

from vl_mamba.utils.io import read_json


def load_config_hf(model_name: str) -> Any:
    """Load model configuration."""
    resolved_archive_file = cached_file(
        model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
    )
    return read_json(resolved_archive_file)


def load_state_dict_hf(
    model_name: str, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> Any:
    """Load model state dict."""
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in {torch.float32, None} else device
    resolved_archive_file = cached_file(
        model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
    )
    return torch.load(resolved_archive_file, map_location=mapped_device)


def load_safe_tensors_state_dict_hf(
    model_name: str, device: torch.device | None = None, dtype: torch.dtype | None = None
):
    """Load model state dict."""
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in {torch.float32, None} else device
    resolved_archive_file = cached_file(
        model_name, "model.safetensors", _raise_exceptions_for_missing_entries=False
    )
    return load_model(resolved_archive_file, map_location=mapped_device)

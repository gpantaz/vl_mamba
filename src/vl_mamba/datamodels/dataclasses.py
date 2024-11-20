from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class DatasetItemCollateFn:
    """Used to determine what to do in the collate function for element in an example."""

    input_ids = "pad"
    labels = "pad"
    attention_mask = "pad"
    pixel_values = "raw"
    task = "stack"
    raw_target = "raw"


@dataclass
class DatasetItem:
    """Output for the dataset reader."""

    input_ids: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    task: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    # Use raw_target for downstream tasks when needed to compute task-specific metrics
    raw_target: Optional[Any] = None


@dataclass
class VisualEncoding:
    """Output of the VisualProcessor."""

    input_ids: Optional[list[torch.Tensor]] = None
    image_patches: Optional[list[torch.Tensor]] = None
    # pixel_values: Optional[list[torch.Tensor]] = None
    images: Optional[list[torch.Tensor]] = None
    bboxes: Optional[list[list[torch.Tensor]]] = None
    bboxes_norm: Optional[list[list[torch.Tensor]]] = None
    image_scale_factors: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class TextEncoding:
    """Output of the ConversationProcessor."""
    # One element per utterance in each conversation
    input_ids: list[torch.Tensor]
    # One element per utterance in each conversation
    labels: Optional[list[torch.Tensor]] = None
    text: Optional[str] = None


@dataclass
class DatasetBatch:
    """Output for the dataset reader."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor
    task: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    raw_target: Optional[Any] = None

    def __len__(self) -> int:
        """Returns the batch size."""
        return self.input_ids.shape[0]


@dataclass
class DatasetPadding:
    """Padding values used by collate."""

    input_ids: int = 0
    pixel_values: int = 0
    attention_mask: int = 0
    labels: int = -100
    task: int = -1


@dataclass
class SpecialTokens:
    """Special tokens.

    These are the special tokens to mark the start/end of image/text tokens in the sequence.

    The format of the input sequence is:
    ### P11 P12 P13 && P21 P22 P23 ###INSTRUCTION PROMPT TARGET TEXT<eos>.
    """

    img_bos_token: str = "###"
    img_eos_token: str = "###"
    img_sep_token: str = "&&"

    text_bos_token: str = "##"
    text_eos_token: str = "<|endoftext|>"

    # Fake placeholder image token id, used to mask the image tokens when training.
    img_token_id: int = -200

    # tokenizer(img_bos_token, return_tensors="pt")["input_ids"][0] = 4118
    img_bos_token_id = 4118
    # tokenizer(img_eos_token, return_tensors="pt")["input_ids"][0] = 4118
    img_eos_token_id = 4118
    # tokenizer(img_sep_token, return_tensors="pt")["input_ids"][0] = 10494
    img_sep_token_id = 10494
    # tokenizer(text_bos_token, return_tensors="pt")["input_ids"][0] = 817
    text_bos_token_id = 817
    # tokenizer(text_eos_token, return_tensors="pt")["input_ids"][0] = 11167
    text_eos_token_id = 0

import dataclasses
from typing import Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from vl_mamba.datamodels.dataclasses import DatasetItem, DatasetItemCollateFn, DatasetPadding


def _pad_sequence(
    seq: list[torch.Tensor],
    padding_value: int,
    padding_side: str = "right",
) -> torch.Tensor:
    """Pad a sequence of tensors.

    IMPORTANT: Use padding_side="left" to pad on the left side when dealing with batch generation.
    """
    if not seq:
        return torch.empty(0)

    if padding_side == "right":
        return pad_sequence(seq, batch_first=True, padding_value=padding_value)
    rev_seq = [s.flip(0) for s in seq]
    rev_padded = pad_sequence(rev_seq, batch_first=True, padding_value=padding_value)
    return rev_padded.flip(-1)


@dataclasses.dataclass
class Collate:
    """Collate class.

    This is a class to ensure that the padding values are correctly passed when creating the batch.
    """

    padding: DatasetPadding
    padding_side: str = "right"
    collate_mode = DatasetItemCollateFn()

    def __call__(self, batch: list[DatasetItem]) -> dict[str, Any]:
        """Collate lists of samples into batches after padding."""
        fields = dataclasses.fields(DatasetItem)

        raw_batch: dict[Any, Any] = {}
        for field in fields:
            field_mode = getattr(self.collate_mode, field.name)
            if field_mode == "raw":
                raw_batch[field.name] = self._process_raw_field(field.name, batch)
            elif field_mode == "stack":
                raw_batch[field.name] = self._process_stack_field(field.name, batch)
            elif field_mode == "pad":
                raw_batch[field.name] = self._process_pad_field(field.name, batch)
        return raw_batch

    def _process_raw_field(self, field_name: str, batch: list[DatasetItem]) -> list[Any]:
        """Raw fields do not require any processing, just return the list of raw items."""
        return [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]

    def _process_stack_field(
        self, field_name: str, batch: list[DatasetItem]
    ) -> Optional[torch.Tensor]:
        """Fields that do not require any padding (ie pixel values) can be stacked."""
        sequence = [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]
        if not sequence:
            return None
        return torch.stack(sequence)

    def _process_pad_field(
        self, field_name: str, batch: list[DatasetItem]
    ) -> Optional[torch.Tensor]:
        """Fields that require padding (ie input_token_ids) need to be padded."""
        sequence = [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]
        if not sequence:
            return None

        return _pad_sequence(
            seq=sequence,
            padding_value=getattr(self.padding, field_name),
            padding_side=self.padding_side,
        )

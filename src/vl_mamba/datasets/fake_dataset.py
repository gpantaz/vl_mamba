from typing import Any

import torch
from torch.utils.data import Dataset

from vl_mamba.datamodels.dataclasses import DatasetItem, SpecialTokens
from vl_mamba.utils.image_preprocessor import VLMambaImageProcessor


class FakeDataset(Dataset[DatasetItem]):
    """Fake dataset."""

    def __init__(
        self,
        dataset_size: int = 1e8,
        text_seq_len: int = 50,
        image_size: int = 336,
        patch_size: int = 14,
        **kwargs: Any,
    ) -> None:
        self.dataset_size = dataset_size
        self.text_seq_len = text_seq_len
        if isinstance(image_size, int):
            image_size = {"height": image_size, "width": image_size}
        self._image_size = image_size

        if isinstance(patch_size, int):
            patch_size = {"height": patch_size, "width": patch_size}

        self.image_processor = VLMambaImageProcessor(
            image_size=self._image_size,
            patch_size=patch_size,
        )

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return int(self.dataset_size)

    def __getitem__(self, index: int) -> DatasetItem:
        """Get a single instance from the dataset."""
        input_ids = torch.randint(1, 1000, (1, self.text_seq_len), dtype=torch.long)
        labels = torch.randint(1, 1000, (1, self.text_seq_len), dtype=torch.long)
        labels[:, : self.text_seq_len // 2] = -100

        image_tokens = self.image_processor.prepare_image_tokens(
            special_tokens=SpecialTokens(),
            image_size=(self._image_size["height"], self._image_size["width"]),
        ).unsqueeze(0)

        input_ids = torch.cat([image_tokens, input_ids], dim=1).squeeze(0)
        labels = torch.cat([torch.ones_like(image_tokens) * -100, labels], dim=1)

        pixel_values = torch.rand(1, 3, self._image_size["height"], self._image_size["width"])

        return DatasetItem(
            input_ids=input_ids.squeeze(0),
            labels=labels.squeeze(0),
            pixel_values=pixel_values,
            raw_target={},
        )


# dataset = FakeDataset()
# from torch.utils.data import DataLoader
# from vl_mamba.datamodels.dataclasses import DatasetPadding
# from vl_mamba.datasets.collate import Collate

# from tqdm import tqdm

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# tokenizer.pad_token_id = 0

# collate_fn = Collate(
#     padding=DatasetPadding(input_ids=tokenizer.pad_token_id),
#     padding_side=tokenizer.padding_side,
# )
# dataloader = DataLoader(dataset, num_workers=0, collate_fn=collate_fn, batch_size=2)

# text_tokens = []
# image_tokens = []
# for batch in tqdm(dataloader, total=len(dataset) // 2):
#     breakpoint()
#     x = 2
# breakpoint()

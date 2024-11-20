from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.datamodules.fake_dataset import BatchEncoding, FakeDataset


class FakeDataModule(LightningDataModule):
    """Defines a PL data module that instantiates the dataloaders."""

    def __init__(
        self,
        model_name: str,
        num_workers: int = 0,
        batch_size: int = 8,
        pin_memory: bool = False,
        dataset_size: int = 100,
        text_seqlen: int = 20,
        video_seqlen: int = 60,
        num_objects: int = 18,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        self._dataset_size = dataset_size
        self._text_seqlen = text_seqlen
        self._video_seqlen = video_seqlen
        self._num_objects = num_objects

    def prepare_data(self) -> None:
        """Download and prepare data."""
        super().prepare_data()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data for dataloaders."""
        self._train_dataset = FakeDataset(
            self._tokenizer,
            self._dataset_size,
            self._text_seqlen,
            self._video_seqlen,
            self._num_objects,
        )
        self._val_dataset = FakeDataset(
            self._tokenizer,
            self._dataset_size,
            self._text_seqlen,
            self._video_seqlen,
            self._num_objects,
        )
        self._test_dataset = FakeDataset(
            self._tokenizer,
            self._dataset_size,
            self._text_seqlen,
            self._video_seqlen,
            self._num_objects,
        )

        self._collate_fn = DataCollatorForLanguageModeling(self._tokenizer)

    def train_dataloader(self) -> DataLoader[BatchEncoding]:
        """Generate train dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader[BatchEncoding]:
        """Generate val dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader[BatchEncoding]:
        """Generate test dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
        )

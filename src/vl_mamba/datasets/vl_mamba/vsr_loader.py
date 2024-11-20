import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from datasets.utils.download_manager import DownloadConfig, DownloadManager  # type: ignore[report]
from loguru import logger

from vl_mamba.datamodels.datamodels import DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import DatasetsLoader
from vl_mamba.utils.io import json_serializer


class VSRLoader(DatasetsLoader):
    """VSR dataset loader."""

    images_url = "https://www.dropbox.com/s/raw/0s3bj25s62crjh2/vsr_images.zip"
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}

    def __init__(
        self,
        split: str,
        cache_dir: Path,
        num_proc: int,
        dataset_name: str = "cambridgeltl/vsr_random",
        datasets_batch_size: int = 1000,
        chunk_size: int = 1,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            datasets_batch_size=datasets_batch_size,
            num_proc=num_proc,
            config_name=None,
        )

        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )
        self.images_folder = (
            Path(self.download_manager.download_and_extract(self.images_url)) / "images"
        )

    def cast_to_vlmamba_features(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Cast the batch to VL-Mamba features."""
        return {
            "task": [Task.itm.value for _ in batch["image"]],
            "image": [str(self.images_folder.joinpath(image)) for image in batch["image"]],
            "caption": batch["caption"],
            "qa_pairs": [None for _ in batch["image"]],
            "region": [None for _ in batch["image"]],
            "relation": [None for _ in batch["image"]],
            "chat": [None for _ in batch["image"]],
            "source": [self.source for _ in batch["image"]],
            "metadata": [
                json.dumps(
                    {
                        "image_path": str(self.images_folder.joinpath(image)),
                        "relation": relation,
                        "label": label,
                    },
                    default=json_serializer,
                    indent=2,
                )
                for image, relation, label in zip(
                    batch["image"], batch["relation"], batch["label"], strict=False
                )
            ],
        }

    def _build_rows_iterator(
        self,
        chunk_size: int,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> Iterator[list[Any]]:
        logger.info("Building VSR dataset.")
        split = "dev" if self.split == DatasetSplits.VALIDATION else self.split
        dataset = load_dataset(  # type: ignore[report]
            self.dataset_name,
            data_files=self.data_files,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )[split]
        buffer = []

        for row in dataset:
            buffer.append(row)
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

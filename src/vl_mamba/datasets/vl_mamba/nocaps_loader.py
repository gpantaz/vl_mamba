import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from loguru import logger
from overrides import overrides
from PIL import ImageFile

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import DatasetsLoader
from vl_mamba.utils.io import json_serializer


ImageFile.LOAD_TRUNCATED_IMAGES = True


class NocapsLoader(DatasetsLoader):
    """Nocaps dataset loader."""

    split_map = {
        DatasetSplits.VALIDATION: "validation",
        DatasetSplits.TEST: "test",
    }

    def __init__(
        self,
        split: str,
        num_proc: int,
        cache_dir: Path,
        datasets_batch_size: int = 1000,
        chunk_size: int = 1,
    ):
        super().__init__(
            dataset_name="HuggingFaceM4/NoCaps",
            split=split,
            datasets_batch_size=datasets_batch_size,
            num_proc=num_proc,
            config_name=None,
        )

        self.chunk_size = chunk_size
        self.cache_dir = cache_dir

    def cast_to_vlmamba_features(  # noqa: WPS231
        self, batch: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Cast the dataset to the format expected by vlmamba."""
        dataset_examples: dict[str, list[Any]] = {
            data_key: [] for data_key in DatasetFeatures.keys()
        }

        df = pd.DataFrame.from_dict(batch)
        for _, instance in df.iterrows():
            dataset_examples["source"].append("nocaps")
            dataset_examples["task"].append(Task.captioning.value)
            metadata = {
                # We need the iamge id to make submissions
                "image_id": instance.image_id,
                "image_coco_url": instance.image_coco_url,
                "image_file_name": instance.image_file_name,
                "image_open_images_id": instance.image_open_images_id,
                # Add these to the metadata since we will only test for these examples
                "captions": instance.annotations_captions,
            }
            dataset_examples["metadata"].append(
                json.dumps(
                    metadata,
                    default=json_serializer,
                    indent=2,
                )
            )
            dataset_examples["image"].append(instance.image)

            dataset_examples["caption"].append(None)
            dataset_examples["qa_pairs"].append(None)
            dataset_examples["region"].append(None)
            dataset_examples["relation"].append(None)
            dataset_examples["chat"].append(None)

        return dataset_examples

    def fetch_dataset_from_hf(self) -> tuple[Dataset, str]:
        """Fetch the nocaps dataset from Hugging Face."""
        dataset = load_dataset(
            self.dataset_name,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )[self.split_map[self.split]]
        return dataset

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:  # noqa: WPS210
        logger.info(f"Building {self.source} dataset for {self.split}.")
        dataset = self.fetch_dataset_from_hf()

        buffer = []
        for row in dataset:
            buffer.append(row)
            if len(buffer) == 1000:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

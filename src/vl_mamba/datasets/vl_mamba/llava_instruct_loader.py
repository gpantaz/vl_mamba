import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, LLavaInstructModel, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class LLAVAInstructLoader(BaseLoader):
    """COCO dataset loader."""

    annotation_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json?download=true"

    images_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
    }

    split_dir = {DatasetSplits.TRAIN: "train2017", DatasetSplits.VALIDATION: "val2017"}

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "llava_instruct",
        chunk_size: int = 1,
        num_proc: int = 1,
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            source=source,
            split=split,
            writer_batch_size=writer_batch_size,
            chunk_size=chunk_size,
            num_proc=num_proc,
            **kwargs,
        )

        Path(cache_dir).mkdir(exist_ok=True)

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir),
        )

        self.image_paths = self.download_manager.download_and_extract(self.images_urls)
        self.annotation_path = self.download_manager.download(self.annotation_url)

    @overrides(check_signature=False)
    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        annotations = read_json(self.annotation_path)

        buffer = []
        for example in annotations:
            instance = LLavaInstructModel.model_validate(example)
            buffer.append(instance)

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _get_image_path(self, example: LLavaInstructModel) -> Path:
        """Get the image path for a coco image."""
        image = example.image
        for split in (DatasetSplits.TRAIN, DatasetSplits.VALIDATION):
            path = Path(self.image_paths[split], self.split_dir[split], image)
            if path.exists():
                return path
        raise FileNotFoundError(f"Image {image} not found.")

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[LLavaInstructModel]) -> dict[str, list[Any]]:
        dataset_examples: dict[str, list[Any]] = {
            data_key: [] for data_key in DatasetFeatures.keys()
        }
        for example in examples:
            image_path = str(self._get_image_path(example))
            metadata = {"id": example.id, "image_path": image_path}

            dataset_examples["task"].append(Task.chat.value)
            dataset_examples["image"].append(image_path)
            dataset_examples["caption"].append(None)
            dataset_examples["qa_pairs"].append(None)
            dataset_examples["region"].append(None)
            dataset_examples["relation"].append(None)
            dataset_examples["source"].append(self.source)
            dataset_examples["chat"].append(
                [conversation.model_dump() for conversation in example.conversations]
            )
            dataset_examples["metadata"].append(
                json.dumps(
                    metadata,
                    default=json_serializer,
                    indent=2,
                )
            )
        return dataset_examples

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

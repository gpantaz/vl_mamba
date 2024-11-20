import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, POPEMetadata, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer


class POPELoader(BaseLoader):
    """POPE dataset loader."""

    pope_version = {
        "pope_adversarial": "https://huggingface.co/datasets/gpantaz/pope/resolve/main/coco_pope_adversarial.json?download=true",  # noqa: B950
        "pope_popular": "https://huggingface.co/datasets/gpantaz/pope/resolve/main/coco_pope_popular.json?download=true",
        "pope_random": "https://huggingface.co/datasets/gpantaz/pope/resolve/main/coco_pope_random.json?download=true",
    }

    image_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
    }

    split_dir = {
        DatasetSplits.TRAIN: "train2017",
        DatasetSplits.VALIDATION: "val2017",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: Literal["pope_adversarial", "pope_popular", "pope_random"] = "pope_adversarial",
        chunk_size: int = 1,
        num_proc: int = 1,
    ):

        super().__init__(
            source=source,
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir),
            record_checksums=False,
        )

        self.annotation_file = Path(
            self.download_manager.download_and_extract(self.pope_version[source]),
        )

        self.image_paths = self.download_manager.download_and_extract(self.image_urls)

    def _get_image_path(self, image: str) -> Path:
        image_name = image.split("_")[-1]
        for split, split_dir in self.split_dir.items():
            image_path = Path(self.image_paths[split], split_dir, image_name)
            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"Image for {image_name} not found.")

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        buffer = []
        annotations = pd.read_json(self.annotation_file, lines=True)

        for _, annotation in annotations.iterrows():
            annotation_metadata = POPEMetadata.model_validate(annotation.to_dict())

            image_path = str(self._get_image_path(annotation_metadata.image))

            buffer.append(
                {
                    "image_path": image_path,
                    "qa_pairs": [
                        {
                            "question": annotation_metadata.text,
                            # Even though the answer is provided in the annotation, we will not add it here
                            # Instead, add it to the metadata field so that we can get the evaluation score
                            "answer": None,
                        }
                    ],
                    "metadata": {
                        "question_id": annotation_metadata.question_id,
                        "image_path": image_path,
                        "answer": annotation_metadata.label,
                    },
                }
            )

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.vqa.value for _ in examples],
            "image": [example["image_path"] for example in examples],
            "caption": [None for _ in examples],
            "qa_pairs": [example["qa_pairs"] for example in examples],
            "region": [None for _ in examples],
            "relation": [None for _ in examples],
            "chat": [None for _ in examples],
            "source": [self.source for _ in examples],
            "metadata": [
                json.dumps(
                    example["metadata"],
                    default=json_serializer,
                    indent=2,
                )
                for example in examples
            ],
        }

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

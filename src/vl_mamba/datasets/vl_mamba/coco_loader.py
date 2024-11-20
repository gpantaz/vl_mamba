import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class COCOLoader(BaseLoader):
    """COCO dataset loader."""

    # This has the coco annotations from 2014 organized in terms of the karpathy indices.
    # The captions for the COCO 2017 dataset and the 2014 version are the same.
    karpathy_url = "https://huggingface.co/datasets/gpantaz/karpathy_split/resolve/main/caption_datasets.zip?download=true"

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
        source: str = "coco",
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
            record_checksums=False,
        )

        self.karpathy_coco_file = Path(
            self.download_manager.download_and_extract(self.karpathy_url), "dataset_coco.json"
        )

        self.image_paths = self.download_manager.download_and_extract(self.images_urls)

    @overrides(check_signature=False)
    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        annotations = read_json(self.karpathy_coco_file)

        buffer = []
        for image_metadata in annotations["images"]:
            if self._should_skip_instance_for_split(image_metadata):
                continue

            image_path = self._get_image_path(image_metadata)

            for caption_id, caption in zip(image_metadata["sentids"], image_metadata["sentences"]):
                buffer.append(
                    {
                        "annotation": caption["raw"],
                        "image_path": str(image_path),
                        "coco_id": image_metadata["cocoid"],
                        "caption_id": caption_id,
                    }
                )
                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _get_image_path(self, image_metadata: dict[str, Any]) -> Path:
        """Get the image path for a coco image.

        With the 2017 annotations the COCO images have been reorganized, meaning that for a coco id
        that is present in the 2014 annotations we may need to look in the train or val folder of
        the 2017 dataset.
        """
        cocoid = image_metadata["cocoid"]
        for split in (DatasetSplits.TRAIN, DatasetSplits.VALIDATION):
            path = Path(self.image_paths[split], self.split_dir[split], f"{cocoid:012d}.jpg")
            if path.exists():
                return path
        raise FileNotFoundError(f"Image for {cocoid} not found.")

    def _should_skip_instance_for_split(self, image_metadata: dict[str, Any]) -> bool:
        """Skip instances that are not in the current split."""
        # By default restval images are attributed to the train set.
        skippable_instances_per_split = {
            datasets.Split.TRAIN: {"train", "restval"},
            datasets.Split.VALIDATION: {"val"},
            datasets.Split.TEST: {"test"},
        }

        skippable_instances = skippable_instances_per_split.get(self.split, None)
        if skippable_instances is None:
            return True

        return image_metadata["split"] not in skippable_instances

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.captioning.value for _ in examples],
            "image": [example["image_path"] for example in examples],
            "caption": [example["annotation"] for example in examples],
            "qa_pairs": [None for _ in examples],
            "region": [None for _ in examples],
            "relation": [None for _ in examples],
            "chat": [None for _ in examples],
            "source": [self.source for _ in examples],
            "metadata": [
                json.dumps(
                    example,
                    default=json_serializer,
                    indent=2,
                )
                for example in examples
            ],
        }

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

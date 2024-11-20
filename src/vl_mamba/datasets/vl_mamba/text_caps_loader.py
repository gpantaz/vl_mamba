import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task, TextCapsMetadata
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class TextCapsLoader(BaseLoader):
    """TextCaps dataset loader."""

    images_url = {
        DatasetSplits.TRAIN: "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        DatasetSplits.TEST: "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
    }

    image_folders = {
        DatasetSplits.TRAIN: "train_images",
        DatasetSplits.VALIDATION: "train_images",
        DatasetSplits.TEST: "test_images",
    }

    caption_urls = {
        DatasetSplits.TRAIN: "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json",
        DatasetSplits.VALIDATION: "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json",
        DatasetSplits.TEST: "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_test.json",
    }

    ocr_tokens_urls = {
        DatasetSplits.TRAIN: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json",
        DatasetSplits.VALIDATION: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json",
        DatasetSplits.TEST: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "text_caps",
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

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir),
            record_checksums=False,
        )

        self.caption_file_paths = self.download_manager.download_and_extract(self.caption_urls)
        self.image_folder_paths = self.download_manager.download_and_extract(self.images_url)
        # The images for the validation split is the same as the train split.
        self.image_folder_paths[DatasetSplits.VALIDATION] = self.image_folder_paths[
            DatasetSplits.TRAIN
        ]
        self.ocr_tokens_paths = self.download_manager.download_and_extract(self.ocr_tokens_urls)

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        captions_metadata = read_json(self.caption_file_paths[self.split])["data"]
        image_folder = Path(self.image_folder_paths[self.split], self.image_folders[self.split])
        ocr_metadata = read_json(self.ocr_tokens_paths[self.split])["data"]
        ocr_metadata_image_ids = {ocr["image_id"]: ocr for ocr in ocr_metadata}

        buffer = []
        for caption_metadata in captions_metadata:
            caption_metadata["ocr"] = ocr_metadata_image_ids[caption_metadata["image_id"]]

            text_caps_metadata = TextCapsMetadata.model_validate(caption_metadata)

            image_name = f"{text_caps_metadata.image_id}.jpg"
            image_path = Path(image_folder / image_name)
            if not image_path.exists():
                raise FileNotFoundError(f"Image {image_path} not found.")

            buffer.append(
                {
                    "annotation": text_caps_metadata,
                    "image_path": str(image_path),
                }
            )
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.captioning.value for _ in examples],
            "image": [example["image_path"] for example in examples],
            "caption": [example["annotation"].caption_str for example in examples],
            "qa_pairs": [None for _ in examples],
            "region": [None for _ in examples],
            "relation": [None for _ in examples],
            "chat": [None for _ in examples],
            "source": [self.source for _ in examples],
            "metadata": [
                json.dumps(
                    {
                        "image_id": example["annotation"].image_id,
                        "caption_id": example["annotation"].caption_id,
                        "image_path": example["image_path"],
                        "references": example["annotation"].reference_strs,
                        "ocr": example["annotation"].ocr,
                        "width": example["annotation"].image_width,
                        "height": example["annotation"].image_height,
                        "flickr_original_url": example["annotation"].flickr_original_url,
                        "flickr_300k_url": example["annotation"].flickr300k_url,
                    },
                    default=json_serializer,
                    indent=2,
                )
                for example in examples
            ],
        }

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

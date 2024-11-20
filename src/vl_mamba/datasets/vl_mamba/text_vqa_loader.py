import json
from collections import Counter
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task, TextVQAMetadata
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class TextVQALoader(BaseLoader):
    """TextVQA dataset loader."""

    images_url = {
        DatasetSplits.TRAIN: "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
        DatasetSplits.TEST: "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
    }

    image_folders = {
        DatasetSplits.TRAIN: "train_images",
        DatasetSplits.VALIDATION: "train_images",
        DatasetSplits.TEST: "test_images",
    }

    question_urls = {
        DatasetSplits.TRAIN: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
        DatasetSplits.VALIDATION: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
        DatasetSplits.TEST: "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json",
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
        source: str = "text_vqa",
        chunk_size: int = 1,
        num_proc: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            source=source,
            split=split,
            writer_batch_size=writer_batch_size,
            chunk_size=chunk_size,
            num_proc=num_proc,
            **kwargs,
        )

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )

        self.question_file_paths = self.download_manager.download_and_extract(self.question_urls)
        self.image_folder_paths = self.download_manager.download_and_extract(self.images_url)
        # The images for the validation split is the same as the train split.
        self.image_folder_paths[DatasetSplits.VALIDATION] = self.image_folder_paths[
            DatasetSplits.TRAIN
        ]
        self.ocr_tokens_paths = self.download_manager.download_and_extract(self.ocr_tokens_urls)

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        questions_metadata = read_json(self.question_file_paths[self.split])["data"]
        image_folder = Path(self.image_folder_paths[self.split], self.image_folders[self.split])
        ocr_metadata = read_json(self.ocr_tokens_paths[self.split])["data"]
        ocr_metadata_image_ids = {ocr["image_id"]: ocr for ocr in ocr_metadata}

        if self.split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            questions_metadata = groupby(questions_metadata, key=lambda x: x["image_id"])
        else:
            questions_metadata = [
                (question["image_id"], [question]) for question in questions_metadata
            ]

        buffer = []
        for image_id, questions_iterator in questions_metadata:
            image_path = Path(image_folder, f"{image_id}.jpg")
            if not image_path.exists():
                raise FileNotFoundError(f"Image {image_path} not found.")

            qa_pairs = []
            ocr = []
            question_ids = []
            answers = []
            for question_metadata in list(questions_iterator):
                question_metadata["ocr"] = ocr_metadata_image_ids[question_metadata["image_id"]]

                text_vqa_metadata = TextVQAMetadata.model_validate(question_metadata)
                answer = (
                    Counter(text_vqa_metadata.answers).most_common(1)[0][0]
                    if text_vqa_metadata.answers
                    else None
                )
                qa_pairs.append(
                    {
                        "question": text_vqa_metadata.question,
                        "answer": answer,
                    }
                )

                answers.append(text_vqa_metadata.answers)
                ocr.append(text_vqa_metadata.ocr)
                question_ids.append(text_vqa_metadata.question_id)

            buffer.append(
                {
                    "qa_pairs": qa_pairs,
                    "ocr": ocr,
                    "image_path": str(image_path),
                    "metadata": {
                        "image_id": text_vqa_metadata.image_id,
                        "question_ids": question_ids,
                        "image_path": str(image_path),
                        "ocr": ocr,
                        "width": text_vqa_metadata.image_width,
                        "height": text_vqa_metadata.image_height,
                        "flickr_original_url": text_vqa_metadata.flickr_original_url,
                        "flickr_300k_url": text_vqa_metadata.flickr300k_url,
                        "answers": answers,
                    },
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

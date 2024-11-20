import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, GQAMetadata, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class GQALoader(BaseLoader):
    """GQA dataset loader."""

    annotation_url = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"

    images_url = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

    question_balanced_files = {
        DatasetSplits.TRAIN: "train_balanced_questions.json",
        DatasetSplits.VALIDATION: "val_balanced_questions.json",
        DatasetSplits.TEST: "test_balanced_questions.json",
        # LLAVA reports results on test dev balanced!
        DatasetSplits.TEST_DEV: "testdev_balanced_questions.json",
    }

    question_all_files = {
        DatasetSplits.TRAIN: "train_questions.json",
        DatasetSplits.VALIDATION: "val_questions.json",
        DatasetSplits.TEST: "test_all_questions.json",
        DatasetSplits.TEST_DEV: "testdev_all_questions.json",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "gqa",
        chunk_size: int = 1,
        num_proc: int = 1,
        use_balanced_version: bool = True,
        use_short_answers: bool = True,
        max_annotations_per_image: int = 20,
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
        self.use_balanced_version = use_balanced_version
        self.use_short_answers = use_short_answers
        self.max_annotations_per_image = max_annotations_per_image

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )

        self.question_paths = self.download_manager.download_and_extract(self.annotation_url)
        self.image_paths = self.download_manager.download_and_extract(self.images_url)

    def get_question_file(self) -> Path:
        """Get the question file for a split."""
        if self.use_balanced_version:
            return Path(self.question_paths, self.question_balanced_files[self.split])
        return Path(self.question_paths, self.question_all_files[self.split])

    def build_annotations(self) -> Iterator[Any]:
        """Build annotations for the dataset split.

        The json files are in format {question_id: metadata}, but since we want to do packing, we
        need to group metadata by image_id.
        """
        question_file = self.get_question_file()
        annotations = read_json(question_file)
        questions_metadata_list = []
        for key, metadata in annotations.items():
            metadata["question_id"] = key
            questions_metadata_list.append(metadata)

        if self.split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            questions_metadata_list = sorted(questions_metadata_list, key=lambda x: x["imageId"])
            questions_iterator = groupby(questions_metadata_list, key=lambda x: x["imageId"])

            annotation_chunks = []
            for image_id, grouped_questions_iterator in questions_iterator:
                grouped_questions = list(grouped_questions_iterator)
                # Split the questions into chunks of max_annotations_per_image
                grouped_questions_per_image = [
                    grouped_questions[step : step + self.max_annotations_per_image]
                    for step in range(0, len(grouped_questions), self.max_annotations_per_image)
                ]
                annotation_chunks.extend(
                    [image_id, questions] for questions in grouped_questions_per_image
                )
            return annotation_chunks
        return [[q_metadata["imageId"], [q_metadata]] for q_metadata in questions_metadata_list]

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:  # noqa: WPS231
        logger.info(f"Building {self.source} dataset for {self.split}.")
        image_folder = Path(self.image_paths, "images")
        annotations = self.build_annotations()

        buffer = []
        for image_id, questions_iterator in annotations:
            metadata: dict[str, Any] = {
                "is_balanced": [],
                "question_ids": [],
                "full_answer": [],
                "answer": [],
            }
            qa_pairs = []
            image_path = Path(image_folder, f"{image_id}.jpg")
            if not image_path.exists():
                raise FileNotFoundError(f"Image for {image_path} not found.")

            for question_metadata in list(questions_iterator):
                gqa_metadata = GQAMetadata.model_validate(question_metadata)

                answer = (
                    gqa_metadata.answer if self.use_short_answers else gqa_metadata.full_answer
                )
                qa_pairs.append({"question": gqa_metadata.question, "answer": answer})
                metadata["is_balanced"].append(gqa_metadata.is_balanced)
                metadata["question_ids"].append(gqa_metadata.question_id)
                metadata["full_answer"].append(gqa_metadata.full_answer)
                metadata["answer"].append(gqa_metadata.answer)

            metadata["image_id"] = image_id
            metadata["use_short_answer"] = self.use_short_answers
            metadata["image_path"] = str(image_path)
            buffer.append(
                {"qa_pairs": qa_pairs, "image_path": str(image_path), "metadata": metadata}
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

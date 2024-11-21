import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import (
    AI2DMetadata,
    AID2Question,
    DatasetFeatures,
    DatasetSplits,
    Task,
)
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class AI2DLoader(BaseLoader):
    """AI2D dataset loader."""

    dataset_url = "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip"
    test_ids_url = "https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv"

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "ai2d",
        chunk_size: int = 1,
        num_proc: int = 1,
        validation_split: float = 0.1,
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

        self.dataset_path = Path(
            self.download_manager.download_and_extract(self.dataset_url),
            "ai2d",
        )

        self.test_ids_file = Path(
            self.download_manager.download_and_extract(self.test_ids_url),
        )
        self.annotation_folder = Path(self.dataset_path, "annotations")
        self.question_folder = Path(self.dataset_path, "questions")
        self.image_folder = Path(self.dataset_path, "images")
        self._validation_split = validation_split

    def _get_train_test_ids(self) -> dict[DatasetSplits, list[str]]:
        with open(self.test_ids_file) as fp:
            data = sorted([line.strip() for line in fp])

        images = os.listdir(str(self.image_folder))

        split_ids = {
            DatasetSplits.TRAIN: [],
            DatasetSplits.TEST: [],
        }
        for image in images:
            image_id = image.split(".")[0]
            if image_id not in data:
                split_ids[DatasetSplits.TRAIN].append(image_id)
            else:
                split_ids[DatasetSplits.TEST].append(image_id)

        train_ids = split_ids[DatasetSplits.TRAIN]
        random.shuffle(train_ids)
        split_ids[DatasetSplits.TRAIN] = train_ids[
            : int(len(train_ids) * (1 - self._validation_split))
        ]
        split_ids[DatasetSplits.VALIDATION] = train_ids[
            int(len(train_ids) * (1 - self._validation_split)) :
        ]
        return split_ids

    def _augment_qa_pairs(self, annotation_metadata: AID2Question) -> list[dict[str, Any]]:
        if annotation_metadata.correct_answer is None:
            raise ValueError("Correct choice index cannot be None.")

        correct_answer = annotation_metadata.answers[annotation_metadata.correct_answer]
        qa_pairs = []
        for idx, swapped_answer in enumerate(annotation_metadata.answers):
            answers = annotation_metadata.answers.copy()
            answers[idx] = correct_answer
            answers[annotation_metadata.correct_answer] = swapped_answer
            qa_pairs.append(
                {
                    "question": annotation_metadata.question,
                    "answer": correct_answer,
                    "answers": answers,
                    "correct_answer": idx,
                    "question_id": annotation_metadata.question_id,
                }
            )

        return qa_pairs

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        split_ids = self._get_train_test_ids()[self.split]

        buffer = []
        images = os.listdir(str(self.image_folder))
        for image in images:
            image_id = image.split(".")[0]
            if image_id not in split_ids:
                continue

            question_path = Path(self.question_folder, f"{image}.json")
            if not question_path.exists():
                continue

            annotation = read_json(Path(self.annotation_folder, f"{image}.json"))
            questions_metadata = AI2DMetadata.model_validate(read_json(question_path))

            image_path = str(Path(self.image_folder, image))

            # 1 example per question during test time
            if self.split == DatasetSplits.TEST:
                qa_pairs = [
                    {
                        "question": question.question,
                        "answer": None,
                        "answers": question.answers,
                        "correct_answer": question.correct_answer,
                        "question_id": question.question_id,
                    }
                    for question in questions_metadata.questions
                ]
            # Augment the qa_pairs during training time
            else:
                qa_pairs_list = [
                    self._augment_qa_pairs(question) for question in questions_metadata.questions
                ]
                # Flatten the list
                qa_pairs = [qa for qa_sublist in qa_pairs_list for qa in qa_sublist]

            for qa_pair in qa_pairs:
                buffer.append(
                    {
                        "qa_pairs": [
                            {
                                "question": qa_pair["question"],
                                "answer": qa_pair["answer"],
                            }
                        ],
                        "metadata": {
                            "answers": [qa_pair["answers"]],
                            "answer": [qa_pair["correct_answer"]],
                            "question_id": qa_pair["question_id"],
                            "image_path": image_path,
                            "annotation": annotation,
                            "ocr_text": self._format_ocr(annotation),
                        },
                    }
                )

                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _format_ocr(self, annotation: dict[str, Any]) -> str:
        """Add the context from the annotation to the question."""
        text_dict = annotation["text"]
        # If there is no text, return an empty string
        if not text_dict:
            return ""

        context_text = []
        for text_value_dict in text_dict.values():
            context_text.append(
                (
                    text_value_dict["replacementText"],
                    text_value_dict["value"],
                )
            )

        # Now sort the context_text by alphabetical order of the replacementText
        context_text = sorted(context_text, key=lambda x: x[0])
        context = "\n".join([f"{text_id}: {text_value}" for text_id, text_value in context_text])
        return f"OCR text:\n{context}"

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.m_vqa.value for _ in examples],
            "image": [example["metadata"]["image_path"] for example in examples],
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

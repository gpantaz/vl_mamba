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

from vl_mamba.datamodels.datamodels import (
    DatasetFeatures,
    DatasetSplits,
    Task,
    VQAAnswerMetadata,
    VQAQuestionMetadata,
)
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class VQAv2Loader(BaseLoader):
    """VQA v2 dataset loader."""

    image_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
        DatasetSplits.TEST_DEV: "http://images.cocodataset.org/zips/test2015.zip",
        DatasetSplits.TEST_STD: "http://images.cocodataset.org/zips/test2015.zip",
    }

    question_urls = {
        DatasetSplits.TRAIN: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        DatasetSplits.VALIDATION: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        DatasetSplits.TEST_DEV: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        DatasetSplits.TEST_STD: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    }

    question_files = {
        DatasetSplits.TRAIN: "v2_OpenEnded_mscoco_train2014_questions.json",
        DatasetSplits.VALIDATION: "v2_OpenEnded_mscoco_val2014_questions.json",
        DatasetSplits.TEST_DEV: "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        DatasetSplits.TEST_STD: "v2_OpenEnded_mscoco_test2015_questions.json",
    }

    answer_urls = {
        DatasetSplits.TRAIN: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        DatasetSplits.VALIDATION: "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    }

    answer_files = {
        DatasetSplits.TRAIN: "v2_mscoco_train2014_annotations.json",
        DatasetSplits.VALIDATION: "v2_mscoco_val2014_annotations.json",
    }

    split_dir = {
        DatasetSplits.TRAIN: "train2017",
        DatasetSplits.VALIDATION: "val2017",
        DatasetSplits.TEST_DEV: "test2015",
        DatasetSplits.TEST_STD: "test2015",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "vqa_v2",
        chunk_size: int = 1,
        num_proc: int = 1,
        max_annotations_per_image: int = 20,
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

        self.max_annotations_per_image = max_annotations_per_image
        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )

        self.image_folder_paths = self.download_manager.download_and_extract(self.image_urls)
        self.question_paths = self.download_manager.download_and_extract(self.question_urls)
        self.answer_paths = self.download_manager.download_and_extract(self.answer_urls)

    def build_annotations(self) -> Iterator[Any]:
        """Build annotations for the dataset split."""
        questions_metadata = read_json(
            Path(self.question_paths[self.split], self.question_files[self.split])
        )["questions"]
        if self.split in {DatasetSplits.TEST_STD, DatasetSplits.TEST_DEV}:
            # For the test set the questions are not grouped by image_id
            questions_metadata = [
                (question["image_id"], [question]) for question in questions_metadata
            ]
        else:
            questions_iterator = groupby(questions_metadata, key=lambda x: x["image_id"])

            questions_metadata = []
            for image_id, grouped_questions_iterator in questions_iterator:
                grouped_questions = list(grouped_questions_iterator)
                # Split the questions into chunks of max_annotations_per_image
                grouped_questions_per_image = [
                    grouped_questions[step : step + self.max_annotations_per_image]
                    for step in range(0, len(grouped_questions), self.max_annotations_per_image)
                ]
                questions_metadata.extend(
                    [image_id, questions] for questions in grouped_questions_per_image
                )
        return questions_metadata

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        questions_metadata = self.build_annotations()
        answer_file = self.answer_paths.get(self.split, None)

        buffer = []
        answers2question_ids = None
        if answer_file is not None:
            answers_metadata = read_json(Path(answer_file, self.answer_files[self.split]))[
                "annotations"
            ]
            answers2question_ids = {
                answer_dict["question_id"]: answer_dict for answer_dict in answers_metadata
            }

        for image_id, questions_iterator in questions_metadata:
            image_path = str(self._get_image_path(image_id))
            qa_pairs = []
            question_types = []
            answer_types = []
            question_ids = []
            all_answers = []
            for question in list(questions_iterator):
                question_metadata = VQAQuestionMetadata.model_validate(question)
                question_type = None
                answers = None
                answer_type = None
                answer = None
                if answers2question_ids is not None:
                    answer_metadata = answers2question_ids[question_metadata.question_id]
                    answer_model = VQAAnswerMetadata.model_validate(answer_metadata)
                    question_type = answer_model.question_type
                    answers = [answer_dict["answer"] for answer_dict in answer_model.answers]
                    answer_type = answer_model.answer_type
                    answer = Counter(answers).most_common(1)[0][0]
                    all_answers.append(answers)

                qa_pairs.append({"question": question_metadata.question, "answer": answer})
                question_types.append(question_type)
                answer_types.append(answer_type)
                question_ids.append(question["question_id"])

            buffer.append(
                {
                    "qa_pairs": qa_pairs,
                    "image_path": image_path,
                    "question_types": question_types,
                    "answers": all_answers,
                    "answer_types": answer_types,
                    "image_id": image_id,
                    "question_ids": question_ids,
                }
            )

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _get_image_path(self, cocoid: int) -> Path:
        """Get the image path for a coco image.

        With the 2017 annotations the COCO images have been reorganized, meaning that for a coco id
        that is present in the 2014 annotations we may need to look in the train or val folder of
        the 2017 dataset.
        """
        splits = (
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST_STD,
            DatasetSplits.TEST_DEV,
        )
        for split in splits:
            if split in {DatasetSplits.TEST_STD, DatasetSplits.TEST_DEV}:
                path = Path(
                    self.image_folder_paths[split],
                    self.split_dir[split],
                    f"COCO_test2015_{cocoid:012d}.jpg",
                )
            else:
                path = Path(
                    self.image_folder_paths[split], self.split_dir[split], f"{cocoid:012d}.jpg"
                )
            if path.exists():
                return path

        raise FileNotFoundError(f"Image for {cocoid} not found.")

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
                    {
                        "image_id": example["image_id"],
                        "question_ids": example["question_ids"],
                        "question_types": example["question_types"],
                        "answer_types": example["answer_types"],
                        "answers": example["answers"],
                        "image_path": example["image_path"],
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

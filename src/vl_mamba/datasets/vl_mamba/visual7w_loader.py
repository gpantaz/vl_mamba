import json
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides
from PIL import Image

from vl_mamba.datamodels.datamodels import (
    DatasetFeatures,
    DatasetSplits,
    Task,
    Visual7WImageMetadata,
)
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.boxes import BoxMode
from vl_mamba.utils.io import json_serializer, read_json


class Visual7WLoader(BaseLoader):
    """Visual7W dataset loader."""

    images_url = "http://vision.stanford.edu/yukezhu/visual7w_images.zip"
    telling_questions_url = (
        "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
    )

    pointing_questions_url = (
        "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_pointing.zip"
    )

    grounding_annotations_url = (
        "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_grounding_annotations.zip"
    )

    telling_questions_file = "dataset_v7w_telling.json"
    pointing_questions_file = "dataset_v7w_pointing.json"

    telling_answers_file = "v7w_telling_answers.json"
    pointing_answers_file = "v7w_pointing_answers.json"

    split_map = {
        "train": DatasetSplits.TRAIN,
        "val": DatasetSplits.VALIDATION,
        "test": DatasetSplits.TEST,
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "visual7w",
        chunk_size: int = 1,
        num_proc: int = 1,
        max_questions_per_image_in_turn: int = 3,
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

        self.images_path = Path(
            self.download_manager.download_and_extract(self.images_url),
            "images",
        )

        self._telling_questions_file_path = Path(
            self.download_manager.download_and_extract(self.telling_questions_url),
            self.telling_questions_file,
        )

        self._pointing_questions_file_path = Path(
            self.download_manager.download_and_extract(self.pointing_questions_url),
            self.pointing_questions_file,
        )

        self._telling_answers_file_path = Path(
            self.download_manager.download_and_extract(self.grounding_annotations_url),
            self.telling_answers_file,
        )

        self._pointing_answers_file_path = Path(
            self.download_manager.download_and_extract(self.grounding_annotations_url),
            self.pointing_answers_file,
        )

        self.max_questions_per_image_in_turn = max_questions_per_image_in_turn

    def load_pointing_annotations(self, filepath: Path) -> dict[int, list[dict[str, Any]]]:
        """Loads the grounding annotations in a dictionary format."""
        grounding_annotations = read_json(filepath)["boxes"]
        return {ground_ann["box_id"]: ground_ann for ground_ann in grounding_annotations}

    def load_telling_annotations(self, filepath: Path) -> dict[int, dict[str, Any]]:
        """Loads the grounding annotations in a dictionary format."""
        grounding_annotations = read_json(filepath)["boxes"]
        grounding_annotations_per_question = defaultdict(list)
        for ground_ann in grounding_annotations:
            grounding_annotations_per_question[ground_ann["qa_id"]].append(ground_ann)
        return grounding_annotations_per_question

    def augment_qa_pairs(
        self, image_metadata: Visual7WImageMetadata
    ) -> list[list[dict[str, Any]]]:
        """Augment all the questions for a single image."""
        # Copy the
        augmented_qa_pairs = []
        for qa_pair in image_metadata.qa_pairs:
            # Put the correct answer in all possible answers
            multiple_choices = qa_pair.multiple_choices
            answer = qa_pair.answer
            augmented_qa_pair = []
            for idx in range(len(multiple_choices) + 1):
                if idx == 0:
                    answers = [answer] + multiple_choices[:idx] + multiple_choices[idx:]
                elif idx == len(multiple_choices):
                    answers = [*multiple_choices, answer]
                else:
                    answers = multiple_choices[:idx] + [answer] + multiple_choices[idx:]

                augmented_qa_pair.append(
                    {
                        "question": qa_pair.question,
                        "answer": answer,
                        "answers": answers,
                        "question_id": qa_pair.qa_id,
                        "question_type": qa_pair.question_type,
                    }
                )

            # Now randomly shuffle the qa_pairs to prevent the model from learning the order
            random.shuffle(augmented_qa_pair)
            augmented_qa_pairs.append(augmented_qa_pair)
        return augmented_qa_pairs

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")

        yield from self._pointing_iterator(
            read_json(self._pointing_questions_file_path)["images"],
            chunk_size,
            source="pointing",
        )

        yield from self._telling_iterator(
            read_json(self._telling_questions_file_path)["images"],
            chunk_size,
            source="telling",
        )

    def _pointing_iterator(
        self,
        data: list[dict[str, Any]],
        chunk_size: int,
        source: str,
    ) -> Iterator[list[Any]]:
        pointing_annotations = self.load_pointing_annotations(self._pointing_answers_file_path)

        buffer = []
        for raw_metadata in data:
            image_metadata = Visual7WImageMetadata.model_validate(raw_metadata)
            if self.split_map[image_metadata.split] != self.split:
                continue

            qa_chunks = self._make_qa_chunks(image_metadata)

            image_path = str(Path(self.images_path, image_metadata.filename))
            image_width, image_height = Image.open(image_path).size

            for qa_chunk in qa_chunks:
                bbox_ids = {answer for qa_pair in qa_chunk for answer in qa_pair["answers"]}
                bboxes = self._prepare_boxes(
                    [pointing_annotations[bbox_id] for bbox_id in bbox_ids],
                    max_width=image_width,
                    max_height=image_height,
                )
                buffer.append(
                    {
                        "task": Task.gm_vqa.value,
                        "qa_pairs": [
                            {
                                "question": qa_pair["question"],
                                "answer": qa_pair["answer"],
                            }
                            for qa_pair in qa_chunk
                        ],
                        "region": [
                            {
                                "phrase": bbox_id,
                                "bbox": bbox,
                            }
                            for bbox_id, bbox in bboxes.items()
                        ],
                        "metadata": {
                            "answers": [
                                [str(ans) for ans in qa_pair["answers"]] for qa_pair in qa_chunk
                            ],
                            "answer": [str(qa_pair["answer"]) for qa_pair in qa_chunk],
                            "question_id": [qa_pair["question_id"] for qa_pair in qa_chunk],
                            "question_type": [qa_pair["question_type"] for qa_pair in qa_chunk],
                            "image_path": image_path,
                            "image_id": image_metadata.image_id,
                            "source": source,
                        },
                    }
                )

                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _telling_iterator(
        self,
        data: list[dict[str, Any]],
        chunk_size: int,
        source: str,
    ) -> Iterator[list[Any]]:
        # telling_annotations = self.load_telling_annotations(self._telling_answers_file_path)

        buffer = []
        for raw_metadata in data:
            image_metadata = Visual7WImageMetadata.model_validate(raw_metadata)
            if self.split_map[image_metadata.split] != self.split:
                continue

            qa_chunks = self._make_qa_chunks(image_metadata)

            image_path = str(Path(self.images_path, image_metadata.filename))
            # image_width, image_height = Image.open(image_path).size

            for qa_chunk in qa_chunks:
                buffer.append(
                    {
                        "task": Task.m_vqa.value,
                        "qa_pairs": [
                            {
                                "question": qa_pair["question"],
                                "answer": qa_pair["answer"],
                            }
                            for qa_pair in qa_chunk
                        ],
                        "region": None,
                        "metadata": {
                            "answers": [qa_pair["answers"] for qa_pair in qa_chunk],
                            "answer": [str(qa_pair["answer"]) for qa_pair in qa_chunk],
                            "question_id": [qa_pair["question_id"] for qa_pair in qa_chunk],
                            "question_type": [qa_pair["question_type"] for qa_pair in qa_chunk],
                            "image_path": image_path,
                            "image_id": image_metadata.image_id,
                            "source": source,
                            # Dont use the boxes for telling questions
                            # "boxes": {
                            #     qa_pair["question_id"]: self._prepare_boxes(
                            #         telling_annotations[qa_pair["question_id"]],
                            #         max_width=image_width,
                            #         max_height=image_height,
                            #         return_as="list",
                            #     )
                            #     for qa_pair in qa_chunk
                            # },
                        },
                    }
                )

                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    def _make_qa_chunks(self, image_metadata: Visual7WImageMetadata) -> list[list[dict[str, Any]]]:
        """Make the qa chunks for the given image."""
        # For the test split we will have a single chunk consisting of a single qa pair per image
        if self.split == DatasetSplits.TEST:
            qa_chunks = []
            for pair in image_metadata.qa_pairs:
                # Unforunately there is no predefined order of the candidate answers
                # In this case we will shuffle the answers to have a more robust evaluation
                answers = [*pair.multiple_choices, pair.answer]
                random.shuffle(answers)
                qa_chunks.append(
                    [
                        {
                            "question": pair.question,
                            "answer": pair.answer,
                            "question_type": pair.question_type,
                            "question_id": pair.qa_id,
                            "answers": answers,
                        }
                    ]
                )
        else:
            augmented_qa_pairs = self.augment_qa_pairs(image_metadata)
            # Shuffle the augmented qa_pairs so that in a single example there are different questions
            random.shuffle(augmented_qa_pairs)
            num_augmentations = len(augmented_qa_pairs[0])

            qa_chunks = []
            for aug_idx in range(num_augmentations):
                aug_pairs = []
                # Get one qa_pair per augmented_qa_pair
                for augmented_qa_pair in augmented_qa_pairs:
                    aug_pairs.append(augmented_qa_pair[aug_idx])

                # Split the qa_pairs into chunks of max_questions_per_image_in_turn
                qa_chunks.extend(
                    [
                        aug_pairs[step : step + self.max_questions_per_image_in_turn]
                        for step in range(0, len(aug_pairs), self.max_questions_per_image_in_turn)
                    ]
                )

        return qa_chunks

    def _prepare_boxes(
        self,
        bbox_list: list[dict[str, Any]],
        max_width: int,
        max_height: int,
        return_as: Literal["dict", "list"] = "dict",
        max_boxes: int | None = None,
    ) -> dict[str, list[float]] | list[list[float]]:
        object_bboxes = {} if return_as == "dict" else []
        if max_boxes is not None:
            # Sort the boxes by area and get the top max_boxes
            bbox_list = sorted(bbox_list, key=lambda x: x["width"] * x["height"], reverse=True)
            bbox_list = bbox_list[:max_boxes]

        for bbox in bbox_list:
            # Some of the bounding boxes are negative and should be rounded to closest edge
            x = max(0, bbox["x"])
            y = max(0, bbox["y"])
            width = min(max_width, bbox["width"])
            height = min(max_height, bbox["height"])

            box = BoxMode.convert(
                [x, y, width, height],
                from_mode=BoxMode.XYWH_ABS,
                to_mode=BoxMode.XYXY_ABS,
            )

            if return_as == "dict":
                object_bboxes[str(bbox["box_id"])] = box
            else:
                object_bboxes.append(box)
        return object_bboxes

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [example["task"] for example in examples],
            "image": [example["metadata"]["image_path"] for example in examples],
            "caption": [None for _ in examples],
            "qa_pairs": [example["qa_pairs"] for example in examples],
            "region": [example["region"] for example in examples],
            "relation": [None for _ in examples],
            "chat": [None for _ in examples],
            "source": [f"{self.source} {example['metadata']['source']}" for example in examples],
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

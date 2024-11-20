import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import AOKVQAMetadata, DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class AOKVQALoader(BaseLoader):
    """AOKVQA dataset loader."""

    annotation_url = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"

    annotation_files = {
        DatasetSplits.TRAIN: "aokvqa_v1p0_train.json",
        DatasetSplits.VALIDATION: "aokvqa_v1p0_val.json",
        DatasetSplits.TEST: "aokvqa_v1p0_test.json",
    }

    image_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
        DatasetSplits.TEST: "http://images.cocodataset.org/zips/test2015.zip",
    }

    split_dir = {
        DatasetSplits.TRAIN: "train2017",
        DatasetSplits.VALIDATION: "val2017",
        DatasetSplits.TEST: "test2015",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "aokvqa",
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
            self.download_manager.download_and_extract(self.annotation_url),
            self.annotation_files[self.split],
        )

        self.image_paths = self.download_manager.download_and_extract(self.image_urls)

    def _get_image_path(self, image_id: int) -> Path:
        for split, split_dir in self.split_dir.items():
            if split == DatasetSplits.TEST:
                image_path = Path(
                    self.image_paths[split],
                    self.split_dir[split],
                    f"COCO_test2015_{image_id:012d}.jpg",
                )
            else:
                image_path = Path(
                    self.image_paths[split], split_dir, f"{image_id}.jpg".zfill(16)  # noqa: WPS432
                )
            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"Image for {image_id} not found.")

    def _augment_qa_pairs(self, annotation_metadata: AOKVQAMetadata) -> list[dict[str, Any]]:
        if annotation_metadata.correct_choice_idx is None:
            raise ValueError("Correct choice index cannot be None.")

        correct_answer = annotation_metadata.choices[annotation_metadata.correct_choice_idx]
        qa_pairs = []
        for idx, swapped_answer in enumerate(annotation_metadata.choices):
            answers = annotation_metadata.choices.copy()
            answers[idx] = correct_answer
            answers[annotation_metadata.correct_choice_idx] = swapped_answer
            qa_pairs.append(
                {
                    "question": annotation_metadata.question,
                    "answer": correct_answer,
                    "answers": answers,
                    "correct_choice_idx": idx,
                }
            )

        return qa_pairs

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        buffer = []
        annotations = read_json(self.annotation_file)

        annotations = sorted(annotations, key=lambda x: x["image_id"])

        for annotation in annotations:
            image_path = self._get_image_path(annotation["image_id"])

            annotation_metadata = AOKVQAMetadata.model_validate(annotation)

            # There is a problem with this question id, the "a" is missing
            if annotation_metadata.question_id == "bCjS888uSeyRMczZageqTp":
                annotation_metadata.choices = ["n", "w", "a", "e"]

            # Training and validation: repeat each qa pair k times following LLaVA
            # https://arxiv.org/pdf/2310.03744.pdf
            if annotation_metadata.correct_choice_idx is not None:
                qa_pairs = self._augment_qa_pairs(annotation_metadata)
            # Testing, where there are no answers in the annotation
            else:
                qa_pairs = [
                    {
                        "question": annotation_metadata.question,
                        "answer": None,
                        "answers": annotation_metadata.choices,
                        "correct_choice_idx": None,
                    }
                ]

            for qa_pair in qa_pairs:
                buffer.append(
                    {
                        "image_path": str(image_path),
                        "qa_pairs": [
                            {"question": qa_pair["question"], "answer": qa_pair["answer"]}
                        ],
                        "metadata": {
                            "rationales": annotation_metadata.rationales,
                            "difficult_direct_answer": annotation_metadata.difficult_direct_answer,
                            "image_path": str(self._get_image_path(annotation_metadata.image_id)),
                            "answers": [qa_pair["answers"]],
                            "correct_choice_idx": qa_pair["correct_choice_idx"],
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
            "task": [Task.m_vqa.value for _ in examples],
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

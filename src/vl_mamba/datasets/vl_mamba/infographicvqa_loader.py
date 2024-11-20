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
    InfographicsVQAMetadata,
    Task,
)
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class InfographicVQALoader(BaseLoader):
    """InfographicVQA dataset loader."""

    image_urls = "https://huggingface.co/datasets/gpantaz/infographicvqa/resolve/main/infographicsvqa_images.tar.gz"

    question_urls = "https://huggingface.co/datasets/gpantaz/infographicvqa/resolve/main/infographicsvqa_qas.zip"

    question_files = {
        DatasetSplits.TRAIN: "infographicsVQA_train_v1.0.json",
        DatasetSplits.VALIDATION: "infographicsVQA_val_v1.0_withQT.json",
        DatasetSplits.TEST: "infographicsVQA_test_v1.0.json",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "infographics_vqa",
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
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )

        self.image_folder_path = Path(self.download_manager.download_and_extract(self.image_urls))

        self.questions_path = Path(
            self.download_manager.download_and_extract(self.question_urls),
            self.question_files[self.split],
        )

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:  # noqa: WPS210
        logger.info(f"Building {self.source} dataset for {self.split}.")
        questions_metadata = read_json(self.questions_path)
        # Pack the questions by image
        if self.split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            grouped_annotations = groupby(
                questions_metadata["data"], key=lambda x: x["image_local_name"]
            )
        # Dont pack the questions in the the test split
        else:
            grouped_annotations = [
                (metadata["image_local_name"], [metadata])
                for metadata in questions_metadata["data"]
            ]

        buffer = []
        for image_id, questions_iterator in grouped_annotations:
            qa_pairs = []
            all_metadata = []
            for raw_question in questions_iterator:
                question_metadata = InfographicsVQAMetadata.model_validate(raw_question)
                answer = None
                # For train/validation get the most common answer
                if question_metadata.answers:
                    answer = Counter(question_metadata.answers).most_common(1)[0][0]

                question = question_metadata.question
                question = f"{question[0].upper()}{question[1:]}"
                qa_pairs.append({"question": question_metadata.question, "answer": answer})

                all_metadata.append(question_metadata)

            buffer.append(
                {
                    "qa_pairs": qa_pairs,
                    "image_path": str(Path(self.image_folder_path, Path(image_id).name)),
                    "image_id": str(Path(image_id).stem),
                    "metadata": {
                        "image_path": str(Path(self.image_folder_path, Path(image_id).name)),
                        "image_url": [metadata.image_url for metadata in all_metadata],
                        "answers": [metadata.answers for metadata in all_metadata],
                        "question": [metadata.question for metadata in all_metadata],
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

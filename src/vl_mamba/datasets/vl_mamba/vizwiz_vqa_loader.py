import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task, VisWizVQAMetadata
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class VizWizVQALoader(BaseLoader):
    """VizWizVQALoader dataset loader."""

    annotation_url = "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"

    annotation_files = {
        DatasetSplits.TRAIN: "train.json",
        DatasetSplits.VALIDATION: "val.json",
        DatasetSplits.TEST: "test.json",
    }

    image_urls = {
        DatasetSplits.TRAIN: "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
        DatasetSplits.VALIDATION: "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
        DatasetSplits.TEST: "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip",
    }

    split_dir = {
        DatasetSplits.TRAIN: "train",
        DatasetSplits.VALIDATION: "val",
        DatasetSplits.TEST: "test",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "viswiz_vqa",
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
            download_config=DownloadConfig(cache_dir=cache_dir), record_checksums=False
        )

        self.annotation_file = Path(
            self.download_manager.download_and_extract(self.annotation_url),
            self.annotation_files[self.split],
        )

        self.image_path = Path(
            self.download_manager.download_and_extract(self.image_urls)[self.split],
            self.split_dir[self.split],
        )

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        """Build a chunk_size examples.

        VizWiz VQA has exactly 1 question per image so we dont need to group the annotations per
        image
        """
        logger.info(f"Building {self.source} dataset for {self.split}.")

        buffer = []
        annotations = read_json(self.annotation_file)
        for annotation in annotations:
            annotation_metadata = VisWizVQAMetadata.model_validate(annotation)
            answers, answer = None, None
            if annotation_metadata.answers:
                answers = [answer_dict["answer"] for answer_dict in annotation_metadata.answers]
                answers_counter = Counter(answers)
                answer = answers_counter.most_common(1)[0][0]

            buffer.append(
                {
                    # 1 question per image
                    "qa_pairs": [{"question": annotation_metadata.question, "answer": answer}],
                    "metadata": {
                        "image_path": str(Path(self.image_path, annotation_metadata.image)),
                        "answers": answers,
                        "answer_type": annotation_metadata.answer_type,
                    },
                }
            )

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {
            "task": [Task.vqa.value for _ in examples],
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

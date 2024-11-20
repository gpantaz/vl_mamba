import json
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request

import gdown
import pyarrow as pa
from loguru import logger
from overrides import overrides
from tqdm import tqdm

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, OCRVQAMetadata, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class OCRVQALoader(BaseLoader):
    """OCRVQA dataset loader."""

    annotation_url = (
        "https://docs.google.com/uc?export=download&id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q"
    )

    split_map = {
        DatasetSplits.TRAIN: 1,
        DatasetSplits.VALIDATION: 2,
        DatasetSplits.TEST: 3,
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "ocr_vqa",
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

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.dataset_json = Path(cache_dir, "dataset.json")
        if not self.dataset_json.exists():
            gdown.download(
                self.annotation_url,
                str(self.dataset_json),
                quiet=False,
            )

        dataset = read_json(self.dataset_json)
        self.dataset = [
            {"id": key, **metadata}
            for key, metadata in dataset.items()
            if self._example_in_split(metadata)
        ]

        self.download_images()

    def download_images(self) -> None:
        """Check if the images are already downloaded, if not download them."""
        with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
            # total argument for tqdm is just the number of submitted tasks:
            with tqdm(total=len(self.dataset), desc="Downloading images") as progress_bar:
                futures = {}
                for idx, example in enumerate(self.dataset):
                    future = executor.submit(self._download_file, example)
                    futures[future] = idx
                for _ in as_completed(futures):
                    progress_bar.update(1)

    def _download_file(self, example: dict[str, Any], repeats: int = 3) -> None:
        out_image = Path(self.cache_dir, f"{Path(example['imageURL']).name}")
        for _ in range(repeats):
            if not out_image.exists():
                request.urlretrieve(example["imageURL"], out_image)  # noqa: S310
            if out_image.exists():
                return

    def _example_in_split(self, metadata: dict[str, Any]) -> bool:
        return metadata["split"] == self.split_map[self.split]

    def _get_all_qa_pairs_for_image(
        self, questions_iterator: Iterator[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        qa_pairs = []
        for question_metadata in list(questions_iterator):
            ocrvqa_metadata = OCRVQAMetadata.model_validate(question_metadata)
            for question, answer in zip(ocrvqa_metadata.questions, ocrvqa_metadata.answers):
                if self._should_keep_question(question):
                    qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    def _should_keep_question(self, question: str) -> bool:
        """Check if the question should be kept.

        After some manual inspection, we are filtering out questions related to the type of the
        book, as it can be generally hard to identify the type of the book its cover.
        """
        should_keep = (
            # 'What type of book is this?'
            "type of book" not in question.lower()
            # Is this a sociopolitical book?
            and "type of the book" not in question.lower()
            # Is this book related to Science & Math?
            and "this book related to" not in question.lower()
        )
        return should_keep

    def _preprocess_dataset(self) -> Iterator[Any]:
        """Preprocess the dataset.

        Just a simple groupby operation to group all questions for the same image together.
        """
        filtered_json = Path(self.cache_dir, "images_filtered.json")
        if not filtered_json.exists():
            raise ValueError(
                "Filtered json does not exist, run: scripts/filter_out_ocrvqa_images.py first!"
            )

        self.filtered_images = read_json(filtered_json)
        # We are going to drop all the examples in the dataset with duplicate image URLs
        # since they have different annotations and we don't want to deal with that
        url_to_keys = defaultdict(list)
        for metadata in self.dataset:
            url_to_keys[metadata["imageURL"]].append(metadata["id"])

        return [example for example in self.dataset if len(url_to_keys[example["imageURL"]]) == 1]

        # self.dataset = sorted(self.dataset, key=lambda x: x["imageURL"])
        # return groupby(self.dataset, key=lambda x: x["imageURL"])

    @overrides(check_signature=False)
    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        annotations = self._preprocess_dataset()

        buffer = []
        for raw_example in annotations:
            ocrvqa_metadata = OCRVQAMetadata.model_validate(raw_example)
            if not self.filtered_images[ocrvqa_metadata.image_url]:
                continue

            qa_pairs = [
                {"question": question, "answer": answer}
                for question, answer in zip(ocrvqa_metadata.questions, ocrvqa_metadata.answers)
            ]
            if not qa_pairs:
                continue
            # At this point we need to group all questions for the same image, each element in the questions_iterator
            # has multiple questions so we need to group all of them together
            # qa_pairs = self._get_all_qa_pairs_for_image(questions_iterator)

            buffer.append(
                {
                    "qa_pairs": qa_pairs,
                    "image_path": str(
                        Path(self.cache_dir, f"{Path(ocrvqa_metadata.image_url).name}")
                    ),
                    "image_url": ocrvqa_metadata.image_url,
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
                    {"image_path": example["image_path"], "image_url": example["image_url"]},
                    default=json_serializer,
                    indent=2,
                )
                for example in examples
            ],
        }

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

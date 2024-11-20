import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class SyntheticVGLoader(BaseLoader):
    """SyntheticVG dataset loader."""

    annotation_urls = {
        DatasetSplits.TRAIN: {
            20: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen20.json?download=true",
            30: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen30.json?download=true",
            50: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen50.json?download=true",
            70: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen70.json?download=true",
            80: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen80.json?download=true",
            100: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen100.json?download=true",
            200: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen200.json?download=true",
            300: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen300.json?download=truen",
            400: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen400.json?download=true",
            500: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen500.json?download=true",
            576: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen576.json?download=true",
            600: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/train_synthetic_vg_seqlen600.json?download=true",
        },
        DatasetSplits.VALIDATION: {
            20: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen20.json?download=true",
            30: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen30.json?download=true",
            50: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen50.json?download=true",
            70: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen70.json?download=true",
            80: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen80.json?download=true",
            100: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen100.json?download=true",
            200: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen200.json?download=true",
            300: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen300.json?download=truen",
            400: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen400.json?download=true",
            500: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen500.json?download=true",
            576: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen576.json?download=true",
            600: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/validation_synthetic_vg_seqlen600.json?download=true",
        },
        DatasetSplits.TEST: {
            20: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen20.json?download=true",
            30: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen30.json?download=true",
            50: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen50.json?download=true",
            70: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen70.json?download=true",
            80: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen80.json?download=true",
            100: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen100.json?download=true",
            200: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen200.json?download=true",
            300: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen300.json?download=truen",
            400: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen400.json?download=true",
            500: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen500.json?download=true",
            576: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen576.json?download=true",
            600: "https://huggingface.co/gpantaz/vl-mamba-synthetic/resolve/main/test_synthetic_vg_seqlen600.json?download=true",
        },
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "synthetic_vg",
        synthetic_prefix: str = "<syn_{token}>",
        positional_prefix: str = "<pos_{token}>",
        seq_len: int = 576,
        chunk_size: int = 1,
        num_proc: int = 1,
    ) -> None:
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

        ann_path = self.annotation_urls[self.split][seq_len]
        self.annotation_file = Path(
            self.download_manager.download_and_extract(ann_path),
        )

        logger.info(f"Annotation file: {self.annotation_file}")

        self.seq_len = seq_len
        self.synthetic_prefix = synthetic_prefix
        self.positional_prefix = positional_prefix

    def verify(self, example: dict[str, Any]) -> None:
        """Verify the example."""
        # Check the sequence length
        seq_len = len(example["sequence"])
        if seq_len != self.seq_len:
            raise AssertionError(f"Sequence length {seq_len} is not {self.seq_len}")

        # Check that all the tokens are unique
        if len(set(example["sequence"])) != self.seq_len:
            raise AssertionError("Tokens are not unique")

        # Check that the target token is within the sequence length
        if example["query"] not in example["sequence"]:
            raise AssertionError(
                f"Target {example['query']} is not in the sequence {example['sequence']}"
            )

        # Check that the position of the target token is correct
        position = f"<pos_{example['sequence'].index(example['query'])}>"
        if example["label"] != position:
            raise AssertionError(f"Label {example['label']} is not correct {position}")

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        buffer = []
        ann = read_json(self.annotation_file)

        for synth_seq, synth_label in zip(
            ann["target_sequences"], ann["target_tokens"], strict=False
        ):
            seq = [self.synthetic_prefix.format(token=token) for token in synth_seq]

            example_metadata = {
                "sequence": seq,
                "label": self.positional_prefix.format(token=synth_label),
                "query": seq[synth_label],
            }

            self.verify(example_metadata)

            buffer.append({"metadata": example_metadata})

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.synthetic_vg.value for _ in examples],
            "image": [None for _ in examples],
            "qa_pairs": [None for _ in examples],
            "caption": [None for _ in examples],
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

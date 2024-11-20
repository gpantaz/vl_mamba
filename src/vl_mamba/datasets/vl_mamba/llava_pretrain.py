import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, LLavaPretrainModel, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class LLAVAPretrainLoader(BaseLoader):
    """Loader for LLAVA pretrain dataset.

    >>> from datasets import load_dataset
    >>> dataset = load_dataset("liuhaotian/LLaVA-Pretrain")

    datasets.exceptions.DatasetGenerationCastError: An error occurred while generating the dataset

    All the data files must have the same columns, but at some point there are 2 new columns (url, blip_caption) and 1 missing columns (conversations).

    This happened while the json dataset builder was generating data using

    hf://datasets/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k_meta.json (at revision 70f9d1e5e1a697fe35830875cfc7de1dd590d727)

    Please either edit the data files to have matching columns, or separate them into different configurations (see docs at https://hf.co/docs/hub/datasets-manual-configuration#multiple-configurations)
    """

    urls = {
        "annotations": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json",
        "metadata": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k_meta.json",
        "images": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true",
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "llava_pretrain",
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
            download_config=DownloadConfig(cache_dir=cache_dir),
        )

        self.annotation_paths = self.download_manager.download_and_extract(self.urls)

    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[LLavaPretrainModel]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        buffer = []

        annotations = read_json(self.annotation_paths["annotations"])
        metadata = read_json(self.annotation_paths["metadata"])

        if len(annotations) != len(metadata):
            raise AssertionError(f"Expecting {len(annotations)} == {len(metadata)}")

        for annotation, annotation_metadata in zip(annotations, metadata):
            instance = LLavaPretrainModel(annotation=annotation, metadata=annotation_metadata)
            buffer.append(instance)
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, batch: list[LLavaPretrainModel]) -> dict[str, list[Any]]:
        """Convert a LLavaPretrainModel instance to the VL-Mamba format."""
        dataset_examples: dict[str, list[Any]] = {
            data_key: [] for data_key in DatasetFeatures.keys()
        }
        for example in batch:
            image_path = str(Path(self.annotation_paths["images"], example.annotation["image"]))
            example.metadata.update({"image_path": image_path, "prompt": example.prompt})

            dataset_examples["task"].append(Task.captioning.value)
            dataset_examples["image"].append(image_path)
            dataset_examples["caption"].append(example.caption)
            dataset_examples["qa_pairs"].append(None)
            dataset_examples["region"].append(None)
            dataset_examples["relation"].append(None)
            dataset_examples["chat"].append(None)
            dataset_examples["source"].append(self.source)
            dataset_examples["metadata"].append(
                json.dumps(
                    example.metadata,
                    default=json_serializer,
                    indent=2,
                )
            )
        return dataset_examples

    def _generate_tables(
        self, examples: list[LLavaPretrainModel], **kwargs: dict[str, Any]
    ) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

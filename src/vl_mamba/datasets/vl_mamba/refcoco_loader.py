import json
import pickle
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides

from vl_mamba.datamodels.datamodels import (
    DatasetFeatures,
    DatasetSplits,
    RefCOCOExpression,
    RefCOCOImageMetadata,
    RefCOCORegion,
    Task,
)
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.boxes import BoxMode
from vl_mamba.utils.io import json_serializer, read_json


class RefCOCOLoader(BaseLoader):
    """RefCOCOg dataset loader."""

    annotation_url = {
        "refcoco": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
        "refcoco+": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
        "refcocog": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
    }

    annotation_files = {
        "refcoco": "refs(unc).p",
        "refcoco+": "refs(unc).p",
        "refcocog": "refs(umd).p",
    }

    image_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
    }

    split_dir = {DatasetSplits.TRAIN: "train2017", DatasetSplits.VALIDATION: "val2017"}

    split_map = {
        "train": DatasetSplits.TRAIN,
        "val": DatasetSplits.VALIDATION,
        "test": DatasetSplits.TEST,
        "testA": DatasetSplits.TEST_DEV,
        "testB": DatasetSplits.TEST_STD,
    }

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: Literal["refcoco", "refcoco+", "refcocog"] = "refcoco",
        chunk_size: int = 1,
        num_proc: int = 1,
        task: str = Task.visual_grounding.value,
    ) -> None:
        super().__init__(
            source=source,
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

        self.source = source
        self.task = task

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir),
            record_checksums=False,
        )

        self.annotation_file = Path(
            self.download_manager.download_and_extract(self.annotation_url[self.source]),
            self.source,
            self.annotation_files[self.source],
        )

        self.instance_file = Path(
            self.download_manager.download_and_extract(self.annotation_url[self.source]),
            self.source,
            "instances.json",
        )

        self.image_paths = self.download_manager.download_and_extract(self.image_urls)

    def read_refcoco_region_annotations(
        self, annotation_path: Path
    ) -> dict[str, list[RefCOCORegion]]:
        """Read the annotations for the regions associated with referring expressions.

        The bbox coordinates are [x,y,w,h] where xy are the cooridinates of the bottom left corner.
        Return metadata as a dictionary with annotation ids as the keys.
        """
        data = read_json(annotation_path)["annotations"]

        regions: dict[str, list[RefCOCORegion]] = defaultdict(list)
        for datum in data:
            regions[datum["image_id"]].append(
                RefCOCORegion(
                    annotation_id=str(datum["id"]),
                    image_id=str(datum["image_id"]),
                    x=datum["bbox"][0],
                    y=datum["bbox"][1],
                    w=datum["bbox"][2],
                    h=datum["bbox"][3],
                    category_id=datum["category_id"],
                )
            )
        return regions

    def read_refcoco_referring_expressions(
        self, referring_expressions_path: Path
    ) -> dict[str, list[RefCOCOExpression]]:
        """Read the RefCOCO referring expressions and group them per split."""
        with open(referring_expressions_path, "rb") as in_file:
            annotations = pickle.load(in_file)  # noqa: S301

        referring_expressions = defaultdict(list)

        for instance in annotations:
            # Get the split of the instance, because all referring expressions are stored in a single file
            split = self.split_map[instance["split"]]

            if split != self.split:
                continue

            # Each instance is associated with multiple referring expressions
            for sentence in instance["sentences"]:
                referring_expressions[instance["image_id"]].append(
                    RefCOCOExpression(
                        sentence=sentence["raw"],
                        sentence_id=str(sentence["sent_id"]),
                        annotation_id=str(instance["ann_id"]),
                    )
                )
        return referring_expressions

    def read_refcoco_image_metadata(
        self, annotation_path: Path
    ) -> dict[str, RefCOCOImageMetadata]:
        """Read the metadata for the RefCOCO images.

        Return metadata as a dictionary with image ids as the keys.
        """
        data = read_json(annotation_path)["images"]

        image_metadata = {}
        for image in data:
            image_path = self._get_image_path(image)

            image_metadata[image["id"]] = RefCOCOImageMetadata(
                image_path=str(image_path),
                image_id=str(image["id"]),
                width=image["width"],
                height=image["height"],
                url=image["coco_url"],
            )
        return image_metadata

    def _get_image_path(self, image: dict[str, Any]) -> Path:
        for split, split_dir in self.split_dir.items():
            image_path = Path(self.image_paths[split], split_dir, f"{image['id']}.jpg".zfill(16))
            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"Image for {image['id']} not found.")

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split} and task {self.task}.")

        # Dict where keys are annotation ids
        regions_metadata = self.read_refcoco_region_annotations(annotation_path=self.instance_file)

        # List of referring expressions
        referring_expressions = self.read_refcoco_referring_expressions(
            referring_expressions_path=self.annotation_file
        )

        # Dict where keys are image ids
        image_metadata = self.read_refcoco_image_metadata(annotation_path=self.instance_file)
        buffer = []
        for image_id, img_metadata in image_metadata.items():
            if not img_metadata:
                continue

            region_metadata = regions_metadata.get(image_id)
            if not region_metadata:
                continue

            ref_expressions = referring_expressions.get(image_id)
            if not ref_expressions:
                continue

            instance = {
                "image_metadata": img_metadata,
                "region": region_metadata,
                "referring_expression": referring_expressions[image_id],
            }

            buffer.append(instance)
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        dataset_examples: dict[str, list[Any]] = {data_key: [] for data_key in DatasetFeatures}
        for example in examples:
            image_path = str(example["image_metadata"].image_path)
            regions, metadata = self._make_region_annotation(example)

            if self.split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
                dataset_examples["task"].append(Task.visual_grounding.value)
                dataset_examples["image"].append(image_path)
                dataset_examples["caption"].append(None)
                dataset_examples["qa_pairs"].append(None)
                dataset_examples["region"].append(regions)
                dataset_examples["relation"].append(None)
                dataset_examples["chat"].append(None)
                dataset_examples["source"].append(self.source)
                dataset_examples["metadata"].append(
                    json.dumps(
                        metadata,
                        default=json_serializer,
                        indent=2,
                    )
                )
            else:
                for region in regions:
                    dataset_examples["task"].append(Task.visual_grounding.value)
                    dataset_examples["image"].append(image_path)
                    dataset_examples["caption"].append(None)
                    dataset_examples["qa_pairs"].append(None)
                    dataset_examples["region"].append([region])
                    dataset_examples["relation"].append(None)
                    dataset_examples["chat"].append(None)
                    dataset_examples["source"].append(self.source)
                    dataset_examples["metadata"].append(
                        json.dumps(
                            metadata,
                            default=json_serializer,
                            indent=2,
                        )
                    )
        return dataset_examples

    def _make_region_annotation(
        self, example: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        regions = []
        annotation2region = {region.annotation_id: region for region in example["region"]}
        expression2region = {
            expression.annotation_id: expression for expression in example["referring_expression"]
        }

        image_path = str(example["image_metadata"].image_path)
        example_metadata = example["image_metadata"].dict()

        category_ids = []
        for annotation_id, region in annotation2region.items():
            expression = expression2region.get(annotation_id)
            if not expression:
                continue

            object_bbox = BoxMode.convert(
                [region.x, region.y, region.w, region.h],
                from_mode=BoxMode.XYWH_ABS,
                to_mode=BoxMode.XYXY_ABS,
            )

            category_ids.append(region.category_id)

            regions.append({"bbox": object_bbox, "phrase": expression.sentence})

        example_metadata["image_path"] = image_path
        example_metadata["category_ids"] = category_ids

        return regions, example_metadata

    def _generate_tables(self, examples: list[Any], **kwargs: dict[str, Any]) -> pa.Table:
        output_batch = self._generate_examples(examples)
        return pa.table(DatasetFeatures.encode_batch(output_batch))

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Image, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.utils.download_manager import DownloadManager
from loguru import logger

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import DatasetsLoader
from vl_mamba.utils.boxes import BoxMode, RawBoxType
from vl_mamba.utils.io import json_serializer


class VisualGenomeLoader(DatasetsLoader):
    """Visual Genome dataset loader."""

    annotation_url = (
        "https://huggingface.co/datasets/gpantaz/karpathy_split/resolve/main/caption_datasets.zip"
    )

    task_config_map = {
        Task.dense_captioning.value: "region_descriptions_v1.2.0",
        Task.visual_grounding.value: "region_descriptions_v1.2.0",
        Task.relationship_detection.value: "relationships_v1.2.0",
        Task.vqa.value: "question_answers_v1.2.0",
        Task.object_detection.value: "objects_v1.2.0",
    }

    # Similar to OFA https://arxiv.org/pdf/2202.03052.pdf
    max_area = 16384

    def __init__(
        self,
        split: str,
        task: str,
        num_proc: int,
        dl_manager: DownloadManager,
        cache_dir: Path,
        datasets_batch_size: int = 1000,
        chunk_size: int = 1,
        max_annotations_per_image: int = 10,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            dataset_name="visual_genome",
            split=split,
            datasets_batch_size=datasets_batch_size,
            num_proc=num_proc,
            config_name=None,
        )
        self.task = task
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.max_annotations_per_image = max_annotations_per_image

        Path(dl_manager.download_config.cache_dir).mkdir(parents=True, exist_ok=True)
        karpathy_coco_file = Path(
            dl_manager.download_and_extract(self.annotation_url), "dataset_coco.json"
        )

        validation_images = []
        with open(karpathy_coco_file, encoding="utf-8") as f:
            annotations = json.load(f)
            for annotation in annotations["images"]:
                if annotation["split"] == "val":
                    validation_images.append(int(annotation["cocoid"]))
        self.validation_images = validation_images

    def cast_to_vlmamba_features(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Cast the dataset to the format expected by vlmamba."""
        dataset_examples: dict[str, list[Any]] = {data_key: [] for data_key in DatasetFeatures}

        df = pd.DataFrame.from_dict(batch)
        for _, instance in df.iterrows():
            if self.task in {Task.dense_captioning.value, Task.visual_grounding.value}:
                annotations = self._process_region_instance(instance)
                for ann_key, ann_value in annotations.items():
                    dataset_examples[ann_key].extend(ann_value)
            elif self.task == Task.vqa.value:
                instance_annotation = self._process_vqa_instance(instance)
                for instance_key, instance_value in instance_annotation.items():
                    dataset_examples[instance_key].extend(instance_value)
            elif self.task == Task.relationship_detection.value:
                instance_annotation = self._process_relation_instance(instance)
                for instance_key, instance_value in instance_annotation.items():
                    dataset_examples[instance_key].extend(instance_value)
            else:
                raise ValueError(f"Task {self.task} not supported by VG.")

        return dataset_examples

    def fetch_dataset_from_hf(self, config_name: str) -> tuple[Dataset, str]:
        """Fetch the visual genome dataset from Hugging Face.

        VG doesnt have a validation split, so we use the COCO validation split.
        """
        dataset = load_dataset(
            self.dataset_name,
            config_name,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        ).cast_column("image", Image(decode=False))["train"]

        if self.split == DatasetSplits.VALIDATION:
            dataset = dataset.filter(
                lambda example: example["coco_id"] is not None
                and example["coco_id"] in self.validation_images
            )
            return dataset

        dataset = dataset.filter(
            lambda example: example["coco_id"] is None
            or example["coco_id"] not in self.validation_images
        )

        return dataset

    def should_skip_instance(self, annotation: dict[str, Any]) -> bool:
        """Determine if an instance should be skipped based on the bbox area.

        Similar to OFA, we skip instances that have a bbox area less than 16384.
        https://arxiv.org/pdf/2202.03052.pdf
        """
        if self.task in {Task.dense_captioning.value, Task.visual_grounding.value}:
            bbox_area = annotation["width"] * annotation["height"]
            return bbox_area > self.max_area
        return False

    def convert_bbox(self, annotation: dict[str, Any]) -> RawBoxType:
        """Converts an annotation from XYWH to XYXY format.

        There is a problem with visual genome where some coordinates have negative values.
        https://github.com/ranjaykrishna/visual_genome_python_driver/issues/21

        The solution is that any out of bounds indices should be rounded to the nearest edge of the image.
        Since the bbox has the XYWH format, we can just set the negative values to 0.
        """
        return BoxMode.convert(
            [
                max(annotation["x"], 0),
                max(annotation["y"], 0),
                annotation["width"],
                annotation["height"],
            ],
            from_mode=BoxMode.XYWH_ABS,
            to_mode=BoxMode.XYXY_ABS,
        )

    def _build_rows_iterator(
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        config_name = self.task_config_map[self.task]
        logger.info(f"Building VG for task: {self.task}.")
        dataset = self.fetch_dataset_from_hf(config_name)

        buffer = []
        for row in dataset:
            buffer.append(row)
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        if buffer:
            yield buffer

    def _process_region_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Process the instance for dense captioning and visual grounding.

        The instance is processed in chunks of where each element in the chunk has at most
        self.max_annotations_per_image annotations.
        """
        instance_annotations: dict[str, list[Any]] = {data_key: [] for data_key in DatasetFeatures}

        annotations = [
            region for region in instance["regions"] if not self.should_skip_instance(region)
        ]
        annotation_chunks = [
            annotations[step : step + self.max_annotations_per_image]
            for step in range(0, len(annotations), self.max_annotations_per_image)
        ]

        if not annotation_chunks:
            return {}

        for annotation_chunk in annotation_chunks:
            chunk_regions = []
            for annotation in annotation_chunk:
                # convert the region from (x, y, w, h) to (xmin, ymin, xmax, ymax)
                bbox = self.convert_bbox(annotation)
                chunk_regions.append({"phrase": annotation["phrase"], "bbox": bbox})

            instance_annotations["region"].append(chunk_regions)

        instance_annotations = self._field_common_fields(instance, instance_annotations)
        instance_annotations = self._fill_empty_fields(instance_annotations)
        return instance_annotations

    def _process_vqa_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Process the instance for visual question answering.

        The instance is processed in chunks of where each element in the chunk has at most
        self.max_annotations_per_image annotations.
        """
        instance_annotations: dict[str, list[Any]] = {data_key: [] for data_key in DatasetFeatures}

        annotation_chunks = [
            instance["qas"][step : step + self.max_annotations_per_image]
            for step in range(0, len(instance["qas"]), self.max_annotations_per_image)
        ]

        if not annotation_chunks:
            return {}

        for annotation_chunk in annotation_chunks:
            chunk_qa_pairs = [
                {"question": annotation["question"], "answer": annotation["answer"]}
                for annotation in annotation_chunk
            ]

            instance_annotations["qa_pairs"].append(chunk_qa_pairs)

        instance_annotations = self._field_common_fields(instance, instance_annotations)
        instance_annotations = self._fill_empty_fields(instance_annotations)
        return instance_annotations

    def _process_relation_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        """Process the instance for relationship detection.

        The instance is processed in chunks of where each element in the chunk has at most
        self.max_annotations_per_image annotations.
        """
        instance_annotations: dict[str, list[Any]] = {data_key: [] for data_key in DatasetFeatures}

        annotation_chunks = [
            instance["relationships"][step : step + self.max_annotations_per_image]
            for step in range(0, len(instance["relationships"]), self.max_annotations_per_image)
        ]

        if not annotation_chunks:
            return {}

        for annotation_chunk in annotation_chunks:
            chunk_relation_pairs = []
            for annotation in annotation_chunk:
                subject_names = annotation["subject"]["names"]
                object_names = annotation["object"]["names"]
                multiple_names = len(subject_names) > 1 or len(object_names) > 1
                if multiple_names:
                    raise ValueError("Expecting a single name for both subject and object.")

                subject_bbox = self.convert_bbox(
                    {
                        "x": annotation["subject"]["x"],
                        "y": annotation["subject"]["y"],
                        "width": annotation["subject"]["w"],
                        "height": annotation["subject"]["h"],
                    },
                )
                object_bbox = self.convert_bbox(
                    {
                        "x": annotation["object"]["x"],
                        "y": annotation["object"]["y"],
                        "width": annotation["object"]["w"],
                        "height": annotation["object"]["h"],
                    },
                )
                chunk_relation_pairs.append(
                    {
                        "subject_name": annotation["subject"]["names"][0],
                        "subject_bbox": subject_bbox,
                        "object_name": annotation["object"]["names"][0],
                        "object_bbox": object_bbox,
                        "predicate": annotation["predicate"],
                    }
                )

            instance_annotations["relation"].append(chunk_relation_pairs)

        instance_annotations = self._field_common_fields(instance, instance_annotations)
        instance_annotations = self._fill_empty_fields(instance_annotations)
        return instance_annotations

    def _field_common_fields(
        self, instance: dict[str, Any], instance_annotations: dict[str, Any]
    ) -> dict[str, Any]:
        """Fill the common fields for the instance.

        These fields are the same for all tasks.
        """
        instance_annotations["image"].append(instance["image"]["path"])
        instance_annotations["task"].append(self.task)

        instance_annotations["source"].append(self.source)
        instance_annotations["metadata"].append(
            json.dumps(
                {
                    "image_id": instance["image_id"],
                    "image_path": instance["image"]["path"],
                    "width": instance["width"],
                    "height": instance["height"],
                    "coco_id": instance["coco_id"],
                    "flickr_id": instance["flickr_id"],
                },
                default=json_serializer,
                indent=2,
            )
        )
        return instance_annotations

    def _fill_empty_fields(
        self, instance_annotations: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Fill empty fields with None."""
        max_len = max([len(ann) for ann in instance_annotations.values()])
        for key in instance_annotations:
            # Use the first value for `task`, `image` and `source`
            if key in {"task", "image", "source", "metadata"}:
                instance_annotations[key] = [instance_annotations[key][0] for _ in range(max_len)]
            # Pad the annotation fields with None
            elif not instance_annotations[key]:
                instance_annotations[key] = [None for _ in range(max_len)]

        return instance_annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from datasets.utils.download_manager import DownloadConfig, DownloadManager
from loguru import logger
from overrides import overrides
from PIL import Image

from vl_mamba.datamodels.datamodels import DatasetFeatures, DatasetSplits, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer


class LocalizedNarrativesLoader(BaseLoader):
    """LocalizedNarratives dataset loader."""

    annotation_urls = {
        DatasetSplits.TRAIN: [
            "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00000-of-00004.jsonl",
            "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00001-of-00004.jsonl",
            "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00002-of-00004.jsonl",
            "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00003-of-00004.jsonl",
        ],
        DatasetSplits.VALIDATION: [
            "https://storage.googleapis.com/localized-narratives/annotations/coco_val_localized_narratives.jsonl",
        ],
    }

    image_urls = {
        DatasetSplits.TRAIN: "http://images.cocodataset.org/zips/train2017.zip",
        DatasetSplits.VALIDATION: "http://images.cocodataset.org/zips/val2017.zip",
    }

    split_dir = {DatasetSplits.TRAIN: "train2017", DatasetSplits.VALIDATION: "val2017"}

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "localized_narratives",
        chunk_size: int = 1,
        num_proc: int = 1,
        segment_window_size: float = 0.4,
    ) -> None:
        super().__init__(
            source=source,
            split=split,
            num_proc=num_proc,
            chunk_size=chunk_size,
            writer_batch_size=writer_batch_size,
        )

        self.segment_window_size = segment_window_size

        self.download_manager = DownloadManager(
            download_config=DownloadConfig(cache_dir=cache_dir),
        )

        self.annotation_files = [
            Path(filepath)
            for filepath in self.download_manager.download_and_extract(
                self.annotation_urls[self.split]
            )
        ]

        self.image_path = Path(
            self.download_manager.download_and_extract(self.image_urls[self.split]),
            self.split_dir[self.split],
        )

    def group_trace_by_segment_window(
        self, trace: list[dict[str, float]]
    ) -> list[list[dict[str, float]]]:
        """Group the trace by segment (in secs)."""
        # If the trace has only one point, return it as a single segment
        if len(trace) == 1:
            return [trace]

        # If the trace lasts less than the window size, return it as a single segment
        if trace[-1]["t"] < self.segment_window_size:
            return [trace]

        segments = []
        idx = 0
        step_window = int(trace[0]["t"] // self.segment_window_size) + 1
        while idx < len(trace) - 1:
            curr_segments = [trace[idx]]
            for jdx in range(idx + 1, len(trace)):  # noqa: WPS518
                end_time = trace[jdx]["t"]
                if end_time > step_window * self.segment_window_size:
                    break
                curr_segments.append(trace[jdx])

            step_window += 1
            idx = jdx  # noqa: WPS441
            segments.append(curr_segments)

        return segments

    def validate_positions_for_trace(
        self, trace: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Validate the positions for a trace.

        A trace is a list of segments, one between each time the mouse pointer enters the image and
        goes away from it. Each trace segment is represented as a list of timed points, i.e. {x, y,
        t}, where x and y are the normalized image coordinates (with origin at the top-left corner
        of the image) and t is the time in seconds since the start of the recording. Please note
        that the coordinates can go a bit beyond the image, i.e. <0 or >1, as we recorded the mouse
        traces including a small band around the image.
        """
        positions = []
        for point in trace:
            point["x"] = max(0, point["x"])
            point["x"] = min(1, point["x"])

            point["y"] = max(0, point["y"])
            point["y"] = min(1, point["y"])

            positions.append({"x": point["x"], "y": point["y"], "t": point["t"]})
        return positions

    def get_absolute_position_from_relative_positions(
        self, trace: list[dict[str, float]], width: int, height: int
    ) -> list[dict[str, float]]:
        """Get the absolute position from a trace."""
        positions = []
        for point in trace:
            point["x"] = min(max(0, point["x"]), 1)
            point["y"] = min(max(0, point["y"]), 1)

            absolute_positions = {
                "x": point["x"] * width,
                "y": point["y"] * height,
                "t": point["t"],
            }
            positions.append(absolute_positions)
        return positions

    def _get_image_path(self, image_id: str) -> Path:
        image_path = Path(self.image_path, f"{image_id:012d}.jpg")
        if image_path.exists():
            return image_path
        raise FileNotFoundError(f"Image {image_path} not found.")

    @overrides(check_signature=False)
    def _build_rows_iterator(  # noqa: WPS231
        self, chunk_size: int, **kwargs: dict[str, Any]
    ) -> Iterator[list[Any]]:
        logger.info(f"Building {self.source} dataset for {self.split}.")
        buffer = []
        for annotation_file in self.annotation_files:
            annotations = pd.read_json(path_or_buf=annotation_file, lines=True)
            all_lens = []
            for _, instance in annotations.iterrows():
                image_path = self._get_image_path(instance.image_id)

                with Image.open(image_path) as image:
                    width, height = image.size

                if not instance.traces:
                    logger.info(
                        f"Skipping instance with image id: {instance.image_id} as it has no traces."
                    )
                    continue

                traces_downsampled = []
                for trace in instance.traces:
                    # Step 1: validate the relative positions
                    trace_with_relative_positions = self.validate_positions_for_trace(trace)

                    # Step 2: convert the relative positions to absolute positions
                    trace_with_absolute_positions = (
                        self.get_absolute_position_from_relative_positions(
                            trace_with_relative_positions, width=width, height=height
                        )
                    )

                    # Step 3: group the trace by segment window
                    grouped = self.group_trace_by_segment_window(trace_with_absolute_positions)

                    # Step 4: Get the middle point of each segment
                    traces_downsampled.append([group[len(group) // 2] for group in grouped])

                all_lens.append(sum([len(x) for x in traces_downsampled]))
                buffer.append(
                    {
                        "image_path": str(image_path),
                        "caption": instance.caption,
                        "metadata": {
                            "traces_downsampled": traces_downsampled,
                            "traces": instance.traces,
                            "timed_caption": instance.timed_caption,
                            "image_path": str(image_path),
                        },
                    }
                )

                if len(buffer) == chunk_size:
                    yield buffer
                    buffer = []

            if len(buffer) == chunk_size:
                yield buffer
                buffer = []

        # Getting the last batch
        if buffer:
            yield buffer

    @overrides(check_signature=False)
    def _generate_examples(self, examples: list[Any]) -> dict[str, list[Any]]:
        return {
            "task": [Task.captioning for _ in examples],
            "image": [example["image_path"] for example in examples],
            "caption": [example["caption"] for example in examples],
            "qa_pairs": [None for _ in examples],
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

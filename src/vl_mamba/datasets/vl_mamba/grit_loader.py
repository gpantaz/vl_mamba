import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
from overrides import overrides

from vl_mamba.datamodels.datamodels import DatasetFeatures, GRITMetadata, Task
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.utils.io import json_serializer, read_json


class GRITLoader(BaseLoader):
    """GRIT dataset loader."""

    def __init__(
        self,
        split: str,
        writer_batch_size: int,
        cache_dir: Path,
        source: str = "grit",
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

        self.image_shard_cache = cache_dir

        self.coords_placeholder = "<coords_{idx}>"
        self.object_sep = "||"

    def modify_caption(self, caption: str, noun_chunks: list[list[float]]) -> str:
        """Add placeholder coordinates to the caption."""
        modified_caption = ""
        noun_chunks = sorted(
            [(noun_chunk[0], noun_chunk[1]) for noun_chunk in noun_chunks], key=lambda x: x[0]
        )

        noun_chunks_counter = Counter(noun_chunks)

        if noun_chunks[0][0] > 0:
            modified_caption = caption[: int(noun_chunks[0][0])]

        idx = 0
        offset = 0
        # Since there can be multiple noun chunks with the same start and end, we need to
        # iterate through the unique noun chunks and add the placeholders to
        # every noun chunk and its duplicates
        unique_noun_chunks = list(set(noun_chunks))
        unique_noun_chunks.sort(key=lambda x: x[0])
        while idx < len(unique_noun_chunks):
            start = int(unique_noun_chunks[idx][0])
            end = int(unique_noun_chunks[idx][1])
            if idx == 0:
                coord_str = "".join(
                    self.coords_placeholder.format(idx=step + offset)
                    for step in range(noun_chunks_counter[unique_noun_chunks[idx]])
                )
                obj_string = f"{self.object_sep}{caption[start:end]} {coord_str}{self.object_sep}"
                modified_caption = f"{modified_caption}{obj_string}"
            elif idx < len(unique_noun_chunks):
                prev_end = int(unique_noun_chunks[idx - 1][1])

                coord_str = "".join(
                    self.coords_placeholder.format(idx=step + offset)
                    for step in range(noun_chunks_counter[unique_noun_chunks[idx]])
                )
                prefix = f"{modified_caption}{caption[prev_end:start]}"
                modified_caption = (
                    f"{prefix}{self.object_sep}{caption[start:end]} {coord_str}{self.object_sep}"
                )
            else:
                prev_end = int(unique_noun_chunks[idx - 1][1])

                coord_str = "".join(
                    self.coords_placeholder.format(idx=step + offset)
                    for step in range(noun_chunks_counter[unique_noun_chunks[idx]])
                )
                prefix = f"{modified_caption}{caption[prev_end:start]}"
                modified_caption = (
                    f"{prefix}{self.object_sep}{caption[start:end]} {coord_str}{self.object_sep}"
                )

            offset += noun_chunks_counter[unique_noun_chunks[idx]]
            idx += 1

        modified_caption = f"{modified_caption.strip()}{caption[end:]}"
        return modified_caption

    def build_region_annotation(self, instance_metadata: GRITMetadata) -> list[dict[str, Any]]:
        """Build the region annotation.

        We will use the region annotations in the conversation processor to replace the coordinates
        """
        width = instance_metadata.width
        height = instance_metadata.height
        region_annotation = []
        for idx, noun_chunk in enumerate(instance_metadata.noun_chunks):
            start, end, *noun_normalized_coords, confidence_score = noun_chunk
            # The coords are normalized and in xyxy format,
            # so we need to multiply them by the width and height to get the actual coordinates
            # IMPORTANT: The width/height should be not the original ones, but the ones after
            # the image has been downloaded. This is because the after the image is downloaded,
            # we no longer have it with the original width and height.
            coords = [
                noun_normalized_coords[0] * width,
                noun_normalized_coords[1] * height,
                noun_normalized_coords[2] * width,
                noun_normalized_coords[3] * height,
            ]
            region_annotation.append(
                {
                    "phrase": self.coords_placeholder.format(idx=idx),
                    "bbox": coords,
                }
            )
        return region_annotation

    @overrides(check_signature=False)
    def _build_rows_iterator(self, chunk_size: int) -> Iterator[list[Any]]:  # noqa: WPS231
        buffer = []
        image_paths = list(self.image_shard_cache.glob("*.jpg"))
        for image_path in image_paths:
            raw_metadata = read_json(Path(image_path).with_suffix(".json"))
            instance_metadata = GRITMetadata.model_validate(raw_metadata)

            buffer.append(
                {
                    "caption": self.modify_caption(
                        instance_metadata.caption, instance_metadata.noun_chunks
                    ),
                    "region": self.build_region_annotation(instance_metadata),
                    "metadata": {
                        "image_path": str(image_path),
                        "original_width": instance_metadata.original_width,
                        "original_height": instance_metadata.original_height,
                        "url": instance_metadata.url,
                        "clip_similarity_vitl14": instance_metadata.clip_similarity_vitl14,
                        "clip_similarity_vitb32": instance_metadata.clip_similarity_vitb32,
                        "noun_chunks": instance_metadata.noun_chunks,
                        "ref_exps": instance_metadata.ref_exps,
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
            "task": [Task.grounded_captioning.value for _ in examples],
            "image": [example["metadata"]["image_path"] for example in examples],
            "caption": [example["caption"] for example in examples],
            "qa_pairs": [None for _ in examples],
            "region": [example["region"] for example in examples],
            "relation": [None for _ in examples],
            "chat": [None for _ in examples],
            "source": [f"{self.source}" for _ in examples],
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

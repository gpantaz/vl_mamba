import argparse
from pathlib import Path
from typing import Any, Union

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vl_mamba.datamodels.datamodels import DatasetSplits
from vl_mamba.datasets.vl_mamba.ocr_vqa_loader import OCRVQALoader
from vl_mamba.utils.io import read_json, write_json


class OCRVQADatasetFilter(Dataset[dict[str, Union[bool, str]]]):
    """Filter out OCRVQA dataset.

    Ideally this would be inside the OCRVQALoader class, but I couldnt make it run fast enough.
    """

    def __init__(
        self,
        cache_dir: Path,
        min_image_size: int = 1,
        uniform_threshold: float = 0.85,
    ) -> None:
        self.cache_dir = Path(cache_dir)

        self.dataset_json = Path(cache_dir, "dataset.json")

        dataset = read_json(self.dataset_json)
        self.dataset = [{"id": key, **metadata} for key, metadata in dataset.items()]

        self.min_image_size = min_image_size
        self.uniform_threshold = uniform_threshold

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Union[bool, str]]:
        """Get the item at the given index."""
        return {
            "url": self.dataset[idx]["imageURL"],
            "should_keep": self.should_keep_image(self.dataset[idx]),
        }

    def can_load_image(self, image_path: Path) -> bool:
        """Check if the image can be loaded."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                return True
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return False

    def should_keep_image(self, example: dict[str, Any]) -> bool:
        """Check if the image should be skipped."""
        image_path = Path(self.cache_dir, f"{Path(example['imageURL']).name}")

        # If the image is not downloaded, we should skip it
        if not image_path.exists():
            return False

        if not self.can_load_image(image_path):
            return False

        with Image.open(image_path) as img:
            # If the image is too small, we should skip it
            if min(img.size) <= self.min_image_size:
                return False

            # If the image is too uniform, we should skip it
            img = img.convert("RGB")
            image_array = np.array(img).reshape(-1, 3)
            _, counts = np.unique(image_array, axis=0, return_counts=True)

            if counts.max() / counts.sum() > self.uniform_threshold:
                return False

        return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        default="../datasets/vl-mamba/ocr_vqa/",
        type=Path,
        help="Path to the dataset folder.",
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default="../datasets/vl-mamba/ocr_vqa/images_filtered.json",
        help="Path to save the filtered images.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use for filtering.",
    )

    parser.add_argument(
        "--min_image_size",
        type=int,
        default=350,  # noqa: WPS432
        help="Minimum size of the image to keep.",
    )

    parser.add_argument(
        "--uniform_threshold",
        type=float,
        default=0.85,  # noqa: WPS432
        help="Threshold for the uniformity of the image.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for split in (DatasetSplits.TRAIN, DatasetSplits.VALIDATION, DatasetSplits.TEST):
        OCRVQALoader(split=split, writer_batch_size=1, cache_dir=args.cache_dir)

    dataset = OCRVQADatasetFilter(
        cache_dir=args.cache_dir,
        min_image_size=args.min_image_size,
        uniform_threshold=args.uniform_threshold,
    )
    data_generator = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers, collate_fn=lambda x: x
    )
    should_keep = {}
    for batch in tqdm(data_generator, total=len(dataset), desc="Filtering out images"):
        for element in batch:
            should_keep[element["url"]] = element["should_keep"]

    logger.info(f"Keeping {sum(should_keep.values())} images out of {len(should_keep)}")
    write_json(args.output_json, should_keep)

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from vl_mamba.utils.io import read_json


def remove_articles(text: str) -> str:
    """Remove articles from the text."""
    # Define a regular expression pattern to match articles
    article_pattern = r"\b(a|an|the|some)\b"
    # Use re.sub() function to replace all occurrences of articles with an empty string
    cleaned_text = re.sub(article_pattern, "", text, flags=re.IGNORECASE)
    # Remove any double spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def overlapping_noun_phrases(noun_chunks_text_span: list[tuple[float, float]]) -> bool:
    """Check if the noun_chunks_text_span overlap."""
    # Sort noun_chunks_text_span based on start times
    noun_chunks_text_span.sort(key=lambda x: x[0])
    for idx in range(len(noun_chunks_text_span) - 1):
        if noun_chunks_text_span[idx][1] > noun_chunks_text_span[idx + 1][0]:
            return True  # Overlapping intervals found
    return False  # No overlapping intervals found


class GRITDatasetPreparator:
    """GRITDatasetPreparator.

    Filters the GRIT dataset based on the min and max counts of the noun_phrases.
    """

    def __init__(
        self,
        cache_dir: Path,
        output_figure: Path,
        output_folder: Path,
        max_count: int = 8,
        min_count: int = 3,
        min_image_size: int = 100,
        check_overlap: bool = False,
        downsample_images: bool = False,
        num_proc: int = 1,
    ) -> None:
        if not Path(cache_dir, "images").exists():
            raise ValueError("The images folder is not found. Please download the images first.")

        self.cache_dir = cache_dir
        self._images_path = Path(cache_dir, "images")
        self._noun_phrases_path = Path(cache_dir, "noun_phrases")
        if not self._noun_phrases_path.exists():
            self._noun_phrases_path.mkdir(parents=True, exist_ok=True)

        self.max_count = max_count
        self.min_count = min_count
        self.min_image_size = min_image_size
        self.output_figure = output_figure
        self.output_folder = Path(output_folder, "images")
        self._num_proc = num_proc
        self._do_check_overlap = check_overlap
        self._do_downsample_images = downsample_images

    def compute_noun_phrase_chunks_for_parquet(self, parquet_file: Path) -> None:
        """Get the noun_phrase chunks for the given parquet file."""
        noun_phrase_path = Path(self._noun_phrases_path, parquet_file.with_suffix(".json").name)
        if noun_phrase_path.exists():
            return

        metadata = pd.read_parquet(parquet_file, engine="fastparquet")
        metadata = metadata[metadata.status == "success"]

        noun_phrases = defaultdict(list)
        images = defaultdict(list)
        shard_id = parquet_file.stem
        for _, row in metadata.iterrows():
            image_filepath = self._images_path.joinpath(shard_id, f"{row.key}.jpg")
            json_filepath = image_filepath.with_suffix(".json")
            data = read_json(json_filepath)
            if self._should_skip_image(data):
                continue

            noun_phrase_chunks = data["noun_phrase_chunks"]
            caption = row.caption
            for noun_phrase in noun_phrase_chunks:
                noun_phrase = remove_articles(caption[int(noun_phrase[0]) : int(noun_phrase[1])])
                noun_phrases[noun_phrase.lower()].append(f"{shard_id}/{row.key}.jpg")
                images[f"{shard_id}/{row.key}.jpg"].append(noun_phrase.lower())

        logger.info(f"Saving noun_phrase chunks for {parquet_file}")
        with open(noun_phrase_path, "w") as fp:
            json.dump(noun_phrases, fp, indent=4)

    def downsample_images(self) -> dict[str, float]:
        """Downsample the images."""
        logger.info("Downsampling images")

        parquet_files = sorted(
            [
                filepath
                for filepath in self._images_path.iterdir()
                if filepath.is_file() and filepath.suffix == ".parquet"
            ],
            key=lambda x: int(x.stem),
        )

        total = 0
        indices_to_remove = []
        logger.info("Checking the status of the parquet files")
        for idx, parquet_file in enumerate(parquet_files):
            try:
                metadata = pd.read_parquet(parquet_file, engine="fastparquet")
                metadata = metadata[metadata.status == "success"]
            except Exception:
                logger.warning(f"Failed to read {parquet_file}, will skip it")
                indices_to_remove.append(idx)
            total += len(metadata)

        logger.info(f"Total image-text pairs: {total}")

        parq_files = [
            pf for index, pf in enumerate(parquet_files) if index not in indices_to_remove
        ]

        with Pool(self._num_proc) as pool, tqdm(total=len(parquet_files)) as pbar:
            fn = self.compute_noun_phrase_chunks_for_parquet
            for _ in pool.imap_unordered(fn, parq_files):
                pbar.update(1)

    def gather_noun_phrase_frequencies(self) -> dict[str, list[str]]:
        """Compute noun_phrase frequencies."""
        noun_phrase_files = self._noun_phrases_path.iterdir()
        noun_phrases = defaultdict(list)
        for noun_phrase_file in tqdm(noun_phrase_files, desc="Gathering noun_phrase frequencies"):
            with open(noun_phrase_file) as fp:
                noun_phrases_for_shard = json.load(fp)
                # Some images may have multiple occurrences of the same noun_phrase
                for noun_phrase, images in noun_phrases_for_shard.items():
                    noun_phrases[noun_phrase].extend(list(set(images)))
        return noun_phrases

    def filter_noun_phrase_frequencies(
        self, noun_phrases: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Filter the noun_phrase frequencies."""
        logger.info("Before filtering")
        self._print_noun_phrases(noun_phrases)

        noun_phrases_sorted = sorted(
            noun_phrases.items(),
            key=lambda x: len(x[1]),
        )

        filtered_noun_phrases = defaultdict(list)
        for noun_phrase, images in noun_phrases_sorted:
            if self.min_count <= len(images) <= self.max_count:
                filtered_noun_phrases[noun_phrase] = images

            elif len(images) > self.max_count:
                random.shuffle(images)
                filtered_noun_phrases[noun_phrase] = images[: self.max_count]

        logger.info("After filtering")
        self._print_noun_phrases(filtered_noun_phrases)
        return filtered_noun_phrases

    def plot_distributions(
        self, noun_phrases: dict[str, list[str]], filtered_noun_phrases: dict[str, list[str]]
    ) -> None:
        """Plot the distributions of the noun_phrases."""
        # Get all images from the filtered noun_phrases
        all_selected_images = {
            image for images in filtered_noun_phrases.values() for image in images
        }
        # We need to get the actual noun_phrase-image map, because even after filtering,
        # the same image may be associated with multiple noun_phrases
        noun_phrase_filtered_plot = defaultdict(list)
        for noun_phrase, images in noun_phrases.items():
            for image in images:
                if image in all_selected_images:
                    noun_phrase_filtered_plot[noun_phrase].append(image)

        logger.info("In actual dataset:")
        self._print_noun_phrases(noun_phrase_filtered_plot)

        fig, ax = plt.subplots()
        noun_phrase_dicts = [noun_phrases, noun_phrase_filtered_plot]
        labels = ["Original", "Filtered"]
        colors = ["blue", "green"]
        for noun_phrase_dict, label, color in zip(noun_phrase_dicts, labels, colors, strict=False):
            x = np.arange(len(noun_phrase_dict))
            y = sorted([len(set(img)) for img in noun_phrase_dict.values()])[::-1]
            logger.info(f"{label}: {sum(y)}")

            total = sum(y) // 1e6
            ax.plot(x, y, color=color, alpha=1, label=f"{label}: {total}M")
            ax.fill_between(x, y, 0, color=color, alpha=0.1)

        ax.set_xlabel("Noun Phrases")
        ax.set_ylabel("Frequency")
        ax.set_title("Noun Phrase Frequency Distribution")
        ax.set_yscale("log")
        plt.grid()
        plt.legend()
        plt.savefig(self.output_figure)

    def check_overlap(self, noun_phrases: dict[str, list[str]]) -> None:
        """Check if there are overlapping noun_phrases."""
        all_selected_images = {image for images in noun_phrases.values() for image in images}

        with Pool(self._num_proc) as pool:
            desc = "Checking for overlapping noun phrases"
            with tqdm(total=len(all_selected_images), desc=desc) as pbar:
                fn = self._check_overlap_for_image
                for _ in pool.imap_unordered(fn, all_selected_images):
                    pbar.update(1)

    def copy_images_and_metadata(self, noun_phrases: dict[str, list[str]]) -> None:
        """Copy the images."""
        all_selected_images = {image for images in noun_phrases.values() for image in images}
        logger.info(f"Copying {len(all_selected_images)} images to {self.output_folder}")

        with Pool(self._num_proc) as pool:
            with tqdm(total=len(all_selected_images), desc="Copying images") as pbar:
                fn = self._copy_image_and_metadata
                for _ in pool.imap_unordered(fn, all_selected_images):
                    pbar.update(1)

    def run(self) -> None:
        """Run the GRITDatasetPreparator."""
        # First downsample the images
        if self._do_downsample_images:
            self.downsample_images()
        # Gather all noun_phrase frequencies from the downsampled images
        noun_phrases = self.gather_noun_phrase_frequencies()
        # Filter the noun_phrases based on the min and max counts
        filtered_noun_phrases = self.filter_noun_phrase_frequencies(noun_phrases)
        # Plot the distributions
        self.plot_distributions(noun_phrases, filtered_noun_phrases)

        missing = 0
        for noun_phrase in noun_phrases:
            if noun_phrase not in filtered_noun_phrases:
                missing += 1

        missing_str = f"{missing} / {len(noun_phrases)}"
        perc = round(missing / len(noun_phrases) * 100, 2)
        logger.info(f"Total missing: {missing_str} {perc}%")

        num_images_before_filter = self._compute_image_text_pairs(noun_phrases)
        logger.info(f"Total image-text pairs before filtering: {num_images_before_filter}")

        num_images_after_filter = self._compute_image_text_pairs(filtered_noun_phrases)
        logger.info(f"Total image-text pairs after filtering: {num_images_after_filter}")

        # Verify that there are no overlapping noun_phrases
        if self._do_check_overlap:
            self.check_overlap(filtered_noun_phrases)

        # Finally, copy the images and the json files for each image to a new root directory
        # This directory PRESERVES the original directory structure, but only contains the images
        self.copy_images_and_metadata(filtered_noun_phrases)

    def _check_overlap_for_image(self, image: str) -> None:
        data = read_json(Path(self._images_path, image).with_suffix(".json"))
        noun_chunks = data["noun_chunks"]
        # Some images have multiple referents for the same noun_phrase
        # For example the noun phrase "happy dogs" may refer to multiple dogs in the image
        # Thats okay though since we are interested in unique overlapping noun_phrases
        noun_chunks_text_span_set = {(int(chunk[0]), int(chunk[1])) for chunk in noun_chunks}
        if overlapping_noun_phrases(list(noun_chunks_text_span_set)):
            raise ValueError(f"Overlapping noun phrases found for {image}")

    def _compute_image_text_pairs(self, noun_phrases: dict[str, list[str]]) -> None:
        """Print the image-text pairs."""
        images = []
        for image_list in noun_phrases.values():
            images.extend(image_list)
        return len(set(images))

    def _print_noun_phrases(self, noun_phrases: dict[str, list[str]]) -> Counter:
        """Count the number of noun_phrases."""
        counts = {noun_phrase: len(images) for noun_phrase, images in noun_phrases.items()}
        most_common = Counter(**counts).most_common()
        for most_common_noun_phrase, most_common_count in most_common[:10]:
            logger.info(f"{most_common_noun_phrase}: {most_common_count}")

        for least_common_noun_phrase, least_common_count in most_common[-10:]:
            logger.info(f"{least_common_noun_phrase}: {least_common_count}")

    def _copy_image_and_metadata(self, image: str) -> None:
        image_path = Path(self._images_path, image)
        image_dest = Path(self.output_folder, image)
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(image_path), str(image_dest))

        json_path = Path(image_path).with_suffix(".json")
        json_dest = Path(self.output_folder, Path(image).parent, json_path.name)
        shutil.copy(str(json_path), str(json_dest))

    def _should_skip_image(self, data: dict[str, Any]) -> bool:
        """Check if the image should be skipped."""
        # If the image is too small, skip it
        is_small_resized_image = (
            data["width"] < self.min_image_size or data["height"] < self.min_image_size
        )
        if is_small_resized_image:
            logger.info(f"Skipping {data}")
            return True

        # If the original image is too small, skip it
        is_small_resized_image = (
            data["original_width"] < self.min_image_size
            or data["original_height"] < self.min_image_size
        )
        if is_small_resized_image:
            logger.info(f"Skipping {data}")
            return True
        return False


def parse_args() -> argparse.Namespace:
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        default=Path("../datasets/vl_mamba/grit"),
        help="Path to save the dataset. This is the path where the pyarrow dataset is stored.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum count of the noun_phrases.",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=8,
        help="Maximum count of the noun_phrases.",
    )
    parser.add_argument(
        "--output_figure",
        default=Path("original_vs_filtered.png"),
        help="Output figure to save the distributions of the noun_phrases.",
    )
    parser.add_argument(
        "--output_folder",
        default=Path("../datasets/vl_mamba/grit_downsampled"),
        help="Output folder to save the images and the json files.",
    )
    parser.add_argument(
        "--downsample_images",
        action="store_true",
        help="Whether to downsample the images or not.",
    )
    parser.add_argument(
        "--check_overlap",
        action="store_true",
        help="Whether to check for overlapping noun_phrases or not.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=32,
        help="Number of processes to use for preprocessing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preparator = GRITDatasetPreparator(
        cache_dir=args.cache_dir,
        min_count=args.min_count,
        max_count=args.max_count,
        output_figure=args.output_figure,
        output_folder=args.output_folder,
        downsample_images=args.downsample_images,
        check_overlap=args.check_overlap,
        num_proc=args.num_proc,
    ).run()

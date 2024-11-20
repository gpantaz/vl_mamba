import argparse

from datasets import load_dataset
from loguru import logger
from vl_mamba.datasets.vl_mamba.data_paths import DatasetPaths


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_subset",
        default="llava_pretrain",
        help="Which dataset subset to use. Look at the options from datamodules/datasets/mm_icl.py.",
    )

    parser.add_argument(
        "--dataset_path",
        default="src/vl_mamba/datasets/vl_mamba",
        help="Path to the dataset folder. Don't change this.",
    )

    parser.add_argument(
        "--root_dataset_path",
        default="storage/datasets",
        help="This is the path that all images / annotations are downloaded.",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use for preprocessing.",
    )

    parser.add_argument(
        "--cache_dir",
        default="storage/datasets/tmp",
        help="Path to save the dataset. This is the path where the pyarrow dataset is stored.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(
        args.dataset_path,
        args.dataset_subset,
        cache_dir=args.cache_dir,
        root_dataset_path=args.root_dataset_path,
        dataset_paths=DatasetPaths(args.root_dataset_path),
        num_proc=args.num_proc,
        verification_mode="no_checks",
        trust_remote_code=True,
    )

    logger.info(
        f"Dataset: {args.dataset_subset} was created successfully and stored at {args.cache_dir}"
    )

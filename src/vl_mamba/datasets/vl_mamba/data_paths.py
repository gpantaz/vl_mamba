from pathlib import Path
from typing import Union


BASE_DIR = Path("storage/datasets")


class DatasetPaths:  # noqa: WPS230
    """Dataclass for data paths.

    Some datasets are downloaded with a different tool (img2dataset, etc), or have separate
    hugginface cache (visual genome). We need to know where they are.
    """

    def __init__(self, base_dir: Union[str, Path] = BASE_DIR) -> None:
        self.storage = Path(base_dir)

        # This needs to be the base dir since the fixtures are within the repo
        self.fixtures_dataset_path = BASE_DIR.joinpath("fixtures")

        self.llava_pretrain_cache_dir = self.storage.joinpath("llava_pretrain")
        # We will use the same directory for datasets that use coco images since they have the share images
        self.coco_cache_dir = self.storage.joinpath("coco")
        self.refcoco_cache_dir = self.storage.joinpath("coco")
        self.aok_vqa_cache_dir = self.storage.joinpath("coco")
        self.localized_narratives_cache_dir = self.storage.joinpath("coco")
        self.pope_cache_dir = self.storage.joinpath("coco")
        self.grec_cache_dir = self.storage.joinpath("coco")

        # We will use the same directory for textvqa and textcaps since they have the same images
        # This will download the images only once and save space
        self.textvqa_cache_dir = self.storage.joinpath("textcaps")
        self.textcaps_cache_dir = self.storage.joinpath("textcaps")

        self.vqa_v2_cache_dir = self.storage.joinpath("vqa_v2")

        self.visual7w_cache_dir = self.storage.joinpath("visual7w")

        self.visual_genome_cache_dir = self.storage.joinpath("visual_genome")

        self.ai2d_cache_dir = self.storage.joinpath("ai2d")

        self.gqa_cache_dir = self.storage.joinpath("gqa")

        self.ocr_vqa_cache_dir = self.storage.joinpath("ocr_vqa")

        self.vizwiz_vqa_cache_dir = self.storage.joinpath("vizwiz_vqa")

        self.nocaps_cache_dir = self.storage.joinpath("nocaps")

        self.vsr_cache_dir = self.storage.joinpath("vsr")

        self.docvqa_cache_dir = self.storage.joinpath("docvqa")
        self.infographicsvqa_cache_dir = self.storage.joinpath("infovqa")

        # These datasets are downloaded with an external tool aka img2dataset
        self.conceptual_captions_cache_dir = self.storage.joinpath(
            "conceptual_captions_3m", "train"
        )
        self.sbu_captions_cache_dir = self.storage.joinpath("sbu_captions", "train")
        self.grit_cache_dir = self.storage.joinpath("grit_downsampled")

        # Hugginface datasets are stored in separate cache directories
        self.visual_genome_cache_dir = self.storage.joinpath("visual_genome")
        self.hateful_memes_cache_dir = self.storage.joinpath("hateful_memes")

        self.synthetic_vg_cache_dir = self.storage.joinpath("synthetic_vg")

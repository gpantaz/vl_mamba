from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseModelArguments:
    """Model arguments."""

    model_name: str = field(default="state-spaces/mamba-790m")
    vision_encoder_name: Literal[
        "openai/clip-vit-large-patch14-336",
        "timm/eva02_large_patch14_clip_224",
        "timm/eva02_large_patch14_clip_336",
    ] = field(default="timm/eva02_large_patch14_clip_336")
    select_layer: int = field(default=-2)
    select_feature: str = field(default="patch")
    train_only_visual_embeddings: bool = field(default=True)
    image_size: int = field(default=336)
    extrapolate_image_size: int | None = field(default=None)
    patch_size: int = field(default=14)
    num_channels: int = field(default=3)

    tokenizer_name: str = field(default="EleutherAI/gpt-neox-20b")
    tokenizer_truncation_side: Literal["right", "left"] = field(default="right")
    tokenizer_padding_side: Literal["right", "left"] = field(default="right")
    tokenizer_add_special_tokens: bool = field(default=True)
    model_max_length: int = field(default=100)
    pixel_based: bool = field(default=False)


@dataclass
class BaseDataArguments:
    """Data arguments."""

    dataset_path: str = field(default="src/vl_mamba/datasets/vl_mamba")
    dataset_cache_dir: str = field(default="/mnt/ceph_rbd/storage/datasets/vl_mamba")
    root_dataset_path: str = field(default="/mnt/ceph_rbd/storage/datasets/")
    image_mean: tuple[float, float, float] = field(default=(0.48145466, 0.4578275, 0.40821073))
    image_std: tuple[float, float, float] = field(default=(0.26862954, 0.26130258, 0.27577711))
    variable_sized: bool = field(default=False)
    box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = field(default="normalize")

    # Optional arguments for evalaution on a subset of the dataset
    start_index: int = field(default=0)
    end_index: int = field(default=-1)


@dataclass
class BaseGenerationArguments:
    """BaseGeneration arguments."""

    max_length: int = 20
    max_new_tokens: int = 20
    top_k: int = 1
    top_p: float = 0
    temperature: float = 1.0
    return_dict_in_generate: bool = False
    output_scores: bool = False
    do_sample: bool = False

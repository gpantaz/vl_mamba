from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

from vl_mamba.datamodels.dataclasses import DatasetItem
from vl_mamba.datamodels.datamodels import TASK_TEMPLATES_MAP, Task
from vl_mamba.datasets.vl_mamba.data_paths import DatasetPaths
from vl_mamba.models.modeling_vlmamba import VLMambaLMHeadModel
from vl_mamba.models.modeling_vlmambaclip import VLMambaCLIPLMHeadModel
from vl_mamba.models.modeling_vlmambaclip_xattn import VLMambaCLIPLXAttnMHeadModel
from vl_mamba.models.modeling_vlpythia import VLGPTNeoXForCausalLM
from vl_mamba.models.modeling_vlpythiaclip import VLCLIPGPTNeoXForCausalLM
from vl_mamba.models.vision_encoder import build_vision_encoder
from vl_mamba.utils.conversation_processor import VLMambaConversationProcessor
from vl_mamba.utils.image_preprocessor import VLMambaImageProcessor

ModelType = (
    VLGPTNeoXForCausalLM
    | VLCLIPGPTNeoXForCausalLM
    | VLMambaCLIPLMHeadModel
    | VLMambaLMHeadModel
    | VLMambaCLIPLXAttnMHeadModel
)


class BaseEvalDataset(Dataset[DatasetItem]):
    """Base evaluation dataset."""

    def __init__(
        self,
        split: str,
        dataset_path: str,
        dataset_cache_dir: str,
        root_dataset_path: str,
        dataset_subset: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_mean: float | tuple[float, float, float],
        image_std: float | tuple[float, float, float],
        image_size: int | tuple[int, int] = 224,
        patch_size: int | dict[str, int] = 16,
        variable_sized: bool = False,
        box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = "normalize",
        needs_attention_mask: bool = False,
        pixel_based: bool = False,
        instruction_first: bool = False,
        has_gated_cross_attention: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._split = split
        self._dataset_path = dataset_path
        self._dataset_cache_dir = dataset_cache_dir
        self._root_dataset_path = root_dataset_path
        self._dataset_subset = dataset_subset

        self.tokenizer = tokenizer

        if isinstance(image_mean, float):
            image_mean = (image_mean, image_mean, image_mean)
        self._image_mean = image_mean

        if isinstance(image_std, float):
            image_std = (image_std, image_std, image_std)
        self._image_std = image_std

        if isinstance(image_size, int):
            image_size = {"height": image_size, "width": image_size}  # type: ignore[assignment]
        self._image_size = image_size

        if isinstance(patch_size, int):
            patch_size = {"height": patch_size, "width": patch_size}
        self._patch_size = patch_size

        self.image_processor = VLMambaImageProcessor(
            image_mean=self._image_mean,
            image_std=self._image_std,
            size=self._image_size,
            patch_size=self._patch_size,
            variable_sized=variable_sized,
            pixel_based=pixel_based,
        )

        self.conversation_processor = VLMambaConversationProcessor(box_mode=box_mode)
        self._needs_attention_mask = needs_attention_mask
        self._pixel_based = pixel_based
        self._instruction_first = instruction_first
        self._has_gated_cross_attention = has_gated_cross_attention

        if instruction_first:
            logger.warning(
                "The instruction_first flag is set to True. Make sure you use a model that supports it"
            )

        self.prepare_dataset()

    def prepare_dataset(self) -> None:
        """Prepare dataset."""
        self.dataset = load_dataset(  # type: ignore[report]
            self._dataset_path,
            self._dataset_subset,
            cache_dir=self._dataset_cache_dir,
            root_dataset_path=self._root_dataset_path,
            dataset_paths=DatasetPaths(self._root_dataset_path),
            verification_mode="no_checks",
            trust_remote_code=True,
        )[self._split]

        self.dataset_size = len(self.dataset)

        logger.info(
            f"Loaded {self.dataset_size} examples from {self._dataset_subset} {self._split} split."
        )

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item from dataset."""
        raise NotImplementedError

    def _get_random_template_for_task(self, task: Task) -> str:
        """Choose a random template for the given task."""
        return TASK_TEMPLATES_MAP[task][0]


@dataclass
class BaseModelArguments:
    """Base Model arguments."""

    model_name: str = field(default="state-spaces/mamba-790m")
    cross_attention_config: str = field(default="configs/model/mamba-790m_layer_interval4.json")
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
    """Base Data arguments."""

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
    """Base Generation arguments."""

    max_new_tokens: int = 120
    top_k: int = 1
    top_p: float = 0
    temperature: float = 1.0
    return_dict_in_generate: bool = False
    output_scores: bool = False
    output_attentions: bool = False
    do_sample: bool = False


def build_model(model_args: BaseModelArguments) -> ModelType:
    """Build model."""
    logger.info(f"Building model: {model_args.model_name}")
    if "mamba" in model_args.model_name and "attn" not in model_args.model_name:
        if model_args.pixel_based:
            model = VLMambaLMHeadModel.from_pretrained(
                pretrained_model_name=model_args.model_name,
                image_size=model_args.image_size,
                patch_size=model_args.patch_size,
                num_channels=model_args.num_channels,
            )
        else:
            model = VLMambaCLIPLMHeadModel.from_pretrained(
                pretrained_model_name=model_args.model_name,
                image_size=model_args.image_size,
                vision_encoder_name=model_args.vision_encoder_name,
                select_layer=model_args.select_layer,
                select_feature=model_args.select_feature,
            )
    elif "mamba" in model_args.model_name and "attn" in model_args.model_name:
        model = VLMambaCLIPLXAttnMHeadModel.from_pretrained(
            pretrained_model_name=model_args.model_name,
            image_size=model_args.image_size,
            vision_encoder_name=model_args.vision_encoder_name,
            select_layer=model_args.select_layer,
            select_feature=model_args.select_feature,
            cross_attention_config=model_args.cross_attention_config,
        )
    else:  # noqa: PLR5501
        if model_args.pixel_based:
            model = VLGPTNeoXForCausalLM.from_pretrained(
                pretrained_model_name=model_args.model_name,
                patch_size=model_args.patch_size,
                num_channels=model_args.num_channels,
            )
        else:
            model = VLCLIPGPTNeoXForCausalLM.from_pretrained(
                pretrained_model_name=model_args.model_name,
                vision_encoder_name=model_args.vision_encoder_name,
                image_size=model_args.image_size,
                select_layer=model_args.select_layer,
                select_feature=model_args.select_feature,
                use_flash_attention_2=True,
            )

    extrapolate = model_args.extrapolate_image_size is not None
    if extrapolate:
        logger.info(f"Extrapolating image size to {model_args.extrapolate_image_size}")
        # We need to build a new vision encoder with the new image size for extrapolation
        # WARNING: This is a hacky way to do this, and it wont work if the model checkpoint
        # has updated the vision encoder (e.g the vision encoder was unfrozen during training)
        # If that is the case we need to iterate over the model state_dict and update the weights
        new_vision_encoder = build_vision_encoder(
            model_args.vision_encoder_name, model_args.extrapolate_image_size
        )
        model.vision_encoder = new_vision_encoder

    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def build_tokenizer(model_args: BaseModelArguments) -> transformers.AutoTokenizer:
    """Build tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Padding and eos tokens are the same
    tokenizer.eos_token = "<|endoftext|>"  # noqa: S105
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = model_args.model_max_length
    tokenizer.truncation_side = model_args.tokenizer_truncation_side
    tokenizer.padding_side = model_args.tokenizer_padding_side

    return tokenizer

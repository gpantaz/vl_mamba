import os
from dataclasses import dataclass, field
from typing import Literal, Union

import transformers
from loguru import logger
from torch.utils.data import Dataset

from vl_mamba.datamodels.dataclasses import DatasetPadding
from vl_mamba.datamodels.datamodels import DatasetSplits
from vl_mamba.datasets.collate import Collate
from vl_mamba.datasets.fake_dataset import FakeDataset
from vl_mamba.datasets.pretrain_dataset import PretrainDataset
from vl_mamba.models.modeling_vlmamba import VLMambaLMHeadModel
from vl_mamba.models.modeling_vlmambaclip import VLMambaCLIPLMHeadModel
from vl_mamba.models.modeling_vlpythia import VLGPTNeoXForCausalLM
from vl_mamba.models.modeling_vlpythiaclip import VLCLIPGPTNeoXForCausalLM
from vl_mamba.models.vision_encoder import build_vision_encoder
from vl_mamba.trainer import CustomTrainer
from vl_mamba.utils.count_model_parameters import compute_trainable_params

ModelType = Union[
    VLGPTNeoXForCausalLM, VLCLIPGPTNeoXForCausalLM, VLMambaCLIPLMHeadModel, VLMambaLMHeadModel
]


@dataclass
class ModelArguments:
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
    image_size: int = field(default=224)
    patch_size: int = field(default=16)
    num_channels: int = field(default=3)

    tokenizer_name: str = field(default="EleutherAI/gpt-neox-20b")
    tokenizer_truncation_side: Literal["right", "left"] = field(default="right")
    tokenizer_padding_side: Literal["right", "left"] = field(default="right")
    tokenizer_add_special_tokens: bool = field(default=True)
    model_max_length: int = field(default=100)
    pixel_based: bool = field(default=True)
    extrapolate_image_size: int | None = field(default=None)


@dataclass
class DataArguments:
    """Data arguments."""

    dataset_path: str = field(default="src/vl_mamba/datasets/vl_mamba")
    dataset_cache_dir: str = field(default="/mnt/ceph_rbd/storage/datasets/vl_mamba")
    root_dataset_path: str = field(default="/mnt/ceph_rbd/storage/datasets/")
    train_dataset_subset: str = field(default="llava_pretrain")
    eval_dataset_subset: str = field(default="llava_pretrain")
    image_mean: tuple[float, float, float] = field(default=(0.48145466, 0.4578275, 0.40821073))
    image_std: tuple[float, float, float] = field(default=(0.26862954, 0.26130258, 0.27577711))
    variable_sized: bool = field(default=False)
    instruction_first: bool = field(default=False)
    box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = field(default="normalize")


@dataclass
class TrainArgs(transformers.TrainingArguments):
    """Training arguments."""

    output_dir: str = field(default="mamba-790m")
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=128)
    gradient_accumulation_steps: int = field(default=3)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.1)
    num_train_epochs: int = field(default=5)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="linear")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    deepspeed: str = field(default="configs/trainer/zero2.json")
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    log_level: str = field(default="debug")
    save_safetensors: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.1)
    seed: int = field(default=12345)
    data_seed: int = field(default=12345)
    dataloader_num_workers: int = field(default=4)
    logging_nan_inf_filter: bool = field(default=False)
    run_name: str = field(default="mamba-790m")
    project_name: str = field(default="vl-mamba")


def build_model(model_args: ModelArguments) -> ModelType:
    """Build model."""
    logger.info(f"Building model: {model_args.model_name}")
    if "mamba" in model_args.model_name:
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
    else:
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

    # Freeze the vision encoder
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        logger.info("Freezing vision encoder")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False

    # Check if should train only the visual embedder
    if model_args.train_only_visual_embeddings:
        logger.info("Freezing all parameters except the vision projector")
        for name, parameter in model.named_parameters():
            if "vision" not in name:
                parameter.requires_grad = False

    extrapolate = (
        model_args.extrapolate_image_size is not None
        and model_args.extrapolate_image_size != model_args.image_size
    )
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

    # Freeze the vision encoder
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        logger.info("Freezing vision encoder")
        for p in model.vision_encoder.parameters():
            p.requires_grad = False

    # Check if should train only the visual embedder
    if model_args.train_only_visual_embeddings:
        logger.info("Freezing all parameters except the vision projector")
        for name, parameter in model.named_parameters():
            if "vision" not in name:
                parameter.requires_grad = False
    compute_trainable_params(model)
    return model


def build_tokenizer(model_args: ModelArguments) -> transformers.AutoTokenizer:
    """Build tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Padding and eos tokens are the same
    tokenizer.eos_token = "<|endoftext|>"  # noqa: S105
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = model_args.model_max_length
    tokenizer.truncation_side = model_args.tokenizer_truncation_side
    tokenizer.padding_side = model_args.tokenizer_padding_side

    return tokenizer


def build_datasets(
    tokenizer: transformers.AutoTokenizer, data_args: DataArguments, model_args: ModelArguments
) -> tuple[Dataset, Dataset | None, Collate]:
    """Build datasets."""
    logger.info(
        f"Building datasets: {data_args.train_dataset_subset} {data_args.eval_dataset_subset}"
    )

    logger.warning(f"The instruction_first flag is set to {data_args.instruction_first}.")

    if data_args.train_dataset_subset == "debug":
        logger.info("Using debug dataset")
        train_dataset = FakeDataset(
            image_size=model_args.image_size,
            patch_size=model_args.patch_size,
        )
        eval_dataset = FakeDataset(
            image_size=model_args.image_size,
            patch_size=model_args.patch_size,
        )

    elif "pretrain" in data_args.train_dataset_subset:
        train_dataset = PretrainDataset(
            dataset_path=data_args.dataset_path,
            dataset_cache_dir=data_args.dataset_cache_dir,
            root_dataset_path=data_args.root_dataset_path,
            dataset_split=DatasetSplits.TRAIN,
            dataset_subset=data_args.train_dataset_subset,
            tokenizer=tokenizer,
            image_mean=data_args.image_mean,
            image_std=data_args.image_std,
            image_size=(
                model_args.extrapolate_image_size
                if model_args.extrapolate_image_size is not None
                else model_args.image_size
            ),
            patch_size=model_args.patch_size,
            variable_sized=data_args.variable_sized,
            box_mode=data_args.box_mode,
            needs_attention_mask="mamba" not in model_args.model_name,
            model_max_length=model_args.model_max_length,
            pixel_based=model_args.pixel_based,
            instruction_first=data_args.instruction_first,
        )

        eval_dataset = None
    else:
        train_dataset = PretrainDataset(
            dataset_path=data_args.dataset_path,
            dataset_cache_dir=data_args.dataset_cache_dir,
            root_dataset_path=data_args.root_dataset_path,
            dataset_split=DatasetSplits.TRAIN,
            dataset_subset=data_args.train_dataset_subset,
            tokenizer=tokenizer,
            image_mean=data_args.image_mean,
            image_std=data_args.image_std,
            image_size=(
                model_args.extrapolate_image_size
                if model_args.extrapolate_image_size is not None
                else model_args.image_size
            ),
            patch_size=model_args.patch_size,
            variable_sized=data_args.variable_sized,
            box_mode=data_args.box_mode,
            needs_attention_mask="mamba" not in model_args.model_name,
            model_max_length=model_args.model_max_length,
            pixel_based=model_args.pixel_based,
            instruction_first=data_args.instruction_first,
        )

        eval_dataset = PretrainDataset(
            dataset_path=data_args.dataset_path,
            dataset_cache_dir=data_args.dataset_cache_dir,
            root_dataset_path=data_args.root_dataset_path,
            dataset_split=DatasetSplits.VALIDATION,
            dataset_subset=data_args.train_dataset_subset,
            tokenizer=tokenizer,
            image_mean=data_args.image_mean,
            image_std=data_args.image_std,
            image_size=(
                model_args.extrapolate_image_size
                if model_args.extrapolate_image_size is not None
                else model_args.image_size
            ),
            patch_size=model_args.patch_size,
            variable_sized=data_args.variable_sized,
            box_mode=data_args.box_mode,
            needs_attention_mask="mamba" not in model_args.model_name,
            model_max_length=model_args.model_max_length,
            pixel_based=model_args.pixel_based,
            instruction_first=data_args.instruction_first,
        )

    collate_fn = Collate(
        padding=DatasetPadding(input_ids=tokenizer.pad_token_id),
        padding_side=tokenizer.padding_side,
    )
    return train_dataset, eval_dataset, collate_fn


def train() -> None:
    """Train."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # Build model and tokenizer
    model = build_model(model_args=model_args)
    tokenizer = build_tokenizer(model_args=model_args)

    # Set the environment variables
    # https://docs.wandb.ai/guides/integrations/huggingface
    if train_args.project_name is not None:
        os.environ["WANDB_PROJECT"] = train_args.project_name

    train_dataset, eval_dataset, data_collator = build_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
    )

    trainer = CustomTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
    # if trainer.deepspeed:
    #     torch.cuda.synchronize()
    #     trainer.save_model(checkpoint_dir)
    # else:
    #     state_dict = trainer.model.state_dict()
    #     if trainer.args.should_save:
    #         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    #         del state_dict
    #         trainer._save(checkpoint_dir, state_dict=cpu_state_dict)


if __name__ == "__main__":
    train()

import glob
import os
from typing import Any

import torch
from loguru import logger
from safetensors.torch import load_model
from transformers import (
    AutoConfig,
    CLIPVisionModel,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import is_flash_attn_2_available

from vl_mamba.models.vision_encoder import build_vision_encoder
from vl_mamba.utils.compute_loss import compute_loss
from vl_mamba.utils.hugginface import load_config_hf, load_state_dict_hf


class VLCLIPGPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    """VLGPTNeoXForCausalLM model."""

    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config) -> None:
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vision_encoder = build_vision_encoder(config.vision_encoder_name, config.image_size)
        self._select_layer = config.select_layer
        self._select_feature = config.select_feature

        vision_encoder_hidden_size = (
            self.vision_encoder.config.hidden_size
            if isinstance(self.vision_encoder, CLIPVisionModel)
            else self.vision_encoder.num_features
        )

        modules = [
            torch.nn.Linear(
                vision_encoder_hidden_size,
                config.hidden_size,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        ]
        self.vision_embed_tokens = torch.nn.Sequential(*modules)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> torch.nn.Linear:
        """Get output embeddings."""
        return self.embed_out

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:
        """Set output embeddings."""
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        """Foward pass for MambaLMHeadModel."""
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        # import time
        # start = time.time()
        patch_embeddings = self.get_patch_embeddings(pixel_values)
        # end = time.time() - start
        # print(f"patch_embeddings: {end}")

        # step 2: concatenate the visual embeddings and the textual embeddings
        masked_input_ids = input_ids.clone()
        masked_input_ids[input_ids < 0] = 0
        inputs_embeds = self.gpt_neox.embed_in(masked_input_ids)

        for batch_idx in range(inputs_embeds.shape[0]):
            dst_indices = (input_ids[batch_idx] < 0).nonzero(as_tuple=True)[0]
            inputs_embeds[batch_idx, dst_indices] = patch_embeddings[batch_idx].to(
                dtype=inputs_embeds.dtype
            )

            # if inputs_embeds.shape[0] != patch_embeddings.shape[0]:
            #     inputs_embeds[batch_idx, dst_indices] = patch_embeddings[0].to(
            #         dtype=inputs_embeds.dtype
            #     )
            # else:
            #     inputs_embeds[batch_idx, dst_indices] = patch_embeddings[batch_idx].to(
            #         dtype=inputs_embeds.dtype
            #     )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # step 3: forward through the language model
        outputs = self.gpt_neox(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            lm_loss = compute_loss(labels, lm_logits, self.config.vocab_size)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss, *output)) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "use_cache": kwargs.get("use_cache", None),
            }
        )

        return model_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        select_layer: int = -2,
        select_feature: str = "patch",
        image_size: int = 336,
        use_flash_attention_2: bool = False,
    ):
        """Load model from pretrained model name."""
        use_flash_attention_2 = use_flash_attention_2 and is_flash_attn_2_available()
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
            config_data = load_config_hf(pretrained_model_name)
            config = PretrainedConfig(**config_data)
            config.vision_encoder_name = vision_encoder_name
            config.select_layer = select_layer
            config.select_feature = select_feature
            config.image_size = image_size
            config._attn_implementation = attn_implementation
            model = cls(config=config)
            logger.info(f"Loading model from {pretrained_model_name}")
            if os.path.exists(os.path.join(pretrained_model_name, "model.safetensors")):
                logger.info("Attempting to load model using safetensors")
                try:
                    load_model(model, os.path.join(pretrained_model_name, "model.safetensors"))
                    return model
                except Exception as safetensor_exception:
                    logger.error(f"Failed to load model using safetensors: {safetensor_exception}")

            if len(glob.glob(os.path.join(pretrained_model_name, "*.safetensors"))) >= 2:
                logger.info("Found multiple safetensors files, using sharded checkpoint loading")
                try:
                    load_sharded_checkpoint(model, pretrained_model_name)
                    return model
                except Exception as safetensor_exception:
                    logger.error(f"Failed to load model using safetensors: {safetensor_exception}")

            if os.path.exists(os.path.join(pretrained_model_name, "pytorch_model.bin")):
                logger.info("Attempting to load model using torch.load")
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(pretrained_model_name, "pytorch_model.bin"))
                    )
                    return model
                except Exception as torch_exception:
                    logger.error(f"Failed to load model using torch.load: {torch_exception}")
            else:
                logger.error(f"Could not load model from {pretrained_model_name}")
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.vision_encoder_name = vision_encoder_name
            config.select_layer = select_layer
            config.select_feature = select_feature
            config.image_size = image_size
            config._attn_implementation = attn_implementation
            model = cls(config=config)
            model.load_state_dict(load_state_dict_hf(pretrained_model_name), strict=False)
        return model

    def get_patch_embeddings(self, pixel_values: torch.tensor) -> torch.tensor:
        """Get patch embeddings."""
        if isinstance(self.vision_encoder, CLIPVisionModel):
            patch_embeddings = self.vision_encoder(
                torch.stack([px[0] for px in pixel_values]), output_hidden_states=True
            )
        else:
            patch_embeddings = self.vision_encoder.forward_features(
                torch.stack([px[0] for px in pixel_values])
            )

        patch_embeddings = self.feature_select(patch_embeddings)
        patch_embeddings = self.vision_embed_tokens(patch_embeddings)
        return patch_embeddings

    def feature_select(self, image_forward_outs: Any) -> torch.tensor:
        """Feature select."""
        # TODO: feature selection from layer is only supported for CLIP
        if isinstance(self.vision_encoder, CLIPVisionModel):
            image_features = image_forward_outs.hidden_states[self._select_layer]
        else:
            image_features = image_forward_outs

        if self._select_feature == "patch":
            return image_features[:, 1:]
        elif self._select_feature == "cls_patch":
            return image_features
        raise ValueError(f"Unexpected select feature: {self._select_feature}")

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.tensor):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


# from dataclasses import dataclass, field
# from loguru import logger
# from typing import Literal

# import transformers
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from vl_mamba.datamodels.dataclasses import DatasetPadding
# from vl_mamba.datamodels.datamodels import DatasetSplits
# from vl_mamba.datasets.collate import Collate
# from vl_mamba.datasets.pretrain_dataset import PretrainDataset


# @dataclass
# class DataArguments:
#     """Data arguments."""

#     dataset_path: str = field(default="src/vl_mamba/datasets/vl_mamba")
#     # dataset_cache_dir: str = field(default="storage/datasets/tmp")
#     # root_dataset_path: str = field(default="../datasets/vl-mamba")
#     dataset_cache_dir: str = field(default="/mnt/ceph_rbd/storage/datasets/vl_mamba")
#     root_dataset_path: str = field(default="/mnt/ceph_rbd/storage/datasets/")
#     dataset_subset: str = field(default="llava_pretrain")
#     image_mean: tuple[float, float, float] = field(default=(0.48145466, 0.4578275, 0.40821073))
#     image_std: tuple[float, float, float] = field(default=(0.26862954, 0.26130258, 0.27577711))
#     rescale_factor: float = field(default=0.00392156862745098)
#     variable_sized: bool = field(default=False)
#     box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = field(default="normalize")


# @dataclass
# class ModelArguments:
#     """Model arguments."""

#     image_size: int = field(default=336)
#     patch_size: int = field(default=14)
#     num_channels: int = field(default=3)

#     tokenizer_name: str = field(default="EleutherAI/gpt-neox-20b")
#     tokenizer_truncation_side: Literal["right", "left"] = field(default="right")
#     tokenizer_padding_side: Literal["right", "left"] = field(default="right")
#     tokenizer_add_special_tokens: bool = field(default=True)
#     model_max_length: int = field(default=1024)


# def build_tokenizer(model_args: ModelArguments) -> transformers.AutoTokenizer:
#     """Build tokenizer."""
#     tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tokenizer_name)

#     # Padding and eos tokens are the same
#     tokenizer.eos_token = "<|endoftext|>"
#     tokenizer.pad_token = tokenizer.eos_token

#     tokenizer.model_max_length = model_args.model_max_length
#     tokenizer.truncation_side = model_args.tokenizer_truncation_side
#     tokenizer.padding_side = model_args.tokenizer_padding_side

#     return tokenizer


# def run() -> None:
#     """Build datasets."""
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
#     model_args, data_args = parser.parse_args_into_dataclasses()

#     tokenizer = build_tokenizer(model_args)

#     dataset = PretrainDataset(
#         dataset_path=data_args.dataset_path,
#         dataset_cache_dir=data_args.dataset_cache_dir,
#         root_dataset_path=data_args.root_dataset_path,
#         dataset_split=DatasetSplits.TRAIN,
#         dataset_subset=data_args.dataset_subset,
#         tokenizer=tokenizer,
#         model_max_length=model_args.model_max_length,
#         image_mean=data_args.image_mean,
#         image_std=data_args.image_std,
#         image_size=model_args.image_size,
#         patch_size=model_args.patch_size,
#         variable_sized=data_args.variable_sized,
#         box_mode=data_args.box_mode,
#         needs_attention_mask=True,
#         pixel_based=False,
#     )

#     collate_fn = Collate(
#         padding=DatasetPadding(input_ids=tokenizer.pad_token_id),
#         padding_side=tokenizer.padding_side,
#     )

#     dataloader = DataLoader(dataset, num_workers=0, collate_fn=collate_fn, batch_size=2)

#     model = VLCLIPGPTNeoXForCausalLM.from_pretrained(
#         "storage/checkpoints/pythiaeva/pythiaeva/checkpoint-10900/",
#         vision_encoder_name="timm/eva02_large_patch14_clip_336",
#         image_size=336,
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     for batch in tqdm(dataloader, total=len(dataset) // 2):
#         breakpoint()
#         with torch.no_grad():
#             outs = model(
#                 input_ids=batch.input_ids.to(model.device),
#                 pixel_values=batch.pixel_values.to(model.device).unsqueeze(0),
#             )
#         continue


# run()

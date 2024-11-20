# Adapted from https://github.com/state-spaces/mamba/blob/maintorc/mamba_ssm/models/mixer_seq_simple.py
# Copyright (c) 2023, Albert Gu, Tri Dao.
import glob
import json
import math
import os
from collections import namedtuple
from functools import partial
from typing import Any

import torch
from loguru import logger
from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from safetensors.torch import load_model
from torch import nn
from transformers import CLIPVisionModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import load_sharded_checkpoint

from vl_mamba.models.gated_xattn import (
    GatedCrossAttentionBlock,
    invert_attention_mask,
)
from vl_mamba.models.vision_encoder import build_vision_encoder
from vl_mamba.utils.compute_loss import compute_loss
from vl_mamba.utils.generation import GenerationMixin, decode

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


from dataclasses import dataclass, field


@dataclass
class VLMambaConfig:
    d_model: int = 2560
    hidden_size: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    image_size: int = 512
    patch_size: int = 16
    num_channels: int = 3


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
) -> None:
    # names = [name for name, p in module.named_parameters()]
    # logger.info(names)
    # breakpoint()
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "gated" in name.lower():
                # The values of the gated cross attention block have already been initialized
                continue

            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        gated_xattn_config=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        # fused_add_norm = False

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.gated_xattn_config = gated_xattn_config
        self.gated_xattn_config.sharable_params = (
            gated_xattn_config.sharable_params
            if hasattr(gated_xattn_config, "sharable_params")
            else False
        )
        self.cross_layer_interval = gated_xattn_config.cross_layer_interval
        num_cross_layers = n_layer // self.cross_layer_interval

        logger.warning(
            f"Number of cross layers: {num_cross_layers}, sharable parames: {gated_xattn_config.sharable_params}"
        )
        if self.gated_xattn_config.sharable_params:
            self.gated_cross_attn_layers = GatedCrossAttentionBlock(config=gated_xattn_config)
        else:
            self.gated_cross_attn_layers = nn.ModuleList(
                [
                    GatedCrossAttentionBlock(config=gated_xattn_config)
                    for _ in range(num_cross_layers)
                ]
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        patch_embeddings=None,
        cross_attention_gate=None,
        image_attention_mask=None,
        inference_params=None,
    ):
        hidden_states = self.embedding(input_ids)

        residual = None
        for _decoder_layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if (_decoder_layer_idx + 1) % self.cross_layer_interval == 0:
                hidden_states = (
                    (hidden_states + residual) if residual is not None else hidden_states
                )

                if self.gated_xattn_config.sharable_params:
                    x_attn_block = self.gated_cross_attn_layers
                else:
                    x_attn_block = self.gated_cross_attn_layers[
                        _decoder_layer_idx // self.cross_layer_interval
                    ]

                outputs = x_attn_block(
                    hidden_states=hidden_states,
                    image_hidden_states=patch_embeddings,
                    cross_attention_gate=cross_attention_gate,
                    image_attention_mask=image_attention_mask,
                )
                # breakpoint()
                hidden_states = outputs[0]
                # Reset the residual so that in can be re-instantiated (as the hidden states)
                # in the mamba block
                residual = None

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class VLMambaCLIPLXAttnMHeadModel(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config,
        gated_xattn_config,
        initializer_cfg=None,
        pad_vocab_size_multiple: int = 1,
        device=None,
        dtype=None,
        **backbone_kwargs,
    ) -> None:
        # TODO: THERE IS A PROBLEM HERE
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config=config)
        # self.config = config
        # d_model = config.d_model
        # n_layer = config.n_layer
        # vocab_size = config.vocab_size
        # ssm_cfg = config.ssm_cfg
        # rms_norm = config.rms_norm
        # residual_in_fp32 = config.residual_in_fp32
        # fused_add_norm = config.fused_add_norm
        # pad_vocab_size_multiple = config.pad_vocab_size_multiple
        # factory_kwargs = {"device": device, "dtype": dtype}

        # super().__init__(config=config)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=config.ssm_cfg,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm,
            gated_xattn_config=gated_xattn_config,
            **backbone_kwargs,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

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

    def tie_weights(self) -> None:
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **kwargs,
    ):
        """Foward pass for MambaLMHeadModel.

        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        (bsz, text_seq_len) = input_ids.shape

        patch_embeddings = self.get_patch_embeddings(pixel_values)

        (_, image_seq_len, _) = patch_embeddings.shape

        # The image attention mask should be (bsz, text_seq_len), and denotes the text tokens that
        # actually attend to the image tokens
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        num_images = 1
        attention_mask = attention_mask.unsqueeze(-1)
        attention_mask = attention_mask.repeat(1, 1, 1, image_seq_len)
        attention_mask = attention_mask.view(bsz, text_seq_len, num_images * image_seq_len)

        attention_mask = invert_attention_mask(attention_mask, dtype=self.dtype)

        # If any of the elements are 0.0, then the token is attending to at least one image and
        # the gate value is 1. Otherwise the gate value is 0.
        # `cross_attention_gate` has shape [bsz, seq_len] with elements equal to either 0.0 or 1.0.
        cross_attention_gate = (
            (((attention_mask == 0).any(dim=-1)).to(dtype=self.dtype)).squeeze(dim=1)
        ).to(attention_mask.device)

        hidden_states = self.backbone(
            input_ids,
            patch_embeddings,
            cross_attention_gate=cross_attention_gate,
            image_attention_mask=attention_mask,
            inference_params=inference_params,
        )

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = compute_loss(labels, lm_logits, self.config.vocab_size)

            # penalty_weight = 0.001
            # log_z = torch.logsumexp(lm_logits, dim=-1)
            # z_loss = (log_z**2).mean()
            # loss = loss + penalty_weight * z_loss

            # # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()

            # # Flatten the tokens
            # loss_fct = nn.CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)

            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
            return (loss,)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     pretrained_model_name,
    #     device=None,
    #     dtype=None,
    #     image_size: int = 512,
    #     patch_size: int = 16,
    #     num_channels: int = 3,
    #     **kwargs,
    # ):
    #     if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
    #         logger.info(f"Loading model from {pretrained_model_name}")
    #         from safetensors.torch import load_model

    #         config_data = load_config_hf(pretrained_model_name)
    #         config = PretrainedConfig(**config_data)
    #         config.hidden_size = config.d_model
    #         config.vocab_size = 50280
    #         config.image_size = image_size
    #         config.patch_size = patch_size
    #         config.num_channels = num_channels
    #         model = cls(config=config, device=device, dtype=dtype, **kwargs)
    #         load_model(model, os.path.join(pretrained_model_name, "model.safetensors"))
    #     else:
    #         config_data = load_config_hf(pretrained_model_name)
    #         config = PretrainedConfig(**config_data)
    #         config.hidden_size = config.d_model
    #         config.vocab_size = 50280
    #         config.image_size = image_size
    #         config.patch_size = patch_size
    #         config.num_channels = num_channels
    #         model = cls(config=config, device=device, dtype=dtype, **kwargs)
    #         model.load_state_dict(
    #             load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype), strict=False
    #         )

    #     return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        image_size: int = 224,
        vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        select_layer: int = -2,
        select_feature: str = "patch",
        cross_attention_config: str | None = None,
        **kwargs,
    ):
        """Load a pretrained model from a given model name or path."""
        config_data = load_config_hf(pretrained_model_name)
        config = PretrainedConfig(**config_data)
        config.hidden_size = config.d_model
        config.vocab_size = 50280
        config.vision_encoder_name = vision_encoder_name
        config.select_layer = select_layer
        config.select_feature = select_feature
        config.image_size = image_size

        logger.info(f"Using {cross_attention_config} config for GatedCrossAttention")
        gated_xattn_config_data = json.load(open(cross_attention_config))
        gated_xattn_config = PretrainedConfig(**gated_xattn_config_data)

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
            model = cls(
                config=config,
                device=device,
                dtype=dtype,
                gated_xattn_config=gated_xattn_config,
                **kwargs,
            )

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
            model = cls(
                config=config,
                device=device,
                dtype=dtype,
                gated_xattn_config=gated_xattn_config,
                **kwargs,
            )

            model.load_state_dict(
                load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype), strict=False
            )

        return model

    # def save_pretrained(self, save_directory):
    #     """
    #     Minimal implementation of save_pretrained for MambaLMHeadModel.
    #     Save the model and its configuration file to a directory.
    #     """
    #     # Ensure save_directory exists
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)

    #     # Save the model's state_dict
    #     model_path = os.path.join(save_directory, "pytorch_model.bin")
    #     torch.save(self.state_dict(), model_path)

    #     # Save the configuration of the model
    #     config_path = os.path.join(save_directory, "config.json")
    #     with open(config_path, "w") as f:
    #         json.dump(self.config.__dict__, f)

    def generate(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        max_length,
        top_k=1,
        top_p=0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pixel_values=pixel_values,
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences

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


# model = VLMambaCLIPLXAttnMHeadModel.from_pretrained(
#     pretrained_model_name="state-spaces/mamba-790m",
#     image_size=336,
#     vision_encoder_name="timm/eva02_large_patch14_clip_336",
#     select_feature="patch",
#     cross_attention_config="configs/model/mamba-790m_layer_interval4.json",
# )

# breakpoint()

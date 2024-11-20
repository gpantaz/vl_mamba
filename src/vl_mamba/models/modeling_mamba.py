# Adapted from https://github.com/state-spaces/mamba/blob/maintorc/mamba_ssm/models/mixer_seq_simple.py
# Copyright (c) 2023, Albert Gu, Tri Dao.
import json
import math
import os
from functools import partial
import glob

import torch
from loguru import logger
from collections import namedtuple
from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.utils.generation import GenerationMixin, decode
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from safetensors.torch import load_model
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import load_sharded_checkpoint

from vl_mamba.models.gated_xattn import invert_attention_mask

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
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)


class LlamaMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class TransformerBlock(LlamaDecoderLayer):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.config = config
        self.pre_downsample_layernorm = LlamaRMSNorm(config.input_size, eps=config.rms_norm_eps)
        self.pre_upsample_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample_mlp = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.upsample_mlp = nn.Linear(config.hidden_size, config.input_size, bias=False)

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.pre_downsample_layernorm(hidden_states)

        hidden_states = self.downsample_mlp(hidden_states)

        position_ids = torch.arange(
            0,
            hidden_states.shape[1],
            device=hidden_states.device,
        ).unsqueeze(0)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.pre_upsample_layernorm(hidden_states)

        hidden_states = self.upsample_mlp(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


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
        attention_config=None,
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

        self.attention_config = attention_config
        if self.attention_config is not None:
            self.cross_layer_interval = attention_config.cross_layer_interval
            num_cross_layers = n_layer // self.cross_layer_interval

            self.transformer_block = nn.ModuleList(
                [
                    TransformerBlock(config=self.attention_config, layer_idx=layer_idx)
                    for layer_idx in range(num_cross_layers)
                ]
            )
        else:
            self.transformer_block = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        cross_attention_input_ids=None,
        attention_mask=None,
        inference_params=None,
    ):
        hidden_states = self.embedding(input_ids)

        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if (
                self.attention_config is not None
                and (layer_idx + 1) % self.cross_layer_interval == 0
            ):
                cross_attn_layer_idx = (layer_idx + 1) // self.cross_layer_interval - 1

                # TODO: try this
                # residual = hidden_states

                # hidden_states = self.transformer_block[cross_attn_layer_idx](
                #     hidden_states,
                #     key_value_states=self.embedding(cross_attention_input_ids)
                #     if cross_attention_input_ids
                #     else None,
                #     attention_mask=attention_mask,
                # )[0]
                hidden_states = self.transformer_block[cross_attn_layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                )[0]

                residual = None

                # hidden_states = hidden_states + residual

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


class MambaLMHeadModel(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config,
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
        self.config = config
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
            initializer_cfg=initializer_cfg,
            attention_config=config.attention_config,
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

    def tie_weights(self) -> None:
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def get_input_embeddings(self):
        return self.backbone.embedding

    def set_input_embeddings(self, new_embeddings) -> None:
        self.backbone.embedding = new_embeddings
        self.lm_head = nn.Linear(
            new_embeddings.weight.shape[1],
            new_embeddings.weight.shape[0],
            bias=False,
            device=self.lm_head.weight.device,
            dtype=self.lm_head.weight.dtype,
        )
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """Foward pass for MambaLMHeadModel.

        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if pixel_values:
            pixel_values = torch.stack(pixel_values).squeeze()

        if self.config.attention_config is not None and attention_mask is None:
            if pixel_values:
                attention_mask = torch.ones_like(input_ids)

                (bsz, text_seq_len) = input_ids.shape
                (_, cross_seq_len) = pixel_values.shape

                attention_mask = attention_mask.unsqueeze(-1)
                attention_mask = attention_mask.repeat(1, 1, 1, cross_seq_len)
                attention_mask = attention_mask.view(bsz, text_seq_len, cross_seq_len)

                attention_mask = invert_attention_mask(attention_mask, dtype=self.dtype)
            else:
                attention_mask = None

        hidden_states = self.backbone(
            input_ids,
            cross_attention_input_ids=pixel_values,
            attention_mask=attention_mask,
            inference_params=inference_params,
        )

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # loss = compute_loss(labels, lm_logits, self.config.vocab_size)

            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

            predicted_tokens = torch.argmax(lm_logits, dim=-1)
            if self.config.prefix:
                # labels = [-100, -100, ..., -100, target_label, 0]
                # predicted_tokens = [0, 0, ..., predicted_label, 0, w/e]
                target_token_positions = labels[:, -2]
                predicted_token_positions = predicted_tokens[:, -3]
            else:
                # labels = [-100, -100, ..., -100, target_label, 0]
                # predicted_tokens = [0, 0, ..., predicted_label, 0, w/e]
                target_token_positions = labels[:, -2]
                predicted_token_positions = predicted_tokens[:, -3]
                
            correct_per_position = torch.zeros(self.config.seq_len)
            for target_token, predicted_token in zip(target_token_positions, predicted_token_positions):
                correct = int(target_token == predicted_token.item())
                correct_per_position[target_token.item() - 2] = correct_per_position[target_token.item() - 2] + correct

            correct = torch.sum(predicted_token_positions == target_token_positions)
            return (loss, correct, correct_per_position)

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
    #         print(f"Loading model from {pretrained_model_name}")
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
        new_num_tokens: int | None = None,
        attention_config: str | None = None,
        seq_len: int = 50,
        is_prefix: bool = False,
        **kwargs,
    ):
        """Load a pretrained model from a given model name or path."""
        config_data = load_config_hf(pretrained_model_name)
        config = PretrainedConfig(**config_data)
        config.hidden_size = config.d_model
        config.vocab_size = 50280
        config.seq_len = seq_len
        config.prefix = is_prefix

        if attention_config is not None:
            attn_config = json.load(open(attention_config))
            config.attention_config = PretrainedConfig(**attn_config)
        else:
            config.attention_config = None

        model = cls(config=config, device=device, dtype=dtype, **kwargs)
        if new_num_tokens is not None:
            embed_size = model.get_input_embeddings().weight.shape[0]
            new_size = embed_size + new_num_tokens
            logger.info(
                f"Resizing token embeddings to {new_size} = {embed_size} + {new_num_tokens}"
            )
            model.resize_token_embeddings(new_size)

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
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
            model.load_state_dict(
                load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype), strict=False
            )

        return model

    def generate(
        self,
        input_ids,
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
            **kwargs,
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences

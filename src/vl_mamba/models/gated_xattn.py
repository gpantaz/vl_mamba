from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from loguru import logger
from torch import nn
from transformers.activations import ACT2FN


@dataclass
class VLMambaGatedXAttnConfig:
    """Configuration class to store the configuration of a `VLMambaGatedXAttn` model.

    These are the default values for mamba-790m
    """

    _name_or_path = None
    alpha_initializer = "zeros"
    alpha_type = "float"
    alphas_initializer_range = 0.0
    architectures = (["MambaVLXAttn"],)
    cross_layer_interval = 4
    dropout = 0.0
    hidden_act = "silu"
    hidden_size = 1536
    intermediate_size = 6144
    num_attention_heads = 8
    qk_layer_norms = True
    rms_norm_eps = 1e-05
    use_cache = False
    vision_config = {"embed_dim": 1024}


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        """LlamaRMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def invert_attention_mask(
    encoder_attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Invert an attention mask (e.g., switches 0. and 1.)."""
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(
        dtype
    ).min

    return encoder_extended_attention_mask


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Possible replace with LLama2 attention
class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: VLMambaGatedXAttnConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dropout = config.dropout
        # We just do cross attention here
        self.is_causal = False
        self.config = config

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ValueError("this model requires pytorch 2.0 or higher")

        kv_input_dim = config.vision_config["embed_dim"]
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(kv_input_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            kv_input_dim,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.qk_layer_norms = config.qk_layer_norms
        if self.qk_layer_norms:
            # self.q_layer_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
            # self.k_layer_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)

            self.q_layer_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    #     return (
    #         tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    #     )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # if key_value_states are provided this layer is used as a cross-attention layer

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # if not is_cross_attention:
        #     key_states = (
        #         self.k_proj(hidden_states)
        #         .view(bsz, q_len, self.num_heads, self.head_dim)
        #         .transpose(1, 2)
        #     )
        #     value_states = (
        #         self.v_proj(hidden_states)
        #         .view(bsz, q_len, self.num_heads, self.head_dim)
        #         .transpose(1, 2)
        #     )
        # else:
        #     (
        #         _,
        #         kv_len,
        #         _,
        #     ) = key_value_states.size()  # Note that, in this case, `kv_len` == `kv_seq_len`
        #     key_states = (
        #         self.k_proj(key_value_states)
        #         .view(bsz, kv_len, self.num_heads, self.head_dim)
        #         .transpose(1, 2)
        #     )
        #     value_states = (
        #         self.v_proj(key_value_states)
        #         .view(bsz, kv_len, self.num_heads, self.head_dim)
        #         .transpose(1, 2)
        #     )
        (_, kv_len, _) = key_value_states.size()

        key_states = (
            self.k_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # if not is_cross_attention:
        #     cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        #     query_states, key_states = apply_rotary_pos_emb(
        #         query_states, key_states, cos, sin, position_ids
        #     )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        attn_weights = None
        if output_attentions:
            logger.warning(
                "attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead"
            )

        return attn_output, attn_weights, past_key_value


class GatedCrossAttentionBlock(nn.Module):
    """Gated Cross Attention Block."""

    def __init__(self, config: VLMambaGatedXAttnConfig) -> None:  # noqa: PLR0912
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = Attention(
            config=config,
            # hidden_size=self.hidden_size,
            # num_heads=config.num_attention_heads,
            # is_cross_attention=True,
            # dropout=config.dropout,
            # qk_layer_norms=config.qk_layer_norms,
        )
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout_p = config.dropout

        self.act_cross_attn = nn.Tanh()
        self.act_dense = nn.Tanh()

        if config.alpha_initializer == "zeros":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1))
                self.alpha_dense = nn.Parameter(torch.zeros(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer == "ones":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.ones(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1))
                self.alpha_dense = nn.Parameter(torch.ones(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer in {"normal", "gaussian", "random"}:
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(
                        mean=0.0,
                        std=config.alphas_initializer_range,
                        size=(1, 1, self.hidden_size),
                    )
                )
                self.alpha_dense = nn.Parameter(
                    torch.normal(
                        mean=0.0,
                        std=config.alphas_initializer_range,
                        size=(1, 1, self.hidden_size),
                    )
                )
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1))
                )
                self.alpha_dense = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1))
                )
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        else:
            raise NotImplementedError(
                f"Alpha initialization scheme {config.alpha_initializer} not yet implemented!"
            )

        if not (hasattr(self, "alpha_cross_attn") and hasattr(self, "alpha_dense")):
            raise ValueError("Alpha parameters not initialized correctly!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_hidden_states: torch.Tensor,
        cross_attention_gate: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        past_key_value: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor]:
        """Forward pass."""
        if past_key_value is not None:
            raise NotImplementedError(
                "Past key value states are not implemented for cross attention module."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Cross Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=image_hidden_states,
            attention_mask=image_attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout_p, training=self.training
        )
        hidden_states[cross_attention_gate == 0] = hidden_states[cross_attention_gate == 0].fill_(
            0
        )

        hidden_states = residual + self.act_cross_attn(self.alpha_cross_attn) * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout_p, training=self.training
        )
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore[report]


# # config = Config.from_pretrained("HuggingFaceM4/idefics-9b")
# config = VLMambaGatedXAttnConfig()
# cross_attn = GatedCrossAttentionBlock(config)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# with torch.autocast(device):
#     cross_attn = cross_attn.to(device)
#     hidden_states = torch.randn(2, 10, config.hidden_size).to(device)
#     image_hidden_states = torch.randn(2, 576, config.vision_config["embed_dim"]).to(device)
#     image_batch_size, image_sequence_length, _ = image_hidden_states.size()
#     image_hidden_shape = (image_batch_size, image_sequence_length)

#     # image_attention_mask = image_attention_mask.unsqueeze(-1)
#     # image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
#     # image_attention_mask = image_attention_mask.view(batch_size, text_seq_len, num_images * image_seq_len)

#     image_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],],device=device)
#     image_attention_mask = image_attention_mask.unsqueeze(-1)
#     image_attention_mask = image_attention_mask.repeat(1, 1, 1, 576)
#     image_attention_mask = image_attention_mask.view(2, 10, 1 * 576)

#     image_attention_mask = invert_attention_mask(image_attention_mask, dtype=hidden_states.dtype)

#     out = cross_attn(
#         hidden_states=hidden_states,
#         image_hidden_states=image_hidden_states,
#         image_attention_mask=image_attention_mask,
#     )

# breakpoint()
# yy = 2

# image_attention_mask = torch.ones(image_hidden_shape, device=device)
# cross_attention_gate = (
#     (((image_attention_mask == 0.0).any(dim=-1)).to(dtype=image_attention_mask.dtype)).squeeze(dim=1)
# ).to(device)

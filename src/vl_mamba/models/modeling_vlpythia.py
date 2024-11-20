import os
from typing import Any, Optional

import torch
from loguru import logger
from safetensors.torch import load_model
from transformers import AutoConfig, GPTNeoXModel, GPTNeoXPreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available

from vl_mamba.utils.compute_loss import compute_loss
from vl_mamba.utils.hugginface import load_config_hf, load_state_dict_hf


class VLGPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    """VLGPTNeoXForCausalLM model."""

    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        modules = [
            torch.nn.Linear(
                config.patch_size * config.patch_size * config.num_channels,
                config.hidden_size,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        ]

        self.vision_embed_tokens = torch.nn.Sequential(*modules)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> torch.nn.Linear:  # noqa: WPS615
        """Get output embeddings."""
        return self.embed_out

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:  # noqa: WPS615
        """Set output embeddings."""
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.tensor] = None,
        pixel_values: Optional[torch.tensor] = None,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        head_mask: Optional[torch.tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.tensor]]] = None,
        labels: Optional[torch.tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Foward pass for MambaLMHeadModel."""
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        patch_embeddings = [self.vision_embed_tokens(patches) for patches in pixel_values]

        # step 2: concatenate the visual embeddings and the textual embeddings
        # language_model_attention_mask = torch.ones(
        #     patch_embeddings.size()[:-1],
        #     dtype=torch.long,
        #     device=patch_embeddings.device,
        # )

        masked_input_ids = input_ids.clone()
        masked_input_ids[input_ids < 0] = 0
        inputs_embeds = self.gpt_neox.embed_in(masked_input_ids)

        for batch_idx in range(inputs_embeds.shape[0]):
            dst_indices = (input_ids[batch_idx] < 0).nonzero(as_tuple=True)[0]
            inputs_embeds[batch_idx, dst_indices] = patch_embeddings[batch_idx][0].to(
                dtype=inputs_embeds.dtype
            )

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
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.tensor,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        pixel_values: Optional[torch.tensor] = None,
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
            }
        )

        return model_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str,
        patch_size: int = 16,
        num_channels: int = 3,
        use_flash_attention_2: bool = False,  # noqa: WPS114
    ):
        """Load model from pretrained model name."""
        use_flash_attention_2 = (  # noqa: WPS114
            use_flash_attention_2 and is_flash_attn_2_available()
        )
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
            config_data = load_config_hf(pretrained_model_name)
            config = PretrainedConfig(**config_data)
            config.patch_size = patch_size
            config.num_channels = num_channels
            config._attn_implementation = attn_implementation
            model = cls(config=config)
            logger.info(f"Loading model from {pretrained_model_name}")
            if os.path.exists(os.path.join(pretrained_model_name, "model.safetensors")):
                logger.info("Attempting to load model using safetensors")
                try:
                    load_model(model, os.path.join(pretrained_model_name, "model.safetensors"))
                except Exception as safetensor_exception:
                    logger.error(f"Failed to load model using safetensors: {safetensor_exception}")

            elif os.path.exists(os.path.join(pretrained_model_name, "pytorch_model.bin")):
                logger.info("Attempting to load model using torch.load")
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(pretrained_model_name, "pytorch_model.bin"))
                    )
                except Exception as torch_exception:
                    logger.error(f"Failed to load model using torch.load: {torch_exception}")
            else:
                logger.error(f"Could not load model from {pretrained_model_name}")
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.patch_size = patch_size
            config.num_channels = num_channels
            config._attn_implementation = attn_implementation
            model = cls(config=config)
            model.load_state_dict(load_state_dict_hf(pretrained_model_name), strict=False)
        return model

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.tensor):
        reordered_past = ()
        for layer_past in past_key_values:  # noqa: WPS519
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


# from transformers import AutoTokenizer

# model_name = "EleutherAI/pythia-410m"
# model = VLGPTNeoXForCausalLM.from_pretrained(model_name)
# model.eval()

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# inputs = tokenizer("Hello, I am", return_tensors="pt")
# tokens = model.generate(input_ids=inputs.input_ids, pixel_values=torch.rand(1, 3, 16 * 16 * 3))
# print(tokenizer.decode(tokens[0]))

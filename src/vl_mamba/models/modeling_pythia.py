import glob
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from loguru import logger
from safetensors.torch import load_model
from transformers import (
    AutoConfig,
    CLIPVisionModel,
    GPTNeoXModel,
    GPTNeoXForCausalLM,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import is_flash_attn_2_available

from vl_mamba.utils.compute_loss import compute_loss
from vl_mamba.utils.hugginface import load_config_hf, load_state_dict_hf


@dataclass
class CausalLMOutputWithPastWithCorrect(CausalLMOutputWithPast):
    """CausalLMOutputWithPast with correct number of exaples per batch."""
    correct: Optional[torch.tensor] = None
    correct_per_position: Optional[torch.tensor] = None


class CustomGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """GPTNeoXForCausalLM model."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def get_input_embeddings(self):
        return self.gpt_neox.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.tensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        correct = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            # if lm_loss < 0.5:
            #     breakpoint()
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
            
            correct = torch.sum(predicted_token_positions == target_token_positions)
            correct_per_position = torch.zeros(self.config.seq_len)
            for target_token, predicted_token in zip(target_token_positions, predicted_token_positions):
                correct = int(target_token == predicted_token.item())
                correct_per_position[target_token.item() - 2] = correct_per_position[target_token.item() - 2] + correct

            correct = torch.sum(predicted_token_positions == target_token_positions)

        if not return_dict:
            output = (lm_logits,) + outputs[1:] + correct
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPastWithCorrect(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            correct=correct,
            correct_per_position=correct_per_position,
        )


    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids: torch.tensor,
    #     past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
    #     attention_mask: Optional[torch.tensor] = None,
    #     inputs_embeds: Optional[torch.tensor] = None,
    #     pixel_values: Optional[torch.tensor] = None,
    #     **kwargs,
    # ):
    #     """Prepare inputs for generation."""
    #     input_shape = input_ids.shape
    #     # cut decoder_input_ids if past is used
    #     if past_key_values is not None:
    #         past_length = past_key_values[0][0].shape[2]

    #         # Some generation methods already pass only the last input ID
    #         if input_ids.shape[1] > past_length:
    #             remove_prefix_length = past_length
    #         else:
    #             # Default to old behavior: keep only final ID
    #             remove_prefix_length = input_ids.shape[1] - 1

    #         input_ids = input_ids[:, remove_prefix_length:]

    #     position_ids = kwargs.get("position_ids", None)
    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         if past_key_values:
    #             position_ids = position_ids[:, -input_ids.shape[1] :]

    #     # if model is used as a decoder in encoder-decoder model,
    #  1

    @classmethod
    def from_pretrained(  # noqa: WPS231
        cls,
        pretrained_model_name: str,
        use_flash_attention_2: bool = False,  # noqa: WPS114
        new_num_tokens: Optional[int] = None,
        seq_len: int = 50,
        is_prefix: bool = False,
    ):
        """Load model from pretrained model name."""
        use_flash_attention_2 = (  # noqa: WPS114
            use_flash_attention_2 and is_flash_attn_2_available()
        )
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
            config_data = load_config_hf(pretrained_model_name)
            config = PretrainedConfig(**config_data)
            config._attn_implementation = attn_implementation  # noqa: WPS437
            config.seq_len = seq_len
            config.is_prefix = is_prefix
            model = cls(config=config)
            if new_num_tokens is not None:
                embed_size = model.get_input_embeddings().weight.shape[0]
                new_size = embed_size + new_num_tokens
                logger.info(f"Resizing token embeddings to {new_size} = {embed_size} + {new_num_tokens}")
                model.resize_token_embeddings(new_size)

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
            config._attn_implementation = attn_implementation  # noqa: WPS437
            config.seq_len = seq_len
            model = cls(config=config)
            if new_num_tokens is not None:
                embed_size = model.get_input_embeddings().weight.shape[0]
                new_size = embed_size + new_num_tokens
                logger.info(f"Resizing token embeddings to {new_size} = {embed_size} + {new_num_tokens}")
                model.resize_token_embeddings(new_size)

            model.load_state_dict(load_state_dict_hf(pretrained_model_name), strict=False)
        return model

import timm
import torch
from loguru import logger
from torch.utils import checkpoint
from transformers import CLIPVisionModel, PreTrainedModel


class PretrainedEva02(timm.models.Eva):
    """Pretrained EVA_02 model."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()

    @classmethod
    def from_pretrained_model(cls, model) -> "PretrainedEva02":
        """Create model from pretrained model."""
        new_model = cls()
        new_model.__dict__ = model.__dict__.copy()
        return new_model

    def forward_features(self, x):
        """Forward features."""
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        outputs = self.forward_features(inputs)
        return outputs

    def get_num_layer(self, var_name: str) -> int:
        """Get number of layers."""
        if var_name in {"cls_token", "mask_token", "pos_embed"}:
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split(".")[1])
            return layer_id + 1
        return len(self.blocks)


def create_eva2_model(eva_name: str, img_size: int = 336) -> PretrainedEva02:
    """Create EVA_02 model."""
    pretrained_model = timm.create_model(
        eva_name,
        pretrained=True,
        img_size=img_size,
    )
    model = PretrainedEva02.from_pretrained_model(pretrained_model)

    # use_fused_attention = timm.layers.use_fused_attn()
    # logger.info(f"Using fused attention: {use_fused_attention}")
    return model


def build_vision_encoder(
    vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
    image_size: int = 336,
) -> PreTrainedModel:
    """Build vision encoder."""
    if "eva" in vision_encoder_name:
        logger.info(f"Loading EVA_02 pretrained: {vision_encoder_name}")
        vision_encoder = timm.create_model(
            vision_encoder_name,
            pretrained=True,
            img_size=image_size,
            num_classes=0,
        )
        # Remove the head since we only want the features
        # This will save some memory from the gpu
        # del vision_encoder.head
        # This is a workaround to prevent ovewriting the weights of the vision
        # encoder in post_init() call
        for module_name, module in vision_encoder.named_modules():
            module._is_hf_initialized = True

    elif "clip" in vision_encoder_name:
        logger.info(f"Loading CLIP pretrained: {vision_encoder_name}")
        vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
    else:
        raise ValueError(f"Unknown vision encoder: {vision_encoder_name}")
    return vision_encoder

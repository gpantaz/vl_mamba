import torch

from vl_mamba.datamodels.dataclasses import SpecialTokens


def format_text(
    text: str,
    strip: bool = True,
    capitalize: bool = True,
    punctuate: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    text_bos_token: str = SpecialTokens.text_bos_token,
    text_eos_token: str = SpecialTokens.text_eos_token,
) -> str:
    """Format the text."""
    if strip:
        text = text.strip()

    if capitalize:
        text = text.capitalize()

    if punctuate and not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
        text = f"{text}."

    if add_bos and not text.startswith(text_bos_token):
        text = f"{text_bos_token}{text}"

    if add_eos and not text.endswith(text_eos_token):
        text = f"{text}{text_eos_token}"
    return text


# def prepare_image_tokens(
#     special_tokens: SpecialTokens, image_size=(224, 224), patch_size=16
# ) -> torch.tensor:
#     row_size, col_size = image_size[1], image_size[0]
#     img_tokens_per_row = torch.concatenate(
#         [
#             torch.tensor([special_tokens.img_token_id] * int(col_size / patch_size)),
#             torch.tensor([special_tokens.img_sep_token_id]),
#         ]
#     )
#     img_tokens = img_tokens_per_row.repeat(int(row_size / patch_size))
#     img_tokens[-1] = special_tokens.img_eos_token_id
#     img_tokens = torch.concatenate([torch.tensor([special_tokens.img_bos_token_id]), img_tokens])
#     return img_tokens


def prepare_image_tokens(
    special_tokens: SpecialTokens, image_size: tuple[int, int] = (224, 224), patch_size: int = 16
) -> torch.Tensor:
    """Prepare the image token sequence."""
    row_size, col_size = image_size[1], image_size[0]

    # Step1: Create a rectangle with image token ids
    dim = (row_size // patch_size, col_size // patch_size)
    img_tokens = torch.full(
        dim,
        special_tokens.img_token_id,
    )

    # Step2: Concatenate the separator token to the end of each row
    img_tokens = torch.concatenate(
        [
            img_tokens,
            torch.full((row_size // patch_size, 1), special_tokens.img_sep_token_id),
        ],
        dim=-1,
    )

    # Step3: Flatten the image tokens
    img_tokens = img_tokens.flatten()

    # Step4: Add the bos and eos tokens
    img_tokens[-1] = special_tokens.img_eos_token_id
    img_tokens = torch.concatenate([torch.tensor([special_tokens.img_bos_token_id]), img_tokens])
    return img_tokens

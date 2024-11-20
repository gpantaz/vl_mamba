import math
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import pad, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
)

from vl_mamba.datamodels.dataclasses import SpecialTokens, VisualEncoding


DEFAULT_SIZE = (1080, 1920)
DEFAULT_PATCH_SIZE = (30, 30)

VERY_IMPORTANT_INFORMATION = """
An image with width x height:
In numpy array, the shape is (height, width, channels),
meaning that the ROWS are the height and the COLUMNS are the width.
In PIL, the size is (width, height)
A bbox has the format (x_min, y_min, x_max, y_max),
where the x-axis is the height and the y-axis is the width
"""


class VLMambaImageProcessor(BaseImageProcessor):  # noqa: WPS230
    """Image processor for VLMamba.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image to `size`.
        size (`dict[str, int]`, *optional*, defaults to `{"height": 1080, "width": 1920}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the image.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to `size`.
        padding_value (`float`, *optional*, defaults to 1.0):
            The value to pad the image with.
        padding_mode (`str`, *optional*, defaults to `"constant"`):
            The padding mode to use when padding the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float`, *optional*, defaults to 0.5):
            The mean to use when normalizing the image.
        image_std (`float`, *optional*, defaults to 0.5):
            The standard deviation to use when normalizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `1 / 255`):
            The factor to use when rescaling the image.
        patch_size (`dict[str, int]`, *optional*, defaults to `{"height": 30, "width": 30}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Union[float, list[float]] = 0.5,
        image_std: Union[float, list[float]] = 0.5,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,  # noqa: WPS404, WPS432
        patch_size: Optional[dict[str, int]] = None,
        variable_sized: bool = True,
        pixel_based: bool = True,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = (
            size if size is not None else {"height": DEFAULT_SIZE[0], "width": DEFAULT_SIZE[1]}
        )
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.patch_size = (
            patch_size
            if patch_size is not None
            else {"height": DEFAULT_PATCH_SIZE[0], "width": DEFAULT_PATCH_SIZE[1]}
        )
        self.variable_sized = variable_sized
        self.pixel_based = pixel_based

    def resize(
        self,
        image: np.ndarray,  # type: ignore[type-arg]
        size: dict[str, int],
        variable_sized: bool,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Resize an image."""
        image_height, image_width = get_image_size(image, input_data_format)
        target_height, target_width = size["height"], size["width"]

        # If we are dealing with variable sized images, we need to find the minimum height and width that
        # exceed the size of the image and are divisible by the patch size
        if variable_sized:
            target_height = min(
                target_height,
                math.ceil(image_height / self.patch_size["height"]) * self.patch_size["height"],
            )
            target_width = min(
                target_width,
                math.ceil(image_width / self.patch_size["width"]) * self.patch_size["width"],
            )
            # return image

        scaled_image = resize(
            image=image,
            size=(target_height, target_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return scaled_image

    def pad_image(
        self,
        image: np.ndarray,  # type: ignore[type-arg]
        size: dict[str, int],
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Pad an image."""
        image_height, image_width = get_image_size(image, input_data_format)
        target_height, target_width = size["height"], size["width"]
        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width
        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def preprocess(  # noqa: WPS231
        self,
        images: ImageInput,
        bboxes: Optional[list[list[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> VisualEncoding:
        """Utility function to preprocess the images."""
        # All transformations expect numpy arrays.
        if isinstance(images, (Image.Image, np.ndarray)):
            batch_images = [np.array(images)]
            batch_bboxes = [np.array(bboxes)] if bboxes is not None else None
        else:
            batch_images = [np.array(image) for image in images]
            batch_bboxes = [np.array(bbox) for bbox in bboxes] if bboxes is not None else None

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(batch_images[0])

        original_image_sizes = [
            get_image_size(image, channel_dim=input_data_format) for image in batch_images
        ]

        if self.do_resize:
            batch_images = [
                self.resize(
                    image,
                    size=self.size,
                    input_data_format=input_data_format,
                    variable_sized=self.variable_sized,
                )
                for image in batch_images
            ]

        image_sizes = [
            get_image_size(image, channel_dim=input_data_format) for image in batch_images
        ]

        if self.do_rescale:
            batch_images = [
                self.rescale(image, scale=self.rescale_factor, input_data_format=input_data_format)
                for image in batch_images
            ]

        if self.do_normalize:
            batch_images = [
                self.normalize(
                    image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format=input_data_format,
                )
                for image in batch_images
            ]

        if data_format is not None:
            batch_images = [
                to_channel_dimension_format(image, data_format, input_data_format)
                for image in batch_images
            ]

        image_scale_factors = [
            (resized_size[0] / original_size[0], resized_size[1] / original_size[1])
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]

        image_patches = [] if self.pixel_based else None
        image_input_ids = []
        image_bboxes = [] if batch_bboxes is not None else None
        image_bboxes_norm = [] if batch_bboxes is not None else None

        for idx, image in enumerate(batch_images):
            resized_size = image_sizes[idx]
            if image_patches:
                num_patches = self.get_num_patches(
                    image_height=resized_size[0], image_width=resized_size[1]
                )
                patches = self.patchify_image(image=torch.tensor(image).unsqueeze(0)).squeeze(0)
                if num_patches != patches.shape[0]:
                    raise AssertionError(
                        f"Expected {num_patches} patches, got {patches.shape[0]} patches."
                    )

                image_patches.append(patches)

            image_input_ids.append(
                self.prepare_image_tokens(special_tokens=SpecialTokens(), image_size=resized_size)
            )

            if batch_bboxes is not None:
                # bboxes have x_min, y_min, x_max, y_max format, we need to expand the scale factor
                scale_factors = image_scale_factors[idx][::-1] * 2
                img_bboxes = batch_bboxes[idx]
                image_bboxes.append(img_bboxes * scale_factors)
                image_bboxes_norm.append(img_bboxes * scale_factors / (resized_size[::-1] * 2))

        return VisualEncoding(
            input_ids=image_input_ids,
            image_patches=image_patches,
            # images=batch_images,
            images=[torch.tensor(image) for image in batch_images],
            bboxes=image_bboxes,
            bboxes_norm=image_bboxes_norm,
            # pixel_values=[torch.tensor(image) for image in batch_images],
            image_scale_factors=image_scale_factors,
        )

    def get_num_patches(self, image_height: int, image_width: int) -> int:
        """Calculate number of patches required to encode an image."""
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height} must be divisible by {patch_height}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width} must be divisible by {patch_width}")

        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        return num_patches

    def patchify_image(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an image into a tensor of patches."""
        # Use the variable_sized_patchify method for variable sized images
        if self.variable_sized:
            return self.variable_sized_patchify(image)
        # Use the square_patchify method for squared images
        # This is faster than the variable_sized_patchify method
        return self.square_patchify(image)

    def variable_sized_patchify(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a batch of any image into a tensor of patches."""
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        # TODO refer to https://github.com/ArthurZucker/transformers/blob/0f0a3fe5ca5697ee58faeb5b53f049af720b5e98/src/transformers/models/vit_mae/modeling_vit_mae.py#L871
        # torch implementation is faster but does not handle non-squares

        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
        return patches

    def square_patchify(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a batch of squared image into a tensor of patches."""
        # By convention, patch height in squared images is the same as patch width
        patch_size = self.patch_size["height"]
        # sanity checks
        not_square = image.shape[2] != image.shape[3]
        not_divisible = image.shape[2] % patch_size != 0
        if not_square or not_divisible:
            raise ValueError(
                "Make sure the pixel values have a squared size that is divisible by the patch size"
            )

        # patchify
        batch_size, num_channels, _, _ = image.shape

        num_patches_one_direction = image.shape[2] // patch_size
        patchified_pixel_values = image.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction,
            patch_size,
            num_patches_one_direction,
            patch_size,
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction * num_patches_one_direction,
            patch_size**2 * num_channels,
        )
        return patchified_pixel_values

    def prepare_image_tokens(
        self, special_tokens: SpecialTokens, image_size: tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """Prepare the image token sequence."""
        row_size, col_size = image_size[1], image_size[0]

        # Step1: Create a rectangle with image token ids
        dim = (row_size // self.patch_size["height"], col_size // self.patch_size["width"])
        img_tokens = torch.full(
            dim,
            special_tokens.img_token_id,
        )

        # Step2: Concatenate the separator token to the end of each row
        img_tokens = torch.concatenate(
            [
                img_tokens,
                torch.full(
                    (row_size // self.patch_size["height"], 1), special_tokens.img_sep_token_id
                ),
            ],
            dim=-1,
        )

        # Step3: Flatten the image tokens
        img_tokens = img_tokens.flatten()

        # Step4: Add the bos and eos tokens
        img_tokens[-1] = special_tokens.img_eos_token_id
        img_tokens = torch.concatenate(
            [torch.tensor([special_tokens.img_bos_token_id]), img_tokens]
        )
        return img_tokens

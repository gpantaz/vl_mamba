import json
import random
from typing import Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer

from vl_mamba.datamodels.dataclasses import DatasetItem, SpecialTokens
from vl_mamba.datamodels.datamodels import TASK_TEMPLATES_MAP, DatasetSplits, Instance, Task
from vl_mamba.datasets.base_dataset import format_text, prepare_image_tokens
from vl_mamba.datasets.vl_mamba.data_paths import DatasetPaths
from vl_mamba.utils.boxes import patchify_image


class InstructionTuningDataset(Dataset[DatasetItem]):
    """Pretrain dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_cache_dir: str,
        dataset_base_dir: str,
        dataset_split: DatasetSplits,
        tokenizer: PreTrainedTokenizer,
        image_mean: Union[float, tuple[float, float, float]],
        image_std: Union[float, tuple[float, float, float]],
        dataset_subset: str = "coco",
        image_size: Union[int, tuple[int, int]] = 224,
        patch_size: int = 16,
    ) -> None:
        dataset_paths = DatasetPaths(base_dir=dataset_base_dir)
        self.dataset = load_dataset(
            dataset_path,
            dataset_subset,
            cache_dir=dataset_cache_dir,
            root_dataset_path=dataset_base_dir,
            dataset_paths=dataset_paths,
            verification_mode="no_checks",
            trust_remote_code=True,
        )[dataset_split]

        self.dataset_split = dataset_split
        self.tokenizer = tokenizer

        if isinstance(image_mean, float):
            image_mean = (image_mean, image_mean, image_mean)
        self._image_mean = image_mean

        if isinstance(image_std, float):
            image_std = (image_std, image_std, image_std)
        self._image_std = image_std

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self._image_size = image_size
        self._patch_size = patch_size

        # Default image-text pair transform.
        self._image_text_pair_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self._image_mean,
                    std=self._image_std,
                ),
            ]
        )

        self._return_tensors = "pt"
        self._special_tokens = SpecialTokens()

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatasetItem:
        """Get a single instance from the dataset."""
        raw_instance = self.dataset[index]
        instance = Instance.model_validate(raw_instance)
        return self.captioning(instance)

    def captioning(self, instance: Instance) -> DatasetItem:
        """Process the instance for the captioning task.

        The format of the input sequence is ### P11 P12 P13 && P21 P22 P23 ### ##INSTRUCTION PROMPT
        TARGET TEXT##. When creating the labels, mask the image tokens and the instruction prompt
        """
        if instance.caption is None:
            raise AssertionError("Captions for this instance must exist!")

        metadata = json.loads(instance.metadata)

        visual_encoding = self._image_text_pair_transform(instance.image.convert("RGB"))
        visual_encoding = patchify_image(
            visual_encoding.unsqueeze(0),
            patch_size={"height": self._patch_size, "width": self._patch_size},
        ).squeeze(0)

        target_text = format_text(
            instance.caption,
            strip=True,
            capitalize=True,
            punctuate=True,
            add_bos=False,
            add_eos=True,
            text_bos_token=self._special_tokens.text_bos_token,
            text_eos_token=self._special_tokens.text_eos_token,
        )

        # Some datasets have included the prompt in their examples, use it if it is available.
        # By default all dataset loaders should have the prompt in the metadata.
        prompt_text = metadata.get("prompt", None)
        prompt_text = (
            prompt_text
            if prompt_text is not None
            else self._get_random_template_for_task(instance.task)
        )

        prompt_text = format_text(
            prompt_text,
            strip=True,
            capitalize=True,
            punctuate=True,
            add_bos=True,
            add_eos=False,
            text_bos_token=self._special_tokens.text_bos_token,
            text_eos_token=self._special_tokens.text_eos_token,
        )

        instruction_prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors=self._return_tensors,
            truncation=True,
            add_special_tokens=True,
        )

        input_encoding = self.tokenizer(
            target_text,
            return_tensors=self._return_tensors,
            truncation=True,
            add_special_tokens=True,
        )

        input_token_ids = torch.concatenate(
            [instruction_prompt_ids.input_ids, input_encoding.input_ids], dim=-1
        ).squeeze(0)

        # img_tokens = ### p1 p2 p3 && p4 p5 p6 ###
        img_tokens = prepare_image_tokens(
            special_tokens=self._special_tokens,
            image_size=self._image_size,
            patch_size=self._patch_size,
        )

        # The input tokens are the concatenation of the image tokens, the instruction prompt, and the target text.
        input_token_ids = torch.concatenate([img_tokens, input_token_ids])
        # The target tokens are the concatenation of the instruction prompt and the target text.
        # The tokens for the images and the instruction prompt are masked
        target_token_ids = torch.concatenate(
            [
                (torch.ones_like(img_tokens) * -100).unsqueeze(0),
                torch.ones_like(instruction_prompt_ids.input_ids) * -100,
                input_encoding.input_ids,
            ],
            dim=-1,
        ).squeeze(0)

        raw_target = {
            "task": instance.task,
            "target_text": target_text,
            "metadata": json.loads(instance.metadata),
        }

        return DatasetItem(
            input_ids=input_token_ids,
            labels=target_token_ids,
            pixel_values=visual_encoding,
            task=self._get_task_as_tensor(Task.captioning),
            raw_target=raw_target,
        )

    def _get_random_template_for_task(self, task: Task) -> str:
        """Choose a random template for the given task."""
        return random.choice(TASK_TEMPLATES_MAP[task])

    def _get_task_as_tensor(self, task: Task) -> torch.Tensor:
        """Convert the given task to a Tensor."""
        return torch.tensor([Task.get_index(task)], dtype=torch.long)


# dataset_base_dir = "storage/datasets"
# dataset_path = "src/vl_mamba/datasets/vl_mamba"
# dataset_cache_dir = "storage/datasets/tmp"
# dataset_subset = "coco"

# from transformers import AutoTokenizer


# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# tokenizer.eos_token = "<|endoftext|>"
# tokenizer.pad_token = tokenizer.eos_token

# tokenizer_truncation_side = "right"
# tokenizer_padding_side = "right"
# tokenizer.model_max_length = 100


# dataset = InstructionTuningDataset(
#     dataset_path=dataset_path,
#     dataset_cache_dir=dataset_cache_dir,
#     dataset_base_dir=dataset_base_dir,
#     dataset_split=DatasetSplits.VALIDATION,
#     dataset_subset=dataset_subset,
#     tokenizer=tokenizer,
#     image_mean=[0.5, 0.5, 0.5],
#     image_std=[0.5, 0.5, 0.5],
# )

# indices = list(range(len(dataset)))
# import random

# random.shuffle(indices)
# for idx in indices[:100]:
#     xx = dataset[idx]
#     yy = xx.input_ids[xx.input_ids > 0]
#     zz = tokenizer.decode(yy)

#     print(dataset[idx])
#     breakpoint()

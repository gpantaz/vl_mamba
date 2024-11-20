import json
from typing import Any, Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from vl_mamba.datamodels.dataclasses import DatasetItem
from vl_mamba.datamodels.datamodels import DatasetSplits, Instance, Task
from vl_mamba.datasets.vl_mamba.data_paths import DatasetPaths
from vl_mamba.utils.conversation_processor import VLMambaConversationProcessor
from vl_mamba.utils.image_preprocessor import VLMambaImageProcessor


class PretrainDataset(Dataset[DatasetItem]):
    """Pretrain dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_cache_dir: str,
        root_dataset_path: str,
        dataset_split: DatasetSplits,
        tokenizer: PreTrainedTokenizer,
        image_mean: float | tuple[float, float, float],
        image_std: float | tuple[float, float, float],
        dataset_subset: str = "llava_pretrain",
        image_size: int | tuple[int, int] = 224,
        patch_size: int | dict[str, int] = 16,
        variable_sized: bool = False,
        box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = "normalize",
        needs_attention_mask: bool = False,
        model_max_length: int = 500,
        pixel_based: bool = True,
        instruction_first: bool = False,
        add_visual_token_ids: bool = True,
    ) -> None:
        dataset_paths = DatasetPaths(base_dir=root_dataset_path)
        self.dataset = load_dataset(
            dataset_path,
            dataset_subset,
            cache_dir=dataset_cache_dir,
            root_dataset_path=root_dataset_path,
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
            image_size = {"height": image_size, "width": image_size}
        self._image_size = image_size

        if isinstance(patch_size, int):
            patch_size = {"height": patch_size, "width": patch_size}
        self._patch_size = patch_size

        self.image_processor = VLMambaImageProcessor(
            image_mean=self._image_mean,
            image_std=self._image_std,
            size=self._image_size,
            patch_size=self._patch_size,
            variable_sized=variable_sized,
            pixel_based=pixel_based,
        )

        self._box_mode = box_mode
        self.conversation_processor = VLMambaConversationProcessor(box_mode=self._box_mode)

        self._needs_attention_mask = needs_attention_mask
        self._pixel_based = pixel_based
        self._return_tensors = "pt"
        self._model_max_length = model_max_length
        self._instruction_first = instruction_first
        self._add_visual_token_ids = add_visual_token_ids

        # if self._instruction_first:
        #     logger.warning(f"Instruction first is enabled, but it is only supported for {Task.visual_grounding.value} task")

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatasetItem:
        """Get a single instance from the dataset."""
        raw_instance = self.dataset[index]
        instance = Instance.model_validate(raw_instance)
        return self.process_instance(instance)

    def process_instance(self, instance: Instance) -> DatasetItem:
        """Process the instance."""
        bboxes = [region.bbox for region in instance.region] if instance.region else None
        image = instance.image.convert("RGB")
        visual_encoding = self.image_processor.preprocess(images=image, bboxes=bboxes)

        text_encoding = self.conversation_processor(
            instance,
            tokenizer=self.tokenizer,
            split=self.dataset_split,
            visual_encoding=visual_encoding,
        )

        # If we have the cross attention setup, we dont need visual token ids at the input
        if not self._add_visual_token_ids:
            input_ids = torch.concatenate(text_encoding.input_ids)
        # If we dont have the cross attention setup, then check if we need to add the instruction
        # before the input image.
        elif self._instruction_first:
            encoding = [
                text_encoding.input_ids[0],
                visual_encoding.input_ids[0],
            ]

            for txt_encoding in text_encoding.input_ids[1:]:
                encoding.append(txt_encoding)

            input_ids = torch.concatenate(encoding)
        # We dont have the cross attention setup, and we dont need to put the instruction before
        # the input image
        else:
            input_ids = torch.concatenate([visual_encoding.input_ids[0], *text_encoding.input_ids])

        input_ids = input_ids[: self._model_max_length]

        labels = text_encoding.labels if text_encoding.labels is not None else None
        if labels is not None:
            # If we have the cross attention setup, we dont need visual token ids at the input
            if not self._add_visual_token_ids:
                labels = torch.concatenate(labels)
            elif self._instruction_first:
                labels = torch.concatenate(
                    [
                        labels[0],
                        torch.ones_like(visual_encoding.input_ids[0]) * -100,
                    ]
                    + labels[1:]
                )

            else:
                labels = torch.concatenate(
                    [torch.ones_like(visual_encoding.input_ids[0]) * -100, *labels]
                )

            labels = labels[: self._model_max_length]

        raw_target = {
            "task": instance.task,
            "task_metadata": self._get_task_info(instance),
            "text": text_encoding.text,
            "metadata": json.loads(instance.metadata),
        }

        # For models that are pixel-based (no visual encoder) we need the raw pixel values
        # For models that are not pixel-based (there is a visual encoder) we need the processed image
        pixel_values = (
            visual_encoding.image_patches[0].unsqueeze(0)
            if self._pixel_based
            else visual_encoding.images[0].unsqueeze(0)
        )
        return DatasetItem(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids) if self._needs_attention_mask else None,
            task=self._get_task_as_tensor(instance.task),
            raw_target=raw_target,
        )

    def _get_task_info(self, instance: Instance) -> dict[str, Any]:
        """Get the task information."""
        switcher = {
            Task.vqa.value: ["qa_pairs"],
            Task.m_vqa.value: ["qa_pairs"],
            Task.captioning.value: ["caption"],
            Task.dense_captioning.value: ["region"],
            Task.visual_grounding.value: ["region"],
            Task.chat.value: ["chat"],
            Task.gm_vqa.value: ["qa_pairs", "region"],
            Task.grounded_captioning.value: ["caption", "region"],
            Task.itm.value: ["caption"],
        }
        raw_instance = instance.model_dump()
        return {field: raw_instance[field] for field in switcher[instance.task.value]}

    def _get_task_as_tensor(self, task: Task) -> torch.Tensor:
        """Convert the given task to a Tensor."""
        return torch.tensor([Task.get_index(task)], dtype=torch.long)


# root_dataset_path = "../datasets/vl_mamba/"
# dataset_path = "src/vl_mamba/datasets/vl_mamba"
# dataset_cache_dir = "../datasets/vl_mamba/"
# dataset_subset = "visual7w"
# split = DatasetSplits.TRAIN

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# print(tokenizer.vocab_size)

# tokenizer.eos_token = "<|endoftext|>"
# tokenizer.pad_token = tokenizer.eos_token

# tokenizer_truncation_side = "right"
# tokenizer_padding_side = "right"
# tokenizer.model_max_length = 1024

# dataset = PretrainDataset(
#     dataset_path=dataset_path,
#     dataset_cache_dir=dataset_cache_dir,
#     root_dataset_path=root_dataset_path,
#     dataset_split=split,
#     dataset_subset=dataset_subset,
#     tokenizer=tokenizer,
#     image_mean=(0.48145466, 0.4578275, 0.40821073),
#     image_std=(0.26862954, 0.26130258, 0.27577711),
#     image_size=336,
#     patch_size=14,
#     variable_sized=False,
#     box_mode="normalize",
#     needs_attention_mask=False,
#     model_max_length=1024,
#     pixel_based=False,
#     instruction_first=True,
# )


# import random
# while True:
#     idx = random.randint(0, len(dataset))
#     xx = dataset[idx]
#     print(xx.input_ids)
#     print("----")
#     print(xx.labels)
#     print("----")
#     print(tokenizer.decode(xx.input_ids[xx.input_ids > 0]))
#     print("----")
#     breakpoint()
#     yy = 2

# from torch.utils.data import DataLoader
# from vl_mamba.datamodels.dataclasses import DatasetPadding
# from vl_mamba.datasets.collate import Collate

# from tqdm import tqdm

# collate_fn = Collate(
#     padding=DatasetPadding(input_ids=tokenizer.pad_token_id),
#     padding_side=tokenizer.padding_side,
# )
# dataloader = DataLoader(dataset, num_workers=0, collate_fn=collate_fn, batch_size=2, shuffle=False)

# text_tokens = []
# image_tokens = []
# for batch in tqdm(dataloader, total=len(dataset) // 16):
#     xx = "".join(tokenizer.decode(batch["input_ids"][0][batch["input_ids"][0] >0]))

#     input_ids = batch["input_ids"]
#     masked_input_ids = input_ids.clone()
#     masked_input_ids[input_ids < 0] = 0
#     # hidden_states = torch.randn(masked_input_ids.shape, dtype=masked_input_ids.dtype)

#     for batch_idx in range(input_ids.shape[0]):
#         dst_indices = (input_ids[batch_idx] < 0).nonzero(as_tuple=True)[0]
#         breakpoint()
#         hidden_states[batch_idx, dst_indices] = patch_embeddings[batch_idx].to(
#             dtype=hidden_states.dtype
#         )
#     breakpoint()
#     x = 2
#     # text_tokens.extend(raw_target["text_tokens"] for raw_target in batch["raw_target"])
#     # image_tokens.extend(raw_target["image_tokens"] for raw_target in batch["raw_target"])
# print(max(text_tokens), max(image_tokens))


# # indices = list(range(len(dataset)))
# import random

# random.shuffle(indices)
# for idx in indices[:100]:
#     xx = dataset[idx]
#     yy = xx.input_ids[xx.input_ids > 0]
#     zz = tokenizer.decode(yy)

#     print(dataset[idx])
#     breakpoint()

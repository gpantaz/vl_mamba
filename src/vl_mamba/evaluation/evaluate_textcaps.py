import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from datasets import load_dataset
from loguru import logger
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm

from vl_mamba.datamodels.dataclasses import DatasetItem
from vl_mamba.datamodels.datamodels import DatasetSplits, Instance, Task
from vl_mamba.datasets.base_dataset import format_text
from vl_mamba.datasets.vl_mamba.data_paths import DatasetPaths
from vl_mamba.evaluation.base_evaluator import (
    BaseDataArguments,
    BaseEvalDataset,
    BaseGenerationArguments,
    BaseModelArguments,
    ModelType,
    build_model,
    build_tokenizer,
)
from vl_mamba.models.modeling_vlmamba import VLMambaLMHeadModel
from vl_mamba.models.modeling_vlmambaclip import VLMambaCLIPLMHeadModel


class TextCapsEvalDataset(BaseEvalDataset):
    """TextCaps evaluation dataset."""

    def __init__(
        self,
        split: str,
        dataset_path: str,
        dataset_cache_dir: str,
        root_dataset_path: str,
        dataset_subset: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_mean: float | tuple[float, float, float],
        image_std: float | tuple[float, float, float],
        image_size: int | tuple[int, int] = 224,
        patch_size: int | dict[str, int] = 16,
        variable_sized: bool = False,
        box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = "normalize",
        needs_attention_mask: bool = False,
        pixel_based: bool = False,
        instruction_first: bool = False,
    ) -> None:
        super().__init__(
            split=split,
            dataset_path=dataset_path,
            dataset_cache_dir=dataset_cache_dir,
            root_dataset_path=root_dataset_path,
            dataset_subset=dataset_subset,
            tokenizer=tokenizer,
            image_mean=image_mean,
            image_std=image_std,
            image_size=image_size,
            patch_size=patch_size,
            variable_sized=variable_sized,
            box_mode=box_mode,
            needs_attention_mask=needs_attention_mask,
            pixel_based=pixel_based,
            instruction_first=instruction_first,
        )

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item from dataset."""
        dataset_index = list(self.dataset_packed_map.keys())[idx]
        instance_indices = self.dataset_packed_map[dataset_index]
        raw_instance = self.dataset[instance_indices[0]]
        instance = Instance.model_validate(raw_instance)
        return self.process_instance(instance, caption_indices=instance_indices)

    def process_instance(self, instance: Instance, caption_indices: list[int]) -> DatasetItem:
        """Captioning."""
        if not instance.caption and self._split != DatasetSplits.TEST:
            raise AssertionError(f"Instance {instance} has no caption.")

        image = instance.image.convert("RGB")  # type: ignore[report]
        visual_encoding = self.image_processor.preprocess(images=image)

        prompt = self._get_random_template_for_task(Task.captioning)

        if self._split == DatasetSplits.TEST:
            target_text = None
        else:
            target_text = [
                format_text(
                    Instance.model_validate(self.dataset[idx]).caption,
                    strip=True,
                    punctuate=True,
                    capitalize=True,
                    add_bos=False,
                    add_eos=False,
                )
                for idx in caption_indices
            ]
        # There are no other turns in captioning, only the prompt and the caption.
        full_conversation = [
            f"{prompt}\n",
        ]

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
            for turn in full_conversation
        ]

        input_ids = torch.concatenate(tokens, dim=-1)

        if self._instruction_first:
            input_ids = torch.concatenate([input_ids, visual_encoding.input_ids[0]])  # type: ignore[report]
        else:
            input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        # input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])

        raw_target = {
            "target_text": target_text,
            "metadata": json.loads(instance.metadata),
        }

        pixel_values = (
            visual_encoding.image_patches[0].unsqueeze(0)  # type: ignore[report]
            if self._pixel_based
            else visual_encoding.images[0].unsqueeze(0)  # type: ignore[report]
        )

        return DatasetItem(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids) if self._needs_attention_mask else None,
            raw_target=raw_target,
        )

    def prepare_dataset(self) -> None:
        """Prepare dataset."""
        self.dataset = load_dataset(  # type: ignore[report]
            self._dataset_path,
            self._dataset_subset,
            cache_dir=self._dataset_cache_dir,
            root_dataset_path=self._root_dataset_path,
            dataset_paths=DatasetPaths(self._root_dataset_path),
            verification_mode="no_checks",
            trust_remote_code=True,
        )[self._split]

        # Pack the dataset into examples with same images
        self.dataset_packed_map = defaultdict(list)
        for idx, metadata in enumerate(self.dataset["metadata"]):
            coco_id = json.loads(metadata)["coco_id"]
            self.dataset_packed_map[coco_id].append(idx)
        self.dataset_size = len(self.dataset_packed_map)

        logger.info(
            f"Loaded {self.dataset_size} examples from {self._dataset_subset} {self._split} split."
        )


@dataclass
class DataArguments(BaseDataArguments):
    """Data arguments."""

    split: Literal["train", "validation", "test"] = field(default="test")
    eval_dataset_subset: str = field(default="text_caps")
    instruction_first: bool = field(default=False)


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    max_new_tokens: int = 20
    prediction_json: str = "textcaps_predictions.json"
    groundtruth_json: str = "textcaps_groundtruths.json"
    metric_json: str = "textcaps_metrics.json"


@torch.no_grad()
def mamba_generate(
    model: ModelType,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    generation_args: GenerationArguments,
) -> torch.Tensor:
    """Generate with Mamba model."""
    return model.generate(  # type: ignore[report]
        input_ids=input_ids,
        pixel_values=[pixel_values],
        max_length=input_ids.shape[-1] + generation_args.max_new_tokens,
        top_k=generation_args.top_k,
        top_p=generation_args.top_p,
        temperature=generation_args.temperature,
        return_dict_in_generate=generation_args.return_dict_in_generate,
        output_scores=generation_args.output_scores,
    )


@torch.no_grad()
def pythia_generate(  # type: ignore[report]
    model: ModelType,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    generation_args: GenerationArguments,
) -> torch.Tensor:
    """Generate with Pythia model."""
    config = transformers.GenerationConfig(
        max_new_tokens=generation_args.max_new_tokens,
        top_p=generation_args.top_p if generation_args.do_sample else None,
        top_k=generation_args.top_k if generation_args.do_sample else None,
        temperature=generation_args.temperature,
        return_dict_in_generate=generation_args.return_dict_in_generate,
        output_scores=generation_args.output_scores,
        use_cache=False,
        eos_token_id=0,
        pad_token_id=0,
        do_sample=generation_args.do_sample,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=[pixel_values],
            generation_config=config,
            use_cache=False,
        )
    return outputs  # type: ignore[report]


def coco_caption_eval(groundtruth_filepath: str, predictions_filepath: str) -> dict[str, Any]:
    """Evaluate COCO captioning."""
    # create coco object and coco_result object
    coco = COCO(groundtruth_filepath)
    coco_result = coco.loadRes(predictions_filepath)
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        logger.info(f"{metric}: {score:.3f}")

    return coco_eval.eval


def evaluate() -> None:
    """Evaluate."""
    parser = transformers.HfArgumentParser(
        (BaseModelArguments, DataArguments, GenerationArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    model = build_model(model_args)

    tokenizer = build_tokenizer(model_args)

    dataset = TextCapsEvalDataset(
        dataset_path=data_args.dataset_path,
        dataset_cache_dir=data_args.dataset_cache_dir,
        root_dataset_path=data_args.root_dataset_path,
        dataset_subset=data_args.eval_dataset_subset,
        split=data_args.split,
        tokenizer=tokenizer,
        image_mean=data_args.image_mean,
        image_std=data_args.image_std,
        image_size=(
            model_args.extrapolate_image_size
            if model_args.extrapolate_image_size is not None
            else model_args.image_size
        ),
        patch_size=model_args.patch_size,
        variable_sized=data_args.variable_sized,
        box_mode=data_args.box_mode,
        needs_attention_mask="mamba" not in model_args.model_name,
        pixel_based=model_args.pixel_based,
        instruction_first=data_args.instruction_first,
    )

    # Return everything in the format that the COCO evaluation script expects
    predictions = []
    groundtruths = []
    image_ids = []
    model = model.to(device="cuda")
    indices = list(
        range(data_args.start_index, data_args.end_index)
        if data_args.end_index > 0
        else range(len(dataset))
    )
    description = (
        f"Generating predictions for {data_args.eval_dataset_subset} {data_args.split} split."
    )
    pbar = tqdm(indices, desc=description)
    for idx in pbar:
        instance = dataset[idx]
        if isinstance(model, VLMambaLMHeadModel | VLMambaCLIPLMHeadModel):
            outputs = mamba_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),  # type: ignore[report]
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )
        else:
            outputs = pythia_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),  # type: ignore[report]
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )

        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :])  # type: ignore[report]
        image_id = instance.raw_target["metadata"]["image_id"]  # type: ignore[report]

        predictions.append(
            {"image_id": image_id, "caption": output_str.split(tokenizer.eos_token)[0]}  # type: ignore[report]
        )

        image_ids.append({"id": image_id})
        if data_args.split == DatasetSplits.VALIDATION:
            groundtruths.extend(
                [
                    {
                        "image_id": image_id,
                        "caption": caption,
                        "id": instance.raw_target["metadata"]["image_id"],  # type: ignore[report]
                    }
                    for caption in instance.raw_target["target_text"]  # type: ignore[report]
                ]
            )
        logger.info(f"Prediction: {predictions[-1]['caption']}, Gt: {groundtruths[-1]['caption']}")

    with open(generation_args.prediction_json, "w") as pred_fp:
        json.dump(predictions, pred_fp, indent=4)

    if data_args.split != DatasetSplits.TEST:
        with open(generation_args.groundtruth_json, "w") as gt_fp:
            json.dump({"annotations": groundtruths, "images": image_ids}, gt_fp, indent=4)

        metrics = coco_caption_eval(
            groundtruth_filepath=generation_args.groundtruth_json,
            predictions_filepath=generation_args.prediction_json,
        )

        with open(generation_args.metric_json, "w") as metric_fp:
            json.dump(metrics, metric_fp, indent=4)


if __name__ == "__main__":
    evaluate()

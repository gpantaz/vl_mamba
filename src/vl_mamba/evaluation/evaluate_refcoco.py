import json
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
from datasets import load_dataset
from loguru import logger
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
from vl_mamba.models.modeling_vlmambaclip_xattn import VLMambaCLIPLXAttnMHeadModel
from vl_mamba.utils.boxes import Boxes, matched_pairwise_iou


class RefCOCOEvalDataset(BaseEvalDataset):
    """RefCOCO evaluation dataset."""

    split_map = {
        "train": DatasetSplits.TRAIN,
        "test": DatasetSplits.TEST,
        "testA": DatasetSplits.TEST_DEV,
        "testB": DatasetSplits.TEST_STD,
    }

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
        pixel_based: bool = True,
        instruction_first: bool = False,
        has_gated_cross_attention: bool = False,
    ) -> None:
        """Initialize the dataset."""
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
            has_gated_cross_attention=has_gated_cross_attention,
        )
        self._instruction_first = instruction_first
        self._has_gated_cross_attention = has_gated_cross_attention

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item from dataset."""
        raw_instance = self.dataset[idx]
        instance = Instance.model_validate(raw_instance)
        return self.process_instance(instance)

    def process_instance(self, instance: Instance) -> DatasetItem:
        """Process the instance."""
        if not instance.region:
            raise AssertionError(f"Instance {instance} has no region.")

        if not instance.image:
            raise AssertionError(f"Instance {instance} has no image.")

        image = instance.image.convert("RGB")

        bboxes = [region.bbox for region in instance.region]

        visual_encoding = self.image_processor.preprocess(images=image, bboxes=bboxes)

        region_text = format_text(
            instance.region[0].phrase,
            strip=True,
            punctuate=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )
        prompt = self._get_random_template_for_task(Task.visual_grounding)

        full_conversation = [
            f"{prompt}\n",
            f"{region_text}\n",
        ]

        full_conversation = [f"{prompt}"]
        for region, _bbox in zip(instance.region, visual_encoding.bboxes_norm[0], strict=False):  # type: ignore[report]
            region_text = format_text(
                region.phrase,
                strip=True,
                punctuate=True,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )

            full_conversation.append(f"\n{region_text}\n")

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [self.tokenizer(turn, return_tensors="pt") for turn in full_conversation]

        if self._instruction_first:
            encoding = [
                tokens[0].input_ids[0],
                visual_encoding.input_ids[0],  # type: ignore[report]
            ]

            for txt_encoding in tokens[1:]:
                encoding.append(txt_encoding.input_ids.squeeze())  # noqa: PERF401

            input_ids = torch.concatenate(encoding)

        else:
            tokens = tokens.input_ids.squeeze(0)  # type: ignore[report]
            input_ids = torch.concatenate(tokens, dim=-1)
            if not self._has_gated_cross_attention:
                input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        raw_target = {
            "region": {
                "bbox": [region.bbox for region in instance.region],
                "bbox_norm": [visual_encoding.bboxes_norm[0]],  # type: ignore[report]
                "phrase": [region.phrase for region in instance.region],
            },
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
        )[self.split_map[self._split]]


@dataclass
class DataArguments(BaseDataArguments):
    """Data arguments."""

    eval_dataset_subset: str = field(default="refcocog")
    split: Literal["test", "testA", "testB"] = field(default="test")
    instruction_first: bool = field(default=False)


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    output_json: str = "refcoco.json"


@torch.no_grad()
def mamba_generate(
    model: ModelType,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    generation_args: GenerationArguments,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate with Mamba model."""
    if isinstance(model, VLMambaCLIPLXAttnMHeadModel):
        return model.generate(  # type: ignore[return]
            input_ids=input_ids,
            pixel_values=[pixel_values],
            attention_mask=attention_mask,
            max_length=input_ids.shape[-1] + generation_args.max_new_tokens,
            top_k=generation_args.top_k,
            top_p=generation_args.top_p,
            temperature=generation_args.temperature,
            return_dict_in_generate=generation_args.return_dict_in_generate,
            output_scores=generation_args.output_scores,
        )

    return model.generate(  # type: ignore[return]
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
def pythia_generate(
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
        output_attentions=generation_args.output_attentions,
        use_cache=False,
        eos_token_id=0,
        pad_token_id=0,
        do_sample=False,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=[pixel_values],
            generation_config=config,
            use_cache=False,
        )
    return outputs  # type: ignore[return]


def compute_score(
    instance: DatasetItem, output_str: str, iou_threshold: float = 0.5
) -> tuple[int, list[float], list[float]]:
    """Compute RefCOCO score."""
    target_bbox = instance.raw_target["region"]["bbox"][0]  # type: ignore[report]
    # [0.54, 0.54, 0.54, 0.54]
    try:
        parts = output_str.split("[")[1].split("]")[0].split(",")
        pred_bbox_norm = [float(c) for c in parts]
    except Exception:  # noqa: BLE001
        logger.error(f"Error parsing output string: {output_str}")
        return 0, [], target_bbox

    if len(pred_bbox_norm) != 4:  # noqa: PLR2004
        logger.error(f"Error parsing output string: {output_str}")
        return 0, [], target_bbox

    image_width = instance.raw_target["metadata"]["width"]  # type: ignore[report]
    image_height = instance.raw_target["metadata"]["height"]  # type: ignore[report]
    pred_bbox = [
        pred_bbox_norm[0] * image_width,
        pred_bbox_norm[1] * image_height,
        pred_bbox_norm[2] * image_width,
        pred_bbox_norm[3] * image_height,
    ]

    iou = matched_pairwise_iou(
        Boxes(torch.tensor(pred_bbox).unsqueeze(0)),
        Boxes(torch.tensor(target_bbox).unsqueeze(0)),
    )[0]
    return (int(iou > iou_threshold), pred_bbox, target_bbox)


def evaluate() -> None:
    """Evaluate."""
    parser = transformers.HfArgumentParser(
        (BaseModelArguments, DataArguments, GenerationArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    # trainer_state = json.load(open(f"{model_args.model_name}/checkpoint-final/trainer_state.json"))

    # best_checkpoint = trainer_state["best_model_checkpoint"]
    # logger.info(f"Best checkpoint: {best_checkpoint}")
    # model_args.model_name = best_checkpoint

    model = build_model(model_args)

    tokenizer = build_tokenizer(model_args)

    dataset = RefCOCOEvalDataset(
        split=data_args.split,
        dataset_path=data_args.dataset_path,
        dataset_cache_dir=data_args.dataset_cache_dir,
        root_dataset_path=data_args.root_dataset_path,
        dataset_subset=data_args.eval_dataset_subset,
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
        needs_attention_mask=not isinstance(model, VLMambaCLIPLMHeadModel),
        has_gated_cross_attention=isinstance(model, VLMambaCLIPLXAttnMHeadModel),
        pixel_based=model_args.pixel_based,
        instruction_first=data_args.instruction_first,
    )

    all_outputs = []
    model = model.to(device="cuda")
    indices = list(
        range(
            data_args.start_index, data_args.end_index if data_args.end_index > 0 else len(dataset)
        )
    )
    correct = 0
    description = (
        f"Generating predictions for {data_args.eval_dataset_subset} {data_args.split} split."
    )
    pbar = tqdm(indices, desc=description)
    for idx in pbar:
        instance = dataset[idx]

        if isinstance(
            model, VLMambaLMHeadModel | VLMambaCLIPLMHeadModel | VLMambaCLIPLXAttnMHeadModel
        ):
            outputs = mamba_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
                attention_mask=instance.attention_mask.to(model.device)
                if instance.attention_mask is not None
                else None,
            )
        else:
            outputs = pythia_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )

        # out_tokens = []
        # sequence = [instance.input_ids.to(model.device)]
        # while True:
        #     out_step = model(
        #         input_ids=torch.cat(sequence).unsqueeze(0),
        #         pixel_values=[instance.pixel_values.to(model.device)],
        #         # attention_mask=instance.attention_mask.to(model.device)
        #         # if instance.attention_mask is not None
        #         # else None,
        #     )
        #     logits = out_step.logits
        #     next_token = torch.argmax(logits[:, -1, :])
        #     out_tokens.append(next_token)
        #     sequence.append(next_token.unsqueeze(0))
        #     if next_token.item() == tokenizer.eos_token_id:
        #         break

        # output_str = tokenizer.decode(outputs[outputs > 0])
        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :]).replace(  # type: ignore[report]
            "<|endoftext|>", ""
        )
        # out_str = tokenizer.decode(out_tokens).replace("<|endoftext|>", "")
        # logger.info(f"Output: {output_str}, Out: {out_str} {out_str == output_str}")
        # breakpoint()
        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :])  # type: ignore[report]

        out = compute_score(
            instance=instance,
            output_str=output_str.split(tokenizer.eos_token)[0][:-1],  # type: ignore[report]
        )

        correct += out[0]
        all_outputs.append(
            {
                "prediction": out[1],
                "target": out[2],
                "phrase": instance.raw_target["region"]["phrase"][0],  # type: ignore[report]
                "image_path": str(instance.raw_target["metadata"]["image_path"]),  # type: ignore[report]
            }
        )

        logger.info(output_str)
        pbar.set_postfix({"RefCOCO Accuracy": f"{(correct / (idx + 1)):.3f}"})

    logger.info(f"RefCOCO Accuracy: {(correct / len(dataset)):.3f}")
    with open(generation_args.output_json, "w") as pred_fp:
        json.dump({"accuracy": correct / len(dataset), "outputs": all_outputs}, pred_fp, indent=4)


if __name__ == "__main__":
    evaluate()

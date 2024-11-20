import json
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from loguru import logger
from tqdm import tqdm

from vl_mamba.datamodels.dataclasses import DatasetItem
from vl_mamba.datamodels.datamodels import DatasetSplits, Instance, Task
from vl_mamba.datasets.base_dataset import format_text
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


class PopeEvalDataset(BaseEvalDataset):
    """POPE evaluation dataset."""

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
        )

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item from dataset."""
        raw_instance = self.dataset[idx]
        instance = Instance.model_validate(raw_instance)
        return self.process_instance(instance)

    def process_instance(self, instance: Instance) -> DatasetItem:
        """Process the POPE instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"Instance {instance} has no qa_pair.")

        image = instance.image.convert("RGB")  # type: ignore[report]
        visual_encoding = self.image_processor.preprocess(images=image)

        prompt = self._get_random_template_for_task(Task.vqa)

        question = format_text(
            instance.qa_pairs[0].question,
            strip=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )

        # There are no other turns in captioning, only the prompt and the caption.
        full_conversation = [f"{prompt}\n{question}\n"]

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
            for turn in full_conversation
        ]

        input_ids = torch.concatenate(tokens, dim=-1)

        input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        metadata = json.loads(instance.metadata)
        raw_target = {
            "target_text": metadata["answer"],
            "metadata": metadata,
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


@dataclass
class DataArguments(BaseDataArguments):
    """Data arguments."""

    eval_dataset_subset: Literal["pope_adversarial", "pope_popular", "pope_random"] = field(
        default="pope_adversarial"
    )
    split: str = field(default="test")


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    max_length: int = 10
    output_json: str = "pope_adversarial.json"


def compute_metrics(pred_list: list[str], label_list: list[str]) -> dict[str, Any]:
    """Compute metrics."""
    labels_numeric = [0 if label == "no" else 1 for label in label_list]
    preds_numeric = [0 if pred == "no" else 1 for pred in pred_list]

    pos = 1
    neg = 0
    yes_ratio = preds_numeric.count(1) / len(pred_list)

    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, label in zip(preds_numeric, labels_numeric, strict=False):
        if pred == pos and label == pos:
            tp += 1
        elif pred == pos and label == neg:
            fp += 1
        elif pred == neg and label == neg:
            tn += 1
        elif pred == neg and label == pos:
            fn += 1

    logger.info("tp\tfp\ttn\tfn\t")
    logger.info(f"{tp}\t{fp}\t{tn}\t{fn}")

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 score: {f1}")
    logger.info(f"Yes ratio: {yes_ratio}")
    logger.info(f"{f1:.3f}, {acc:3f} {precision:3f} {recall:3f}, {yes_ratio:3f}")
    return {"f1": f1, "acc": acc, "precision": precision, "recall": recall, "yes_ratio": yes_ratio}


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


def evaluate() -> None:
    """Evaluate."""
    parser = transformers.HfArgumentParser(
        (BaseModelArguments, DataArguments, GenerationArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    model = build_model(model_args)

    tokenizer = build_tokenizer(model_args)

    dataset = PopeEvalDataset(
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
        pixel_based=model_args.pixel_based,
    )

    predictions = []
    groundtruths = []
    model = model.to(device="cuda")
    indices = list(
        range(data_args.start_index, data_args.end_index)
        if data_args.end_index > 0
        else range(len(dataset))
    )
    correct = 0
    description = (
        f"Generating predictions for {data_args.eval_dataset_subset} {DatasetSplits.TEST} split."
    )
    pbar = tqdm(indices, desc=description)
    for idx in pbar:
        instance = dataset[idx]
        if isinstance(model, VLMambaLMHeadModel | VLMambaCLIPLMHeadModel):
            outputs = mamba_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )
        else:
            outputs = pythia_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )

        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :])  # type: ignore[report]
        output_str = output_str.split(tokenizer.eos_token)[0]  # type: ignore[report]
        predictions.append(output_str.lower())
        groundtruths.append(instance.raw_target["target_text"])  # type: ignore[report]
        correct += int(output_str.lower() == instance.raw_target["target_text"])  # type: ignore[report]
        pbar.set_postfix({"Accuracy": f"{(correct / (idx + 1)):.3f}"})

    metrics = compute_metrics(predictions, groundtruths)
    with open(generation_args.output_json, "w") as metric_fp:
        json.dump(
            {"metrics": metrics, "predictions": predictions, "grountruths": groundtruths},
            metric_fp,
            indent=4,
        )


if __name__ == "__main__":
    evaluate()

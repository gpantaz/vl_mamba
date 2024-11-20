import json
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from vl_mamba.datamodels.dataclasses import DatasetItem
from vl_mamba.datamodels.datamodels import DatasetSplits, Instance
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


class GQAEvalDataset(BaseEvalDataset):
    """GQA evaluation dataset."""

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
        image_size: int | tuple[int, int] = 336,
        patch_size: int | dict[str, int] = 14,
        variable_sized: bool = False,
        box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = "normalize",
        needs_attention_mask: bool = False,
        pixel_based: bool = False,
        has_gated_cross_attention: bool = False,
        instruction_first: bool = False,
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

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.dataset_size

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get item from dataset."""
        if self.dataset_indices_unpacked is not None:
            raw_instance = self.dataset[self.dataset_indices_unpacked[idx][0]]
            question_idx = self.dataset_indices_unpacked[idx][1]
        else:
            raw_instance = self.dataset[idx]
            question_idx = None

        instance = Instance.model_validate(raw_instance)
        return self.process_instance(instance, question_idx)

    def process_instance(self, instance: Instance, question_idx: int | None = None) -> DatasetItem:
        """Process the instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"Instance {instance} has no qa_pair.")

        image = instance.image.convert("RGB")  # type: ignore[report]
        visual_encoding = self.image_processor.preprocess(images=image)

        prompt = self._get_random_template_for_task(instance.task)

        metadata = json.loads(instance.metadata)

        qa_pair = instance.qa_pairs[0] if question_idx is None else instance.qa_pairs[question_idx]
        question = format_text(
            qa_pair.question,
            strip=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )

        # Tokenize the conversation and prepare the input and target tensors.
        input_ids = self.tokenizer(
            f"{prompt}\nQuestion: {question}\nAnswer: ", return_tensors="pt"
        ).input_ids.squeeze(0)

        # Add the visual tokens to the input_ids only for models without gated xattn
        if not self._has_gated_cross_attention:
            if self._instruction_first:
                full_conversation = [f"{prompt}", f"\nQuestion: {question}\nAnswer: "]
                tokens = [
                    self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
                    for turn in full_conversation
                ]
                input_ids = torch.concatenate([tokens[0], visual_encoding.input_ids[0], tokens[1]])  # type: ignore[report]
            else:
                input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        question_id = (
            metadata["question_ids"][0]
            if question_idx is None
            else metadata["question_ids"][question_idx]
        )

        raw_target = {
            "answer": qa_pair.answer,
            "metadata": metadata,
            "question": question,
            "question_id": question_id,
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
        """Prepare the dataset."""
        self.dataset = load_dataset(  # type: ignore[report]
            self._dataset_path,
            self._dataset_subset,
            cache_dir=self._dataset_cache_dir,
            root_dataset_path=self._root_dataset_path,
            dataset_paths=DatasetPaths(self._root_dataset_path),
            verification_mode="no_checks",
            trust_remote_code=True,
        )[self._split]

        (dataset_indices_unpacked, dataset_size) = self._compute_dataset_size()
        self.dataset_indices_unpacked = dataset_indices_unpacked
        self.dataset_size = dataset_size

    def _compute_dataset_size(self) -> tuple[dict[int, tuple[int, int]] | None, int]:
        """Compute the size of the dataset."""
        packed_instances = self._split in {DatasetSplits.VALIDATION, DatasetSplits.TRAIN}

        dataset_size = 0
        dataset_indices_unpacked = {} if packed_instances else None

        # We need to unpack the examples for validation
        if packed_instances:
            for idx, example in enumerate(tqdm(self.dataset, total=len(self.dataset))):
                for question_idx, _ in enumerate(example["qa_pairs"]):
                    pos = dataset_size + question_idx
                    dataset_indices_unpacked[pos] = (  # type: ignore[report]
                        idx,
                        question_idx,
                    )
                dataset_size += len(example["qa_pairs"])
        # For test splits, we don't need to unpack the examples
        else:
            dataset_size = len(self.dataset)

        return dataset_indices_unpacked, dataset_size


@dataclass
class DataArguments(BaseDataArguments):
    """Data arguments."""

    eval_dataset_subset: str = field(default="gqa")
    split: Literal["train", "validation", "test", "testdev"] = field(default="test")
    instruction_first: bool = field(default=False)


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    max_new_tokens: int = 10
    output_json: str = "gqa.json"


def gqa_score(prediction: str, answer: str) -> float:
    """Compute GQA score."""
    return prediction.lower() == answer.lower()


def compute_metrics(pred_list: list[str], label_list: list[str]) -> dict[str, Any]:
    """Compute metrics."""
    score = sum(
        gqa_score(prediction, answer)
        for prediction, answer in zip(pred_list, label_list, strict=False)
    )

    # log the score for the VQA-v2 metric in percentage format
    score = round(score / len(pred_list) * 100, 2)
    logger.info(f"GQA accuracy: {score}")
    return {"accuracy": score}


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


def evaluate() -> None:
    """Evaluate."""
    parser = transformers.HfArgumentParser(
        (BaseModelArguments, DataArguments, GenerationArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    model = build_model(model_args)

    tokenizer = build_tokenizer(model_args)

    dataset = GQAEvalDataset(
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

    predictions = []
    groundtruths = []
    question_ids = []
    model = model.to(device="cuda")
    indices = list(
        range(data_args.start_index, data_args.end_index)
        if data_args.end_index > 0
        else range(len(dataset))
    )

    score = 0
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
                attention_mask=None,  # instance.attention_mask.to(model.device),
            )
        else:
            outputs = pythia_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),  # type: ignore[report]
                generation_args=generation_args,
            )

        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :])  # type: ignore[report]
        output_str = output_str.split(f".{tokenizer.eos_token}")[0].lower()  # type: ignore[report]

        raw_target = instance.raw_target
        predictions.append(output_str)
        question_ids.append(raw_target["question_id"])  # type: ignore[report]
        if raw_target.get("answer", None):  # type: ignore[report]
            groundtruths.append(raw_target["answer"])  # type: ignore[report]
            score += gqa_score(output_str, raw_target["answer"])  # type: ignore[report]

            logger.info(f"Prediction: {output_str}, Groundtruth: {raw_target['answer']}")  # type: ignore[report]
            pbar.set_postfix({"Accuracy": f"{(score / (idx + 1)):.3f}"})

    if data_args.split in {DatasetSplits.VALIDATION, DatasetSplits.TEST_DEV}:
        metrics = compute_metrics(predictions, groundtruths) if groundtruths else None
        output_result = {
            "metrics": metrics,
            "predictions": predictions,
            "groundtruths": groundtruths,
            "question_ids": question_ids,
        }
    else:
        output_result = [
            {"questionId": question_id, "prediction": prediction}
            for question_id, prediction in zip(question_ids, predictions, strict=False)
        ]

    with open(generation_args.output_json, "w") as fp:
        json.dump(output_result, fp, indent=4)


if __name__ == "__main__":
    evaluate()

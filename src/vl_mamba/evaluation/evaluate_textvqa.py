import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
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
from vl_mamba.utils.evalai_answer_processor import EvalAIAnswerProcessor


class TextVQAEvalDataset(BaseEvalDataset):
    """TextVQA evaluation dataset."""

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
        """Process instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"Instance {instance} has no qa_pair.")

        image = instance.image.convert("RGB")  # type: ignore[optional]
        visual_encoding = self.image_processor.preprocess(images=image)
        metadata = json.loads(instance.metadata)

        prompt = self._get_random_template_for_task(Task.vqa)

        qa_pair = instance.qa_pairs[0] if question_idx is None else instance.qa_pairs[question_idx]
        question = format_text(
            qa_pair.question,
            strip=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )

        ocr_metadata = metadata["ocr"]
        if len(ocr_metadata) != 1 and question_idx is None:
            raise AssertionError(f"Instance {instance} has more than one ocr metadata.")

        ocr_metadata_dict = ocr_metadata[0] if question_idx is None else ocr_metadata[question_idx]
        ocr_tokens_str = " ".join(ocr_metadata_dict["ocr_tokens"])
        ocr_text = f"OCR text:\n{ocr_tokens_str}"
        full_conversation = [f"{prompt}\n{ocr_text}\nQuestion: {question}\nAnswer: "]

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
            for turn in full_conversation
        ]

        input_ids = torch.concatenate(tokens, dim=-1)

        input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[optional]

        question_id = (
            metadata["question_ids"][0]
            if question_idx is None
            else metadata["question_ids"][question_idx]
        )

        answers = metadata.get("answers", None)
        # Get only the set of answers for the current question
        if answers:
            answers = answers[question_idx]

        raw_target = {
            "target_text": qa_pair.answer,
            "answers": answers,
            "metadata": metadata,
            "question": question,
            "question_id": question_id,
        }

        pixel_values = (
            visual_encoding.image_patches[0].unsqueeze(0)  # type: ignore[optional]
            if self._pixel_based
            else visual_encoding.images[0].unsqueeze(0)  # type: ignore[optional]
        )

        return DatasetItem(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids) if self._needs_attention_mask else None,
            raw_target=raw_target,
        )

    def prepare_dataset(self) -> None:
        """Prepare dataset."""
        if self._split == DatasetSplits.TEST:
            logger.info("Loading teststd and testdev splits.")
            dataset = load_dataset(  # type: ignore[report]
                self._dataset_path,
                self._dataset_subset,
                cache_dir=self._dataset_cache_dir,
                root_dataset_path=self._root_dataset_path,
                dataset_paths=DatasetPaths(self._root_dataset_path),
                verification_mode="no_checks",
                trust_remote_code=True,
            )
            self.dataset = concatenate_datasets([dataset["teststd"], dataset["testdev"]])  # type: ignore[report]

        else:
            self.dataset = load_dataset(  # type: ignore[report]
                self._dataset_path,
                self._dataset_subset,
                cache_dir=self._dataset_cache_dir,
                root_dataset_path=self._root_dataset_path,
                dataset_paths=DatasetPaths(self._root_dataset_path),
                verification_mode="no_checks",
                trust_remote_code=True,
            )[self._split]

        (dataset_indices_unpacked, dataset_size) = self._compute_dataset_size(self._split)
        self.dataset_indices_unpacked = dataset_indices_unpacked
        self.dataset_size = dataset_size
        logger.info(
            f"Loaded {self.dataset_size} examples from {self._dataset_subset} {self._split} split."
        )

    def _compute_dataset_size(self, split: str) -> tuple[dict[int, tuple[int, int]] | None, int]:
        """Compute the size of the dataset."""
        dataset_size = 0
        dataset_indices_unpacked = {} if split == DatasetSplits.VALIDATION else None
        # We need to unpack the examples for validation
        if split == DatasetSplits.VALIDATION:
            for idx, example in enumerate(tqdm(self.dataset, total=len(self.dataset))):
                for question_idx, _ in enumerate(example["qa_pairs"]):
                    pos = dataset_size + question_idx
                    dataset_indices_unpacked[pos] = (  # type: ignore[optional]
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

    eval_dataset_subset: str = field(default="text_vqa")
    split: Literal["validation", "test"] = field(default="validation")


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    max_new_tokens: int = 20
    output_json: str = "textvqa.json"


def vqa_v2_score(count: int) -> float:
    """VQA-v2 includes 10 answers for each question.

    Scores are assigned as follows:
    - 0.3 if the answer appears once
    - 0.6 if the answer appears twice
    - 0.9 if the answer appears three times
    - 1.0 if the answer appears more than three times
    """
    return min(1.0, round(0.3 * count, 1))


def compute_metrics(
    pred_list: list[str],
    label_list: list[list[str]],
) -> dict[str, Any]:
    """Compute metrics."""
    score = 0
    for prediction, answers in zip(pred_list, label_list, strict=False):
        ground_truth_counts = Counter(answers)
        score += vqa_v2_score(ground_truth_counts.get(prediction, 0))

    # log the score for the VQA-v2 metric in percentage format
    score = round(score / len(pred_list) * 100, 5)
    logger.info(f"VQAv2 accuracy: {score}")
    return {"accuracy": score}


@torch.no_grad()
def mamba_generate(
    model: ModelType,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    generation_args: GenerationArguments,
) -> torch.Tensor:
    """Generate with Mamba model."""
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
        use_cache=False,
        eos_token_id=0,
        pad_token_id=0,
        do_sample=generation_args.do_sample,
        # num_beams=5,
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

    dataset = TextVQAEvalDataset(
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
        pixel_based=model_args.pixel_based,
    )

    answer_processor = EvalAIAnswerProcessor()

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
        if instance.pixel_values is None:
            raise AssertionError(f"Instance {instance} has no pixel_values.")
        if instance.raw_target is None:
            raise AssertionError(f"Instance {instance} has no raw_target.")

        if isinstance(model, VLMambaLMHeadModel | VLMambaCLIPLMHeadModel):
            outputs = mamba_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),
                generation_args=generation_args,
            )
        else:
            outputs = pythia_generate(
                model,
                input_ids=instance.input_ids.to(model.device).unsqueeze(0),
                pixel_values=instance.pixel_values.to(model.device),
                generation_args=generation_args,
            )

        output_str = tokenizer.decode(outputs[0, instance.input_ids.shape[0] :])  # type: ignore[attribute]
        output_str = output_str.split(f".{tokenizer.eos_token}")[0].lower()  # type: ignore[attribute]
        output_str = answer_processor(output_str)
        predictions.append(output_str.lower())
        question_ids.append(instance.raw_target["question_id"])
        if instance.raw_target.get("answers", None):
            groundtruths.append(instance.raw_target["answers"])
            score += vqa_v2_score(Counter(instance.raw_target["answers"]).get(output_str, 0))
            pbar.set_postfix({"Accuracy": f"{(score / (idx + 1)):.5f}"})

    if data_args.split == DatasetSplits.VALIDATION:
        metrics = compute_metrics(predictions, groundtruths) if groundtruths else None
        output_result = {
            "metrics": metrics,
            "predictions": predictions,
            "groundtruths": groundtruths,
        }
    else:
        output_result = [
            {"question_id": question_id, "answer": prediction}
            for question_id, prediction in zip(question_ids, predictions, strict=False)
        ]

    with open(generation_args.output_json, "w") as fp:
        json.dump(output_result, fp, indent=4)


if __name__ == "__main__":
    evaluate()

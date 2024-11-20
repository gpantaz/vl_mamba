import json
import string
from dataclasses import dataclass, field
from typing import Any, Literal

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


class Visual7WEvalDataset(BaseEvalDataset):
    """Visual7W evaluation dataset."""

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
        )

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
        """Process the Visual7W instance."""
        if instance.task == Task.gm_vqa:
            return self.preprocess_grounding_multiple_choice_vqa(instance, question_idx)
        return self.process_multiple_choice_vqa(instance, question_idx)

    def preprocess_grounding_multiple_choice_vqa(
        self, instance: Instance, question_idx: int | None = None
    ) -> DatasetItem:
        """Process the Visual7W instance for grounding multiple choice VQA."""
        if not instance.qa_pairs:
            raise AssertionError(f"Instance {instance} has no qa_pair.")

        if not instance.region:
            raise AssertionError(f"{instance} has no region.")

        image = instance.image.convert("RGB")  # type: ignore[report]
        visual_encoding = self.image_processor.preprocess(
            images=image, bboxes=[region.bbox for region in instance.region]
        )

        prompt = self._get_random_template_for_task(instance.task)

        qa_pair = instance.qa_pairs[0] if question_idx is None else instance.qa_pairs[question_idx]
        metadata = json.loads(instance.metadata)
        answers = (
            metadata["answers"][0] if question_idx is None else metadata["answers"][question_idx]
        )

        question = format_text(
            qa_pair.question,
            strip=True,
            capitalize=False,
            add_bos=False,
            add_eos=False,
        )

        full_conversation = [f"{prompt}"]

        bbox_ids2position = {region.phrase: idx for idx, region in enumerate(instance.region)}

        candidate_answers = [
            self.conversation_processor.format_bbox(
                bbox=visual_encoding.bboxes_norm[0][bbox_ids2position[region_id]]  # type: ignore[report]
            )
            for region_id in answers
        ]

        candidate_answers = [
            f"{choice}: {answer}\n"
            for choice, answer in zip(string.ascii_uppercase, candidate_answers, strict=False)
        ]

        candidate_answers[-1] = f"{candidate_answers[-1]}Answer: "

        full_conversation.extend(
            [
                f"\nQuestion: {question}\n",
                *candidate_answers,
            ]
        )

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
            for turn in full_conversation
        ]

        if self._instruction_first:
            input_ids = torch.concatenate([tokens[0], visual_encoding.input_ids[0], *tokens[1:]])  # type: ignore[report]
        else:
            input_ids = torch.concatenate(tokens, dim=-1)
            input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])

        raw_target = {
            "target_text": qa_pair.answer,
            "answers": answers,
            "answer": qa_pair.answer,
            "metadata": metadata,
            "question": question,
            "source": "telling" if "telling" in instance.source else "pointing",
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

    def process_multiple_choice_vqa(
        self, instance: Instance, question_idx: int | None = None
    ) -> DatasetItem:
        """Process the Visual7W instance for multiple choice VQA."""
        if not instance.qa_pairs:
            raise AssertionError(f"Instance {instance} has no qa_pair.")

        image = instance.image.convert("RGB")  # type: ignore[report]
        visual_encoding = self.image_processor.preprocess(images=image)

        prompt = self._get_random_template_for_task(instance.task)

        qa_pair = instance.qa_pairs[0] if question_idx is None else instance.qa_pairs[question_idx]
        question = format_text(
            qa_pair.question,
            strip=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )

        metadata = json.loads(instance.metadata)

        full_conversation = [f"{prompt}"]

        answers = (
            metadata["answers"][0] if question_idx is None else metadata["answers"][question_idx]
        )

        candidate_answers = [
            format_text(
                answer,
                strip=True,
                punctuate=False,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )
            for answer in answers
        ]

        candidate_answers = [
            f"{choice}: {answer}\n"
            for choice, answer in zip(string.ascii_uppercase, candidate_answers, strict=False)
        ]

        candidate_answers[-1] = f"{candidate_answers[-1]}Answer: "

        full_conversation.extend(
            [
                f"\nQuestion: {question}\n",
                *candidate_answers,
            ]
        )

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            self.tokenizer(turn, return_tensors="pt").input_ids.squeeze(0)
            for turn in full_conversation
        ]

        input_ids = torch.concatenate(tokens, dim=-1)

        input_ids = torch.concatenate([visual_encoding.input_ids[0], input_ids])  # type: ignore[report]

        raw_target = {
            "target_text": qa_pair.answer,
            "answers": answers,
            "answer": (
                metadata["answer"][0] if question_idx is None else metadata["answer"][question_idx]
            ),
            "metadata": metadata,
            "question": question,
            "source": "telling" if "telling" in instance.source else "pointing",
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

    eval_dataset_subset: str = field(default="visual7w")
    split: Literal["train", "validation", "test"] = field(default="test")
    instruction_first: bool = field(default=False)


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Generation arguments."""

    max_new_tokens: int = 10
    output_json: str = "visual7w.json"


def visual7w_score(prediction: str, answer: str) -> int:
    """Compute Visual7W score."""
    index_map = {idx: chara for idx, chara in enumerate(string.ascii_uppercase)}  # noqa: C416

    answer_character = index_map[answer]
    return int(prediction == answer_character)


def compute_metrics(
    pred_list: list[str], label_list: list[str], sources: list[str]
) -> dict[str, Any]:
    """Compute metrics."""
    scores = {source: 0 for source in set(sources)}
    for source in set(sources):
        source_pred_list = [
            pred for pred, src in zip(pred_list, sources, strict=False) if src == source
        ]
        source_label_list = [
            label for label, src in zip(label_list, sources, strict=False) if src == source
        ]
        score = sum(
            visual7w_score(prediction, answer)
            for prediction, answer in zip(source_pred_list, source_label_list, strict=False)
        )

        # log the score for the Visual7W metric in percentage format
        score = round(score / len(source_pred_list) * 100, 2)
        logger.info(f"{source.capitalize()} Accuracy: {score}")
        scores[source] = score

    return scores


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
        do_sample=False,
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

    dataset = Visual7WEvalDataset(
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
        needs_attention_mask="mamba" not in model_args.model_name,
        pixel_based=model_args.pixel_based,
        instruction_first=data_args.instruction_first,
    )

    predictions = []
    groundtruths = []
    sources = []
    model = model.to(device="cuda")
    indices = list(
        range(
            data_args.start_index, data_args.end_index if data_args.end_index > 0 else len(dataset)
        )
    )
    scores = {"pointing": 0, "telling": 0}
    description = (
        f"Generating predictions for {data_args.eval_dataset_subset} {data_args.split} split."
    )
    pbar = tqdm(indices, desc=description)
    for idx in pbar:
        instance = dataset[idx]
        if instance.raw_target["source"] == "pointing":  # type: ignore[report]
            continue

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
        output_str = output_str.replace(tokenizer.eos_token, "")  # type: ignore[report]
        output_str = output_str.split(".")[0]

        answer = instance.raw_target["answers"].index(instance.raw_target["answer"])  # type: ignore[report]
        predictions.append(output_str)
        groundtruths.append(answer)
        sources.append(instance.raw_target["source"])  # type: ignore[report]
        score = visual7w_score(output_str, answer)
        scores[instance.raw_target["source"]] += score  # type: ignore[report]

        prefix = {
            f"{source.capitalize()} Accuracy": f"{(source_score / (sources.count(source) + 1)):.3f}"
            for source, source_score in scores.items()
        }
        pbar.set_postfix(prefix)

    metrics = compute_metrics(predictions, groundtruths, sources)
    output_result = {
        "metrics": metrics,
        "predictions": predictions,
        "groundtruths": groundtruths,
        "sources": sources,
    }

    with open(generation_args.output_json, "w") as fp:
        json.dump(output_result, fp, indent=4)


if __name__ == "__main__":
    evaluate()

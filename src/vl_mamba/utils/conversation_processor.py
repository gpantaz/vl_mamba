import json
import random
import string
from typing import Literal

import numpy as np
import torch
from transformers import AutoTokenizer

from vl_mamba.datamodels.dataclasses import SpecialTokens, TextEncoding, VisualEncoding
from vl_mamba.datamodels.datamodels import (
    TASK_TEMPLATES_MAP,
    DatasetNames,
    DatasetSplits,
    Instance,
    Task,
)
from vl_mamba.datasets.base_dataset import format_text


class VLMambaConversationProcessor:
    """VLMambaConversation preprocessor."""

    def __init__(
        self, box_mode: Literal["normalize", "quantize", "quantize_with_tokens"] = "normalize"
    ) -> None:
        self.switcher = {
            Task.captioning: self.preprocess_captioning,
            Task.grounded_captioning: self.preprocess_grounded_captioning,
            Task.dense_captioning: self.preprocess_dense_captioning,
            Task.chat: self.preprocess_chat,
            Task.itm: self.process_itm,
            Task.m_vqa: self.preprocess_multiple_choice_vqa,
            Task.gm_vqa: self.preprocess_grounding_multiple_choice_vqa,
            Task.visual_grounding: self.preprocess_visual_grounding,
            Task.vqa: self.preprocess_vqa,
        }
        self.box_mode = box_mode

    def __call__(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process the conversation."""
        return self.switcher[instance.task](instance, tokenizer, split, visual_encoding)

    def format_bbox(
        self, bbox: list[float] | torch.Tensor | np.ndarray, punctuate: bool = True
    ) -> str:
        """Format the bounding box."""
        if self.box_mode == "normalize":
            bbox_str = ",".join([f"{coord:.2f}" for coord in bbox])
            return f"[{bbox_str}]." if punctuate else f"[{bbox_str}]"
        raise NotImplementedError("Only normalize mode is supported for now.")

    def process_itm(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process ITM instance."""
        if not instance.caption:
            raise AssertionError(f"{instance} has no caption.")

        metadata = json.loads(instance.metadata)
        if "label" not in metadata:
            raise AssertionError(f"{instance} has no label in metadata.")

        prompt = self._get_random_template_for_task(Task.itm)
        caption = format_text(
            instance.caption,
            strip=True,
            punctuate=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )
        full_conversation = [
            f"{prompt}\n",
            f"{caption}\n",
        ]

        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            label = "True" if metadata["label"] else "False"
            full_conversation.append(f"{label}.{SpecialTokens.text_eos_token}")

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            masked_positions = len(tokens[0]) + len(tokens[1])
            # labels = torch.concatenate(
            #     [
            #         torch.ones(masked_positions) * -100,
            #         tokens[2],
            #     ],
            #     dim=-1,
            # )
            labels = [
                torch.ones(masked_positions, dtype=tokens[0].dtype) * -100,
                tokens[2],
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_dense_captioning(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process visual grounding instance."""
        if not instance.region:
            raise AssertionError(f"{instance} has no region.")

        if not visual_encoding:
            raise AssertionError("For visual grounding we need to preprocess the image first.")

        prompt = self._get_random_template_for_task(Task.dense_captioning)
        full_conversation = [f"{prompt}\n"]
        for region, bbox in zip(instance.region, visual_encoding.bboxes_norm[0], strict=False):
            bbox_text = self.format_bbox(bbox=bbox)

            region_text = format_text(
                region.phrase,
                strip=True,
                punctuate=True,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )
            if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
                full_conversation.extend([f"{bbox_text}\n", f"{region_text}\n"])
            else:
                full_conversation.append(f"{bbox_text}\n")

        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            full_conversation[-1] = f"{full_conversation[-1]}{SpecialTokens.text_eos_token}"

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            target_indices = [c + 2 for c, _ in enumerate(full_conversation) if c % 2 == 0]
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )

            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_grounded_captioning(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process captioning instance."""
        if not instance.caption:
            raise AssertionError(f"{instance} has no caption.")

        if not instance.region:
            raise AssertionError(f"{instance} has no region.")

        if not visual_encoding:
            raise AssertionError("For grounded captioning we need to preprocess the image first.")

        prompt = self._get_random_template_for_task(Task.grounded_captioning)

        caption = format_text(
            instance.caption,
            strip=True,
            punctuate=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )

        # This is the point where we need to replace the placeholder region
        # tokens in the caption with the bounding box coordinates.
        for idx, bbox in enumerate(visual_encoding.bboxes_norm[0]):
            bbox_text = self.format_bbox(bbox=bbox, punctuate=False)
            caption = caption.replace(f"<coords_{idx}>", bbox_text)

        # There are no other turns in captioning, only the prompt and the caption.
        full_conversation = [
            f"{prompt}\n",
            f"{caption}{SpecialTokens.text_eos_token}",
        ]

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # labels = torch.concatenate(
            #     [torch.ones_like(tokens[0]) * -100, tokens[1]],
            #     dim=-1,
            # )
            labels = [torch.ones_like(tokens[0]) * -100, tokens[1]]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_captioning(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process captioning instance."""
        if not instance.caption:
            raise AssertionError(f"{instance} has no caption.")

        metadata = json.loads(instance.metadata)
        prompt = metadata.get("prompt", None)
        if prompt is None:
            prompt = self._get_random_template_for_task(Task.captioning)
        # This is a temporary workaround to handle the LLaVA pretrain dataset.
        else:
            prompt = prompt.replace("\n<image>", "").replace("<image>\n", "")

        caption = format_text(
            instance.caption,
            strip=True,
            punctuate=True,
            capitalize=True,
            add_bos=False,
            add_eos=False,
        )
        # There are no other turns in captioning, only the prompt and the caption.
        full_conversation = [
            f"{prompt}\n",
            f"{caption}{SpecialTokens.text_eos_token}",
        ]

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # labels = torch.concatenate(
            #     [torch.ones_like(tokens[0]) * -100, tokens[1]],
            #     dim=-1,
            # )
            labels = [torch.ones_like(tokens[0]) * -100, tokens[1]]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_chat(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,  # noqa: ARG002
    ) -> TextEncoding:
        """Process chat instance."""
        conversation = instance.chat
        if not conversation:
            raise AssertionError(f"{instance} has no conversation.")

        full_conversation = []
        target_indices = []
        for idx, conversation_turn in enumerate(conversation):
            text = conversation_turn.text
            text = format_text(
                text,
                strip=True,
                punctuate=True,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )

            if idx == 0:
                text = text.replace("\n<image>", "").replace("<image>\n", "")
                text = f"{text}"
            full_conversation.append(f"{text}\n")

            if conversation_turn.is_response_turn:
                target_indices.append(idx)

        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            full_conversation[-1] = f"{full_conversation[-1]}{SpecialTokens.text_eos_token}"

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )

            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_multiple_choice_vqa(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,  # noqa: ARG002
    ) -> TextEncoding:
        """Process multiple choice VQA instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"{instance} has no QA pairs.")

        metadata = json.loads(instance.metadata)

        prompt = self._get_random_template_for_task(Task.m_vqa)

        target_pos = 0
        target_indices = []
        # Some tasks (for example AI2D) contain an OCR text, which needs to be included in the prompt.
        ocr_text = metadata.get("ocr_text", "")
        full_conversation = [f"{prompt}\n{ocr_text}"] if ocr_text else [f"{prompt}"]

        for qa_pair, raw_candidate_answers in zip(
            instance.qa_pairs, metadata["answers"], strict=False
        ):
            question = format_text(
                qa_pair.question,
                strip=True,
                punctuate=True,
                # Only capitalize the question if it is not from AI2D.
                # This is because capitalization results in lowercasing the entire question
                # But we dont want to do this since the question might contain a capitalized letter
                # that refers to the ocr text.
                capitalize=instance.source != DatasetNames.ai2d.value,
                add_bos=False,
                add_eos=False,
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
                for answer in raw_candidate_answers
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

            if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
                correct_letter = string.ascii_uppercase[
                    raw_candidate_answers.index(qa_pair.answer)
                ]
                full_conversation.append(f"{correct_letter}.{SpecialTokens.text_eos_token}")
                # Increase the target pos by 1 (question) + 1 (prompt) + the number of candidate answers.
                target_pos += 2 + len(candidate_answers)
                target_indices.append(target_pos)

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )

            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_grounding_multiple_choice_vqa(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process grounding multiple choice VQA instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"{instance} has no qa_pairs.")

        if not instance.region:
            raise AssertionError(f"{instance} has no region.")

        if not visual_encoding:
            raise AssertionError(
                "For grounding_multiple_choice we need to preprocess the image first."
            )

        prompt = self._get_random_template_for_task(Task.gm_vqa)
        metadata = json.loads(instance.metadata)

        target_pos = 0
        target_indices = []
        full_conversation = [f"{prompt}\n"]

        for qa_pair, candidate_answers_ids in zip(
            instance.qa_pairs, metadata["answers"], strict=False
        ):
            question = format_text(
                qa_pair.question,
                strip=True,
                punctuate=True,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )

            bbox_ids2position = {region.phrase: idx for idx, region in enumerate(instance.region)}

            candidate_answers = [
                self.format_bbox(bbox=visual_encoding.bboxes_norm[0][bbox_ids2position[region_id]])
                for region_id in candidate_answers_ids
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

            if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
                correct_letter = string.ascii_uppercase[
                    candidate_answers_ids.index(qa_pair.answer)
                ]
                full_conversation.append(f"{correct_letter}.{SpecialTokens.text_eos_token}")
                # Increase the target pos by 1 (question) + 1 (prompt) + the number of candidate answers.
                target_pos += 2 + len(candidate_answers)
                target_indices.append(target_pos)

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )
            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_visual_grounding(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process visual grounding instance."""
        if not instance.region:
            raise AssertionError(f"{instance} has no region.")

        if not visual_encoding:
            raise AssertionError("For visual grounding we need to preprocess the image first.")

        prompt = self._get_random_template_for_task(Task.visual_grounding)
        full_conversation = [f"{prompt}"]
        for region, bbox in zip(instance.region, visual_encoding.bboxes_norm[0], strict=False):
            bbox_text = self.format_bbox(bbox=bbox)

            region_text = format_text(
                region.phrase,
                strip=True,
                punctuate=True,
                capitalize=True,
                add_bos=False,
                add_eos=False,
            )
            if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
                full_conversation.extend(
                    [f"\n{region_text}\n", f"{bbox_text}{SpecialTokens.text_eos_token}"]
                )
            else:
                full_conversation.append(f"{region_text}")

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            target_indices = [c + 2 for c, _ in enumerate(full_conversation) if c % 2 == 0]
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )
            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def preprocess_vqa(
        self,
        instance: Instance,
        tokenizer: AutoTokenizer,
        split: DatasetSplits,
        visual_encoding: VisualEncoding | None = None,
    ) -> TextEncoding:
        """Process VQA instance."""
        if not instance.qa_pairs:
            raise AssertionError(f"{instance} has no QA pairs.")
        prompt = self._get_random_template_for_task(Task.vqa)
        conversation = self._get_conversation_from_vqa_instance(instance)

        full_conversation = [f"{prompt}"]
        for conversation_turn in conversation:
            # Only capitalize the answer if it is not from DocVQA / Infographics VQA.
            # This is because capitalization results in lowercasing the entire question
            # But we dont want to do this since the question might contain a capitalized letter
            # that refers to the ocr text.
            capitalize = instance.source not in {
                DatasetNames.docvqa.value,
                DatasetNames.infographics_vqa.value,
            }
            question = format_text(
                conversation_turn[0],
                strip=True,
                punctuate=True,
                capitalize=capitalize,
                add_bos=False,
                add_eos=False,
            )
            # If there is a response, add it to the conversation.
            if len(conversation_turn) > 1:
                response = format_text(
                    conversation_turn[1],
                    strip=True,
                    punctuate=True,
                    capitalize=capitalize,
                    add_bos=False,
                    add_eos=False,
                )
                full_conversation.extend(
                    [
                        f"\nQuestion: {question}\nAnswer: ",
                        f"{response}{SpecialTokens.text_eos_token}",
                    ]
                )
            # If there is no response, add the question to the conversation.
            else:
                full_conversation.append(f"\nQuestion: {question}\nAnswer: ")

        # Tokenize the conversation and prepare the input and target tensors.
        tokens = [
            tokenizer(turn, return_tensors="pt").input_ids.squeeze(0) for turn in full_conversation
        ]

        # input_ids = torch.concatenate(tokens, dim=-1)

        # Mask the target tokens, only the response for each question should be predicted.
        labels = None
        if split in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}:
            # We need every 2nd turn to not be masked, as it is the response to the question.
            target_indices = [c for c, _ in enumerate(full_conversation, 1) if c % 2 == 0]
            # labels = torch.concatenate(
            #     [
            #         turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
            #         for index, turn_tokens in enumerate(tokens)
            #     ],
            #     dim=-1,
            # )

            labels = [
                turn_tokens if index in target_indices else torch.ones_like(turn_tokens) * -100
                for index, turn_tokens in enumerate(tokens)
            ]

        return TextEncoding(input_ids=tokens, labels=labels, text="".join(full_conversation))

    def _get_conversation_from_vqa_instance(self, instance: Instance) -> list[list[str]]:
        """Create the conversation for the VQA instance."""
        qa_pairs = instance.qa_pairs
        conversation: list[list[str]] = []
        for qa_pair in qa_pairs:  # type: ignore[union-attr]
            question = qa_pair.question
            answer = qa_pair.answer
            if answer:
                conversation.append([question, answer])
            else:
                conversation.append([question])
        return conversation

    def _get_random_template_for_task(self, task: Task) -> str:
        """Choose a random instruction template for the given task."""
        return random.choice(TASK_TEMPLATES_MAP[task])

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import datasets
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


STRING_VALUE = "string"

DatasetFeatures = datasets.Features(
    {
        "task": datasets.Value(STRING_VALUE),
        "image": datasets.Image(),
        "caption": datasets.Value(STRING_VALUE),
        "qa_pairs": [
            {
                "question": datasets.Value(STRING_VALUE),
                "answer": datasets.Value(STRING_VALUE),
            }
        ],
        # For some reason you cannot set None as default value dictionary.
        # The workaround is to the `region` and the `relation` as lists
        "region": [
            {
                "bbox": [datasets.Value("float")],
                "phrase": datasets.Value(STRING_VALUE),
            }
        ],
        "relation": [
            {
                "subject_name": datasets.Value(STRING_VALUE),
                "subject_bbox": [datasets.Value("float")],
                "object_name": datasets.Value(STRING_VALUE),
                "object_bbox": [datasets.Value("float")],
                "predicate": datasets.Value(STRING_VALUE),
            }
        ],
        "chat": [
            {
                "role": datasets.Value(STRING_VALUE),
                "text": datasets.Value(STRING_VALUE),
            }
        ],
        # Define where the sample comes from.
        "source": datasets.Value(STRING_VALUE),
        # We commit any kind of additional information in json format in `metadata`
        "metadata": datasets.Value(STRING_VALUE),
    }
)


class Task(Enum):
    """Task for the an instance."""

    vqa = "vqa"
    m_vqa = "multiple_choice_vqa"
    gm_vqa = "grounded_multiple_choice_vqa"
    captioning = "captioning"
    grounded_captioning = "grounded_captioning"
    controlled_captioning = "controlled_captioning"
    dense_captioning = "dense_captioning"
    visual_grounding = "visual_grounding"
    relationship_detection = "relationship_detection"
    object_detection = "object_detection"
    itm = "image_text_match"
    chat = "chat"
    synthetic_vg = "synthetic_vg"

    @classmethod
    def get_index(cls, task: Any) -> int:
        """Get task index."""
        return list(cls).index(task)


class ConversationTurn(BaseModel):
    """Chat message."""

    model_config = ConfigDict(populate_by_name=True)

    role: str = Field(..., alias="from")
    text: str = Field(..., alias="value")

    @property
    def is_instruction_turn(self) -> bool:
        """Check if the turn is an instruction turn."""
        return self.role == "human"

    @property
    def is_response_turn(self) -> bool:
        """Check if the turn is a response turn."""
        return self.role == "gpt"


class Chat(BaseModel):
    """Chat."""

    chat: list[ConversationTurn]


class Region(BaseModel):
    """Dataset region."""

    bbox: list[float]
    phrase: str

    @field_validator("bbox")
    @classmethod
    def validate_answers(cls, bbox: list[float]) -> list[float]:
        """Bounding box coordinates in XYXWY format must have exactly 4 positive coordinates."""
        if len(bbox) != 4:
            raise ValueError(f"Expecting exactly 4 coordinates, got {len(bbox)}!")
        if any(coord < 0 for coord in bbox):
            raise ValueError(f"Expecting all coordinates to be positive, got {bbox}!")
        return bbox


class Relation(BaseModel):
    """Dataset relationship."""

    subject_name: str
    subject_bbox: list[float]
    object_name: str
    object_bbox: list[float]
    predicate: str


class QAPair(BaseModel):
    """Dataset QA pair."""

    question: str
    answer: Optional[str]


class Instance(BaseModel):
    """Dataset instance."""

    task: Task
    image: Optional[Image.Image]
    caption: Optional[str]
    qa_pairs: Optional[list[QAPair]]
    # question: Optional[str]
    # answers: Optional[Union[str, list[str]]]
    region: Optional[list[Region]]
    relation: Optional[list[Relation]]
    chat: Optional[list[ConversationTurn]]
    source: str
    metadata: str

    class Config:
        """Updated config."""

        arbitrary_types_allowed = True


class DatasetNames(Enum):
    """Dataset names."""

    ai2d = "ai2d"
    aok_vqa = "aok_vqa"
    cc3m = "conceptual_captions_3m"
    coco = "coco"
    docvqa = "docvqa"
    gqa = "gqa"
    grit = "grit"
    infographics_vqa = "infographics_vqa"
    llava_pretrain = "llava_pretrain"
    llava_instruct = "llava_instruct"
    localized_narratives = "localized_narratives"
    nocaps = "nocaps"
    ocr_vqa = "ocr_vqa"
    ok_vqa = "ok_vqa"
    pope_adversarial = "pope_adversarial"
    pope_popular = "pope_popular"
    pope_random = "pope_random"
    refcoco = "refcoco"
    refcoco_plus = "refcoco+"
    refcocog = "refcocog"
    sbu_captions = "sbu_captions"
    text_caps = "text_caps"
    text_vqa = "text_vqa"
    visual7w = "visual7w"
    visual_genome = "visual_genome"
    vizwiz_vqa = "viswiz_vqa"
    vqa_v2 = "vqa_v2"
    vsr = "visual_spatial_reasoning"
    instruction_tuning = "instruction_tuning"
    pretrain = "pretrain"
    # This is language only task
    synthetic_vg = "synthetic_vg"

    @classmethod
    def list_all_dataset_names(cls) -> list[str]:
        """List all dataset names."""
        return [name.value for name in cls]


class DatasetSplits(datasets.Split):  # type: ignore[misc]
    """Dataset splits."""

    TEST_STD = datasets.NamedSplit("teststd")  # noqa: WPS115
    TEST_DEV = datasets.NamedSplit("testdev")  # noqa: WPS115

    @classmethod
    def list_splits(cls) -> list[datasets.Split]:
        """List all splits."""
        return [
            cls.TRAIN,
            cls.VALIDATION,
            cls.TEST,
            cls.TEST_STD,
            cls.TEST_DEV,
        ]


@dataclass
class PretrainDatasets:
    """A dataclass that contains all datasets that are used for pretraining."""

    pretrain_dataset_map: Mapping[str, bool] = MappingProxyType(
        {
            DatasetNames.ai2d.value: False,
            DatasetNames.aok_vqa.value: False,
            DatasetNames.cc3m.value: False,
            DatasetNames.coco.value: False,
            DatasetNames.docvqa.value: False,
            DatasetNames.gqa.value: False,
            DatasetNames.grit.value: False,
            DatasetNames.infographics_vqa.value: False,
            DatasetNames.llava_pretrain.value: True,
            DatasetNames.llava_instruct.value: False,
            DatasetNames.localized_narratives.value: False,
            DatasetNames.nocaps.value: False,
            DatasetNames.ocr_vqa.value: False,
            DatasetNames.ok_vqa.value: False,
            DatasetNames.pope_adversarial.value: False,
            DatasetNames.pope_popular.value: False,
            DatasetNames.pope_random.value: False,
            DatasetNames.refcoco.value: False,
            DatasetNames.refcoco_plus.value: False,
            DatasetNames.refcocog.value: False,
            DatasetNames.sbu_captions.value: False,
            DatasetNames.text_caps.value: False,
            DatasetNames.text_vqa.value: False,
            DatasetNames.visual7w.value: False,
            DatasetNames.visual_genome.value: False,
            DatasetNames.vizwiz_vqa.value: False,
            DatasetNames.vqa_v2.value: False,
            DatasetNames.vsr.value: False,
            DatasetNames.synthetic_vg.value: False,
        }
    )

    def __init__(self) -> None:
        """Verify that all datasets are in the map."""
        if len(self.pretrain_dataset_map) != len(DatasetNames) - 2:
            raise AssertionError("Pretrain dataset map is not equal to all datasets!")

        for key, _ in self.pretrain_dataset_map.items():
            missing = (
                key not in DatasetNames.list_all_dataset_names()
                and key != DatasetNames.pretrain.value
                and key != DatasetNames.instruction_tuning.value
            )
            if missing:
                raise ValueError(f"Dataset {key} not in {DatasetNames.list_all_dataset_names()}!")

    def is_in_pretrain_dataset(self, dataset_name: str) -> bool:
        """Check if the dataset is in the pretrain dataset."""
        return self.pretrain_dataset_map[dataset_name]

    def list_all_pretrain_dataset_names(self) -> list[str]:
        """List all pretrain datasets."""
        return [
            name for name, is_in_pretrain in self.pretrain_dataset_map.items() if is_in_pretrain
        ]


@dataclass
class InstructionTuningDatasets:
    """A dataclass that contains all datasets that are used for instruction tuning."""

    instruction_tuning_dataset_map: Mapping[str, bool] = MappingProxyType(
        {
            DatasetNames.ai2d.value: True,
            DatasetNames.aok_vqa.value: True,
            DatasetNames.cc3m.value: False,
            DatasetNames.coco.value: True,
            DatasetNames.docvqa.value: True,
            DatasetNames.gqa.value: True,
            DatasetNames.grit.value: True,
            DatasetNames.infographics_vqa.value: True,
            DatasetNames.llava_pretrain.value: False,
            DatasetNames.llava_instruct.value: True,
            DatasetNames.localized_narratives.value: False,
            DatasetNames.nocaps.value: False,
            DatasetNames.ocr_vqa.value: True,
            DatasetNames.ok_vqa.value: True,
            DatasetNames.pope_adversarial.value: False,
            DatasetNames.pope_popular.value: False,
            DatasetNames.pope_random.value: False,
            DatasetNames.refcoco.value: True,
            DatasetNames.refcoco_plus.value: True,
            DatasetNames.refcocog.value: True,
            DatasetNames.sbu_captions.value: False,
            DatasetNames.text_caps.value: True,
            DatasetNames.text_vqa.value: False,
            DatasetNames.visual7w.value: True,
            DatasetNames.visual_genome.value: True,
            DatasetNames.vizwiz_vqa.value: False,
            DatasetNames.vqa_v2.value: True,
            DatasetNames.vsr.value: True,
            DatasetNames.synthetic_vg.value: False,
        }
    )

    def __init__(self) -> None:
        """Verify that all datasets are in the map."""
        if len(self.instruction_tuning_dataset_map) != len(DatasetNames) - 2:
            raise AssertionError("instruction_tuning dataset map is not equal to all datasets!")

        for key, _ in self.instruction_tuning_dataset_map.items():
            missing = (
                key not in DatasetNames.list_all_dataset_names()
                and key != DatasetNames.pretrain.value
                and key != DatasetNames.instruction_tuning.value
            )
            if missing:
                raise ValueError(f"Dataset {key} not in {DatasetNames.list_all_dataset_names()}!")

    def is_in_instruction_tuning_dataset(self, dataset_name: str) -> bool:
        """Check if the dataset is in the pretrain dataset."""
        return self.instruction_tuning_dataset_map[dataset_name]

    def list_all_instruction_tuning_dataset_names(self) -> list[str]:
        """List all instruction_tuning datasets."""
        return [
            name
            for name, is_in_instruction_tuning in self.instruction_tuning_dataset_map.items()
            if is_in_instruction_tuning
        ]


class LLavaPretrainModel(BaseModel):
    """LLava pretrain basemodel."""

    annotation: dict[str, Any]
    metadata: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def validate_instance(cls, field_values: dict[str, Any]) -> dict[str, Any]:  # noqa: WPS238
        """Validate that the annotation and the metadata for an instance are correct."""
        conversations = field_values["annotation"]["conversations"]
        if len(conversations) != 2:
            raise ValueError(f"Expecting exactly 2 conversation turns! {conversations}")

        from_value = [turn["from"] for turn in field_values["annotation"]["conversations"]]
        if "human" not in from_value or "gpt" not in from_value:
            raise ValueError(f"Expecting exactly 1 human and 1 gpt turn! {conversations}")

        image = field_values["annotation"]["image"]
        metadata = field_values["metadata"]["image"]
        if metadata != field_values["metadata"]["image"]:
            raise ValueError(f"Images dont match! {image} {metadata}")

        if field_values["annotation"]["id"] != field_values["metadata"]["id"]:
            raise ValueError(
                f"IDs dont match! {field_values['annotation']['id']} {field_values['metadata']['id']}"
            )
        return field_values

    @property
    def caption(self) -> str:
        """Return the caption."""
        conversations = self.annotation["conversations"]
        for conversation_turn in conversations:
            if conversation_turn["from"] == "gpt":
                return conversation_turn["value"]

        raise ValueError(f"Expecting a gpt turn! {conversations}")

    @property
    def prompt(self) -> str:
        """Return the prompt."""
        conversations = self.annotation["conversations"]
        for conversation_turn in conversations:
            if conversation_turn["from"] == "human":
                return conversation_turn["value"]

        raise ValueError(f"Expecting a human turn! {conversations}")


class LLavaInstructModel(BaseModel):
    """LLava instruct basemodel."""

    id: str
    image: str
    conversations: list[ConversationTurn]


class TextVQAMetadata(BaseModel):
    """TextVQA metadata."""

    question: str
    image_id: str
    question_id: int
    answers: Optional[list[str]] = None
    image_height: float
    image_width: float
    ocr: dict[str, Any]
    flickr_original_url: str
    flickr300k_url: str = Field(..., alias="flickr_300k_url")

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, answers: list[str]) -> list[str]:
        """VQA v2 instances have exactly 10 answers."""
        if len(answers) != 10:
            raise ValueError(f"Expecting exactly 10 answers, got {len(answers)}!")
        return answers


class TextCapsMetadata(BaseModel):
    """TextCaps metadata."""

    caption_str: Optional[str] = None
    image_id: str
    caption_id: Optional[int] = None
    image_height: float
    image_width: float
    set_name: str
    # 5 captions for each example, the references_strs are the other 4 captions, these are absent from test.
    reference_strs: Optional[list[str]] = None
    ocr: dict[str, Any]
    flickr_original_url: str
    flickr300k_url: str = Field(..., alias="flickr_300k_url")


class GQAMetadata(BaseModel):
    """GQA metadata."""

    question: str
    question_id: str
    image_id: str = Field(..., alias="imageId")
    answer: Optional[str] = None
    full_answer: Optional[str] = Field(default=None, alias="fullAnswer")
    is_balanced: bool = Field(..., alias="isBalanced")


class OCRVQAMetadata(BaseModel):
    """OCRVQA metadata."""

    id: str
    image_url: str = Field(..., alias="imageURL")
    questions: list[str]
    answers: list[str]
    title: str
    author_name: str = Field(..., alias="authorName")
    genre: str

    @model_validator(mode="before")
    @classmethod
    def validate_instance(cls, field_values: dict[str, Any]) -> dict[str, Any]:
        """Validate the instance.

        Apparently there is a 1-1 correspondance between the questions and the answers. Ensure that
        the questions and answers are the same length.
        """
        questions = field_values["questions"]
        answers = field_values["answers"]
        if len(questions) != len(answers):
            raise ValueError(
                f"The questions {len(questions)} and answers {len(answers)} must be the same length"
            )
        return field_values


class RefCOCOImageMetadata(BaseModel, frozen=True):
    """Image metadata for RefCOCO scene."""

    image_path: str
    image_id: str
    width: int
    height: int
    url: str


class RefCOCORegion(BaseModel):
    """RefCOCO region."""

    annotation_id: str
    image_id: str
    x: float
    y: float
    w: float
    h: float
    category_id: int

    @field_validator("x", "y", "w", "h")
    @classmethod
    def validate_region(cls, bbox_val: float) -> float:
        """Bbox coordinates in XYXWY format must have exactly 4 positive coordinates."""
        if bbox_val < 0:
            raise ValueError(f"Expecting exactly 4 coordinates, got {bbox_val}!")
        return bbox_val


class RefCOCOExpression(BaseModel):
    """RefCOCO referring expression."""

    sentence: str
    sentence_id: str
    annotation_id: str


class VQAQuestionMetadata(BaseModel):
    """VQA question metadata."""

    question: str
    question_id: int
    image_id: int


class VQAAnswerMetadata(BaseModel):
    """VQA answer metadata."""

    question_type: str
    multiple_choice_answer: str
    answers: list[dict[str, Any]]
    image_id: int
    answer_type: str
    question_id: int

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, answers: list[str]) -> list[str]:
        """VQA v2 instances have exactly 10 answers."""
        if len(answers) != 10:
            raise ValueError(f"Expecting exactly 10 answers, got {len(answers)}!")
        return answers


class AOKVQAMetadata(BaseModel):
    """AOKVQA metadata."""

    split: str
    image_id: int
    question_id: str
    question: str
    choices: list[str]
    correct_choice_idx: Optional[int] = None
    direct_answers: Optional[list[str]] = None
    difficult_direct_answer: bool
    rationales: Optional[list[str]] = None


class VisWizVQAMetadata(BaseModel):
    """VisWizVQA metadata."""

    image: str
    question: str
    answerable: Optional[int] = None
    answer_type: Optional[str] = None
    answers: Optional[list[dict[str, str]]] = None


class AID2Question(BaseModel):
    """AI2D2 question."""

    question: str
    abclabel: bool = Field(..., alias="abcLabel")
    answers: list[str] = Field(..., alias="answerTexts")
    correct_answer: int = Field(..., alias="correctAnswer")
    question_id: str = Field(..., alias="questionId")


class AI2DMetadata(BaseModel):
    """AI2D2 metadata."""

    image_name: str = Field(..., alias="imageName")
    questions: list[AID2Question]

    @field_validator("questions", mode="before")
    @classmethod
    def transform_questions(cls, questions_dict: dict[str, Any]) -> str:
        """Transform the questions."""
        questions = []
        for question, question_dict in questions_dict.items():
            question_dict["question"] = question
            questions.append(AID2Question(**question_dict))
        return questions


class Visual7WQuestion(BaseModel):
    """Visual7W question."""

    question: str
    # Pointing QA pairs do not have the image in the metadata
    image_id: Optional[int] = None
    qa_id: int
    multiple_choices: list[Union[str, int]]
    answer: Union[str, int]
    question_type: str = Field(..., alias="type")


class Visual7WImageMetadata(BaseModel):
    """AI2D2 question."""

    qa_pairs: list[Visual7WQuestion]
    image_id: int
    split: str
    filename: str


class POPEMetadata(BaseModel):
    """POPE metadata."""

    question_id: int
    image: str
    text: str
    label: Literal["yes", "no"]


class GRITMetadata(BaseModel):
    """GRIT metadata."""

    noun_chunks: list[list[float]]
    ref_exps: list[list[float]]
    caption: str
    original_width: int
    original_height: int
    width: int
    height: int
    url: str
    clip_similarity_vitl14: float
    clip_similarity_vitb32: float


class DocVQAMetadata(BaseModel):
    """DocVQA metadata."""

    question_id: int = Field(..., alias="questionId")
    question: str
    question_types: Optional[list[str]] = None
    image: str
    doc_id: int = Field(..., alias="docId")
    ucsf_document_id: str
    ucsf_document_page_no: str
    answers: Optional[list[str]] = None
    data_split: str

    @field_validator("image", mode="before")
    @classmethod
    def fix_image_field(cls, image: str) -> str:
        """The image has the 'documents' root directory."""
        return os.path.basename(image)


class InfographicsVQAMetadata(BaseModel):
    """InfographicsVQA metadata."""

    question_id: int = Field(..., alias="questionId")
    question: str
    image: str = Field(..., alias="image_local_name")
    image_url: str
    answers: Optional[list[str]] = None
    ocr_output_file: str
    data_split: str


TASK_TEMPLATES_MAP: Mapping[Task, list[str]] = MappingProxyType(
    {
        Task.grounded_captioning: [
            "Provide a one-sentence caption for the image and mention each entity.",
        ],
        Task.captioning: [
            "Provide a one-sentence caption for the provided image.",
            # "Write a short caption for the image.",
            # "Briefly describe this image.",
            # "Caption this image.",
            # "Caption the image.",
            # "Come up with a caption for the image.",
            # "Create a fitting caption.",
            # "Describe the image.",
            # "Describe this image.",
            # "Provide a caption for the image.",
            # "Provide a caption relevant to the image.",
            # "Sum up the image in a sentence.",
            # "What does this image show?",
            # "What is shown in the image?",
            # "Write a caption that depicts the content of the image.",
        ],
        Task.vqa: [
            "Answer the question using a single word or phrase."
            # "Answer the question: {question}",
            # "{question}",
            # "Answer: {question}",
            # "Provide an answer to the following question: {question}",
            # "What is the answer to: {question}",
            # "What is the answer to the question: {question}",
        ],
        Task.m_vqa: ["Answer with the option's letter from the given choices directly."],
        Task.gm_vqa: ["Answer with the option's letter from the given choices directly."],
        Task.dense_captioning: [
            "Provide a short description of the region:",
            # "Caption {region}.",
            # "Caption object {region}.",
            # "Describe {region}.",
            # "Describe object {region}.",
            # "Describe the region {region}.",
            # "Provide a caption for {region}.",
            # "What is shown in the {region}?",
            # "Write a caption for {region}.",
        ],
        Task.visual_grounding: [
            "Locate the region that is described by:",
            # "Locate the object that is described by: {caption}.",
            # "Find the object: {caption}.",
            # "Find the object that is described by: {caption}.",
            # "Find the object that matches the description: {caption}.",
            # "Locate the object: {caption}.",
            # "Locate the object that matches the description: {caption}.",
            # "Pick the object: {caption}.",
            # "Pick the object that is described by: {caption}.",
            # "Pick the object that matches the description: {caption}.",
            # "Select the object: {caption}.",
            # "Select the object that is described by: {caption}.",
            # "Select the object that matches the description: {caption}.",
            # "Which object is described by: {caption}?",
            # "Which object matches the description: {caption}?",
        ],
        Task.itm: [
            "Determine if the image matches the description:",
        ],
    },
)

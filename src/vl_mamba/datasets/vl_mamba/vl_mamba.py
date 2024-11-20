from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, Optional, Union

import datasets
from loguru import logger

from vl_mamba.datamodels.datamodels import (
    DatasetFeatures,
    DatasetNames,
    DatasetSplits,
    InstructionTuningDatasets,
    PretrainDatasets,
    Task,
)
from vl_mamba.datasets.vl_mamba.ai2d_loader import AI2DLoader
from vl_mamba.datasets.vl_mamba.aok_vqa_loader import AOKVQALoader
from vl_mamba.datasets.vl_mamba.base_loader import BaseLoader
from vl_mamba.datasets.vl_mamba.coco_loader import COCOLoader
from vl_mamba.datasets.vl_mamba.data_paths import BASE_DIR, DatasetPaths
from vl_mamba.datasets.vl_mamba.docvqa_loader import DocVQALoader
from vl_mamba.datasets.vl_mamba.gqa_loader import GQALoader
from vl_mamba.datasets.vl_mamba.grit_loader import GRITLoader
from vl_mamba.datasets.vl_mamba.infographicvqa_loader import InfographicVQALoader
from vl_mamba.datasets.vl_mamba.llava_instruct_loader import LLAVAInstructLoader
from vl_mamba.datasets.vl_mamba.llava_pretrain import LLAVAPretrainLoader
from vl_mamba.datasets.vl_mamba.localized_narratives_loader import LocalizedNarrativesLoader
from vl_mamba.datasets.vl_mamba.nocaps_loader import NocapsLoader
from vl_mamba.datasets.vl_mamba.ocr_vqa_loader import OCRVQALoader
from vl_mamba.datasets.vl_mamba.pope_loader import POPELoader
from vl_mamba.datasets.vl_mamba.refcoco_loader import RefCOCOLoader
from vl_mamba.datasets.vl_mamba.synthetic_vg_loader import SyntheticVGLoader
from vl_mamba.datasets.vl_mamba.text_caps_loader import TextCapsLoader
from vl_mamba.datasets.vl_mamba.text_vqa_loader import TextVQALoader
from vl_mamba.datasets.vl_mamba.visual7w_loader import Visual7WLoader
from vl_mamba.datasets.vl_mamba.visual_genome_loader import VisualGenomeLoader
from vl_mamba.datasets.vl_mamba.vizwiz_vqa_loader import VizWizVQALoader
from vl_mamba.datasets.vl_mamba.vqa_v2_loader import VQAv2Loader
from vl_mamba.datasets.vl_mamba.vsr_loader import VSRLoader


DEFAULT_NAME = DatasetNames.pretrain.value

_DESCRIPTION = "VL-Mamba Dataset"

_LICENSE = "Please refer to individual LICENSES of each datasets."


class VLMambaConfig(datasets.BuilderConfig):  # type: ignore[misc] # noqa: WPS230
    """BuilderConfig for VLMamba."""

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        subset: Optional[str] = None,
        num_proc: Optional[int] = None,
        datasets_batch_size: int = 1000,
        sqlite3_batch_size: int = 10000,
        chunk_size: int = 10000,
        writer_batch_size: int = 10000,
        use_flickr30k_ln: bool = False,
        use_gqa_balanced_version: bool = True,
        use_gqa_short_answers: bool = True,
        gqa_max_annotations_per_image: int = 20,
        vqa_max_annotations_per_image: int = 20,
        root_dataset_path: Union[str, Path] = BASE_DIR,
        refcoco_tasks: Optional[list[str]] = None,
        visual_genome_tasks: Optional[list[str]] = None,
        visual_genome_max_annotations_per_image: int = 10,
        localized_narratives_segment_window_size: float = 0.4,
        visual7w_max_questions_per_image_in_turn: int = 3,
        vsr_version: str = "cambridgeltl/vsr_random",
        synthetic_vg_seq_len: Literal[20, 100, 200, 300, 500, 576] = 576,
        **kwargs: dict[str, Any],
    ) -> None:
        # We disable multiprocessing.
        if num_proc is None:
            num_proc = 1
        super().__init__(**kwargs)

        self.name = name
        self.subset = subset

        # determines how much we can load
        self.datasets_batch_size = datasets_batch_size

        self.sqlite3_batch_size = sqlite3_batch_size

        # Some datasets should be loaded via multiprocessing.
        self.num_proc = num_proc
        self.chunk_size = chunk_size

        # Batch writing
        self.writer_batch_size = writer_batch_size

        # LN
        self.use_flickr30k_ln = use_flickr30k_ln

        self.root_dataset_path = root_dataset_path
        self.dataset_paths = DatasetPaths(base_dir=root_dataset_path)

        self.use_gqa_balanced_version = use_gqa_balanced_version

        self.use_gqa_short_answers = use_gqa_short_answers
        self.gqa_max_annotations_per_image = gqa_max_annotations_per_image

        self.vqa_max_annotations_per_image = vqa_max_annotations_per_image

        # RefCOCO versions. If no version is provided we will load all of them.
        self.use_refcoco_datasets = ["refcoco", "refcoco+", "refcocog"]

        # RefCOCO tasks. If no task is provided we will load all of them.
        if refcoco_tasks is None:
            refcoco_tasks = [Task.visual_grounding.value, Task.dense_captioning.value]
        self.refcoco_tasks = refcoco_tasks

        # Visual Genome tasks. If no task is provided we will load all of them.
        if visual_genome_tasks is None:
            visual_genome_tasks = [
                Task.visual_grounding.value,
                Task.dense_captioning.value,
                Task.vqa.value,
            ]
        self.visual_genome_tasks = visual_genome_tasks
        self.visual_genome_max_annotations_per_image = visual_genome_max_annotations_per_image

        self.visual7w_max_questions_per_image_in_turn = visual7w_max_questions_per_image_in_turn

        self.vsr_version = vsr_version
        self.synthetic_vg_seq_len = synthetic_vg_seq_len

        self.localized_narratives_segment_window_size = localized_narratives_segment_window_size

        self.instruction_tuning_datasets = InstructionTuningDatasets()
        self.pretraining_datasets = PretrainDatasets()


class VLMamba(datasets.ArrowBasedBuilder):  # type: ignore[misc]
    """Builder VLMamba.

    datasets-cli test src/mm_icl/datamodules/datasets/mm_icl/mm_icl.py --save_info --all_configs
    --cache_dir storage/datasets/mm_icl/
    """

    BUILDER_CONFIG_CLASS = VLMambaConfig  # noqa: WPS115

    BUILDER_CONFIGS = [  # noqa: WPS115
        # Datasets-specific configs
        VLMambaConfig(name=DatasetNames.ai2d.value),
        VLMambaConfig(name=DatasetNames.aok_vqa.value),
        VLMambaConfig(name=DatasetNames.cc3m.value),
        VLMambaConfig(name=DatasetNames.coco.value),
        VLMambaConfig(name=DatasetNames.docvqa.value),
        VLMambaConfig(name=DatasetNames.gqa.value),
        VLMambaConfig(name=DatasetNames.grit.value),
        VLMambaConfig(name=DatasetNames.infographics_vqa.value),
        VLMambaConfig(name=DatasetNames.llava_pretrain.value),
        VLMambaConfig(name=DatasetNames.llava_instruct.value),
        VLMambaConfig(name=DatasetNames.localized_narratives.value),
        VLMambaConfig(name=DatasetNames.nocaps.value),
        VLMambaConfig(name=DatasetNames.ocr_vqa.value),
        VLMambaConfig(name=DatasetNames.ok_vqa.value),
        VLMambaConfig(name=DatasetNames.pope_adversarial.value),
        VLMambaConfig(name=DatasetNames.pope_popular.value),
        VLMambaConfig(name=DatasetNames.pope_random.value),
        VLMambaConfig(name=DatasetNames.refcoco.value),
        VLMambaConfig(name=DatasetNames.refcoco_plus.value),
        VLMambaConfig(name=DatasetNames.refcocog.value),
        VLMambaConfig(name=DatasetNames.sbu_captions.value),
        VLMambaConfig(name=DatasetNames.text_caps.value),
        VLMambaConfig(name=DatasetNames.text_vqa.value),
        VLMambaConfig(name=DatasetNames.visual7w.value),
        VLMambaConfig(name=DatasetNames.visual_genome.value),
        VLMambaConfig(name=DatasetNames.vizwiz_vqa.value),
        VLMambaConfig(name=DatasetNames.vqa_v2.value),
        VLMambaConfig(name=DatasetNames.vsr.value),
        # Pretraining subset: builds all the loaders from all datasets used for pretraining
        VLMambaConfig(name=DatasetNames.pretrain.value),
        # Instruction tuning subset: builds all the loaders from all datasets used for instruction tuning
        VLMambaConfig(name=DatasetNames.instruction_tuning.value),
        # Synthetic VG dataset
        VLMambaConfig(name=DatasetNames.synthetic_vg.value),
    ]

    def _info(self) -> datasets.DatasetInfo:  # noqa: WPS110
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=DatasetFeatures,
            license=_LICENSE,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"loaders": self._build_loaders(dl_manager, split_name)},
            )
            for split_name in DatasetSplits.list_splits()
        ]

    def _add_dataset_for_instruction_tuning(
        self, config_name: str, dataset_name: str, split_name: str
    ) -> bool:
        """Check if should a dataset for instruction tuning.

        A dataset will be added to the instruction tuning subset if:
        1. The config name is instruction_tuning AND
        2. The dataset name is in the instruction tuning dataset map AND
        3. The split is either DatasetSplits.TRAIN or DatasetSplits.VALIDATION.
        """
        return (
            config_name == DatasetNames.instruction_tuning.value
            and self.config.instruction_tuning_datasets.is_in_instruction_tuning_dataset(
                dataset_name
            )
            and split_name in {DatasetSplits.TRAIN, DatasetSplits.VALIDATION}
        )

    def _should_add_ai2d(self, config_name: str, split_name: str) -> bool:
        """Check if should add the AI2D dataset.

        AOKVQA should be added if:
        1. Downstream: the config name is aok_vqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.ai2d.value and split_in_dataset

        from_task = config_name == Task.m_vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.ai2d.value, split_name
            )
        )

    def _should_add_aokvqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the aokvqa dataset.

        AOKVQA should be added if:
        1. Downstream: the config name is aok_vqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.aok_vqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.aok_vqa.value, split_name
            )
        )

    def _should_add_coco(self, config_name: str, split_name: str) -> bool:
        """Check if should add the vqa dataset.

        COCO should be added if:
        1. Downstream: the config name is coco.
        2. Task-specific: the config name is captioning.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.coco.value and split_in_dataset

        from_task = config_name == Task.captioning.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.coco.value, split_name
            )
        )

    def _should_add_docvqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the docvqa dataset.

        DocVQA should be added if:
        1. Downstream: the config name is docvqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.docvqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.docvqa.value, split_name
            )
        )

    def _should_add_gqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the GQA dataset.

        GQA should be added if:
        1. Downstream: the config name is gqa.
        2. Task-specific: the config name is gqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST_DEV,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.gqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.gqa.value, split_name
            )
        )

    def _should_add_grit(self, config_name: str, split_name: str) -> bool:
        """Check if should add the Grit dataset.

        Grit should be added if:
        1. Downstream: the config name is grit.
        2. Task-specific: the config name is visual_grounding.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name == DatasetSplits.TRAIN

        from_name = config_name == DatasetNames.grit.value and split_in_dataset

        from_task = config_name == Task.visual_grounding.value and split_in_dataset

        # Unlike other instruction tuning datasets, GRIT has only a train split.
        from_instruction_tuning = (
            self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.grit.value, split_name
            )
            and split_in_dataset
        )

        return from_name or from_task or from_instruction_tuning

    def _should_add_llava_pretrain(self, config_name: str, split_name: str) -> bool:
        """Check if should add the llava pretrain dataset.

        LLava should be added if:
        1. The config name is gqa and the split is train (the only split available).
        2. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name == DatasetSplits.TRAIN

        from_name = config_name == DatasetNames.llava_pretrain.value and split_in_dataset

        return from_name

    def _should_add_llava_instruct(self, config_name: str, split_name: str) -> bool:
        """Check if should add the llava instruct dataset.

        LLava should be added if:
        1. The config name is gqa and the split is train (the only split available).
        2. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name == DatasetSplits.TRAIN

        from_name = config_name == DatasetNames.llava_instruct.value and split_in_dataset

        # Unlike other instruction tuning datasets, LLaVA instruct has only a train split.
        from_instruction_tuning = (
            self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.grit.value, split_name
            )
            and split_in_dataset
        )

        return from_name or from_instruction_tuning

    def _should_add_localized_narratives(self, config_name: str, split_name: str) -> bool:
        """Check if should add the localized narratives dataset.

        Localized Narratives should be added if:
        1. Downstream: the config name is localized_narratives.
        2. Task-specific: the config name is controlled captioning.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
        }

        from_name = config_name == DatasetNames.localized_narratives.value and split_in_dataset

        from_task = config_name == Task.controlled_captioning.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.localized_narratives.value, split_name
            )
        )

    def _should_add_nocaps(self, config_name: str, split_name: str) -> bool:
        """Check if should add the nocaps dataset.

        NoCaps should be added if:
        1. Downstream: the config name is nocaps.
        2. Task-specific: the config name is captioning.
        """
        split_in_dataset = split_name in {
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.nocaps.value and split_in_dataset

        return from_name

    def _should_add_infographicvqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the infographicvqa dataset.

        InfographicVQA should be added if:
        1. Downstream: the config name is infographicvqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.infographics_vqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.infographics_vqa.value, split_name
            )
        )

    def _should_add_ocr_vqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the ocr vqa dataset.

        OCR VQA should be added if:
        1. Downstream: the config name is ocr_vqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.ocr_vqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.ocr_vqa.value, split_name
            )
        )

    def _should_add_refcoco_version_for_task(
        self, config_name: str, split_name: str, version: str, task: str
    ) -> bool:
        """Check if should add a RefCOCO.

        A RefCOCO should be added if:
        1. Downstream (VG): The config name is one of the refcoco datasets (refcoco, refcoco+, refcocog).
        2. Task-specific: The config name is either visual grounding or dense captioning
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
            DatasetSplits.TEST_DEV,
            DatasetSplits.TEST_STD,
        }

        from_name = (
            config_name == version and split_in_dataset and task == Task.visual_grounding.value
        )

        from_task = config_name == task and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(config_name, version, split_name)
        )

    def _should_add_pope(self, config_name: str, split_name: str) -> bool:
        """Check if should add the pope dataset.

        POPE should be added if:
        1. Downstream: the config name is pope_adversarial, pope_popular or pope_random.
        """
        split_in_dataset = split_name == DatasetSplits.TEST

        from_name = (
            config_name
            in {
                DatasetNames.pope_adversarial.value,
                DatasetNames.pope_popular.value,
                DatasetNames.pope_random.value,
            }
            and split_in_dataset
        )

        return from_name

    def _should_add_text_caps(self, config_name: str, split_name: str) -> bool:
        """Check if should add the text_caps dataset.

        TextCaps should be added if:
        1. Downstream: the config name is text_caps.
        2. Task-specific: the config name is caps.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.text_caps.value and split_in_dataset

        from_task = config_name == Task.captioning.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.text_caps.value, split_name
            )
        )

    def _should_add_text_vqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the text_vqa dataset.

        TextVQA should be added if:
        1. Downstream: the config name is text_vqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.text_vqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.text_vqa.value, split_name
            )
        )

    def _should_add_visual7w(self, config_name: str, split_name: str) -> bool:
        """Check if should add the visual7w dataset.

        Visual7W should be added if:
        1. Downstream: the config name is visual7w.
        2. Task-specific: the config name is visual_grounding.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.visual7w.value and split_in_dataset

        from_task = config_name == Task.visual_grounding.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.visual7w.value, split_name
            )
        )

    def _should_add_visual_genome(self, config_name: str, split_name: str) -> bool:
        """Check if should add the visual genome dataset.

        Visual Genome should be added if:
        1. Downstream: the config name is visual_genome.
        2. Task-specific: the config name is visual_grounding, dense_captioning or vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
        }

        from_name = config_name == DatasetNames.visual_genome.value and split_in_dataset
        from_task = (
            config_name
            in {
                Task.visual_grounding.value,
                Task.dense_captioning.value,
                Task.vqa.value,
                Task.relationship_detection.value,
                Task.object_detection.value,
            }
            and split_in_dataset
        )

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.visual_genome.value, split_name
            )
        )

    def _should_add_vizwiz_vqa(self, config_name: str, split_name: str) -> bool:
        """Check if should add the text_vqa dataset.

        TextVQA should be added if:
        1. Downstream: the config name is text_vqa.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.vizwiz_vqa.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.vizwiz_vqa.value, split_name
            )
        )

    def _should_add_vqav2(self, config_name: str, split_name: str) -> bool:
        """Check if should add the vqa dataset.

        VQA should be added if:
        1. Downstream: the config name is vqa_v2.
        2. Task-specific: the config name is vqa.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST_DEV,
            DatasetSplits.TEST_STD,
        }

        from_name = config_name == DatasetNames.vqa_v2.value and split_in_dataset

        from_task = config_name == Task.vqa.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.vqa_v2.value, split_name
            )
        )

    def _should_add_vsr(self, config_name: str, split_name: str) -> bool:
        """Check if should add the vsr dataset.

        VSR should be added if:
        1. Downstream: the config name is vsr.
        2. Task-specific: the config name is itm.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.vsr.value and split_in_dataset

        from_task = config_name == Task.itm.value and split_in_dataset

        return (
            from_name
            or from_task
            or self._add_dataset_for_instruction_tuning(
                config_name, DatasetNames.vsr.value, split_name
            )
        )

    def _should_add_synthetic_vg(self, config_name: str, split_name: str) -> bool:
        """Check if should add the synthetic vg dataset.

        Synthetic VG should be added if:
        1. Downstream: the config name is synthetic_vg.
        2. Task-specific: the config name is visual_grounding.
        3. Instruction tuning: the config name is instruction_tuning and the dataset is used for that.
        """
        split_in_dataset = split_name in {
            DatasetSplits.TRAIN,
            DatasetSplits.VALIDATION,
            DatasetSplits.TEST,
        }

        from_name = config_name == DatasetNames.synthetic_vg.value and split_in_dataset

        from_task = config_name == Task.synthetic_vg.value and split_in_dataset

        return from_name or from_task

    def _build_loaders(  # noqa: C901, WPS231, WPS213
        self, dl_manager: datasets.DownloadManager, split_name: str
    ) -> list[BaseLoader]:
        loaders: list[BaseLoader] = []
        logger.info(f"Creating loaders for {self.config.name} {split_name}")

        if self._should_add_llava_pretrain(self.config.name, split_name):
            loaders.append(
                LLAVAPretrainLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.llava_pretrain_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_llava_instruct(self.config.name, split_name):
            loaders.append(
                LLAVAInstructLoader(
                    split=split_name,
                    # Since the instruct dataset is composed of coco images use the same cache directory.
                    # This will download the LLaVA instruct annotations in the coco cache
                    # but we will save alot of space by not downloading the images again.
                    cache_dir=self.config.dataset_paths.coco_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_vqav2(self.config.name, split_name):
            loaders.append(
                VQAv2Loader(
                    split=split_name,
                    # Since the instruct dataset is composed of coco images use the same cache directory.
                    # This will download the VQAv2 annotations in the coco cache
                    # but we will save alot of space by not downloading the images again.
                    cache_dir=self.config.dataset_paths.coco_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                    max_annotations_per_image=self.config.vqa_max_annotations_per_image,
                )
            )

        if self._should_add_vsr(self.config.name, split_name):
            loaders.append(
                VSRLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.vsr_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                    dataset_name=self.config.vsr_version,
                )
            )

        if self._should_add_gqa(self.config.name, split_name):
            loaders.append(
                GQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.gqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                    max_annotations_per_image=self.config.gqa_max_annotations_per_image,
                )
            )

        if self._should_add_grit(self.config.name, split_name):
            if not Path(self.config.dataset_paths.grit_cache_dir, "images").exists():
                raise ValueError("The images folder is not found. Please download the images first.")

            images_folder = Path(self.config.dataset_paths.grit_cache_dir, "images")
            for image_shard in images_folder.iterdir():
                loaders.append(
                    GRITLoader(
                        split=split_name,
                        cache_dir=image_shard,
                        writer_batch_size=self.config.writer_batch_size,
                        chunk_size=self.config.chunk_size,
                        num_proc=self.config.num_proc,
                    )
                )

        if self._should_add_ai2d(self.config.name, split_name):
            loaders.append(
                AI2DLoader(
                    split=split_name,
                    source=DatasetNames.ai2d.value,
                    cache_dir=self.config.dataset_paths.ai2d_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_aokvqa(self.config.name, split_name):
            loaders.append(
                AOKVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.aok_vqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_pope(self.config.name, split_name):
            loaders.append(
                POPELoader(
                    split=split_name,
                    source=self.config.name,
                    cache_dir=self.config.dataset_paths.pope_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_ocr_vqa(self.config.name, split_name):
            loaders.append(
                OCRVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.ocr_vqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_coco(self.config.name, split_name):
            loaders.append(
                COCOLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.coco_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_docvqa(self.config.name, split_name):
            loaders.append(
                DocVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.docvqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_text_vqa(self.config.name, split_name):
            loaders.append(
                TextVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.textvqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_text_caps(self.config.name, split_name):
            loaders.append(
                TextCapsLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.textcaps_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        # Check first if we need to add a refcoco dataset by name.
        # By default only add the visual grounding task
        if self.config.name in self.config.use_refcoco_datasets:
            should_add = self._should_add_refcoco_version_for_task(
                config_name=self.config.name,
                split_name=split_name,
                version=self.config.name,
                task=Task.visual_grounding.value,
            )

            if should_add:
                loaders.append(
                    RefCOCOLoader(
                        split=split_name,
                        source=self.config.name,
                        cache_dir=self.config.dataset_paths.refcoco_cache_dir,
                        task=Task.visual_grounding.value,
                        num_proc=self.config.num_proc,
                        writer_batch_size=self.config.writer_batch_size,
                        chunk_size=self.config.chunk_size,
                    )
                )
        else:
            for version in self.config.use_refcoco_datasets:
                for refcoco_task in self.config.refcoco_tasks:
                    should_add = self._should_add_refcoco_version_for_task(
                        self.config.name, split_name, version, refcoco_task
                    )
                    if should_add:
                        loaders.append(  # noqa: WPS220
                            RefCOCOLoader(
                                split=split_name,
                                source=version,
                                cache_dir=self.config.dataset_paths.refcoco_cache_dir,
                                task=refcoco_task,
                                num_proc=self.config.num_proc,
                                writer_batch_size=self.config.writer_batch_size,
                                chunk_size=self.config.chunk_size,
                            )
                        )

        if self._should_add_visual7w(self.config.name, split_name):
            loaders.append(
                Visual7WLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.visual7w_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                    max_questions_per_image_in_turn=self.config.visual7w_max_questions_per_image_in_turn,
                )
            )

        if self._should_add_visual_genome(self.config.name, split_name):
            # Add a single visual genome task
            if self.config.name in self.config.visual_genome_tasks:
                loaders.append(
                    VisualGenomeLoader(
                        split=split_name,
                        task=self.config.name,
                        num_proc=self.config.num_proc,
                        dl_manager=dl_manager,
                        cache_dir=self.config.dataset_paths.visual_genome_cache_dir,
                        datasets_batch_size=self.config.datasets_batch_size,
                        chunk_size=self.config.chunk_size,
                        max_annotations_per_image=self.config.visual_genome_max_annotations_per_image,
                    )
                )
            # Add all visual genome tasks
            else:
                for vg_task in self.config.visual_genome_tasks:
                    loaders.append(
                        VisualGenomeLoader(
                            split=split_name,
                            task=vg_task,
                            num_proc=self.config.num_proc,
                            dl_manager=dl_manager,
                            cache_dir=self.config.dataset_paths.visual_genome_cache_dir,
                            datasets_batch_size=self.config.datasets_batch_size,
                            chunk_size=self.config.chunk_size,
                            max_annotations_per_image=self.config.visual_genome_max_annotations_per_image,
                        )
                    )

        if self._should_add_vizwiz_vqa(self.config.name, split_name):
            loaders.append(
                VizWizVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.vizwiz_vqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_localized_narratives(self.config.name, split_name):
            loaders.append(
                LocalizedNarrativesLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.localized_narratives_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                    segment_window_size=self.config.localized_narratives_segment_window_size,
                )
            )

        if self._should_add_nocaps(self.config.name, split_name):
            loaders.append(
                NocapsLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.nocaps_cache_dir,
                    datasets_batch_size=self.config.datasets_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_infographicvqa(self.config.name, split_name):
            loaders.append(
                InfographicVQALoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.infographicsvqa_cache_dir,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if self._should_add_synthetic_vg(self.config.name, split_name):
            loaders.append(
                SyntheticVGLoader(
                    split=split_name,
                    cache_dir=self.config.dataset_paths.synthetic_vg_cache_dir,
                    seq_len=self.config.synthetic_vg_seq_len,
                    writer_batch_size=self.config.writer_batch_size,
                    chunk_size=self.config.chunk_size,
                    num_proc=self.config.num_proc,
                )
            )

        if loaders:
            logger.info(f"Found {len(loaders)} loaders")
            for loader in loaders:
                info_msg = f"Loader: {loader}, {loader.source}"
                task = getattr(loader, "task", None)
                if task is not None:
                    info_msg = f"{info_msg}, {loader.task}"
                logger.info(info_msg)
        else:
            logger.info(f"No loaders found for {self.config.name}, {split_name}")
        return loaders

    def _generate_tables(self, loaders: list[BaseLoader]) -> Iterator[tuple[int, Any]]:
        idx = 0
        for loader in loaders:
            for elt in loader._generate_batches():  # noqa: WPS437
                yield idx, elt
                idx += 1

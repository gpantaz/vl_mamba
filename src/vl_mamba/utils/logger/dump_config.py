from collections.abc import Sequence
from typing import Optional

import rich
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import StyleType, Tree


DEFAULT_SEQUENCE_ORDER = (
    "trainer",
    "model",
    "datamodule",
    "callbacks",
    "logger",
    "test_after_training",
    "seed",
    "name",
)

TREE_STYLE: StyleType = "dim"


@rank_zero_only
def dump_config(
    config: DictConfig,
    fields: Sequence[str] = DEFAULT_SEQUENCE_ORDER,
    resolve: bool = True,
    save_path: Optional[str] = "config_tree.log",
) -> None:
    """Prints content of DictConfig.

    Args:
        config (DictConfig): Config from Hydra
        fields (Sequence[str]): Which fields to print from the config, and in what
            order. Defaults to DEFAULT_SEQUENCE_ORDER.
        resolve (bool): Whether to resolve reference fields of config. Defaults to True.
        save_path (Optional[str]): Save path for tree. Defaults to "config_tree.log".
    """
    tree = Tree("CONFIG", style="dim", guide_style="dim")

    for field_name in fields:
        field_config = config.get(field_name)

        if isinstance(field_config, DictConfig):
            branch_content = OmegaConf.to_yaml(field_config, resolve=resolve)
        else:
            branch_content = str(field_config)

        branch = tree.add(field_name)
        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_path is not None:
        with open(save_path, "w") as save_file:
            rich.print(tree, file=save_file)

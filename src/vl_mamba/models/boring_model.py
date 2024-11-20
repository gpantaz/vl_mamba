from typing import Any

from overrides import overrides
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AdamW, AutoModelForMaskedLM


class BoringModel(LightningModule):
    """Example of LightningModule for training an Huggingface Transformers model.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Model forward (forward).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    """

    def __init__(self, model_name: str, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

    @overrides(check_signature=False)
    def forward(self, batch: Any) -> Any:
        """Used for inference only."""
        return self._model(**batch)

    @overrides(check_signature=False)
    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Full training loop."""
        outputs = self.forward(batch)

        self.log("train_loss", outputs.loss, on_step=True, on_epoch=True)

        return outputs.loss

    def configure_optimizers(self) -> Any:
        """Define optimizers and LR schedulers."""
        no_decay = ["bias", "LayerNorm.weight"]

        parameters_with_weight_decay = [
            parameter
            for name, parameter in self._model.named_parameters()
            if name not in no_decay and parameter.requires_grad
        ]

        parameters_without_weight_decay = [
            parameter
            for name, parameter in self._model.named_parameters()
            if name in no_decay and parameter.requires_grad
        ]

        grouped_parameters = [
            {
                "params": parameters_with_weight_decay,
                "weight_decay": self.hparams.get("weight_decay"),
            },
            {
                "params": parameters_without_weight_decay,
                "weight_decay": 0,
            },
        ]

        optimizer = AdamW(grouped_parameters)
        return optimizer

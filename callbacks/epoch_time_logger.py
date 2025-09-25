# %%
from lightning.pytorch import Callback, Trainer, LightningModule
import time
from typing import Any


# %%
class EpochTimeLogger(Callback):
    def __init__(self, log_every_n_epochs: int = 1, prog_bar: bool = True):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.prog_bar = prog_bar
        self.epoch_start_time = -1.0
        self.epoch_end_time = -1.0
        self.epoch_time_sum = 0.0
        self.epoch_time_count = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self.epoch_start_time = time.time()
            self.epoch_time_count += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self.epoch_end_time = time.time()
            epoch_duration = self.epoch_end_time - self.epoch_start_time
            pl_module.log("epoch_duration", epoch_duration)

            self.epoch_time_sum += epoch_duration
            pl_module.log(
                "epoch_time_mean",
                self.epoch_time_sum / self.epoch_time_count,
                prog_bar=self.prog_bar,
            )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "epoch_start_time": self.epoch_start_time,
            "epoch_end_time": self.epoch_end_time,
            "epoch_time_sum": self.epoch_time_sum,
            "epoch_time_count": self.epoch_time_count,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epoch_start_time = state_dict["epoch_start_time"]
        self.epoch_end_time = state_dict["epoch_end_time"]
        self.epoch_time_sum = state_dict["epoch_time_sum"]
        self.epoch_time_count = state_dict["epoch_time_count"]

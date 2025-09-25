# %%
from lightning.pytorch import Callback, Trainer, LightningModule
import time
from typing import Any, Optional


# %%
class IterationFrequencyLogger(Callback):
    def __init__(self, log_every_n_steps: Optional[int] = None, prog_bar: bool = False):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.prog_bar = prog_bar
        self.step_start_time = -1.0
        self.step_end_time = -1.0
        self.step_time_sum = 0.0
        self.step_time_count = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_every_n_steps is None:
            self.log_every_n_steps = int(trainer.log_every_n_steps)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if batch_idx % self.log_every_n_steps == 0:
            self.step_start_time = time.time()
            self.step_time_count += 1

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_every_n_steps == 0:
            self.step_end_time = time.time()
            step_duration = self.step_end_time - self.step_start_time
            self.step_time_sum += step_duration

            pl_module.log("iteration_per_second", 1 / step_duration, prog_bar=self.prog_bar)
            pl_module.log(
                "iteration_per_second_mean",
                self.step_time_count / self.step_time_sum,
                prog_bar=self.prog_bar,
            )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "step_start_time": self.step_start_time,
            "step_end_time": self.step_end_time,
            "step_time_sum": self.step_time_sum,
            "step_time_count": self.step_time_count,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_start_time = state_dict["step_start_time"]
        self.step_end_time = state_dict["step_end_time"]
        self.step_time_sum = state_dict["step_time_sum"]
        self.step_time_count = state_dict["step_time_count"]

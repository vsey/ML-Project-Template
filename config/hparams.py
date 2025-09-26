from pydantic import BaseModel, Field
from typing import Type, Optional, Union, Literal
from lightning.pytorch import LightningDataModule
from torch.optim import Optimizer, Adam
import pathlib

SupportedLoggers = Literal["tensorboard"]

class DataModuleConfig(BaseModel):
    """Configuration for the DataModule."""

    model_config = {"arbitrary_types_allowed": True}  # Needed to allow a class type as a field

    datamodule_class: Type[LightningDataModule] = 5
    # These are the arguments that will be passed to the datamodule
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


class OptimizerConfig(BaseModel):
    """Configuration for the Optimizer."""

    optimizer_class: Type[Optimizer] = Adam
    lr: float = 1e-3


class TensorboardConfig(BaseModel):
    """Configuration for Tensorboard Logger."""

    log_graph: bool = False


class TrainerConfig(BaseModel):
    """Configuration for the Trainer."""

    max_epochs: int = 150
    precision: Optional[str] = None


class MainConfig(BaseModel):
    experiment_name: str = "default_experiment"
    log_root_dir: Union[str, pathlib.Path] = "./logs"

    loggers: Optional[list[SupportedLoggers]] = None

    #     # --- Load Checkpoint ---
    #     checkpoint_path: Optional[str] = None

    data_params: DataModuleConfig = Field(default_factory=DataModuleConfig)
    optimizer_params: OptimizerConfig = Field(default_factory=OptimizerConfig)
    tensorboard_params: TensorboardConfig = Field(default_factory=TensorboardConfig)
    trainer_params: TrainerConfig = Field(default_factory=TrainerConfig)




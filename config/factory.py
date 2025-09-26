from typing import Optional, Union
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Optimizer
from .hparams import MainConfig


class ConfigFactory:
    def __init__(self, config: MainConfig) -> None:
        self.config = config

    def get_datamodule(self, **kwargs) -> LightningDataModule:
        """Factory that builds and returns an instance of the specified datamodule class."""

        hparams = self.config.data_params

        # Update params with kwargs if present
        if kwargs:
            hparams = hparams.model_copy(update=kwargs)

        # Get a dictionary of all arguments excluding the class type
        args = hparams.model_dump(exclude={"datamodule_class"})

        # Instantiate the class by unpacking the arguments dictionary
        return hparams.datamodule_class(**args)

    def get_optimizer(self, parameters, **kwargs) -> Optimizer:
        """Factory that builds and returns an instance of the specified optimizer class."""

        hparams = self.config.optimizer_params

        # Update params with kwargs if present
        if kwargs:
            hparams = hparams.model_copy(update=kwargs)

        # Get a dictionary of all arguments excluding the class type
        args = hparams.model_dump(exclude={"optimizer_class"})

        # Instantiate the class by unpacking the arguments dictionary
        return hparams.optimizer_class(parameters, **args)

    def _get_tensorboard(self) -> TensorBoardLogger:
        """Factory that builds and returns an instance of Tensorboard Logger."""

        # Get the validated, specific arguments from our Pydantic model.
        logger_args = self.config.tensorboard_params.model_dump()

        return TensorBoardLogger(
            save_dir=str(self.config.log_root_dir),
            name=TensorBoardLogger.__name__,
            version=self.config.experiment_name,
            **logger_args,
        )

    def get_loggers(self) -> Optional[list[Union[TensorBoardLogger]]]:
        """Factory that builds and returns a List of instances of all specified loggers."""

        if self.config.loggers is None:
            return None

        loggers = []
        for logger_type in self.config.loggers:
            if logger_type == "tensorboard":
                loggers.append(self._get_tensorboard())
            else:
                raise ValueError(f"Unsupported logger_type: {logger_type}")
        return loggers

    def get_trainer(self, **kwargs) -> Trainer:
        """Factory that builds the trainer and correctly wires up the logger."""

        params = self.config.trainer_params

        if kwargs:
            params = params.model_copy(update=kwargs)

        args = params.model_dump()
        args["logger"] = self.get_loggers()
        return Trainer(**args)

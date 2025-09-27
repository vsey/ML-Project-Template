# %%
from lightning.pytorch import LightningDataModule
import pathlib
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from typing import Union, Any, Optional
from pydantic.types import PositiveInt


# %%
class MNISTDM(LightningDataModule):
    def __init__(
        self,
        root_path: Union[str, pathlib.Path] = "./datasets",
        batch_size: PositiveInt = 32,
        num_workers: PositiveInt = 4,
        pin_memory: bool = True,
        mean: tuple[float] = (0.1307,),
        std: tuple[float] = (0.3079,),
        shape: tuple[int, int, int] = (1, 28, 28),
        transforms: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.path = pathlib.Path(root_path) / str(self.__class__.__name__)

    def prepare_data(self) -> None:
        MNIST(root=self.path, train=True, download=True)
        MNIST(root=self.path, train=False, download=True)

    def setup(self, stage: str = None):
        transforms = self.hparams.transforms

        self.train_ds = MNIST(
            root=self.path, train=True, download=False, transform=transforms
        )
        self.val_ds = MNIST(
            root=self.path, train=False, download=False, transform=transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

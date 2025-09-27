from datamodules.mnist import MNISTDM
from pydantic import BaseModel
from hydra_zen import builds, make_config, instantiate
from torchvision.transforms import Normalize, ToTensor, Compose


class DataConf(BaseModel):
    mean: tuple[float] = (0.2,)
    std: tuple[float] = (0.5,)
    shape: tuple[int, int, int] = (1, 28, 28)


to_tensor = builds(ToTensor)
normalize = builds(Normalize, mean="${data.mean}", std="${data.std}")
compose = builds(Compose, transforms=[to_tensor, normalize])

MNISTDM_Conf = builds(
    MNISTDM,
    populate_full_signature=True,
    # Link to other configs
    mean="${data.mean}",
    std="${data.std}",
    shape="${data.shape}",
    transforms=compose,
    # Set dataloader params
    batch_size=64,
)


ExperimentConf = make_config(
    data=DataConf(),
    datamodule=MNISTDM_Conf(),
)

# print(to_yaml(ExperimentConf))

dm = instantiate(ExperimentConf)


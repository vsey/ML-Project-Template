from datamodules.mnist import MNISTDM
from hydra_zen import builds, make_config, instantiate, to_yaml, make_custom_builds_fn
from torchvision.transforms import Normalize, ToTensor, Compose
import torch.optim as optim

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


data_conf = make_config(mean=(0.13,), std=(0.3,), shape=(1, 28, 28))


to_tensor = builds(ToTensor)
normalize = builds(Normalize, mean="${data.mean}", std="${data.std}")
compose = builds(Compose, transforms=[to_tensor, normalize])

datamodule_conf = builds(
    MNISTDM,
    populate_full_signature=True,
    # Link to other configs
    transforms=compose,
    # Set dataloader params
    batch_size=128,
)


# If you fancy types
# Caution needs type annotations otherwise fails silently
# @hydrated_dataclass(target=MNISTDM,  populate_full_signature=True)
# class MNISTDM_Conf:
#     # Link to other configs
#     mean: tuple[float] = "${data.mean}"
#     std: tuple[float, ] = "${data.std}"
#     shape: tuple[int, int, int] = "${data.shape}"
#     transforms: Any = compose
#     # Set dataloader params
#     batch_size: int = 64

optimizer_conf = pbuilds(optim.AdamW, lr=0.3)

ExperimentConf = make_config(
    defaults=["_self_"],
    data=data_conf,
    datamodule=datamodule_conf,
    optimizer=optimizer_conf,
)

print(to_yaml(ExperimentConf))


experimental_conf = instantiate(ExperimentConf)

dm = experimental_conf.datamodule

dm.prepare_data()
dm.setup()

dl = dm.train_dataloader()

optimizer = experimental_conf.optimizer
print(optimizer)

# running_mean = 0
# running_std = 0
# count = 0
# for x, y in dl:
#     count += 1
#     running_mean += x.mean()
#     running_std += x.std()

# print(running_mean / count)
# print(running_std / count)

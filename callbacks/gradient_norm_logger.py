# %%
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger


# %%
class GradientNormLogger(Callback):
    def __init__(self, log_hist: bool = False):
        super().__init__()
        self.log_hist = log_hist

    def on_after_backward(self, trainer, pl_module):
        # Log gradient norms for each parameter
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                # log grad_norm logger independent as line plot
                pl_module.log(f"grad_norm_{name}", grad_norm)

                # if logger is tb_logger and log hist true log histogram of grad
                if isinstance(trainer.logger, TensorBoardLogger) and self.log_hist:
                    trainer.logger.experiment.add_histogram(
                        f"grad_norm_{name}_hist", grad_norm
                    )

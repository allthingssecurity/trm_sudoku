from lightning import Callback


class ResetCarryCallback(Callback):
    """Resets stateful carry on epoch boundaries to improve stability."""

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module, "carry"):
            pl_module.carry = None


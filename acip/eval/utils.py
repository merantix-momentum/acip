from contextlib import contextmanager

from torch import nn


@contextmanager
def eval_mode(model: nn.Module):
    """
    Context manager to put a model in eval mode.
    """
    was_training = model.training
    if was_training:
        model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()

from logging import getLogger

import torch

logger = getLogger(__name__)


def clear_cache(cuda: bool = True, gc: bool = False):
    if cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if gc:
        import gc

        gc.collect()
    logger.debug("Cache cleared.")

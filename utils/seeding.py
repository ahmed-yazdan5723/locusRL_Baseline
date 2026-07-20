"""Central place to control randomness so results are reproducible."""
import random


def set_global_seed(seed: int) -> None:
    """Seed every RNG source we currently use. Extend this if a new
    agent/adapter pulls in numpy/torch — add the seeding call here only,
    nothing else in the codebase should call random.seed() directly.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

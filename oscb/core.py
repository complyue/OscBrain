__all__ = [
    "random_seed",
    "SYNAPSE_LINK_DTYPE",
]

import numpy as np
import pandas as pd
from numba import njit


SYNAPSE_LINK_DTYPE = np.dtype(
    [
        ("i0", "uint32"),  # flatterned index of presynaptic cell
        ("i1", "uint32"),  # flatterned index of postsynaptic cell
    ],
    align=True,
)


@njit
def random_seed(seed):
    """
    Numba has PRNG states independent of that of Numpy, we use numba's PRNG exclusively,
    call this before a batch for reproducible result.
    """
    np.random.seed(seed)

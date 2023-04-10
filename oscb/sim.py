__all__ = [
    "LetterNetSim",
]

import numpy as np
import pandas as pd
from numba import njit


from .core import *


class LetterNetSim:
    """
    Simulator with a LetterNet

    """

    def __init__(self, lnet: LetterNet):
        self.lnet = lnet

        N_COLS = lnet.ALPHABET_SIZE * lnet.N_SPARSE_COLS_PER_LETTER

        # cell potentials
        self.cell_potns = np.zeros((N_COLS, lnet.N_CELLS_PER_COL), "f4")
        # last timestep number each cell ever fired at
        self.cell_ftns = np.full((N_COLS, lnet.N_CELLS_PER_COL), -1, "int32")

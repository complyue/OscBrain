__all__ = [
    "LetterNetSim",
]

import numpy as np
import pandas as pd
from numba import njit


from .core import *
from .lnet import *


class LetterNetSim:
    """
    Simulator with a LetterNet

    """

    def __init__(
        self,
        # LetterNet, unit synaptic efficacy assumed to be valued 1.0
        lnet: LetterNet,
        # rest voltage, simplified to be 0.0
        VOLT_REST=0.0,
        # threshold voltage for a spike, simplified to be 1.0
        SPIKE_THRES=1.0,
        # reset voltage, equal to VOLT_REST, or lower to enable refractory period
        VOLT_RESET=-0.1,
        # membrane time constant
        τ_m=10,
        # global scaling factor, to accommodate a unit synaptic efficacy value of 1.0
        # roughly this specifies that:
        #   how many presynaptic spikes is enough to trigger a postsynaptic spike,
        #   when each synapse has a unit efficacy value of 1.0
        SYNAP_FACTOR=500,
    ):
        self.lnet = lnet
        self.VOLT_REST = VOLT_REST
        self.SPIKE_THRES = SPIKE_THRES
        self.VOLT_RESET = VOLT_RESET
        self.τ_m = τ_m
        self.SYNAP_FACTOR = SYNAP_FACTOR

        # cell voltage, ranging
        self.cell_volts = np.full(lnet.CELLS_SHAPE, VOLT_REST, "f4")

        # timestep
        self.ts = 0


@njit
def _simulate_lnet(
    n_steps,  # total number of time steps to simulate
    cell_volts,  # initial neuron states
    excit_links,
    excit_effis,
    inhib_links,
    inhib_effis,
    sdr_indices,  # letter SDR indices
    prompt_lcodes,  # letter code sequence
    prompt_col_density,  # how many columns per letter to spike
    prompt_cel_density,  # how many cells per column to spike
    prompt_pace=0,  # time (in steps) gap between letters
    # rest voltage, simplified to be 0.0
    VOLT_REST=0.0,
    # threshold voltage for a spike, simplified to be 1.0
    SPIKE_THRES=1.0,
    # reset voltage, equal to VOLT_REST, or lower to enable refractory period
    VOLT_RESET=-0.1,
    # membrane time constant
    τ_m=10,
    # global scaling factor, to facilitate a unit synaptic efficacy value of 1.0
    # roughly this specifies that:
    #   how many presynaptic spikes is enough to trigger a postsynaptic spike,
    #   when each synapse has a unit efficacy value of 1.0
    #   todo: justify, this is sorta global inhibition?
    SYNAP_FACTOR=500,
):
    N_COLS, N_CELLS_PER_COL = cell_volts.shape

    return

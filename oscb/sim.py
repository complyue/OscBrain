__all__ = [
    "LetterNetSim",
]

import numpy as np
import pandas as pd
from numba import njit


from .core import *
from .lnet import *

from . import bkh


class LetterNetSim:
    """
    Simulator with a LetterNet

    """

    # the ratio of minicolumn span, on the x-axis of plot, against a unit time step
    COL_PLOT_WIDTH = 0.8

    def __init__(
        self,
        # LetterNet, unit synaptic efficacy assumed to be valued 1.0
        lnet: LetterNet,
        # global scaling factor, to accommodate a unit synaptic efficacy value of 1.0
        # roughly this specifies that:
        #   how many presynaptic spikes is enough to trigger a postsynaptic spike,
        #   when each incoming firing synapse has a unit efficacy value of 1.0
        SYNAP_FACTOR=5,
        # rest voltage, simplified to be 0.0
        VOLT_REST=0.0,
        # threshold voltage for a spike, simplified to be 1.0
        SPIKE_THRES=1.0,
        # reset voltage, equal to VOLT_REST, or lower to enable refractory period
        VOLT_RESET=-0.1,
        # membrane time constant
        τ_m=10,
        # fire plot params
        plot_width=800,
        plot_height=600,
        plot_n_steps=100,
        fire_dots_glyph="square",
        fire_dots_alpha=0.01,
        fire_dots_size=3,
        fire_dots_color="#0000FF",
    ):
        self.lnet = lnet
        self.VOLT_REST = VOLT_REST
        self.SPIKE_THRES = SPIKE_THRES
        self.VOLT_RESET = VOLT_RESET
        self.τ_m = τ_m
        self.SYNAP_FACTOR = SYNAP_FACTOR

        self.cell_volts = np.full(lnet.CELLS_SHAPE, VOLT_REST, "f4")

        self.done_n_steps = 0
        self.ds_spikes = bkh.ColumnDataSource(
            {
                "x": [],
                "y": [],
            }
        )

        p = bkh.figure(
            title="Letter SDR Spikes",
            x_axis_label="Time Step",
            y_axis_label="Column (with Letter Spans)",
            width=plot_width,
            height=plot_height,
            tools=[
                "pan",
                "box_zoom",
                "xwheel_zoom",
                # "ywheel_zoom",
                "undo",
                "redo",
                "reset",
                "crosshair",
            ],
            y_range=(0, lnet.CELLS_SHAPE[0]),
            x_range=(0, plot_n_steps),
        )

        label_ys = (
            np.arange(1, lnet.ALPHABET_SIZE + 1) * lnet.N_SPARSE_COLS_PER_LETTER - 1
        )
        p.yaxis.ticker = bkh.FixedTicker(ticks=label_ys)
        p.yaxis.formatter = bkh.CustomJSTickFormatter(
            code=f"""
return {list(lnet.ALPHABET.alphabet())!r}[(tick+1)/{lnet.N_SPARSE_COLS_PER_LETTER}-1];
"""
        )

        p.scatter(
            source=self.ds_spikes,
            marker=fire_dots_glyph,
            alpha=fire_dots_alpha,
            size=fire_dots_size,
            color=fire_dots_color,
        )

        self.fig = p

    def simulate(
        self,
        n_steps,
        prompt_words,  # a single word or list of words to prompt
        prompt_col_density=0.5,  # how many columns per letter to spike
        prompt_cel_density=0.5,  # how many cells per column to spike
        prompt_pace=1,  # time step distance between letter spikes
    ):
        lnet = self.lnet

        _w_bound, w_lcode = lnet.ALPHABET.encode_words(
            [prompt_words] if isinstance(prompt_words, str) else prompt_words
        )

        if isinstance(prompt_col_density, float):
            assert 0 < prompt_col_density <= 1.0
            prompt_col_density = int(prompt_col_density * lnet.N_COLS_PER_LETTER)
        if isinstance(prompt_cel_density, float):
            assert 0 < prompt_cel_density <= 1.0
            prompt_cel_density = int(prompt_cel_density * lnet.N_CELLS_PER_COL)

        spikes, step_n_spikes = _simulate_lnet(
            n_steps,
            self.cell_volts,
            *lnet._excitatory_synapses(),
            *lnet._inhibitory_synapses(),
            lnet.sdr_indices,
            w_lcode,
            prompt_col_density,
            prompt_cel_density,
            prompt_pace,
            self.VOLT_RESET,
            self.SPIKE_THRES,
            self.VOLT_REST,
            self.τ_m,
            self.SYNAP_FACTOR,
        )

        x_base = self.done_n_steps
        ci, ici = np.divmod(spikes, lnet.N_CELLS_PER_COL)
        si = 0
        xs, ys = [], []
        for n_spikes in step_n_spikes:
            if n_spikes > 0:
                next_si = si + n_spikes
                xs.append(
                    x_base
                    + self.COL_PLOT_WIDTH * ici[si:next_si] / lnet.N_CELLS_PER_COL
                )
                ys.append(ci[si:next_si])
                si = next_si
            x_base += 1
        self.done_n_steps = x_base
        self.ds_spikes.stream({"x": np.concatenate(xs), "y": np.concatenate(ys)})


@njit
def _simulate_lnet(
    n_steps,  # total number of time steps to simulate
    # neuron voltages, both as input for initial states, and as output for final states
    cell_volts,
    # excitatory synapse links/efficacies
    excit_links,
    excit_effis,
    # inhibitory synapse links/efficacies
    inhib_links,
    inhib_effis,
    sdr_indices,  # letter SDR indices
    prompt_lcodes,  # letter code sequence
    prompt_col_density,  # how many columns per letter to spike
    prompt_cel_density,  # how many cells per column to spike
    prompt_pace=1,  # time step distance between letter spikes
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
    assert 0 < n_steps <= 20000
    assert excit_links.ndim == excit_effis.ndim == 1
    assert excit_links.shape == excit_effis.shape
    assert inhib_links.ndim == inhib_effis.ndim == 1
    assert inhib_links.shape == inhib_effis.shape
    assert prompt_lcodes.ndim == 1
    assert 0 <= prompt_lcodes.size <= n_steps
    assert 1 <= prompt_pace < n_steps

    N_COLS, N_CELLS_PER_COL = cell_volts.shape
    excit_presynap_ci, excit_presynap_ici = np.divmod(
        excit_links["i0"], N_CELLS_PER_COL
    )
    excit_postsynap_ci, excit_postsynap_ici = np.divmod(
        excit_links["i1"], N_CELLS_PER_COL
    )
    inhib_presynap_ci, inhib_presynap_ici = np.divmod(
        inhib_links["i0"], N_CELLS_PER_COL
    )
    inhib_postsynap_ci, inhib_postsynap_ici = np.divmod(
        inhib_links["i1"], N_CELLS_PER_COL
    )

    def ensure_prompted_spikes(lcode):
        for ci in np.random.choice(
            sdr_indices[lcode], prompt_col_density, replace=False
        ):
            for ici in np.random.choice(
                N_CELLS_PER_COL, prompt_cel_density, replace=False
            ):
                if not (cell_volts[ci, ici] >= SPIKE_THRES):
                    cell_volts[ci, ici] = SPIKE_THRES

    # we serialize the indices of spiked cells as the output record of simulation
    # pre-allocate sufficient capacity to store maximumally possible spike info
    spikes = np.empty(n_steps * cell_volts.size, "int32")
    n_spikes = 0  # total number of individual spikes as recorded
    # record number of spikes per each time step, it may vary across steps
    step_n_spikes = np.zeros(n_steps, "int32")

    # intermediate state data for cell voltages
    cell_volts_tobe = np.empty_like(cell_volts)

    next_prompt_i, last_prompt_ts = 0, -prompt_pace
    for i_step in range(n_steps):
        # apply prompt appropriately, force spikes at beginning of current time step
        if next_prompt_i < prompt_lcodes.size:
            if i_step - last_prompt_ts >= prompt_pace:
                ensure_prompted_spikes(prompt_lcodes[next_prompt_i])
                last_prompt_ts = i_step
                next_prompt_i += 1

        # accumulate input current, according to presynaptic spikes
        cell_volts_tobe[:] = 0
        for i in range(excit_links.size):
            v = cell_volts[excit_presynap_ci[i], excit_presynap_ici[i]]
            if v >= SPIKE_THRES:
                cell_volts_tobe[
                    excit_postsynap_ci[i], excit_postsynap_ici[i]
                ] += excit_effis[i]
        for i in range(inhib_links.size):
            v = cell_volts[inhib_presynap_ci[i], inhib_presynap_ici[i]]
            if v >= SPIKE_THRES:
                cell_volts_tobe[
                    inhib_postsynap_ci[i], inhib_postsynap_ici[i]
                ] -= inhib_effis[i]
        # apply the global scaling factor
        cell_volts_tobe[:] /= SYNAP_FACTOR

        # reset voltage if fired, or update the voltage
        for ci in range(N_COLS):
            for ici in range(N_CELLS_PER_COL):
                v = cell_volts[ci, ici]
                if v >= SPIKE_THRES:
                    # fired, reset
                    cell_volts_tobe[ci, ici] = VOLT_RESET

                    # record the spike
                    spikes[n_spikes] = ci * N_CELLS_PER_COL + ici
                    n_spikes += 1
                    step_n_spikes[i_step] += 1

                else:  # add back previous-voltage, plus leakage
                    # note it's just input-current before this update
                    cell_volts_tobe[ci, ici] += v + (VOLT_REST - v) / τ_m

        # update the final state at end of this time step
        cell_volts[:] = cell_volts_tobe

    assert n_spikes == np.sum(step_n_spikes), "bug?!"
    return (
        # return a copy of valid slice, to release extraneous memory allocated
        spikes[:n_spikes].copy(),
        step_n_spikes,
    )

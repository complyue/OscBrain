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
        # reset voltage, negative to enable refractory period
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
        self.VOLT_RESET = VOLT_RESET
        self.τ_m = τ_m
        self.SYNAP_FACTOR = SYNAP_FACTOR

        self.cell_volts = np.full(lnet.CELLS_SHAPE, 0, "f4")

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
        prompt_blur=0.8,  # reduce voltage of other cells than the prompted letter
    ):
        lnet = self.lnet

        _w_bound, w_lcode = lnet.ALPHABET.encode_words(
            [prompt_words] if isinstance(prompt_words, str) else prompt_words
        )

        spikes, step_n_spikes = _simulate_lnet(
            n_steps,
            self.cell_volts,
            *lnet._excitatory_synapses(),
            *lnet._inhibitory_synapses(),
            lnet.sdr_indices,
            w_lcode,
            prompt_blur,
            self.VOLT_RESET,
            self.τ_m,
            self.SYNAP_FACTOR,
        )

        if spikes.size <= 0:
            return  # no spike at all

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


# threshold voltage for a spike, simplified to be 1.0 globally, as constant
SPIKE_THRES = 1.0


@njit  # (debug=True)
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
    prompt_blur=0.8,  # reduce voltage of other cells than the prompted letter
    VOLT_RESET=-0.1,  # reset voltage, negative to enable refractory period
    τ_m=10,  # membrane time constant
    # global scaling factor, to facilitate a unit synaptic efficacy value of 1.0
    # roughly this specifies that:
    #   how many presynaptic spikes is enough to trigger a postsynaptic spike,
    #   when each synapse has a unit efficacy value of 1.0
    #   todo: justify, this is sorta global inhibition?
    SYNAP_FACTOR=5,
):
    assert 0 < n_steps <= 20000
    assert excit_links.ndim == excit_effis.ndim == 1
    assert excit_links.shape == excit_effis.shape
    assert inhib_links.ndim == inhib_effis.ndim == 1
    assert inhib_links.shape == inhib_effis.shape
    assert prompt_lcodes.ndim == 1
    assert 0 <= prompt_lcodes.size <= n_steps
    assert 0 <= prompt_blur <= 1.0

    # ALPHABET_SIZE, N_COLS_PER_LETTER = sdr_indices.shape
    # N_COLS, N_CELLS_PER_COL = cell_volts.shape
    # cvs be a flattened view of cell_volts
    cvs = cell_volts.ravel()

    def prompt_letter(lcode):
        letter_volts = cell_volts[sdr_indices[lcode], :]

        # suppress all (except those in refractory period) cells first
        cvs[cvs > 0] *= prompt_blur

        if np.any(letter_volts >= SPIKE_THRES):
            # some cell(s) of prompted letter would fire
            # restore letter cell voltages
            cell_volts[sdr_indices[lcode], :] = letter_volts
        else:  # no cell of prompted letter would fire
            # force fire all of the letter's cells
            cell_volts[sdr_indices[lcode], :] = SPIKE_THRES

    # we serialize the indices of spiked cells as the output record of simulation
    # pre-allocate sufficient capacity to store maximumally possible spike info
    spikes = np.empty(n_steps * cvs.size, np.uint32)
    n_spikes = 0  # total number of individual spikes as recorded
    # record number of spikes per each time step, it may vary across steps
    step_n_spikes = np.zeros(n_steps, np.uint32)

    # intermediate state data for cell voltages
    cvs_tobe = np.empty_like(cvs)

    prompt_i = 0
    for i_step in range(n_steps):
        if prompt_i < prompt_lcodes.size:  # apply prompt
            prompt_letter(prompt_lcodes[prompt_i])
            prompt_i += 1

        # accumulate input current, according to presynaptic spikes
        cvs_tobe[:] = 0
        for i in range(excit_links.size):
            v = cvs[excit_links[i]["i0"]]
            if v >= SPIKE_THRES:
                cvs_tobe[excit_links[i]["i1"]] += excit_effis[i]
        for i in range(inhib_links.size):
            v = cvs[inhib_links[i]["i0"]]
            if v >= SPIKE_THRES:
                cvs_tobe[inhib_links[i]["i1"]] -= inhib_effis[i]
        # apply the global scaling factor
        cvs_tobe[:] /= SYNAP_FACTOR

        # reset voltage if fired, or update the voltage
        for i in range(cvs.size):
            v = cvs[i]
            if v >= SPIKE_THRES:
                # fired, reset
                cvs_tobe[i] = VOLT_RESET

                # record the spike
                spikes[n_spikes] = i
                n_spikes += 1
                step_n_spikes[i_step] += 1

            else:  # add back previous-voltage, plus leakage
                # note it's just input-current before this update
                cvs_tobe[i] += v + (0 - v) / τ_m

        # update the final state at end of this time step
        cvs[:] = cvs_tobe

    assert n_spikes == np.sum(step_n_spikes), "bug?!"
    return (
        # return a copy of valid slice, to release extraneous memory allocated
        spikes[:n_spikes].copy(),
        step_n_spikes,
    )

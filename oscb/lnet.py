__all__ = [
    "LetterNet",
    "Alphabet",
]

import numpy as np
import pandas as pd
from numba import njit

from .core import *


class Alphabet:
    """
    Simply encode lower-cased a~z into 0~25

    Subclasses can override attributes/methods for different schemas

    """

    size = 26

    @staticmethod
    def alphabet():
        return np.array([chr(c) for c in range(ord("a"), ord("z") + 1)], "U1")

    @staticmethod
    def encode_words(words):
        if len(words) < 1:  # specialize special case
            return np.empty(0, np.int32), np.empty(0, np.int8)

        lcode_base = ord("a")

        reserve_cap = len(words) * max(len(word) for word in words)
        w_bound = np.zeros(len(words), np.int32)
        w_lcode = np.zeros(reserve_cap, np.uint8)
        n_words, n_letters = 0, 0
        for word in words:
            for letter in word.lower():
                lcode = ord(letter) - lcode_base
                if not (0 <= lcode < 26):
                    continue  # discard non-letter chars
                w_lcode[n_letters] = lcode
                n_letters += 1
            if n_words > 0 and w_bound[n_words - 1] == n_letters:
                continue  # don't encounter empty words
            w_bound[n_words] = n_letters
            n_words += 1

        data = w_bound[:n_words].copy(), w_lcode[:n_letters].copy()
        return data

    @staticmethod
    def decode_words(data):
        w_bound, w_lcode = data
        assert w_bound.ndim == w_lcode.ndim == 1
        assert w_bound[-1] == w_lcode.size
        assert np.all((0 <= w_lcode) & (w_lcode < 26))

        words = []
        l_bound = 0
        for r_bound in w_bound:
            words.append(
                "".join(chr(ord("a") + lcode) for lcode in w_lcode[l_bound:r_bound])
            )
            l_bound = r_bound
        return words


class LetterNet:
    """
    Simplified Spiking Neuron Network with:

      * hardcoded letter encoding in SDRs

      * capped sparse excitatory/inhibitory synaptic connections

    """

    def __init__(
        self,
        MAX_SYNAPSES=1_000_000,  # max number of synapses, one million by default
        N_COLS_PER_LETTER=10,  # distributedness of letter SDR
        SPARSE_FACTOR=5,  # sparseness of letter SDR
        N_CELLS_PER_COL=100,  # per mini-column capacity
        ALPHABET=Alphabet(),  # alphabet
    ):
        self.SPARSE_FACTOR = SPARSE_FACTOR
        self.N_CELLS_PER_COL = N_CELLS_PER_COL
        self.ALPHABET = ALPHABET
        N_SPARSE_COLS_PER_LETTER = N_COLS_PER_LETTER * SPARSE_FACTOR
        self.CELLS_SHAPE = ALPHABET.size * N_SPARSE_COLS_PER_LETTER, N_CELLS_PER_COL

        # each letter's representational cell indices up to column addressing
        self.sdr_indices = np.full(
            (
                ALPHABET.size,
                N_COLS_PER_LETTER,
            ),
            -1,
            np.uint32,
        )
        for lcode in range(ALPHABET.size):
            lbase = lcode * N_SPARSE_COLS_PER_LETTER
            for l_col in range(N_COLS_PER_LETTER):
                self.sdr_indices[lcode, l_col] = lbase + l_col * SPARSE_FACTOR

        # excitatory synapse links/efficacies
        self.excit_links = np.zeros(MAX_SYNAPSES, dtype=SYNAPSE_LINK_DTYPE)
        self.excit_effis = np.zeros(MAX_SYNAPSES, dtype="f4")
        self.excit_synap = 0

        # inhibitory synapse links/efficacies
        self.inhib_links = np.zeros(MAX_SYNAPSES, dtype=SYNAPSE_LINK_DTYPE)
        self.inhib_effis = np.zeros(MAX_SYNAPSES, dtype="f4")
        self.inhib_synap = 0

    """
    load factor affects synapse dropout behavior

    once number of synapses exceeds MAX_SYNAPSES, weakest synapse links will
    be dropped out, so that the strongest synaptic links per this ratio is kept.
    """
    LOAD_FACTOR = 0.8

    @property
    def MAX_SYNAPSES(self):
        return self.excit_links.size

    @property
    def ALPHABET_SIZE(self):
        return self.sdr_indices.shape[0]

    @property
    def N_COLS_PER_LETTER(self):
        return self.sdr_indices.shape[1]

    @property
    def MAX_SYNAPSES(self):
        return self.excit_links.size

    @property
    def N_SPARSE_COLS_PER_LETTER(self):
        return self.N_COLS_PER_LETTER * self.SPARSE_FACTOR

    def _excitatory_synapses(self):
        return (
            self.excit_links[: self.excit_synap],
            self.excit_effis[: self.excit_synap],
        )

    def excitatory_synapses(self):
        links, effis = self._excitatory_synapses()
        ci0, ici0 = np.divmod(links["i0"], self.N_CELLS_PER_COL)
        ci1, ici1 = np.divmod(links["i1"], self.N_CELLS_PER_COL)
        return pd.DataFrame(
            {
                "from_column": ci0,
                "from_cell": ici0,
                "to_column": ci1,
                "to_cell": ici1,
                "efficacy": effis,
            }
        )

    def _inhibitory_synapses(self):
        return (
            self.inhib_links[: self.inhib_synap],
            self.inhib_effis[: self.inhib_synap],
        )

    def inhibitory_synapses(self):
        links, effis = self._inhibitory_synapses()
        ci0, ici0 = np.divmod(links["i0"], self.N_CELLS_PER_COL)
        ci1, ici1 = np.divmod(links["i1"], self.N_CELLS_PER_COL)
        return pd.DataFrame(
            {
                "from_column": ci0,
                "from_cell": ici0,
                "to_column": ci1,
                "to_cell": ici1,
                "efficacy": effis,
            }
        )

    def create_inhibitory_links_randomly(self, n, compact=True, normalize=False):
        assert 0 < n
        self.inhib_synap = _connect_synapses_randomly(
            n,
            self.inhib_links,
            self.inhib_effis,
            self.inhib_synap,
            self.LOAD_FACTOR,
            self.ALPHABET_SIZE,
            self.N_COLS_PER_LETTER,
            self.SPARSE_FACTOR,
            self.N_CELLS_PER_COL,
        )
        if compact:
            self.inhib_synap = _compact_synapses(
                self.inhib_links,
                self.inhib_effis,
                self.inhib_synap,
                self.LOAD_FACTOR,
            )
        if normalize:
            _normalize_synapse_efficacies(
                self.inhib_links, self.inhib_effis, self.inhib_synap
            )

    def create_excitatory_links_randomly(self, n, compact=True, normalize=False):
        assert 0 < n
        self.excit_synap = _connect_synapses_randomly(
            n,
            self.excit_links,
            self.excit_effis,
            self.excit_synap,
            self.LOAD_FACTOR,
            self.ALPHABET_SIZE,
            self.N_COLS_PER_LETTER,
            self.SPARSE_FACTOR,
            self.N_CELLS_PER_COL,
        )
        if compact:
            self.excit_synap = _compact_synapses(
                self.excit_links,
                self.excit_effis,
                self.excit_synap,
                self.LOAD_FACTOR,
            )
        if normalize:
            _normalize_synapse_efficacies(
                self.excit_links, self.excit_effis, self.excit_synap
            )

    def learn_words_as_sequence(
        self,
        words,
        sp_width=(3, 10),  # width of spike train: [n_columns, n_cells]
        sp_thick=15,  # thickness of spike train
        link_back=False,  # create backward synapses too
        compact=True,
        normalize=False,
    ):
        _, w_lcode = self.ALPHABET.encode_words(words)
        self.excit_synap = _connect_letter_sequence(
            self.sdr_indices,
            self.excit_links,
            self.excit_effis,
            self.excit_synap,
            w_lcode,
            sp_width,
            sp_thick,
            link_back,
            self.LOAD_FACTOR,
            self.N_CELLS_PER_COL,
        )
        if compact:
            self.excit_synap = _compact_synapses(
                self.excit_links,
                self.excit_effis,
                self.excit_synap,
                self.LOAD_FACTOR,
            )
        if normalize:
            _normalize_synapse_efficacies(
                self.excit_links, self.excit_effis, self.excit_synap
            )

    def learn_words(
        self,
        words,
        sp_width=(3, 10),  # width of spike train: [n_columns, n_cells]
        sp_thick=15,  # thickness of spike train
        link_back=False,  # create backward synapses too
        compact=True,
        normalize=False,
    ):
        w_bound, w_lcode = self.ALPHABET.encode_words(words)
        self.excit_synap = _connect_per_words(
            self.sdr_indices,
            self.excit_links,
            self.excit_effis,
            self.excit_synap,
            w_bound,
            w_lcode,
            sp_width,
            sp_thick,
            link_back,
            self.LOAD_FACTOR,
            self.N_CELLS_PER_COL,
        )
        if compact:
            self.excit_synap = _compact_synapses(
                self.excit_links,
                self.excit_effis,
                self.excit_synap,
                self.LOAD_FACTOR,
            )
        if normalize:
            _normalize_synapse_efficacies(
                self.excit_links, self.excit_effis, self.excit_synap
            )


@njit
def _normalize_synapse_efficacies(links, effis, vlen):
    """
    normalize efficacies, by scaling the smallest value to be 1.0
    """
    assert effis.ndim == 1
    assert 0 <= vlen <= effis.size

    if vlen < 1:  # specialize special case
        return

    if vlen == 1:  # specialize special case
        effis[0] = 1.0
        return

    sorted = True
    for i in range(1, effis.size):
        if effis[i] < effis[i - 1]:
            sorted = False
            break
    if not sorted:
        sidxs = np.argsort(effis[:vlen])
        links[:vlen] = links[sidxs]
        effis[:vlen] = effis[sidxs]

    effis[:vlen] /= effis[0]


@njit  # (debug=True)
def _compact_synapses(links, effis, vlen, LOAD_FACTOR=0.8):
    assert links.ndim == effis.ndim == 1
    assert links.shape == effis.shape

    if vlen < 1:  # specialize special case
        return 0

    # merge duplicate links
    new_links = np.empty_like(links)
    new_effis = np.empty_like(effis)
    new_vlen = 0

    # view the link as a whole
    links_ho = links[:vlen].view(np.byte).view(np.uint64)
    new_links_ho = new_links.view(np.byte).view(np.uint64)
    for i in np.argsort(links_ho):
        if new_vlen > 0 and new_links_ho[new_vlen - 1] == links_ho[i]:
            # accumulate efficacy
            new_effis[new_vlen - 1] += effis[i]
        else:
            # encounter a new distinct link
            new_links_ho[new_vlen] = links_ho[i]
            new_effis[new_vlen] = effis[i]
            new_vlen += 1

    # store new data back inplace
    n2drop = new_vlen - int(links.size * LOAD_FACTOR)
    if n2drop > 0:  # drop synapses with smallest efficacies
        keep_idxs = np.argsort(new_effis[:new_vlen])[n2drop:]
        assert keep_idxs.size == new_vlen - n2drop  # so obvious
        new_vlen = keep_idxs.size
        # store back those with large efficacies
        links[:new_vlen] = new_links[keep_idxs]
        effis[:new_vlen] = new_effis[keep_idxs]
    else:  # not overloaded yet, simply store back
        links[:new_vlen] = new_links[:new_vlen]
        effis[:new_vlen] = new_effis[:new_vlen]

    return new_vlen


@njit
def _connect_per_words(
    sdr_indices,
    links,
    effis,
    vlen,
    w_bound,
    w_lcode,
    sp_width=(3, 10),  # width of spike train: [n_columns, n_cells]
    sp_thick=15,  # thickness of spike train
    link_back=False,  # create backward synapses too
    LOAD_FACTOR=0.8,
    N_CELLS_PER_COL=100,  # per mini-column capacity
):
    ALPHABET_SIZE, N_COLS_PER_LETTER = sdr_indices.shape
    assert np.all((0 <= w_lcode) & (w_lcode < ALPHABET_SIZE))
    assert len(sp_width) == 2
    assert 1 <= sp_width[0] <= N_COLS_PER_LETTER
    assert 1 <= sp_width[1] <= N_CELLS_PER_COL
    assert 1 <= sp_thick <= sp_width[0] * sp_width[1]

    if w_lcode.size < 1:  # specialize the special case
        return vlen

    def random_idxs_for_letter(lcode):
        idxs = np.empty(sp_width, np.uint32)
        for i, ci in enumerate(
            np.random.choice(sdr_indices[lcode], sp_width[0], replace=False)
        ):
            idxs[i, :] = ci * N_CELLS_PER_COL + np.random.choice(
                np.arange(N_CELLS_PER_COL, dtype=np.uint32), sp_width[1], replace=False
            )
        return idxs.ravel()

    l_bound = 0
    for r_bound in w_bound:
        assert r_bound > l_bound, "bug?!"

        for pre_lcode in w_lcode[l_bound:]:
            break

        pre_idxs = random_idxs_for_letter(pre_lcode)

        for post_lcode in w_lcode[l_bound + 1 : r_bound]:
            post_idxs = random_idxs_for_letter(post_lcode)

            # TODO: justify the decision here
            #
            # Alternative 1 - every presynaptic cell connect forward
            #
            # for pre_i in pre_idxs:
            #     for post_i in np.random.choice(post_idxs, sp_thick):
            #
            # Alternative 2 - every postsynaptic cell connect backward
            #
            for post_i in post_idxs:
                for pre_i in np.random.choice(pre_idxs, sp_thick):
                    if vlen >= links.size - 1:
                        # compat the synapses once exceeding allowed maximum
                        vlen = _compact_synapses(links, effis, vlen, LOAD_FACTOR)

                    links[vlen]["i0"] = pre_i
                    links[vlen]["i1"] = post_i
                    effis[vlen] = 1.0

                    vlen += 1

                    if link_back:
                        links[vlen]["i0"] = post_i
                        links[vlen]["i1"] = pre_i
                        effis[vlen] = 1.0

                        vlen += 1

            pre_lcode, pre_idxs = post_lcode, post_idxs

        l_bound = r_bound

    return vlen


@njit  # (debug=True)
def _connect_letter_sequence(
    sdr_indices,
    links,
    effis,
    vlen,
    lcode_seq,
    sp_width=(3, 10),  # width of spike train: [n_columns, n_cells]
    sp_thick=15,  # thickness of spike train
    link_back=False,  # create backward synapses too
    LOAD_FACTOR=0.8,
    N_CELLS_PER_COL=100,  # per mini-column capacity
):
    ALPHABET_SIZE, N_COLS_PER_LETTER = sdr_indices.shape
    assert np.all((0 <= lcode_seq) & (lcode_seq < ALPHABET_SIZE))
    assert len(sp_width) == 2
    assert 1 <= sp_width[0] <= N_COLS_PER_LETTER
    assert 1 <= sp_width[1] <= N_CELLS_PER_COL
    assert 1 <= sp_thick <= sp_width[0] * sp_width[1]

    if lcode_seq.size < 1:  # specialize the special case
        return vlen

    def random_idxs_for_letter(lcode):
        idxs = np.empty(sp_width, np.uint32)
        for i, ci in enumerate(
            np.random.choice(sdr_indices[lcode], sp_width[0], replace=False)
        ):
            idxs[i, :] = ci * N_CELLS_PER_COL + np.random.choice(
                np.arange(N_CELLS_PER_COL, dtype=np.uint32), sp_width[1], replace=False
            )
        return idxs.ravel()

    # pre_lcode = lcode_seq[0]
    # above seems to have signed/unsigned conversion bug, a Numba one?
    # below workarounds that
    for pre_lcode in lcode_seq:
        break

    pre_idxs = random_idxs_for_letter(pre_lcode)

    for post_lcode in lcode_seq[1:]:
        post_idxs = random_idxs_for_letter(post_lcode)

        # TODO: justify the decision here
        #
        # Alternative 1 - every presynaptic cell connect forward
        #
        # for pre_i in pre_idxs:
        #     for post_i in np.random.choice(post_idxs, sp_thick):
        #
        # Alternative 2 - every postsynaptic cell connect backward
        #
        for post_i in post_idxs:
            for pre_i in np.random.choice(pre_idxs, sp_thick):
                if vlen >= links.size - 1:
                    # compat the synapses once exceeding allowed maximum
                    vlen = _compact_synapses(links, effis, vlen, LOAD_FACTOR)

                links[vlen]["i0"] = pre_i
                links[vlen]["i1"] = post_i
                effis[vlen] = 1.0

                vlen += 1

                if link_back:
                    links[vlen]["i0"] = post_i
                    links[vlen]["i1"] = pre_i
                    effis[vlen] = 1.0

                    vlen += 1

        pre_lcode, pre_idxs = post_lcode, post_idxs

    return vlen


@njit
def _connect_synapses_randomly(
    n,  # number of synapses to create
    links,
    effis,
    vlen,
    LOAD_FACTOR=0.8,
    ALPHABET_SIZE=26,  # size of alphabet
    N_COLS_PER_LETTER=10,  # distributedness of letter SDR
    SPARSE_FACTOR=5,  # sparseness of letter SDR
    N_CELLS_PER_COL=100,  # per mini-column capacity
):
    N_SPARSE_COLS_PER_LETTER = N_COLS_PER_LETTER * SPARSE_FACTOR

    N_FULL_CELLS = ALPHABET_SIZE * N_SPARSE_COLS_PER_LETTER * N_CELLS_PER_COL

    for _ in range(n):
        if vlen >= links.size:
            # compat the synapses once exceeding allowed maximum
            vlen = _compact_synapses(links, effis, vlen, LOAD_FACTOR)

        # randomly pick 2 cells and make a synapse
        links[vlen]["i0"] = np.random.randint(N_FULL_CELLS)
        links[vlen]["i1"] = np.random.randint(N_FULL_CELLS)
        effis[vlen] = 1.0

        vlen += 1

    return vlen

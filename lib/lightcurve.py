# -*- coding: utf-8 -*-
import numpy as np


class LightCurve(object):
    def __init__(
        self,
        bins: np.array,
        states: np.array,
        thr: np.float = 0.0,
        name: str = "Name",
    ):
        if len(states) != len(bins) - 1:
            print("Wrong dim!")
            self._bins = None
            self._states = None
        else:
            self._bins = bins
            self._states = states

        self._threshold = thr
        self._name = name

        length = np.abs(np.max(self._bins) - np.min(self._bins))
        flare_length = 0.0
        for i, state in enumerate(self._states):
            if state > self._threshold:
                flare_length += np.abs(self._bins[i + 1] - self._bins[i])

        self._flare_probability = flare_length / length

    def __call__(self, t: np.array):
        if np.any(t > np.max(self._bins)) or np.any(t < np.min(self._bins)):
            print("t outside range")
            return None

        index = np.array(
            (np.sum(self._bins <= t[:, np.newaxis], axis=1) - 1), dtype=np.int
        )
        index[index >= len(self._states)] = len(self._states) - 1

        return [self._states[i] > self._threshold for i in index]

    @property
    def flare_probability(self):
        return self._flare_probability

    @property
    def bins(self):
        return self._bins

    @property
    def states(self):
        return self._states

    @property
    def threshold(self):
        return self._threshold

    @property
    def name(self):
        return self._name

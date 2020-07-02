# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from .lightcurve import LightCurve


def make_rnd_lc(bins, state, thr):
    rand_idx = np.random.choice(len(state), len(state))
    s_rand = LightCurve(bins, state[rand_idx[:-1]], thr)
    return s_rand


def background_trials(ntrials, sources, seed=1):
    np.random.seed(seed)
    trials = []
    for i in tqdm(range(ntrials)):
        t_nu = []
        for i, s in enumerate(sources):
            t_nu.append(
                np.random.uniform(
                    low=np.min(s._bins), high=np.max(s._bins), size=1
                )
            )
        t_nu = np.array(t_nu)
        coincidences = np.array(
            [s(t) for s, t in zip(sources, t_nu)]
        ).flatten()
        trials.append(np.sum(coincidences))
    return np.array(trials)


def signal_trials(ntrials, n_inj_mean, sources, seed=1):
    np.random.seed(seed)
    trials = []
    for i in tqdm(range(ntrials)):
        n_inj = np.random.poisson(n_inj_mean)
        if n_inj > len(sources):
            n_inj = len(sources)
        n_bkg = len(sources) - n_inj
        rnd_idx = np.random.choice(len(sources), n_bkg, replace=False)
        t_nu = []
        for i, s in enumerate(sources[rnd_idx]):
            t_nu.append(
                np.random.uniform(
                    low=np.min(s._bins), high=np.max(s._bins), size=1
                )
            )
        t_nu = np.array(t_nu)
        coincidences = np.array(
            [s(t) for s, t in zip(sources[rnd_idx], t_nu)]
        ).flatten()
        trials.append(np.sum(coincidences) + n_inj)
    return np.array(trials)


def test_statistic(n, pdf):
    p_zero = pdf[0, n]
    p_h = np.max(pdf[1:, n])
    ind = np.argmax(pdf[1:, n])
    return -2 * np.log(p_zero / p_h) if p_h != 0.0 else 0.0, ind


def prof_likelihood(n, pdf):
    p_zero = np.array(pdf[0, n])
    p_h = np.array(pdf[1:, n])
    m = p_h != 0.0
    p_h = p_h[m]
    p_h_max = np.max(pdf[1:, n])
    best = np.log(p_zero / p_h_max)
    lik = np.log(p_zero / p_h)
    prof_lik = -2 * (best - lik)
    return prof_lik

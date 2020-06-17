# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from .lightcurve import LightCurve

def make_rnd_lc(bins, state, thr):
    rand_idx = np.random.choice(len(state), len(state))
    s_rand = LightCurve(bins, state[rand_idx[:-1]], thr)
    return s_rand

def background_trials(ntrials, sources, bins, seed=1):
    np.random.seed(seed)
    trials = []
    for i in tqdm(range(ntrials)):
        t_nu = np.random.uniform(low=np.min(bins), high=np.max(bins), size=(len(sources),1))
        coincidences = np.array([s(t) for s,t in zip(sources, t_nu)]).flatten()
        trials.append(np.sum(coincidences))
    return np.array(trials)

def signal_trials(ntrials, n_inj, sources, bins, seed=1):
    np.random.seed(seed)
    n_bkg = len(sources) - n_inj
    trials = []
    for i in tqdm(range(ntrials)):
        t_nu = np.random.uniform(low=np.min(bins), high=np.max(bins), size=(n_bkg,1))
        rnd_idx = np.random.choice(len(sources), n_bkg, replace=False)
        coincidences = np.array([s(t) for s,t in zip(sources[rnd_idx], t_nu)]).flatten()
        trials.append(np.sum(coincidences) + n_inj)
    return np.array(trials)

def test_statistic(n, pdf):
    p_zero = pdf[0, n]
    p_h = np.max(pdf[1:, n])
    ind = np.argmax(pdf[1:, n])
    return -2 * np.log(p_zero/p_h) if p_h !=0. else 0., ind

def prof_likelihood(n, pdf):
    p_zero = pdf[0, n]
    p_h_max = np.max(pdf[1:, n])
    p_h = pdf[1:, n]
    best = np.log(p_zero/p_h_max)
    lik = np.log(p_zero/p_h)
    prof_lik = -2 * (best - lik)
    return prof_lik

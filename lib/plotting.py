import numpy as np
import matplotlib.pyplot as plt

def plot_lightcurve(ax, lc):
    ax.set_title(lc.name)
    ax.errorbar(lc.bins[:-1]+3.5, lc.states, xerr=3.5, ls="", marker='+')
    ax.axhline(lc.threshold, ls='--', color='gray')
    ax.set_ylabel(r"ph/cm$^{2}$/s")

    x = np.linspace(np.min(lc.bins),np.max(lc.bins),10000)
    y = lc(x)

    #min_x = []
    #max_x = []
    #for i, ics in enumerate(x):
    #    if y[i-1]<1. and y[i] == 1.:
    #        min_x.append(ics)
    #    elif y[i-1]==1. and y[i] == 0.:
    #        max_x.append(ics)
    #
    #    if i == len(x-1) and y[i]==1.:
    #        max_x.append(ics)

    variations = []
    if y[0]==1.:
        variations.append(x[0])

    for i, fl in enumerate(y):
        if i==0 or i==len(y)-1:
            continue

        if fl != y[i-1]:
            variations.append(x[i])

    if y[-1]==1.:
        variations.append(x[-1])

    min_x = variations[::2]
    max_x = variations[1::2]

    lims = ax.get_ylim()
    for min_val, max_val in zip(min_x, max_x):
        ax.fill_betweenx(np.linspace(lims[0],lims[1],1000), min_val, max_val, color='red', alpha=0.1)

    return ax

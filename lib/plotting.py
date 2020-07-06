import numpy as np
import matplotlib.pyplot as plt


def plot_lightcurve(ax, lc):
    ax.set_title(f"{lc.name}")
    bin_c = (lc.bins[1:] + lc.bins[:-1]) / 2.0
    meas = np.array(lc(bin_c))
    flux_plot = ax.errorbar(
        bin_c[meas],
        lc.states[meas],
        xerr=(np.diff(lc.bins) / 2.0)[meas],
        yerr=lc.err[meas],
        ls="",
        marker="o",
    )
    ax.errorbar(
        bin_c[~meas],
        lc.states[~meas],
        xerr=(np.diff(lc.bins) / 2.0)[~meas],
        yerr=1e-8,
        ls="",
        marker="+",
        color=flux_plot[0].get_color(),
        alpha=0.6,
        uplims=True,
    )
    # ax.axhline(lc.threshold, ls='--', color='gray')
    ax.set_ylabel(r"ph/cm$^{2}$/s")
    ax.set_xlabel(r"MJD")

    x = np.linspace(np.min(lc.bins), np.max(lc.bins), 10000)
    y = lc(x)

    variations = []
    if y[0] == 1.0:
        variations.append(x[0])

    for i, fl in enumerate(y):
        if i == 0 or i == len(y) - 1:
            continue

        if fl != y[i - 1]:
            variations.append(x[i])

    if y[-1] == 1.0:
        variations.append(x[-1])

    min_x = variations[::2]
    max_x = variations[1::2]

    lims = ax.get_ylim()
    for min_val, max_val in zip(min_x, max_x):
        ax.fill_betweenx(
            np.linspace(lims[0], lims[1], 1000),
            min_val,
            max_val,
            color="red",
            alpha=0.1,
        )

    return ax

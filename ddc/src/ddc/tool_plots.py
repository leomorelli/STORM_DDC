"""Figures for the preprocessing tools; matplotlib is imported only on use."""

from __future__ import annotations

import numpy as np

from .tools import NDiagnostic, ResolutionDiagnostic, ResolutionEstimate


def plot_n(result: NDiagnostic):
    """Return a matplotlib Figure with Z and pair support at each exact lag."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    axes[0].plot(result.frame_differences, result.z, ".-", linewidth=1.25)
    axes[0].set(ylabel="Z (CDF difference from lag 1)",
                title="Choose N_f where Z reaches a sustained plateau")
    axes[1].plot(result.frame_differences, result.pair_counts, color="tab:gray")
    axes[1].set(xlabel="Frame difference", ylabel="Pair count")
    for ax in axes:
        ax.grid(alpha=.2)
    return fig


def plot_resolution(result: ResolutionDiagnostic | ResolutionEstimate):
    """Return the manual per-image curves or automatic search distributions."""
    import matplotlib.pyplot as plt

    if isinstance(result, ResolutionEstimate):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for trial in result.trials:
            ax.plot(trial.bins[:-1], trial.blink_probability,
                    label=f"{trial.resolution:g} nm")
        ax.set(xlabel="Distance bin lower edge (nm)", ylabel="Blinking probability",
               title=f"Recommended DDC bin width: {result.resolution:g} nm")
        ax.legend(title="Tested bin widths", ncol=3, fontsize="small")
        ax.grid(alpha=.2)
        return fig
    has_truth = result.true_probabilities is not None and np.isfinite(result.true_probabilities).any()
    fig, axes = plt.subplots(2 if has_truth else 1, 1, figsize=(9, 7 if has_truth else 4),
                             squeeze=False, layout="constrained")
    for ax, probabilities, label in zip(axes[:, 0],
                                       (result.probabilities, result.true_probabilities),
                                       ("Measured localizations", "Matched ground truth")):
        ax.plot(result.bins[:-1], probabilities.T)
        ax.set(xlabel="Distance bin lower edge (nm)", ylabel="Pair probability",
               title=f"{label}: bin width {result.resolution:g} nm, lag > {result.n_f:g}")
        ax.grid(alpha=.2)
    return fig


def plot_split(dataset, *, image: int = 1):
    """Preview buffered regions and their unbuffered xy rectangles (1-based image)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if dataset.came_from_image is None or any(c is None for c in dataset.cuts):
        raise ValueError("plot_split requires split-image metadata")
    selected = np.flatnonzero(dataset.came_from_image == image)
    if not selected.size:
        raise ValueError(f"no regions from image {image}")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for j, i in enumerate(selected):
        loc = dataset.localizations[i]
        color = plt.get_cmap("tab20")(j % 20)
        ax.scatter(loc[:, 0], loc[:, 1], s=1, alpha=.2, color=color, rasterized=True)
        x, y = dataset.cuts[0][i], dataset.cuts[1][i]
        ax.add_patch(Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0],
                               fill=False, edgecolor=color))
    ax.set(xlabel="x (nm)", ylabel="y (nm)",
           title=f"Image {image}: {len(selected)} overlapping regions (xy projection)")
    ax.set_aspect("equal")
    return fig

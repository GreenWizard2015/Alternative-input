#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization utilities for dataset statistics.

Provides helper functions for plotting and visualizing dataset statistics,
such as histograms of frame time deltas and session durations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data: np.ndarray, title: str, filename: str) -> None:
    """Plot and save histogram of data to file.

    Helper function to visualize distribution statistics (e.g., frame time deltas).
    Creates a histogram visualization with 100 bins, saves to disk, and cleans up.
    Uses matplotlib for plotting.

    Args:
        data: Array of values to plot (1D numpy array).
        title: Histogram title string.
        filename: Output filename for the saved plot image.

    Returns:
        None (saves plot as side effect).

    Raises:
        ValueError: If data is empty or title/filename are empty strings.

    Example:
        >>> data = np.random.randn(1000)
        >>> plot_histogram(data, "Distribution", "output.png")
    """
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    if not title:
        raise ValueError("title cannot be empty")
    if not filename:
        raise ValueError("filename cannot be empty")

    plt.hist(data, bins=100)
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.close()
    plt.clf()

import matplotlib.pyplot as plt


def fig_config(latex=False, nrows=1, ncols=1, scaler=1.0, text_size=16,
               line_width=3, marker_size=5):
    """Configure matplotlib parameters for better visualization style.

    Parameters
    ----------
    latex : bool, optional
        If true, LaTeX backend will be used.
    nrows : int, optional
        Number of figures row-wise.
    ncols : int, optional
        Number of figures column-wise.
    scaler : float, optional
        Scaler for each figure.
    text_size : int, optional
        Font size for textual elements in figure.
    line_width : int, optional
        Line width.
    marker_size : int, optional
        Marker size for scatter plots.

    Returns
    -------
    None
    """
    plt.rcParams.update({
        'text.usetex': latex,
        'font.family': 'serif',
        'font.size': text_size,
        'figure.figsize': (4.774 * scaler * ncols, 2.950 * scaler * nrows),
        'lines.linewidth': line_width,
        'lines.dashed_pattern': (3, 5),
        'lines.markersize': marker_size,
        'lines.markeredgecolor': 'k',
        'lines.markeredgewidth': 0.5,
        'image.origin': 'lower',
        'axes.labelsize': text_size,
        'axes.titlesize': text_size,
        'grid.linewidth': 0.5,
        'legend.fontsize': text_size,
        'xtick.labelsize': text_size,
        'ytick.labelsize': text_size,
    })


def fig_config_reset():
    """Recover matplotlib default parameters.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    plt.rcParams.update(plt.rcParamsDefault)

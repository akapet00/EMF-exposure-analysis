import matplotlib.pyplot as plt


def fig_config(latex=False, nrows=1, ncols=1, scaler=1.0):
    """Configure matplotlib parameters for better visualization style.

    Parameters
    ----------
    latex : bool, optional
        If true, LaTeX backend will be used
    nrows : int, optional
        number of figures row-wise
    ncols : int, optional
        number of figures column-wise
    scaler : float, optional
        scaler for each figure

    Returns
    -------
    None
    """
    plt.style.use('seaborn-paper')
    plt.rcParams.update({
        'text.usetex': latex,
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'grid.linewidth': 0.7,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': [4.774 * scaler * ncols, 2.950 * scaler * nrows],
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

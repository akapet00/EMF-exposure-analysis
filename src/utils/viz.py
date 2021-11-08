import matplotlib.pyplot as plt
import numpy as np


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
        'lines.dashed_pattern': (3, 3),
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


def set_axes_equal(ax):
    """Return 3-D axes with equal scale.

    Note: This function is implemented as in https://stackoverflow.com/a/31364297/15005103
    because matplotlib currently does not support setting ``ax.axis('equal')``
    for 3-D plotting.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Axes with 'auto' scale settings.

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Axes as if the scale settings were defined as 'equal'.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # bounding box is a sphere in the sense of the infinity norm
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return ax


def save_fig(fig, fname, formats=['pdf']):
    """Save the current figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to be saved.
    fname : str or path-like or binary file-like
        A path, or a Python file-like objec without the format
        extension - it will automatically be added depending on the
        `formats` list.
    formats : list, optional
        The file formats in a list. Figure will be saved in pdf format
        by default.

    Returns
    -------
    None
    """
    for format in formats:
        fname = f'{fname}.{format}'
        fig.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format=format, transparent=True,
                    bbox_inches='tight', pad_inches=0.1)

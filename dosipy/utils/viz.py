from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def fig_config(latex=False, nrows=1, ncols=1, scaler=1.0, text_size=16,
               line_width=3, marker_size=5):
    """Configure visualization parameters.

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
        'text.latex.preamble': r'\usepackage{amsmath}',
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


def set_colorblind():
    """Colorblind coloring.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError('`seaborn` is not installed.')
    sns.set(style='ticks', palette='colorblind')


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

    Note: This function is implemented as in:
    https://stackoverflow.com/a/31364297/15005103 because there is no
    support setting that would enable `ax.axis('equal')` in 3-D.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3-D axes subplot with scale settings set to `auto`.

    Returns
    -------
    matplotlib.axes._subplots.Axes3DSubplot
        Axes as if the scale settings were defined as `equal`.
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


def _minmax_scale(x, _range=(0, 1)):
    """Min-max scaler.

    Parameters
    ----------
    x : numpy.ndarray
        The data to be scaled.
    _range : tuple, optional
        Desired range of transformed data.

    Returns
    -------
    numpy.ndarray
        Scaled data.
    """
    scaler = (x - x.min()) / (x.max() - x.min())
    x_scaled = scaler * (_range[1] - _range[0]) + _range[0]
    return x_scaled


def colormap_from_array(x, cmap='viridis', alpha=None, bytes=False):
    """Return array values mapped into corresponding RGB values.

    Parameters
    ----------
    x : numpy.ndarray
        The data with values to be converted to RGB values.
    cmap : string, optional
        Name of the colormap.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1
        (opaque).
    bytes : bool, optional
        If False (default), the returned RGB values will be floats in
        the interval [0, 1] otherwise they will be integers in the
        interval [0, 255].

    Returns
    -------
    numpy.ndarray
        RGB values.
    """
    from matplotlib import cm
    x_scaled = _minmax_scale(x)
    try:
        cs = eval(f'cm.{cmap}')(x_scaled, alpha, bytes)
    except Exception as e:
        print(e, 'Falling to default colormap')
        cs = cm.viridis(x_scaled, alpha, bytes)
    finally:
        if alpha is None:
            cs = cs[:, :3]
    return cs


def scatter_2d(xy_dict, figsize=None, s=20, c=None, alpha=1):
    """2-D scatter plot.

    Parameters
    ----------
    xy_dict : dictionary
        Keys are label names, values are the corresponding values.
        First input goes on the x-axis, the second one goes on the
        y-axis, and the third one, if exists, defines the color of each
        marker.
    figsize : tuple, optional
        Width and height in inches.
    s : float or array-like, optional
        The marker size.
    c : string, optional
        The marker color. If set, it overrides the third input of
        `xy_dict`.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1
        (opaque).

    Returns
    -------
    tuple
        Figure and axes of the 2-D scatter plot.
    """
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
    assert isinstance(figsize, (tuple, list)), 'Invalid figure size.'
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    keys = list(xy_dict.keys())
    vals = list(xy_dict.values())
    if (len(vals) == 3) and not(c):
        cs = ax.scatter(vals[0], vals[1], s=s, c=vals[2], cmap='viridis')
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel(keys[2])
    else:
        if not(c):
            c = 'k'
        cs = ax.scatter(vals[0], vals[1], s=s, c=c, alpha=alpha)
    ax.set(xlabel=keys[0], ylabel=keys[1])
    ax.axis('equal')
    fig.tight_layout()
    return fig, ax


def scatter_3d(xyz_dict, figsize=None, azim=[45], elev=[9], c=None, alpha=1):
    """3-D scatter plot.

    Parameters
    ----------
    xyz_dict : dictionary
        Keys are label names, values are the corresponding values.
        First three inputs go on the x-, y-, and z-axis, respectively,
        while the forth one, if exists, defines the color of each
        marker.
    figsize : tuple, optional
        Width and height in inches.
    azim : list of floats, optional
        Azimuthal viewing angle. If there are more than 1 element in
        the list, the multiple subplots will be generated.
    elev : list of floats, optional
        Elevation viewing angle. If there are more than 1 element in
        the list, the multiple subplots will be generated.
    c : string, optional
        The marker color. If set, it overrides the forth input in
        `xyz_dict`.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1
        (opaque).

    Returns
    -------
    tuple
        Figure and axes of the 3-D scatter plot.
    """
    num_figs = len(elev) * len(azim)
    if num_figs > 4:
        raise ValueError('The max number of subplots is 4.')
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
    if num_figs != 1:
        figsize = (figsize[0] * num_figs / 2, figsize[1] * num_figs / 2)
    fig = plt.figure(figsize=figsize)
    keys = list(xyz_dict.keys())
    vals = list(xyz_dict.values())
    for i, (e, a) in enumerate(product(elev, azim)):
        ax = fig.add_subplot(num_figs, 1, i+1, projection='3d')
        if (len(vals) == 4) and not(c):
            cs = ax.scatter(vals[0], vals[1], vals[2], c=vals[3],
                            cmap='viridis')
            cbar = fig.colorbar(cs, shrink=0.5, pad=0.1)
            cbar.ax.set_ylabel(keys[3])
        else:
            if not(c):
                c = 'k'
            cs = ax.plot(vals[0], vals[1], vals[2], '.', c=c, alpha=alpha)
        ax.set(xlabel=keys[0], ylabel=keys[1], zlabel=keys[2])
        ax = set_axes_equal(ax)
        ax.view_init(elev=e, azim=a)
    fig.tight_layout()
    return fig, ax


def save_fig(fig, fname, formats=['pdf']):
    """Save the current figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to be saved.
    fname : str or path-like or binary file-like
        A path, or a Python file-like object without the format
        extension - it will automatically be added depending on the
        `formats` list.
    formats : list, optional
        The file format(s). Figure will be saved in pdf by default.

    Returns
    -------
    None
    """
    for format in formats:
        fig.savefig(f'{fname}.{format}', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format=format, transparent=True,
                    bbox_inches='tight', pad_inches=0.1)

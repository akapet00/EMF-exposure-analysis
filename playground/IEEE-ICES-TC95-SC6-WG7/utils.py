import numpy as np


# dielectric properties
def reflection_coefficient(eps, theta_i=0, polarization='parallel'):
    """Return reflection coefficient for oblique plane wave incidence.
    
    Parameters
    ----------
    eps : float
        Relative complex permittivity.
    theta_i : float
        Angle of incidence in °.
    polarization : str
        Either parallel or perpendicular/normal polarization.
    
    Returns
    -------
    float
        Reflection coefficient.
    """
    polarization = polarization.lower()
    SUPPORTED_POLARIZATIONS = ['parallel', 'normal']
    if polarization not in SUPPORTED_POLARIZATIONS:
        raise ValueError(
            f'Unsupported tissue. Choose from: {SUPPORTED_POLARIZATIONS}.'
            )
    scaler = np.sqrt(eps - np.sin(theta_i) ** 2)
    if polarization == 'parallel':
        return np.abs(
            (-eps * np.cos(theta_i) + scaler)
            / (eps * np.cos(theta_i) + scaler)
        )
    return np.abs(
        (np.cos(theta_i) - scaler)
        / (np.cos(theta_i) + scaler)
    )


# visualization
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
    return


def update_matplotlib_rc_parameters(is_3d=False):
    """Run and configure visualization parameters.
    
    Parameters
    ----------
    is_3d : bool, optional
        Set to true for 3d plotting adjustments.
    
    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if is_3d:
        sns.set(style='ticks', font='serif', font_scale=1.5)
        plt.rcParams.update({
            'axes.labelpad': 9
        })
    else:
        sns.set(style='ticks', font='serif', font_scale=1.25)
    plt.rcParams.update({
        'lines.linewidth': 3,
        'lines.markersize': 10,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif'
        })

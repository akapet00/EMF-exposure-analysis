import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import quadpy

from dosipy.utils.viz import set_colorblind, fig_config, save_fig


def visualize_integration_scheme(patch, pts, radii, colors):
    """Return figure and axis of the integration scheme.

    Parameters
    ----------
    patch : matplotlib.patches.Patch
        A 2-D artist with a face and an edge color.
    pts : numpy.ndarray
        Array of points over which the integration will be performed.
    radii : numpy.ndarray
        Array of radii that defines the weight for each corresponding
        integration point.
    colors : list
        Color of the circular patch that defines the area of influence
        at each integration point.

    Returns
    -------
    tuple
        Figure and axis.
    """
    set_colorblind()
    fig_config(scaler=1.5)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis("equal")
    ax.set_axis_off()
    ax.add_patch(patch)
    for pt, radius, color in zip(pts, radii, colors):
        plt.plot([pt[0]], [pt[1]], linestyle="None", marker=".", color=color)
        circ = Circle((pt[0], pt[1]), radius, color=color, alpha=0.7, lw=0)
        ax.add_patch(circ)
    return fig, ax


# rectangular integration scheme
deg = 6
bnd_l, bnd_r = -0.5, 0.5
x, w = np.polynomial.legendre.leggauss(deg)
x = 0.5 * (x + 1.) * (bnd_r - bnd_l) + bnd_l
w = 0.5 * w * (bnd_r - bnd_l)
Xx, Xy = np.meshgrid(x, x)
Wx, Wy = np.meshgrid(w, w)
pts = np.c_[Xx.ravel(), Xy.ravel()]
wts = np.c_[Wx.ravel(), Wy.ravel()]
wts_norm = np.linalg.norm(wts, axis=1, ord=sum(abs(x)**2)**(1./2))
area_tot = 1.2
radii = np.sqrt(abs(wts_norm) / np.sum(wts_norm) * area_tot / np.pi)
colors = ['gray' if w >= 0 else 'r' for w in wts_norm]
rect = Rectangle((bnd_l, bnd_l), 1, 1, ec='k', fc='None')
fig, ax = visualize_integration_scheme(rect, pts, radii, colors)
fname = os.path.join('figures', 'rect_integration_scheme')
save_fig(fig, fname=fname)

# circular integration scheme
scheme = quadpy.s2.get_good_scheme(12)
total_area = np.pi
flt = np.vectorize(float)
pts = flt(scheme.points.T)
weights = flt(scheme.weights.T)
radii = np.sqrt(abs(weights) / np.sum(weights) * total_area / np.pi)
colors = ['gray' if weight >= 0 else 'r' for weight in weights]
circ = Circle((0, 0), 1, ec='k', fc='None')
fig, ax = visualize_integration_scheme(circ, pts, radii, colors)
fname = os.path.join('figures', 'circ_integration_scheme')
save_fig(fig, fname=fname)

import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np

from dosipy.constants import eps_0
from dosipy.utils.dataloader import load_tissue_diel_properties

from utils import reflection_coefficient, load_dipole_data


sns.set_theme(style='ticks',
              font_scale=1.5,
              palette='colorblind',
              rc={'text.usetex': True,
                  'text.latex.preamble': r'\usepackage{amsmath}',
                  'font.family': 'serif'})


def main():
    ## dipole configuration
    # working frequency, Hz
    f = 30e9
    
    # antenna electric properties, free space (Poljak 2005)
    dipole_props = load_dipole_data(f)
    
    # current through the antenna
    Is = dipole_props.Ir.to_numpy() + dipole_props.Ii.to_numpy() * 1j
    
    # dipole-to-tissue separation distance
    d = 5 / 1000  # meters
    
    # dipole antenna position
    xs = dipole_props.x.to_numpy()
    xs = xs - xs.max() / 2  # center the dipole
    xs = np.asarray(xs)
    ys = np.zeros_like(xs)
    zs = np.full_like(xs, d)
    
    ## model configuration
    # dry skin density, kg/m3
    rho = 1109

    # conductivity, relative permitivitya and penetration depth
    sigma, eps_r, _, pen_depth = load_tissue_diel_properties('skin_dry', f)

    # reflection coefficient
    eps_i = sigma / (2 * np.pi * f * eps_0)
    eps = eps_r - 1j * eps_i
    gamma = reflection_coefficient(eps)

    # power transmission coefficient
    T_tr = 1 - gamma ** 2
    
    # exposed surface extent
    exposure_extent = (0.02, 0.02)  # meters x meters

    # exposed volume coordinates
    z_max = 0.02  # depth of the model in meters
    xt = np.linspace(-exposure_extent[0]/2, exposure_extent[0]/2)
    yt = np.linspace(-exposure_extent[1]/2, exposure_extent[1]/2)
    zt = np.linspace(0, z_max)
    
    # visualization
    fig, ax = plt.subplots()
    width = (np.ptp(xs) / np.ptp(xt) * 100).item()
    axin = inset_axes(ax, width=f'{width}%', height='20%', loc='center',
                      bbox_to_anchor=(0, 0.1, 1, 1),
                      bbox_transform=ax.transAxes)

    # main axes, exposed surface
    bbox = [xt.min(), xt.max(), yt.min(), yt.max()]
    xmin, xmax = bbox[:2]
    ymin, ymax = bbox[2:]
    # 2 x 2 cm2 area
    ax.vlines(x=[xmin, xmax], ymin=ymin, ymax=ymax,
              color='k', lw=2)
    ax.hlines(y=[ymin, ymax], xmin=xmin, xmax=xmax,
              color='k', lw=2)
    ax.fill_between(x=bbox[:2], y1=ymin, y2=ymax, color='k', alpha=0.075)
    ax.text(xmin+abs(xmin)*0.05, ymin+abs(ymin)*0.05, s='$A$ = 2 cm$^2$')
    # 1 x 1 cm2 area
    ax.vlines(x=[xmin/2, xmax/2], ymin=ymin/2, ymax=ymax/2,
              color='k', ls='--', lw=2, alpha=0.2)
    ax.hlines(y=[ymin/2, ymax/2], xmin=xmin/2, xmax=xmax/2,
              color='k', ls='--', lw=2, alpha=0.2)
    ax.fill_between(x=[xmin/2, xmax/2], y1=ymin/2, y2=ymax/2,
                    color='k', alpha=0.05)
    ax.text(xmin/2+abs(xmin/2)*0.1, ymin/2+abs(ymin/2)*0.1,
            s='$A$ = 1 cm$^2$')
    # dipole
    ax.hlines(y=ys[0], xmin=xs.min(), xmax=xs.max(),
              color='r', lw=4)
    # axes settings
    ax.set_aspect('equal', 'box')
    ax.set(xlabel='$x$ (cm)',
           ylabel='$y$ (cm)',
           xticks=[xmin, xmin/2, xs.min(), xs.max(), xmax/2, xmax],
           yticks=[ymin, ymin/2, 0, ymax/2, ymax],
           xticklabels=[xmin * 100,
                        xmin / 2 * 100,
                        np.round(xs.min() * 100, 2),
                        np.round(xs.max() * 100, 2),
                        xmax / 2 * 100,
                        xmax * 100],
           yticklabels=[ymin*100, ymin/2*100, 0, ymax/2*100, ymax*100])
    sns.despine(fig=fig, ax=ax)

    # inserted axes - current over the dipole
    axin.plot(xs, np.abs(Is))
    axin.set(xlabel='dipole',
             ylabel='$|I|$ (mA)',
             xticks=[xs.min(), xs.max()],
             yticks=[0, np.abs(Is).max()],
             xticklabels=[],
             yticklabels=[0, np.round(np.abs(Is).max() * 1000, 1)])
    axin.patch.set_alpha(0)
    plt.show()
    return fig


if __name__ == '__main__':
    fname = os.path.basename(__file__).removesuffix('.py')
    fig = main()
    fig.savefig(os.path.join('figures', f'{fname}.pdf'), bbox_inches='tight')

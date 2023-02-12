import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dosipy.constants import eps_0
from dosipy.utils.dataloader import load_tissue_diel_properties

from utils import reflection_coefficient


sns.set_theme(style='ticks',
              font_scale=1.5,
              palette='colorblind',
              rc={'text.usetex': True,
                  'text.latex.preamble': r'\usepackage{amsmath}',
                  'font.family': 'serif'})


def main():
    # frequencies in Hz
    f = np.array([1, 2, 5, 10, 20, 50, 100]) * 1e9  

    # load the dielectric data for dry skin
    sigma, eps_r, _, penetration_depth = np.vectorize(
        load_tissue_diel_properties
        )('skin_dry', f)

    # compute the power transmission coefficient
    eps_i = sigma / (2 * np.pi * f * eps_0)
    eps = eps_r - 1j * eps_i
    gamma = reflection_coefficient(eps)
    T_tr = 1 - gamma ** 2

    # visualize
    cs = sns.color_palette('colorblind', 2)
    fig, ax1 = plt.subplots()
    ax1.plot(f/1e9, penetration_depth * 1000, 'o-', lw=3, ms=7, c=cs[0],
             label='penetration\ndepth')
    ax1.tick_params(axis='y', labelcolor=cs[0])
    ax1.set(
        xscale='log',
        xlabel='frequency [GHz]',
        ylabel='penetration depth (mm)',
        yticks=[0, 25, 50],
        yticklabels=[0, 25, 50])
    ax1.legend(loc='upper left', frameon=False)
    ax2 = ax1.twinx()
    ax2.plot(f / 1e9, np.abs(T_tr), 's--',lw=3, ms=7, c=cs[1],
            label='transmission\ncoefficient')
    ax2.tick_params(axis='y', labelcolor=cs[1])
    ax2.set(
        xscale='log',
        ylabel='transmission coefficient',
        xticks=[1, 10, 100],
        xticklabels=[1, 10, 100],
        yticks=[0.4, 0.6, 0.8],
        yticklabels=[0.4, 0.6, 0.8],
        ylim=[0.385, 0.8])
    ax2.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    plt.show()
    return fig


if __name__ == '__main__':
    fname = os.path.basename(__file__).removesuffix('.py')
    fig = main()
    fig.savefig(os.path.join('figures', f'{fname}.pdf'), bbox_inches='tight')

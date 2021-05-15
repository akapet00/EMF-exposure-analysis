import os

import pandas as pd
from scipy.io import loadmat


SUPPORTED_TISSUES = ['air', 'blood', 'blood_vessel', 'body_fluid',
                     'bone_cancellous', 'bone_cortical', 'bone_marrow',
                     'brain_grey_matter', 'brain_white_matter', 'cerebellum',
                     'cerebro_spinal_fluid', 'dura', 'fat', 'muscle',
                     'skin_dry', 'skin_wet']
SUPPORTED_FREQS = [3., 3.5, 6., 10., 15., 20., 30., 40., 60., 80., 100.]


def load_tissue_diel_properties(tissue, f):
    """Return conductivity, relative permitivity, loss tangent and
    penetration depth of a given tissue based on a given frequency.

    Ref: Hasgall, PA; Di Gennaro, F; Baumgartner, C; Neufeld, E; Lloyd,
    B; Gosselin, MC; Payne, D; Klingenböck, A; Kuster, N. IT'IS
    Database for thermal and electromagnetic parameters of biological
    tissues, Version 4.0, May 15, 2018, DOI: 10.13099/VIP21000-04-0

    Parameters
    ----------
    tissue : str
        type of human tissue
    f : float
        radiation frequency

    Returns
    -------
    tuple
        tuple of 4 float values which represent conductivity, relative
        permitivity, loss tangent and penetration depth, respectively
    """
    if tissue not in SUPPORTED_TISSUES:
        raise ValueError(f'Unsupported tissue. Choose {SUPPORTED_TISSUES}.')
    if 1e9 > f > 100e9:
        raise ValueError('Invalid frequency. Choose in range [1, 100] GHz')
    tissue_diel_properties_path = os.path.join('data', 'tissue_properties',
                                               'tissue_diel_properties.csv')
    df = pd.read_csv(tissue_diel_properties_path)
    df = df[(df.frequency == f) & (df.tissue == tissue)]
    _, _, sigma, eps_r, tan_loss, pen_depth = df.to_numpy()[0]
    return (sigma, eps_r, tan_loss, pen_depth)


def load_antenna_el_properties(f):
    """Return the current distribution over the thin wire half-dipole
    antenna. The data are obtained by solving the Pocklington integro-
    differential equation by using the indirect-boundary element
    method.

    Ref: Poljak, D. Advanced modeling in computational electromagnetic
    compatibility, Wiley-Interscience; 1st edition (March 16, 2007)

    Parameters
    ----------
    f : float
        operating frequency in GHz

    Returns
    -------
    pandas.DataFrame
        current distribution over the wire alongside additional
        configuration details
    """
    assert f / 1e9 in SUPPORTED_FREQS, \
        (f'{f / 1e9} is not in supported. '
         f'Supported frequency values: {SUPPORTED_FREQS}.')
    data = loadmat(os.path.join('data', 'dipole', 'fs_current.mat'))['output']
    df = pd.DataFrame(data,
                      columns=['N', 'f', 'L', 'v', 'x', 'ireal', 'iimag'])
    df_f = df[df.f == f]
    df_f.reset_index(drop=True, inplace=True)
    return df_f


def load_sphere_coords(N):
    """Return the coordinates of a sphere representing a homogenous
    head model with diameter of 0.09 m.

    Parameters
    ----------
    N : int
        number of finite elements of a mesh

    Returns
    -------
    pandas.DataFrame
        (x, y, z) coordinates in m
    """
    try:
        filename = f'sphere_coord_n{N}.mat'
        coord_dict = loadmat(os.path.join('data', 'sphere', filename))
    except FileNotFoundError as e:
        print(e)
    else:
        df = pd.DataFrame(coord_dict['r_c'] / 100., columns=['x', 'y', 'z'])
    finally:
        pass
    return df


def load_head_coords():
    """Return the coordinates of a head model.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        (x, y, z) coordinates in m
    """
    try:
        filename = 'head_ijnme_simpl.csv'
        df = pd.read_csv(os.path.join('data', 'head', filename),
                         names=['y', 'x', 'z'])
    except FileNotFoundError as e:
        print(e)
    else:
        df = df + 0.05  # adjust
    finally:
        pass
    return df

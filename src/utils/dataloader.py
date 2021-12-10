import os

import pandas as pd
from scipy.io import loadmat


SUPPORTED_TISSUES = ['air', 'blood', 'blood_vessel', 'body_fluid',
                     'bone_cancellous', 'bone_cortical', 'bone_marrow',
                     'brain_grey_matter', 'brain_white_matter', 'cerebellum',
                     'cerebro_spinal_fluid', 'dura', 'fat', 'muscle',
                     'skin_dry', 'skin_wet']
SUPPORTED_FREQS = [3., 3.5, 6., 10., 15., 20., 26., 30., 40., 60., 80., 100.]


def load_tissue_diel_properties(tissue, f):
    """Return conductivity, relative permitivity, loss tangent and
    penetration depth of a given tissue based on a given frequency.

    Ref: Hasgall, PA; Di Gennaro, F; Baumgartner, C; Neufeld, E; Lloyd,
    B; Gosselin, MC; Payne, D; Klingenböck, A; Kuster, N. IT'IS
    Database for thermal and electromagnetic parameters of biological
    tissues, Version 4.0, 2018, DOI: 10.13099/VIP21000-04-0

    Parameters
    ----------
    tissue : str
        Type of human tissue.
    f : float
        Radiation frequency in Hz.

    Returns
    -------
    tuple
        Conductivity, relative permitivity, loss tangent and
        penetration depth.
    """
    if tissue not in SUPPORTED_TISSUES:
        raise ValueError(f'Unsupported tissue. Choose {SUPPORTED_TISSUES}.')
    if 1e9 > f > 100e9:
        raise ValueError('Invalid frequency. Choose in range [1, 100] GHz')
    tissue_diel_properties_path = os.path.join('data', 'target-model',
                                               'tissue_properties',
                                               'tissue_diel_properties.csv')
    df = pd.read_csv(tissue_diel_properties_path)
    df = df[(df.frequency == f) & (df.tissue == tissue)]
    _, _, sigma, eps_r, tan_loss, pen_depth = df.to_numpy()[0]
    return sigma, eps_r, tan_loss, pen_depth


def load_antenna_el_properties(f):
    """Return the current distribution over a thin wire, half-wave
    dipole antenna. The data are obtained by solving the Pocklington
    integro-differential equation by using the indirect-boundary
    element method.

    Ref: Poljak, D. Advanced modeling in computational electromagnetic
    compatibility, Wiley-Interscience, 1st edition, 2007

    Parameters
    ----------
    f : float
        Operating frequency in GHz.

    Returns
    -------
    pandas.DataFrame
        Current distribution over the wire alongside additional
        configuration details.
    """
    assert f / 1e9 in SUPPORTED_FREQS, \
        (f'{f / 1e9} is not in supported. '
         f'Supported frequency values: {SUPPORTED_FREQS}.')
    data = loadmat(os.path.join('data', 'source-model',
                                'dipole', 'fs_current.mat'))['output']
    df = pd.DataFrame(data,
                      columns=['N', 'f', 'L', 'v', 'x', 'ireal', 'iimag'])
    df_f = df[df.f == f]
    df_f.reset_index(drop=True, inplace=True)
    return df_f


def load_sphere_coords(N):
    """Return the coordinates of a sphere representing a homogenous
    head model with diameter that equals to 18 cm.

    Parameters
    ----------
    N : int
        Number of finite elements.

    Returns
    -------
    pandas.DataFrame
        x-, y- and z-coordinates.
    """
    try:
        fname = f'sphere_coord_n{N}.mat'
        coord_dict = loadmat(os.path.join('data', 'target-model',
                                          'spherical-head', fname))
    except FileNotFoundError as e:
        print(e)
    else:
        df = pd.DataFrame(coord_dict['r_c'] / 100., columns=['x', 'y', 'z'])
    finally:
        pass
    return df


def load_head_coords():
    """Return the coordinates of the head model.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        x-, y- and z-coordinates.
    """
    try:
        fname = 'head_ijnme_simpl.csv'
        df = pd.read_csv(os.path.join('data', 'target-model',
                                      'realistic-head', fname),
                         names=['y', 'x', 'z'])
    except FileNotFoundError as e:
        print(e)
    else:
        df = df + 0.05  # adjust
    finally:
        pass
    return df


def load_ear_data(mode, f):
    """Return the coordinates for the ear model, where each point in
    space has E and H field values precomputed for a given mode and
    a frequency of plane wave.

    Parameters
    ----------
    mode : str
        Either `TE` (transversal electric) or `TM` (transversal
        magnetic).
    f : float
        Frequency in GHz.

    Returns
    -------
    pandas.DataFrame
        x-, y- and z-coordinates with associated field components.
    """
    mode = str(mode).upper()
    f = int(f)
    assert mode in ['TE', 'TM'], 'Unrecognized mode.'
    assert f in [26, 60], 'Currently unsupported frequency.'
    try:
        dirname = os.path.join('data', 'target-model', 'realistic-ear')
        fname_E = f'E_3D_ear_{f}GHz_{mode}_surface.txt'
        fname_H = f'H_3D_ear_{f}GHz_{mode}_surface.txt'
        df_E = pd.read_csv(os.path.join(dirname, fname_E),
                           names=['x [mm]', 'y [mm]', 'z [mm]',
                                  'ExRe [V/m]', 'ExIm [V/m]',
                                  'EyRe [V/m]', 'EyIm [V/m]',
                                  'EzRe [V/m]', 'EzIm [V/m]',
                                  'area [mm^2]'],
                           header=None, delim_whitespace=True, skiprows=[0, 1])
        df_H = pd.read_csv(os.path.join(dirname, fname_H),
                           names=['x [mm]', 'y [mm]', 'z [mm]',
                                  'HxRe [A/m]', 'HxIm [A/m]',
                                  'HyRe [A/m]', 'HyIm [A/m]',
                                  'HzRe [A/m]', 'HzIm [A/m]',
                                  'area [mm^2]'],
                           header=None, delim_whitespace=True, skiprows=[0, 1])
    except FileNotFoundError as e:
        print(e)
    else:
        df = pd.concat([df_E[['x [mm]', 'y [mm]', 'z [mm]',
                              'ExRe [V/m]', 'ExIm [V/m]',
                              'EyRe [V/m]', 'EyIm [V/m]',
                              'EzRe [V/m]', 'EzIm [V/m]']],
                        df_H[['HxRe [A/m]', 'HxIm [A/m]',
                              'HyRe [A/m]', 'HyIm [A/m]',
                              'HzRe [A/m]', 'HzIm [A/m]',
                              'area [mm^2]']]],
                       axis=1, copy=False)
    finally:
        pass
    return df

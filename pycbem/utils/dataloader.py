import csv
import os

import pandas as pd
from scipy.io import loadmat


SUPPORTED_TISSUES = ['air', 'blood', 'blood_vessel', 'body_fluid',
                     'bone_cancellous', 'bone_cortical', 'bone_marrow',
                     'brain_grey_matter', 'brain_white_matter', 'cerebellum',
                     'cerebro_spinal_fluid', 'dura', 'fat', 'muscle',
                     'skin_dry', 'skin_wet']
SUPPORTED_FREQS = [3., 6., 10., 15., 20., 30., 40., 60., 80., 100.]


def load_tissue_diel_properties(tissue, frequency):
    r"""Return conductivity, relative permitivity, loss tangent and
    penetration depth of a given tissue based on a given frequency.

    Ref: Hasgall, PA; Di Gennaro, F; Baumgartner, C; Neufeld, E; Lloyd,
    B; Gosselin, MC; Payne, D; Klingenböck, A; Kuster, N. IT'IS
    Database for thermal and electromagnetic parameters of biological
    tissues, Version 4.0, May 15, 2018, DOI: 10.13099/VIP21000-04-0

    Parameters
    ----------
    tissue : str
        type of human tissue
    frequency : float
        radiation frequency

    Returns
    -------
    tuple
        tuple of 4 float values which represent conductivity, relative
        permitivity, loss tangent and penetration depth, respectively
    """
    if tissue not in SUPPORTED_TISSUES:
        raise ValueError(f'Unsupported tissue. Choose {SUPPORTED_TISSUES}.')
    if 1e9 > frequency > 100e9:
        raise ValueError('Invalid frequency. Choose in range [1, 100] GHz')
    tissue_diel_properties_path = os.path.join('data',
                                               'tissue_diel_properties.csv')
    with open(tissue_diel_properties_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if str(row[0]) == tissue and float(row[1]) == frequency:
                conductivity = float(row[2])
                relative_permitivity = float(row[3])
                loss_tangent = float(row[4])
                penetration_depth = float(row[5])
        return (conductivity, relative_permitivity, loss_tangent,
                penetration_depth)


def load_antenna_el_properties(frequency):
    r"""Return the current distribution over the thin wire half-dipole
    antenna. The data are obtained by solving the Pocklington integro-
    differential equation by using the indirect-boundary element
    method.

    Ref: Poljak, D. Advanced modeling in computational electromagnetic
    compatibility, Wiley-Interscience; 1st edition (March 16, 2007)

    Parameters
    ----------
    frequency : float
        operating frequency in GHz

    Returns
    -------
    numpy.ndarray
        current distribution over the wire
    """
    assert frequency / 1e9 in SUPPORTED_FREQS, \
        (f'{frequency / 1e9} is not in supported. '
         f'Supported frequency values: {SUPPORTED_FREQS}.')
    data = loadmat(os.path.join('data', 'current.mat'))['output']
    df = pd.DataFrame(data,
                      columns=['L', 'N', 'r', 'f', 'x', 'ireal', 'iimag'])
    df_f = df[df.f == frequency]
    df_f.reset_index(drop=True, inplace=True)
    return df_f

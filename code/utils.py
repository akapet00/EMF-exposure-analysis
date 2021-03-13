import csv
import os


SUPPORTED_TISSUES = [
    'air', 'blood', 'blood_vessel', 'body_fluid', 'bone_cancellous',
    'bone_cortical', 'bone_marrow', 'brain_grey_matter', 'brain_white_matter',
    'cerebellum', 'cerebro_spinal_fluid', 'dura', 'fat', 'muscle', 'skin_dry',
    'skin_wet',
    ]


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
    tissue_diel_properties_path = os.path.join('data', 'tissue_diel_properties.csv')
    with open(tissue_diel_properties_path) as f: 
        reader = csv.reader(f) 
        for row in reader:
            if str(row[0])==tissue and float(row[1])==frequency: 
                conductivity = float(row[2]) 
                relative_permitivity = float(row[3]) 
                loss_tangent = float(row[4]) 
                penetration_depth = float(row[5])
        return (conductivity, relative_permitivity, loss_tangent, penetration_depth)
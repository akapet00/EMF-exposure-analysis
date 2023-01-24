import os

import numpy as np
import pandas as pd


def load_raw_data(antenna, distance, drop_idx=None):
    """Load coordinates and electromagnetic field components as
    exported from CST.

    Parameters
    ----------
    antenna : str
        Which antenna.
    distance : int
        Antenna-to-ear separation distance in mm.
    drop_idx : int, optional
        Drop specific index in some edge cases.

    Returns
    -------
    tuple
        Tree element tuple contains dataframe with coordinates, and two
        series with x-, y-, and z-component of the electric and
        magnetic field, respectively.
    """
    # data path
    path = os.path.join('data', 'raw')
    
    # E field components
    ExRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Ex_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, names=['x', 'y', 'z', 'value']
    )
    ExIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Ex_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    EyRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Ey_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    EyIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Ey_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    EzRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Ez_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    EzIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Ez_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    
    # H field components
    HxRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Hx_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    HxIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Hx_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    HyRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Hy_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    HyIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Hy_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    HzRe_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Re_Hz_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    HzIm_df = pd.read_csv(
        os.path.join(path, f'{antenna}_Im_Hz_d{distance}mm.txt'),
        sep='\s+', comment='%', header=None, usecols=[3], names=['value']
    )
    
    # coordinates
    xyz = ExRe_df.loc[:, ['x', 'y', 'z']]
    
    if drop_idx:
        if (antenna == 'DipoleVertical') and (distance == 5):
            xyz = xyz.drop(index=drop_idx).reset_index(drop=True)
            ExRe_df = ExRe_df.drop(index=drop_idx).reset_index(drop=True)
            ExIm_df = ExIm_df.drop(index=drop_idx).reset_index(drop=True)
            EyRe_df = EyRe_df.drop(index=drop_idx).reset_index(drop=True)
            EzRe_df = EzRe_df.drop(index=drop_idx).reset_index(drop=True)
            EzIm_df = EzIm_df.drop(index=drop_idx).reset_index(drop=True)
            HxRe_df = HxRe_df.drop(index=drop_idx).reset_index(drop=True)
            HxIm_df = HxIm_df.drop(index=drop_idx).reset_index(drop=True)
            HyRe_df = HyRe_df.drop(index=drop_idx).reset_index(drop=True)
            HyIm_df = HyIm_df.drop(index=drop_idx).reset_index(drop=True)
            HzRe_df = HzRe_df.drop(index=drop_idx).reset_index(drop=True)
            HzIm_df = HzIm_df.drop(index=drop_idx).reset_index(drop=True)
        elif (antenna == 'DipoleVertical') and (distance == 15):
            xyz = xyz.drop(index=drop_idx).reset_index(drop=True)
            ExRe_df = ExRe_df.drop(index=drop_idx).reset_index(drop=True)
            ExIm_df = ExIm_df.drop(index=drop_idx).reset_index(drop=True)
            EyRe_df = EyRe_df.drop(index=drop_idx).reset_index(drop=True)
            EyIm_df = EyIm_df.drop(index=drop_idx).reset_index(drop=True)
            EzRe_df = EzRe_df.drop(index=drop_idx).reset_index(drop=True)
            EzIm_df = EzIm_df.drop(index=drop_idx).reset_index(drop=True)
            HxRe_df = HxRe_df.drop(index=drop_idx).reset_index(drop=True)
            HxIm_df = HxIm_df.drop(index=drop_idx).reset_index(drop=True)
            HyRe_df = HyRe_df.drop(index=drop_idx).reset_index(drop=True)
            HzRe_df = HzRe_df.drop(index=drop_idx).reset_index(drop=True)
            HzIm_df = HzIm_df.drop(index=drop_idx).reset_index(drop=True)
        elif (antenna == 'DipoleHorizontal') and (distance == 15):
            xyz = xyz.drop(index=drop_idx).reset_index(drop=True)
            ExRe_df = ExRe_df.drop(index=drop_idx).reset_index(drop=True)
            ExIm_df = ExIm_df.drop(index=drop_idx).reset_index(drop=True)
            EyRe_df = EyRe_df.drop(index=drop_idx).reset_index(drop=True)
            EyIm_df = EyIm_df.drop(index=drop_idx).reset_index(drop=True)
            EzRe_df = EzRe_df.drop(index=drop_idx).reset_index(drop=True)
            EzIm_df = EzIm_df.drop(index=drop_idx).reset_index(drop=True)
            HxIm_df = HxIm_df.drop(index=drop_idx).reset_index(drop=True)
            HyRe_df = HyRe_df.drop(index=drop_idx).reset_index(drop=True)
            HyIm_df = HyIm_df.drop(index=drop_idx).reset_index(drop=True)
            HzRe_df = HzRe_df.drop(index=drop_idx).reset_index(drop=True)
            HzIm_df = HzIm_df.drop(index=drop_idx).reset_index(drop=True)
            
    # assemble all
    Ex = ExRe_df['value'] + 1j * ExIm_df['value']
    Ey = EyRe_df['value'] + 1j * EyIm_df['value']
    Ez = EzRe_df['value'] + 1j * EzIm_df['value']
    Hx = HxRe_df['value'] + 1j * HxIm_df['value']
    Hy = HyRe_df['value'] + 1j * HyIm_df['value']
    Hz = HzRe_df['value'] + 1j * HzIm_df['value']
    
    return xyz, (Ex, Ey, Ez), (Hx, Hy, Hz)


def load_clean_data(antenna, distance):
    """Load clean dataset.

    Parameters
    ----------
    antenna : str
        Which antenna.
    distance : int
        Antenna-to-ear separation distance in mm.

    Returns
    -------
    pandas.DataFrame
        Clean data.
    """
    path = os.path.join('data', 'clean')
    df = pd.read_csv(os.path.join(path, f'{antenna}_d{distance}mm.csv'),
                     index_col=0)
    return df


def load_processed_data(antenna, distance):
    """Load an augmented dataset.

    Parameters
    ----------
    antenna : str
        Which antenna.
    distance : int
        Antenna-to-ear separation distance in mm.

    Returns
    -------
    pandas.DataFrame
        Augmented data.
    """
    path = os.path.join('data', 'processed')
    df = pd.read_csv(os.path.join(path, f'{antenna}_d{distance}mm.csv'),
                     index_col=0)
    return df


def load_apd_data():
    """Load APD values.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        APD data.
    """
    path = os.path.join('data', 'results')
    df = pd.read_csv(os.path.join(path, 'Results_APD.csv'),
                     index_col=0)
    return df


def compute_power_density(E, H):
    """Return the complex power density vector from the electric and
    magnetic field. This function assumes that field components are
    given as peak values.

    Parameters
    ----------
    E : tuple
        3 arrays for 3 components of the electric field vector.
    H : tuple
        3 arrays for 3 components of the magnetic field vector.

    Returns
    -------
    tuple
        Containg 3 arrays for 3 components of the Poynting vector.
    """
    Sx = 0.5 * (E[1] * H[2].conjugate() - E[2] * H[1].conjugate())
    Sy = 0.5 * (E[2] * H[0].conjugate() - E[0] * H[2].conjugate())
    Sz = 0.5 * (E[0] * H[1].conjugate() - E[1] * H[0].conjugate())
    return Sx, Sy, Sz


def estimate_normals(xyz, take_every=1, knn=30, fast=True):
    """Return estimated normals for a given point cloud.

    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    take_every : int, optional
        How many points to skip in the point cloud when estimating
        normal vectors.
    knn : int, optional
        Number of neighbors for KDTree search.
    fast : bool, optional
        If True, the normal estimation uses a non-iterative method to
        extract the eigenvector from the covariance matrix. This is
        faster, but is not as numerical stable.

    Returns
    -------
    numpy.ndarray
        The number of rows correspond to the number of rows of a given
        point cloud, and each column corresponds to each component of
        the normal vector.
    """
    import open3d as o3d
    xyz = xyz[::take_every, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn),
                         fast_normal_computation=fast)
    pcd.normalize_normals()
    n = np.asarray(pcd.normals)
    return n


def normals_to_rgb(n):
    """Return RGB color representation of unit vectors.
    
    Ref: Ben-Shabat et al., in proceedings of CVPR 2019, pp. 10104-10112,
         doi: 10.1109/CVPR.2019.01035.
    
    Parameters
    ----------
    n : numpy.ndsarray
        The number of rows correspond to the number of rows of a given
        point cloud, each column corresponds to each component of a
        (normalized) normal vector.

    Returns
    -------
    numpy.ndarray
        corresponding RGB color on the RGB cube
    """
    if n.shape[1] != 3:
        raise ValueError('`n` should be a 3-dimensional vector.')
    n = np.divide(
        n, np.tile(
            np.expand_dims(
                np.sqrt(np.sum(np.square(n), axis=1)), axis=1), [1, 3]))
    rgb = 127.5 + 127.5 * n
    return rgb / 255.0


def export_rect_idx(xyz, center, edge_length, view='xy'):
    """Extract specific points that correspond to a rectangle, defined
    with a central point and its edge length, from a point cloud.

    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    center : tuple or list
        z- and y-coordinate that defines the center of a desired
        rectangle.
    edge_length : float
        Edge length of a desired rectangle.
    view : string
        Point of view for point extraction. Currently supported `xy`
        and `yz`.

    Returns
    -------
    tuple
        Origin of a desired rectangle and indexes of all points from a point
        cloud that falls into a rectangle.
    """
    x_bound = [center[0] - edge_length / 2, center[0] + edge_length / 2]
    y_bound = [center[1] - edge_length / 2, center[1] + edge_length / 2]
    origin = [x_bound[0], y_bound[0]]
    if view == 'xy':
        col_idx = 0
    elif view == 'yz':
        col_idx = 1
    else:
        raise ValueError(f'Not supported view: {view}')
    idx_rect = np.where((xyz[:, col_idx] > x_bound[0])
                        & (xyz[:, col_idx] < x_bound[1])
                        & (xyz[:, 1] > y_bound[0])
                        & (xyz[:, 1] < y_bound[1]))[0]
    return origin, idx_rect


def diff(fun, arg=0):
    """Central finite differentiation of 2-D function.
    
    Parameters
    ----------
    fun : callable
        2-D function which is differentiated.
    arg : int, optional
        Argument over which `fun` is differentiated.
    
    Returns
    -------
    callable
        First order numerical derivative of `fun`.
    """
    def df(points, eps=1e-3):
        if arg == 0:
            return (fun(np.c_[points[:, 0] + eps, points[:, 1]])
                    - fun(np.c_[points[:, 0] - eps, points[:, 1]])) / (2 * eps)
        elif arg == 1:
            return (fun(np.c_[points[:, 0], points[:, 1] + eps])
                    - fun(np.c_[points[:, 0], points[:, 1] - eps])) / (2 * eps)
        else:
            raise ValueError('Unsupported `arg`.')
    return df

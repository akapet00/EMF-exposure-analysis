"""Utility functions."""

import numpy as np
import open3d as o3d


def clean_df(df):
    """Remove the points that correspond to the external surface of the
    Simulia/CST simulation domain.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw output Simulia/CST data.

    Returns
    -------
    pandas.DataFrame
        Electric and magnetic field components only at the ear surface.
    """
    df = df[(df['x [mm]'] != df['x [mm]'].min())
            & (df['x [mm]'] != df['x [mm]'].max())
            & (df['y [mm]'] != df['y [mm]'].min())
            & (df['y [mm]'] != df['y [mm]'].max())
            & (df['z [mm]'] != df['z [mm]'].min())
            & (df['z [mm]'] != df['z [mm]'].max())]
    df.reset_index(drop=True, inplace=True)
    return df


def export_pcd(df, area=False):
    """Convert the clean dataframe to point cloud.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean version of the dataframe consisting the Simulia/CST
        output data.
    area : Bool, optional
        If True, the output will have area of the finite element
        corresponding to the specific point.

    Returns
    -------
    numpy.ndarray
        Either array of shape (n, 3) where each column corresponds to
        x-, y-, and z-coordinates of the ear model, or array of shape
        (n, 4) where additional column corresponds to the area of the
        finite element corresponding to the specific point.
    """
    if area:
        pcd = np.c_[df['x [mm]'].to_numpy(),
                    df['y [mm]'].to_numpy(),
                    df['z [mm]'].to_numpy(),
                    df['area [mm^2]'].to_numpy()]
    else:
        pcd = np.c_[df['x [mm]'].to_numpy(),
                    df['y [mm]'].to_numpy(),
                    df['z [mm]'].to_numpy()]
    return pcd


def visualize_ear(xyz, show=False, **kwargs):
    """Return point cloud object and visualize the ear model in open3d
    visualizer.

    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    show : bool, optional
        Fire up `open3d` visualizer.
    kwargs : dict, optional
        Additional keyword arguments for
        `open3d.visualization.draw_geometries` function.

    Returns
    -------
    None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    center = pcd.get_center()
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=9, origin=center+np.array([6, -25, -20])
    )
    pcd.estimate_normals()
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    if show:
        o3d.visualization.draw_geometries([cframe, pcd, rec_mesh], **kwargs)
    return cframe, pcd


def get_imcolors(geometries, config):
    """Return colors from given 3-D point cloud objects.

    Parameters
    ----------
    geometries : list
        List of 3-D point cloud models defined as
        `open3d.geometry.PointCloud` objects.
    config : dict
        View controls for visualizer.

    Returns
    -------
    open3d.geometry.Image
        Captured image RGB.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.update_renderer()
    vis.get_view_control().set_zoom(config['zoom'])
    vis.get_view_control().set_front(config['front'])
    vis.get_view_control().set_lookat(config['lookat'])
    vis.get_view_control().set_up(config['up'])
    vis.poll_events()
    color = vis.capture_screen_float_buffer()
    vis.destroy_window()
    return color


def export_fields(df):
    """Convert the clean dataframe to array of electromagnetic field
    components.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean version of the dataframe consisting the Simulia/CST
        output data.

    Returns
    -------
    tuple
        First element of the tuple holds the 3 arrays for 3 components
        of the electric field vector, while the second element holds
        the 3 arrays each corresponding the the 3 components of the
        magnetic field vector.
    """
    Ex = df['ExRe [V/m]'].to_numpy() + 1j * df['ExIm [V/m]'].to_numpy()
    Ey = df['EyRe [V/m]'].to_numpy() + 1j * df['EyIm [V/m]'].to_numpy()
    Ez = df['EzRe [V/m]'].to_numpy() + 1j * df['EzIm [V/m]'].to_numpy()
    Hx = df['HxRe [A/m]'].to_numpy() + 1j * df['HxIm [A/m]'].to_numpy()
    Hy = df['HyRe [A/m]'].to_numpy() + 1j * df['HyIm [A/m]'].to_numpy()
    Hz = df['HzRe [A/m]'].to_numpy() + 1j * df['HzIm [A/m]'].to_numpy()
    return (Ex, Ey, Ez), (Hx, Hy, Hz)


def poynting_vector(E, H):
    """Return power density given x-, y-, and z-component of electric
    and magnetic field. It assumes that the given components are max
    values and all components of the power dansity are scaled by 1/2.

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
    xyz = xyz[::take_every, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn),
                         fast_normal_computation=fast)
    pcd.normalize_normals()
    n = np.asarray(pcd.normals)
    return n


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
        and `zy`.

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
    elif view == 'zy':
        col_idx = 2
    else:
        raise ValueError(f'Not supported view: {view}')
    idx_rect = np.where((xyz[:, col_idx] > x_bound[0])
                        & (xyz[:, col_idx] < x_bound[1])
                        & (xyz[:, 1] > y_bound[0])
                        & (xyz[:, 1] < y_bound[1]))[0]
    return origin, idx_rect


def export_circ_idx(xyz, center, radius, view='xy'):
    """Extract specific points that correspond to a disk, defined
    with a central point and its radius, from a point cloud.

    Parameters
    ----------
    xyz : numpy.ndarray
        Point cloud defining a model in 3-D.
    center : tuple or list
        z- and y-coordinate that defines the center of a desired
        disk.
    radius : float
        Radius of a desired disk.
    view : string
        Point of view for point extraction. Currently supported `xy`
        and `zy`.

    Returns
    -------
    numpy.ndarray
        Indexes of all points from a point cloud that falls into a
        circle describing a disk.
    """
    cx, cy = center
    if view == 'xy':
        col_idx = 0
    elif view == 'zy':
        col_idx = 2
    else:
        raise ValueError(f'Not supported view: {view}')
    idx_circ = np.where(
        (xyz[:, col_idx] - cx) ** 2 + (xyz[:, 1] - cy) ** 2 < radius ** 2)[0]
    return idx_circ


def diff_in_dB(apd1, apd2):
    """Return the difference in dB between two values for the power.

    Parameters
    ----------
    apd1 : float or numpy.ndarray
        Absorbed power density, or anything else measured in Watts.
    apd2 : float or numpy.ndarray
        Absorbed power density, or anything else measured in Watts to
        be compared to `apd1`.

    Returns
    -------
    float or numpy.ndarray
        Difference(s) in dB between two (arrays of) values of the
        power.
    """
    diff = np.abs(10 * np.log10(apd1 / apd2))
    return diff

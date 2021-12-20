import os

from matplotlib.patches import Rectangle, Circle
import numpy as np
try:
    import open3d as o3d
except ImportError:
    import sys
    print(sys.exc_info())

from dosipy.utils.dataloader import load_ear_data
from dosipy.utils.integrate import elementwise_dblquad, elementwise_circquad
from dosipy.utils.viz import (set_colorblind, colormap_from_array, scatter_2d,
                              fig_config, save_fig)
from helpers import (clean_df, export_pcd, export_fields, poynting_vector,
                     estimate_normals, diff_in_dB)


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


# fetch the data
mode = 'tm'
frequency = 60
df = load_ear_data(mode, frequency)
df = clean_df(df)
xyz = export_pcd(df)
E, H = export_fields(df)
Sx, Sy, Sz = poynting_vector(E, H)

# removing the sharp-edged region of the model and computing the real
# part of the power density normal to the surface
crop_idxs = np.where(xyz[:, 0] > 67)[0]
xyz_crop = xyz[crop_idxs]
Sx_crop, Sy_crop, Sz_crop = Sx[crop_idxs], Sy[crop_idxs], Sz[crop_idxs]
n_crop = estimate_normals(xyz_crop, knn=30, fast=True)
Sr_crop = abs(Sx_crop.real * n_crop[:, 0]
              + Sy_crop.real * n_crop[:, 1]
              + Sz_crop.real * n_crop[:, 2])

# define coordinate frame in open3d for cropped ear model
pcd_crop = o3d.geometry.PointCloud()
pcd_crop.points = o3d.utility.Vector3dVector(xyz_crop)
center_crop = pcd_crop.get_center()
pcd_crop.paint_uniform_color([0.5, 0.5, 0.5])

# translate the coordinates of the model to have the center at (0, 0, 0)
xyz_crop_t = np.c_[xyz_crop[:, 0] - center_crop[0],
                   xyz_crop[:, 1] - center_crop[1],
                   xyz_crop[:, 2] - center_crop[2]]
pcd_crop_t = o3d.geometry.PointCloud()
pcd_crop_t.points = o3d.utility.Vector3dVector(xyz_crop_t)
center_crop_t = pcd_crop_t.get_center()
pcd_crop_t.paint_uniform_color([0.5, 0.5, 0.5])
cframe_crop_t = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=center_crop_t
)

# extract the points visible from the xy-plane point of view
diameter = np.linalg.norm(
    pcd_crop_t.get_max_bound() - pcd_crop_t.get_min_bound()
)
radius = 10 ** 5.5
camera = [0, 0, -diameter]
_, pt_map = pcd_crop_t.hidden_point_removal(camera, radius)
xyz_crop_t_xy = xyz_crop_t[pt_map]
Sr_crop_t_xy = Sr_crop[pt_map]
pcd_crop_t_xy = o3d.geometry.PointCloud()
pcd_crop_t_xy.points = o3d.utility.Vector3dVector(xyz_crop_t_xy)
colors = colormap_from_array(Sr_crop_t_xy)
pcd_crop_t_xy.colors = o3d.utility.Vector3dVector(colors)

# extract rectangular averaging suface area
avg_center = [0.21, 3.25]
edge_length = 10
area = edge_length ** 2
origin, idx_rect = export_rect_idx(xyz=xyz_crop_t_xy,
                                   center=avg_center,
                                   edge_length=edge_length,
                                   view='xy')
xyz_rect = xyz_crop_t_xy[idx_rect]
Sr_rect = Sr_crop_t_xy[idx_rect]

# extract circular averaging suface area
radius = np.sqrt(area / np.pi)
idx_circ = export_circ_idx(xyz=xyz_crop_t_xy,
                           center=avg_center,
                           radius=radius,
                           view='xy')
xyz_circ = xyz_crop_t_xy[idx_circ]
Sr_circ = Sr_crop_t_xy[idx_circ]

# visualize the real part of the power density normal to the extracted
# rectangular and circular averaging surfaces
set_colorblind()
fig_config(latex=True, scaler=1.5, text_size=18)
Sr_label = r'$1/2$ $\Re{[\vec{E}\times\vec{H}^{*}]} \cdot \vec{n}$ [W/m$^2$]'
fig, ax = scatter_2d({'$x$ [mm]': xyz_crop_t_xy[:, 0],
                      '$y$ [mm]': xyz_crop_t_xy[:, 1],
                      Sr_label: Sr_crop_t_xy}, s=0.1)
patch_rect = Rectangle(origin, edge_length, edge_length, fc='None', lw=2)
patch_circ = Circle(avg_center, radius, fc='None', lw=2)
ax.add_patch(patch_rect)
ax.add_patch(patch_circ)
ax.invert_xaxis()
fname = os.path.join('figures', f'apd_max_ear_model_{mode}{frequency}')
save_fig(fig, fname=fname, formats=['png'])

# zoom in to the rectangular integration surface
set_colorblind()
fig_config(latex=True, scaler=1.5, text_size=18)
fig, ax = scatter_2d({'$x$ [mm]': xyz_rect[:, 0],
                      '$y$ [mm]': xyz_rect[:, 1],
                      Sr_label: Sr_rect}, s=20)
APD_rect = elementwise_dblquad(points=np.c_[xyz_rect[:, 0], xyz_rect[:, 1]],
                               values=Sr_rect,
                               degree=11) / area
ax.set_title(f'APD = ${APD_rect:.4f}$ W/m$^2$')
ax.invert_xaxis()
fname = os.path.join('figures', f'apd_max_rectangular_surf_{mode}{frequency}')
save_fig(fig, fname=fname, formats=['png'])

# zoom in to the circular integration surface
set_colorblind()
fig_config(latex=True, scaler=1.5, text_size=18)
fig, ax = scatter_2d({'$x$ [mm]': xyz_circ[:, 0],
                      '$y$ [mm]': xyz_circ[:, 1],
                      Sr_label: Sr_circ}, s=20)
APD_circ = elementwise_circquad(points=np.c_[xyz_circ[:, 0], xyz_circ[:, 1]],
                                values=Sr_circ,
                                radius=radius,
                                center=avg_center,
                                degree=11) / area
ax.set_title(f'APD = ${APD_circ:.4f}$ W/m$^2$')
ax.invert_xaxis()
fname = os.path.join('figures', f'apd_max_circular_surf_{mode}{frequency}')
save_fig(fig, fname=fname, formats=['png'])

# difference in dB between APDs computed on rectangular-
# and circular-shaped averaging surface
APD_diff = diff_in_dB(APD_rect, APD_circ)
print(f'APD_diff = {APD_diff:.7f} dB')

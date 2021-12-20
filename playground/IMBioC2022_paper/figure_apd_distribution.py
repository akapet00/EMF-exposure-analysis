import os

import numpy as np
try:
    import open3d as o3d
except ImportError:
    import sys
    print(sys.exc_info())

from dosipy.utils.dataloader import load_ear_data
from dosipy.utils.viz import set_colorblind, fig_config, scatter_3d, save_fig
from helpers import (clean_df, export_pcd, export_fields, poynting_vector,
                     estimate_normals)


# fetch the data
mode = 'te'
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

# translate the coordinates of the model to have the center at (0, 0, 0)
xyz_crop_t = np.c_[xyz_crop[:, 0] - center_crop[0],
                   xyz_crop[:, 1] - center_crop[1],
                   xyz_crop[:, 2] - center_crop[2]]

# scatter plot
set_colorblind()
fig_config(latex=True, scaler=2, text_size=18)
Sr_label = '$1/2$ $\\Re{[\\vec{E}\\times\\vec{H}^{*}]} \\cdot \\vec{n}$ [W/m2]'
fig, ax = scatter_3d({'$z$ [mm]': xyz_crop_t[:, 2],
                      '$x$ [mm]': xyz_crop_t[:, 0],
                      '$y$ [mm]': xyz_crop_t[:, 1],
                      Sr_label: Sr_crop},
                     elev=[15], azim=[150])
# manually turn off visibility of panes -.-
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# and also manually move labels a bit
ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 12

# save figure
fname = os.path.join('figures', f'apd_distribution_{mode}{frequency}')
save_fig(fig, fname=fname, formats=['png'])

import os

import matplotlib.pyplot as plt
import numpy as np
try:
    import open3d as o3d
except ImportError:
    import sys
    print(sys.exc_info())

from dosipy.utils.dataloader import load_ear_data
from dosipy.utils.viz import set_colorblind, save_fig, fig_config
from helpers import clean_df, export_pcd, get_imcolors


# fetch full dataset
df = load_ear_data('tm', 60)
df = clean_df(df)
xyz = export_pcd(df)

# extract colors from open3d visualizer
view_config = {
    'zoom': 0.69999999999999996,
    'front': [0.92231085610160646, 0.17546582462905541, 0.34431733779228646],
    'lookat': [71.236805645281521, 22.531933429935712, -8.12589641784127],
    'up': [-0.16595758534247468, 0.98447554162242001, -0.057148821637356101],
}
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
color = get_imcolors([cframe, pcd, rec_mesh], view_config)

# map colors into matplotlib image
set_colorblind()
fig_config()
fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(color, origin='upper')

# save figure
fname = os.path.join('figures', 'ear_model')
save_fig(fig, fname=fname)

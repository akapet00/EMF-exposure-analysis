import os
import datetime
import time
import itertools
import logging
from multiprocessing import log_to_stderr
from concurrent.futures import ProcessPoolExecutor


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
try:
    import open3d as o3d
except ImportError:
    import sys
    print(sys.exc_info())
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

from dosipy.utils.integrate import elementwise_dblquad
from dosipy.utils.viz import scatter_2d, scatter_3d
from utils import load_processed_data, export_rect_idx


def remove_hidden_yz(xyz, radius=100):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    diameter = np.linalg.norm(
        pcd.get_max_bound() - pcd.get_min_bound()
    )
    center = pcd.get_center()
    camera = [center[0] - diameter, center[1], center[2]]
    _, mask = pcd.hidden_point_removal(camera, radius)
    return mask, center


def find_ps_apd(xyz_yz, pd_yz, center, edge_length=0.02):
    res = {'origin': [], 'apd': []}
    area = edge_length ** 2
    for yc in np.linspace(center[1] - edge_length / 3,
                          center[1] + edge_length / 3, 21):
        for zc in np.linspace(center[2] - edge_length / 3,
                              center[2] + edge_length / 3, 21):
            origin, rect_idx = export_rect_idx(xyz=xyz_yz,
                                               center=[yc, zc],
                                               edge_length=edge_length,
                                               view='yz')
            xyz_rect = xyz_yz[rect_idx]
            pd_rect = pd_yz[rect_idx]
            integration_points = np.c_[xyz_rect[:, 2], xyz_rect[:, 1]]
            apd = elementwise_dblquad(integration_points,
                                      values=pd_rect,
                                      degree=31,
                                      interp_func=LinearNDInterpolator,
                                      fill_value=0
                                      ) / area
            res['origin'].append(origin)
            res['apd'].append(apd)
    ps_idx = np.argmax(res['apd'])
    ps_apd = res['apd'][ps_idx]
    ps_origin = res['origin'][ps_idx]
    return ps_origin, ps_apd


def worker(args):
    # unpack params
    antenna, distance = args
    
    # load data
    df = load_processed_data(antenna, distance)
    xyz = df[['x', 'y', 'z']].values
    n = df[['nx', 'ny', 'nz']].values
    pd = df['PD'].values

    # hidden points removal, yz-plane
    mask, center = remove_hidden_yz(xyz)
    xyz_yz = xyz[mask]
    pd_yz = pd[mask]

    # find the peak spatially-averaged absorbed power density
    ps_origin, ps_apd = find_ps_apd(xyz_yz, pd_yz, center)
    print(f'{antenna} at {distance} mm | '
          f'APD = {ps_apd:.2f} W/m2 at {ps_origin})')
    return ps_origin, ps_apd


antennas = ['DipoleVertical', 'DipoleHorizontal',
            'ArrayVertical', 'ArrayHorizontal']
distances = [5, 10, 15]
args = [p for p in itertools.product(antennas, distances)]
logger = log_to_stderr(logging.INFO)
start_time = time.perf_counter()
with ProcessPoolExecutor() as executor:
    res = list(tqdm(executor.map(worker, args), total=len(args)))
elapsed = time.perf_counter() - start_time

import os
import datetime
import time
import itertools
import logging
from multiprocessing import log_to_stderr
from concurrent.futures import ProcessPoolExecutor

import jax
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from pycbem.field import efield, hfield
from pycbem.utils.dataloader import load_antenna_el_properties


def _pd(row):
    """Power density."""
    return (row['Ey'] * row['Hz'].conjugate()
            - row['Ez'] * row['Hy'].conjugate(),
            row['Ex'] * row['Hz'].conjugate(),
            row['Ex'] * row['Hy'].conjugate())


def _worker(iter_args):
    """Single process worker for multiprocessing pool.

    Parameters
    ----------
    iter_args : tuple
        (number of finite elements of a meshed sphere,
         half-wave dipole operating frequency in Hz,
         distance between the antenna and the head model in m)

    Returns
    -------
    None
    """
    N, f, d = iter_args
    try:
        in_filename = f'sphere_coord_n{N}.mat'
        coord_dict = sio.loadmat(os.path.join('data', 'sphere', in_filename))
    except FileNotFoundError as e:
        print(e)
    else:
        r_c = pd.DataFrame(coord_dict['r_c'] / 100., columns=['x', 'y', 'z'])
    finally:
        pass
    antenna_data = load_antenna_el_properties(f)
    Is = antenna_data.ireal.to_numpy() + antenna_data.iimag.to_numpy() * 1j
    xs = antenna_data.x.to_numpy()
    xs = xs - xs.max() / 2
    ys = np.zeros_like(xs) + r_c['y'].min() + d
    zs = np.zeros_like(xs)

    # E field
    E = r_c.apply(
        lambda row: efield(row['x'], row['y'], row['z'], xs, ys, zs, Is, f),
        axis=1, result_type='expand')
    E.columns = ['Ex', 'Ey', 'Ez']
    E_abs = E.apply(
        lambda row: np.sqrt(row['Ex'] ** 2 + row['Ey'] ** 2 + row['Ez'] ** 2),
        axis=1)
    E.loc[:, 'E_abs'] = E_abs

    # H field
    H = r_c.apply(
        lambda row: hfield(row['x'], row['y'], row['z'], xs, ys, zs, Is, f),
        axis=1, result_type='expand')
    H.columns = ['Hx', 'Hy', 'Hz']
    H_abs = H.apply(
        lambda row: np.sqrt(row['Hx'] ** 2 + row['Hy'] ** 2 + row['Hz'] ** 2),
        axis=1)
    H.loc[:, 'H_abs'] = H_abs

    # update dataframe
    r_c_calc = pd.concat([r_c, E, H], axis=1)

    # EM power density
    S = r_c_calc.apply(_pd, axis=1, result_type='expand')
    S.columns = ['Sx', 'Sy', 'Sz']
    S_abs = S.apply(
        lambda row: np.sqrt(row['Sx'] ** 2 + row['Sy'] ** 2 + row['Sz'] ** 2),
        axis=1)
    S.loc[:, 'S_abs'] = S_abs

    # re-update dataframe
    r_c_calc = pd.concat([r_c_calc, S], axis=1)
    r_c_calc = r_c_calc.astype({c: np.complex128
                                for c in r_c_calc.columns[3:]})
    r_c_calc_dict = {name: col.values for name, col in r_c_calc.items()}

    # store dataframe into hdf5 and .mat
    out_filename = f'sphere_n{N}_d{d * 1e3}_f{f / 1e9}'
    r_c_calc.to_hdf(os.path.join('simulations', out_filename + '.h5'),
                    key='df', mode='w')
    sio.savemat(os.path.join('simulations', out_filename + '.mat'),
                r_c_calc_dict)


def main():
    jax.config.update("jax_enable_x64", True)
    logger = log_to_stderr(logging.INFO)
    logger.info(f'Execution started at {datetime.datetime.now()}')
    start_time = time.perf_counter()
    N = [480, 656, 1304, 1512, 2312]
    f = [3.5e9, 6e9, 10e9, 15e9, 30e9, 60e9, 80e9, 100e9]
    d = [0.002, 0.005, 0.01, 0.05, 0.15]
    iter_args = [p for p in itertools.product(N, f, d)]
    with ProcessPoolExecutor() as executor:
        _ = list(tqdm(executor.map(_worker, iter_args), total=len(iter_args)))
    elapsed = time.perf_counter() - start_time
    logger.info(f'Execution finished at {datetime.datetime.now()}')
    logger.info(f'Elapsed time: {elapsed:.4f}s')


if __name__ == '__main__':
    main()

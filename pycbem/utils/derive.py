import numpy as np
from scipy.special import binom


def holoborodko(y, dx=1):
    """Return the 1st order numerical difference on a sampled data. If
    `dx` is not given, it is assumed to be 1. This function is to be
    used when noise is present in the data. Filter length of size 5 is
    used in this implementation. For more details check:
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/

    Parameters
    ----------
    y : numpy.ndarray
        data to derive w.r.t. a single variable
    dx : float, optional
        elementwise distance

    Returns
    -------
    numpy.ndarray
        1st order numerical differentiation
    """
    N = 5
    M = (N-1) // 2
    m = (N - 3) // 2
    ck = [(1 / 2 ** (2 * m + 1) * (binom(2 * m, m - k + 1)
           - binom(2 * m, m - k - 1))) for k in range(1, M + 1)]
    if np.iscomplex(y).any():
        diff_type = 'complex_'
    else:
        diff_type = 'float'
    y_x = np.empty((y.size, ), dtype=diff_type)
    # since the filter is of length 5, the first two and the last two elements
    # have to be calculated using forward, central and backward difference
    y_x[0] = (y[1] - y[0]) / dx
    y_x[1] = (y[2] - y[0]) / (2 * dx)
    y_x[-2] = (y[-1] - y[-3]) / (2 * dx)
    y_x[-1] = (y[-1] - y[-2]) / dx
    for i in range(M, len(y) - M):
        y_x[i] = 1 / dx * sum([ck[k - 1] * (y[i + k] - y[i - k]) for k
                               in range(1, M + 1)])
    return y_x

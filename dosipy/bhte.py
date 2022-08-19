"""Bioheat transfer equation solvers."""

import numpy as np
from scipy.integrate import odeint

from .constants import pi


class BHTE(object):
    """Bio-heat transfer equation class."""

    def __init__(self, sim_time, t_res, X, s_res, h=10., Ta=37., Tc=37.,
                 Tf=25., Qm=33800., SAR=None):
        """Initialize the bio-heat transfer equation solver for skin.

        Parameters
        ----------
        T : scalar
            Simulation time in seconds.
        t_res : int
            Time resolution.
        X : tuple
            Either 1-, 2-. or 3-element tuple corresponding to either
            depth, length and width, or length, width and depth of
            solution domain in meters, respectively.
        s_res : int
            Spatial resolution.
        h : scalar, optional
            Heat convection coefficient between the skin surface and
            air in W/m^2/°C.
        T_a : scalar, optional
            Arterial temperature in degrees Celsius.
        T_c : scalar, optional
            Body core temperature in degrees Celsius.
        T_f : scalar, optional
            Surrounding air temperature in degrees Celsius.
        Q_m : scalar, optional
            Metabolic heat generation in W/m^3.
        SAR : numpy.ndarray, optional
            Specific absorption rate values in W/m3 with the shape that
            corresponds to `s_res`. If SAR is not specified,
            temperature is computed without taking external source of
            radiation into account.

        Returns
        -------
        numpy.ndarray
            Temperature distribution in time and space.
        """
        if not isinstance(sim_time, (int, float, )):
            raise ValueError('Simulation time should be a real-value number.')
        if not isinstance(t_res, (int, )) & (t_res > 0):
            raise ValueError('Time resolution must be a positive integer.')
        if isinstance(X, (tuple, list, )):
            ndim = len(X)
            if ndim not in [1, 2, 3]:
                raise ValueError('Tuple should have either 1, 2 or 3 elements.')
            for dim in X:
                if not isinstance(dim, (int, float, )) & (dim > 0):
                    raise ValueError('Dimension components must be positive scalar.')
        else:
            raise ValueError('Spatial dimensions must be given in a tuple.')
        if ndim == 1:
            self.depth, = X
        elif ndim == 2:
            self.length, self.width = X
        else:
            self.length, self.width, self.depth = X
        if not isinstance(s_res, (int, )) & (s_res > 0):
            raise ValueError('Spatial resolution must be a positive integer.')
        args = [h, Ta, Tc, Tf, Qm]
        if not all(isinstance(arg, (int, float, )) for arg in args):
            raise ValueError('Please check the values of skin-specific'
                             ' optional arguments.')
        if SAR is not None and not isinstance(SAR, (np.ndarray, )):
            raise ValueError('`SAR` should be given as a numpy.ndarray.')
        if SAR is not None and SAR.shape.count(s_res) != ndim:
            raise ValueError('`SAR` should have the number of dimensions'
                             ' corresponding to the `len(X)` and with number'
                             ' of elements per dimension corresponding to'
                             ' `s_res`.')
        if SAR is None:
            SAR = np.zeros(s_res**ndim).reshape([s_res] * ndim)
        self.sim_time = sim_time
        self.t_res = t_res
        self._create_time_domain()
        self.ndim = ndim
        self.s_res = s_res
        self._generate_lap_operator()
        self.h = h
        self.Ta = Ta
        self.Tc = Tc
        self.Tf = Tf
        self.Qm = Qm
        self.SAR = SAR
        self.k = 0.37  # thermal conductivity of skin in W/m/°C
        self.rho = 1109.  # skin density in kg/m^3
        self.C = 3391.  # specific heat of skin in Ws/kg/°C
        self.mb = 1.76e-6  # blood perfusion in m^3/kg/s = 106 mL/min/kg
        self.kb = 0.52  # thermal conductivity of blood in W/m/°C
        self.rhob = 1000.  # blood density in kg/m^3
        self.Cb = 3617.  # specific heat of blood in J/kg/°C

    def _create_time_domain(self):
        """Initialize the time domain for the solver.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Time domain.
        """
        t = np.linspace(0, self.sim_time, num=self.t_res)
        self.t = t

    def _generate_lap_operator(self):
        """Initialize the Laplace operator for the solver.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.ndim == 1:
            dz = self.depth / self.s_res
            kz = 2 * pi * np.fft.fftfreq(self.s_res, d=dz)
            self.lap = kz ** 2
        elif self.ndim == 2:
            dx = self.length / self.s_res
            dy = self.width / self.s_res
            kx = 2 * pi * np.fft.fftfreq(self.s_res, d=dx)
            ky = 2 * pi * np.fft.fftfreq(self.s_res, d=dy)
            KX, KY = np.meshgrid(kx, ky)
            lap = KX ** 2 + KY ** 2
            lapinv = np.zeros_like(lap)
            lapinv[lap != 0] = 1. / lap[lap != 0]
            DX = 1j * KX * lapinv
            DY = 1j * KY * lapinv
            self.lap = DX + DY
        else:
            dx = self.length / self.s_res
            dy = self.width / self.s_res
            dz = self.depth / self.s_res
            kx = 2 * pi * np.fft.fftfreq(self.s_res, d=dx)
            ky = 2 * pi * np.fft.fftfreq(self.s_res, d=dy)
            kz = 2 * pi * np.fft.fftfreq(self.s_res, d=dz)
            KX, KY, KZ = np.meshgrid(kx, ky, kz)
            lap = KX ** 2 + KY ** 2 + KZ ** 2
            lapinv = np.zeros_like(lap)
            lapinv[lap != 0] = 1. / lap[lap != 0]
            DX = 1j * KX * lapinv
            DY = 1j * KY * lapinv
            DZ = 1j * KZ * lapinv
            self.lap = DX + DY + DZ

    def solve(self, T0, **kwargs):
        """Solve the bio-heat transfer equation.

        Parameters
        ----------
        T0 : numpy.ndarray
            Initial conditions - temperature distribtuion over the
            spatial domain at time t = 0. The shape of the array
            should correspond to (`s_res`, `ndim`) where `ndim` is the
            number of spatial dimensions, i.e., the size of the tuple
            `X` defined in the constructor.
        **kwargs : dict, optional
            Additional keyword arguments for `scipy.integrate.odeint`
            differential equation solver.

        Returns
        -------
        None
        """
        if not isinstance(T0, (np.ndarray, )):
            raise ValueError('`T0` must be a `numpy.ndarray`.')
        if self.ndim != T0.ndim:
            raise ValueError(f'`T0` should have {self.ndim}-dimensional.')
        if ((T0.shape.count(T0.shape[0]) != self.ndim)
            | (T0.shape[0] != self.s_res)):
            raise ValueError(f'All spatial components should have {self.s_res}'
                             ' elements.')
        self.T0 = T0.ravel()
        target_shape = [self.s_res] * self.ndim
        if self.ndim == 1:
            axes = (0, )
        elif self.ndim == 2:
            axes = (0, 1)
        else:
            axes = (0, 1, 2)

        def rhs(T, t):
            T = T.reshape(*target_shape)
            T_fft = np.fft.fftn(T, axes=axes)
            lapT_fft = self.lap * T_fft
            lapT = np.fft.ifftn(lapT_fft, axes=axes)

            dTdt = (self.k * lapT
                    + self.rhob * self.rho * self.mb * self.Cb * (self.Ta - T)
                    + self.Qm
                    + self.SAR * self.rho) / (self.rho * self.C)
            return dTdt.real.ravel()
        
        T = odeint(func=rhs, y0=self.T0, t=self.t, **kwargs)
        return T.reshape(-1, *target_shape)

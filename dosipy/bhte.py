"""Bioheat transfer equation solver."""

import numpy as np
from scipy import integrate

from .constants import pi


supported_ode_solvers = ['solve_ivp', 'RK23', 'RK45', 'DOP853', 'Radau',
                         'BDF', 'LSODA']


class BHTE(object):
    """Bio-heat transfer equation class. Solution of the equation is
    carried out by using pseudo-spectral time domain method. Spatial
    gradients are evaluated in spectral domain via Fourier transform.
    Time derivatives are handled via standard ODE solvers as
    implemented in `scipy.integrate` module."""

    def __init__(self, sim_time, t_res, X, s_res, h=10., Ta=37., Tc=37.,
                 Tf=25., Qm=33800., SAR=None):
        """Initialize the bio-heat transfer equation solver for skin.

        Parameters
        ----------
        sim_time : scalar
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
        self._k = 0.37  # thermal conductivity of skin in W/m/°C
        self._rho = 1109.  # skin density in kg/m^3
        self._C = 3391.  # specific heat of skin in Ws/kg/°C
        self._mb = 1.76e-6  # blood perfusion in m^3/kg/s = 106 mL/min/kg
        self.kb = 0.52  # thermal conductivity of blood in W/m/°C
        self.rhob = 1000.  # blood density in kg/m^3
        self.Cb = 3617.  # specific heat of blood in J/kg/°C

    def __str__(self):
        addon_info = self.__repr__()
        bhte_lhs = '(rho * C) * dTdt'
        bhte_rhs = 'k * grad(y) + (rhob * mb * Cb) * (Ta - T) + Qm + rho * SAR'
        if self.ndim == 1:
            dim = 'z'
        elif self.ndim == 2:
            dim = 'x, y'
        else:
            dim = 'x, y, z'
        parameters = (f'h = {self.h}\n'
                      f'Ta = {self.Ta}\n'
                      f'Tc = {self.Tc}\n'
                      f'Tf = {self.Tf}\n'
                      f'Qm = {self.Qm}\n'
                      f'SAR({dim}) = {self.SAR}\n'
                      f'k({dim}) = {self._k}\n'
                      f'rho({dim}) = {self._rho}\n'
                      f'C({dim}) = {self._C}\n'
                      f'mb({dim}) = {self._mb}\n'
                      f'kb = {self.kb}\n'
                      f'rhob = {self.rhob}\n'
                      f'Cb = {self.Cb}\n')
        output = ('repr\n----\n' + addon_info + '\n\n'
                  + 'equation\n--------\n' + bhte_lhs + ' = ' + bhte_rhs + '\n\n' 
                  + 'parameters\n----------\n' + parameters)
        return output

    @property
    def k(self):
        """Get the thermal conductivity of the skin.
        
        Parameters
        ----------
        None

        Returns
        -------
        number or numpy.ndarray
            Thermal conductivity of the skin in W/m/°C.
        """
        return self._k

    @k.setter
    def k(self, value):
        """Change predefined thermal conductivity of the skin.
        
        Parameters
        ----------
        k_value : number or numpy.ndarray
            Thermal conductivity of the skin in W/m/°C.

        Returns
        -------
        None
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise ValueError('Thermal conductivity must be numerical value(s).')
        if (isinstance(value, (np.ndarray, ))
            and (value.shape.count(self.s_res) != self.ndim)):
            raise ValueError('`k` should have the number of dimensions'
                             ' corresponding to the `len(X)` and with number'
                             ' of elements per dimension corresponding to'
                             ' `s_res`.')
        self._k = value

    @property
    def rho(self):
        """Get the skin density.
        
        Parameters
        ----------
        None

        Returns
        -------
        number of numpy.ndarray
            Skin density in kg/m^3.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """Change predefined skin density.
        
        Parameters
        ----------
        value : number or numpy.ndarray
            Skin density in kg/m^3.

        Returns
        -------
        None
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise ValueError('Thermal conductivity must be numerical value(s).')
        if (isinstance(value, (np.ndarray, ))
            and (value.shape.count(self.s_res) != self.ndim)):
            raise ValueError('`k` should have the number of dimensions'
                             ' corresponding to the `len(X)` and with number'
                             ' of elements per dimension corresponding to'
                             ' `s_res`.')
        self._rho = value

    @property
    def C(self):
        """Get the specific heat of skin.
        
        Parameters
        ----------
        None

        Returns
        -------
        number or numpy.ndarray
            Specific heat of skin in Ws/kg/°C.
        """
        return self._C

    @C.setter
    def C(self, value):
        """Change predefined specific heat of skin.
        
        Parameters
        ----------
        value : number or numpy.ndarray
            Specific heat of skin in Ws/kg/°C.

        Returns
        -------
        None
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise ValueError('Thermal conductivity must be numerical value(s).')
        if (isinstance(value, (np.ndarray, ))
            and (value.shape.count(self.s_res) != self.ndim)):
            raise ValueError('`k` should have the number of dimensions'
                             ' corresponding to the `len(X)` and with number'
                             ' of elements per dimension corresponding to'
                             ' `s_res`.')
        self._C = value
    
    @property
    def mb(self):
        """Get the blood perfusion.
        
        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Volumetric blood perfusion in m^3/kg/s.
        """
        return self._mb

    @mb.setter
    def mb(self, value):
        """Change predefined blood perfusion.
        
        Parameters
        ----------
        value : number or numpy.ndarray
            Volumetric blood perfusion in m^3/kg/s.

        Returns
        -------
        None
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise ValueError('Thermal conductivity must be numerical value(s).')
        if (isinstance(value, (np.ndarray, ))
            and (value.shape.count(self.s_res) != self.ndim)):
            raise ValueError('`k` should have the number of dimensions'
                             ' corresponding to the `len(X)` and with number'
                             ' of elements per dimension corresponding to'
                             ' `s_res`.')
        self._mb = value

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
            self.lap = - kz ** 2
        elif self.ndim == 2:
            dx = self.length / self.s_res
            dy = self.width / self.s_res
            kx = 2 * pi * np.fft.fftfreq(self.s_res, d=dx)
            ky = 2 * pi * np.fft.fftfreq(self.s_res, d=dy)
            KX, KY = np.meshgrid(kx, ky)
            lap = -(KX ** 2 + KY ** 2)
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
            lap = -(KX ** 2 + KY ** 2 + KZ ** 2)
            lapinv = np.zeros_like(lap)
            lapinv[lap != 0] = 1. / lap[lap != 0]
            DX = 1j * KX * lapinv
            DY = 1j * KY * lapinv
            DZ = 1j * KZ * lapinv
            self.lap = DX + DY + DZ

    def solve(self, T0, solver='legacy', **kwargs):
        """Solve the bio-heat transfer equation.

        Parameters
        ----------
        T0 : number or numpy.ndarray
            Initial conditions - temperature distribtuion over the
            spatial domain at time t = 0. The shape of the array
            should correspond to (`s_res`, `ndim`) where `ndim` is the
            number of spatial dimensions, i.e., the size of the tuple
            `X` defined in the constructor. If initial conditions are
            set as a scalar value, it is assumed the entire spatial
            domain is at the same temperature.
        solver : string, optional
            Type of initial value problem solver for ODE systems
            provided by `scipy.integrate` module.
        kwargs : dict, optional
            Additional keyword arguments for the initial value problem
            ODE systems solver.

        Returns
        -------
        numpy.ndarray
            Temperature distribution in time (first axis) and space
            (remaining axes).
        """
        if isinstance(T0, (np.ndarray, )):
            if self.ndim != T0.ndim:
                raise ValueError(f'`T0` should have {self.ndim}-dimensional.')
            if ((T0.shape.count(T0.shape[0]) != self.ndim)
                | (T0.shape[0] != self.s_res)):
                raise ValueError('All spatial components should have'
                                 f'{self.s_res} elements.')
        elif isinstance(T0, (int, float, )):
            T0 = np.ones_like(self.SAR) * T0
        else:
            raise ValueError('`T0` must be either a number or an array.')
        if (solver != 'legacy') and (solver not in supported_ode_solvers):
            print(f'Solver {solver} is not supported. Falling back to default.')
            solver = 'legacy'
            solver_fn = integrate.odeint
        elif solver == 'legacy':
            solver_fn = integrate.odeint
        else:
            solver_fn = integrate.solve_ivp
        self.T0 = T0
        target_shape = [self.s_res] * self.ndim
        if self.ndim == 1:
            axes = (0, )
        elif self.ndim == 2:
            axes = (0, 1)
        else:
            axes = (0, 1, 2)

        if solver == 'legacy':
            def rhs(y, t):
                y = y.reshape(*target_shape)
                y_fft = np.fft.fftn(y, axes=axes)
                lap_y_fft = self.lap * y_fft
                lap_y = np.fft.ifftn(lap_y_fft, axes=axes).real
                dydt = (self._k * lap_y
                        + self.rhob * self._mb * self.Cb * (self.Ta - y)
                        + self.Qm
                        + self.SAR * self._rho) / (self._rho * self._C)
                return dydt.ravel()
            
            sol = solver_fn(rhs, y0=self.T0.ravel(), t=self.t, **kwargs)
            return sol.reshape(-1, *target_shape)
        else:
            def rhs(t, y):
                y = y.reshape(*target_shape)
                y_fft = np.fft.fftn(y, axes=axes)
                lap_y_fft = self.lap * y_fft
                lap_y = np.fft.ifftn(lap_y_fft, axes=axes).real
                dydt = (self._k * lap_y
                        + self.rhob * self._mb * self.Cb * (self.Ta - y)
                        + self.Qm
                        + self.SAR * self._rho) / (self._rho * self._C)
                return dydt.ravel()

            sol = solver_fn(rhs, y0=self.T0.ravel(),
                            t_span=(self.t[0], self.t[-1]), t_eval=self.t,
                            method=solver, **kwargs)
            return np.moveaxis(sol.y.reshape(*target_shape, -1), -1, 0)

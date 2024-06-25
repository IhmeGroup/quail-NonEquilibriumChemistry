# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/base/transport.py
#
#       Contains functions for computing transport properties.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
from quail.general import ThermoType
from quail.external.optional_cantera import ct
from quail.external.optional_mutationpp import mpp


class TransportBase(ABC):
    '''
    This is a base class for transport property calculations.
    '''
    def __init__(self, **kwargs):
        self.init_params = kwargs
        pass

    def reinitialize(self):
        self.__init__(**self.init_params)
        pass

    def get_viscosity(self):
        pass

    def get_thermal_conductivity(self):
        pass

    def get_diffusion_coefficients(self):
        pass

class ConstantTransport(TransportBase):
    '''
    Returns viscosity and thermal conductivity for constant transport
    properties.

    Inputs:
    -------
        physics: physics object
        Uq: solution state evaluated at quadrature points [ne, nq, ns]
        flag_non_physical: boolean to check for physicality of transport

    Attributes:
    --------
        mu: viscosity [ne, nq]
        kappa: thermal conductivity [ne, nq]
    '''
    def __init__(self, PrandtlNumber=0.7, Viscosity=1.0, **kwargs):
        super().__init__(**kwargs)
        self.Pr = PrandtlNumber
        self.mu0 = Viscosity

    def get_viscosity(self, thermo):
        return self.mu0

    def get_thermal_conductivity(self, thermo):
        mu = self.get_viscosity(thermo)

        return mu * thermo.cp / self.Pr

    def get_diffusion_coefficients(self, thermo):
        return np.zeros(thermo.Y.shape)


class SutherlandTransport(TransportBase):
    '''
    Calculates the viscosity and thermal conductivity using Sutherland's
    law.

    Inputs:
    -------
        physics: physics object
        Uq: solution state evaluated at quadrature points [ne, nq, ns]
        flag_non_physical: boolean to check for physicality of transport

    Attributes:
    --------
        mu: viscosity [ne, nq]
        kappa: thermal conductivity [ne, nq]
    '''
    def __init__(self, PrandtlNumber=0.7, Viscosity=1.0, s=1.0, T0=1.0, beta=1., **kwargs):
        super().__init__(**kwargs)
        self.Pr = PrandtlNumber
        self.mu0 = Viscosity
        self.s = s
        self.T0 = T0
        self.beta = beta

    def get_viscosity(self, thermo):
        T = thermo.T

        return (self.mu0 * (T / self.T0)**self.beta *
                ((self.T0 + self.s) / (T + self.s)))

    def get_thermal_conductivity(self, thermo):
        mu = self.get_viscosity(thermo)

        return mu * thermo.cp / self.Pr

    def get_diffusion_coefficients(self, thermo):
        return np.zeros(thermo.Y.shape)


class CanteraTransport(TransportBase):
    '''
    Interface to Cantera to compute transport properties.
    '''
    def __init__(self, Mechanism='air.yaml', ThermoModel='NotNeeded', **kwargs):
        super().__init__(**kwargs)
        if ThermoModel != 'Cantera':
            self.gas = ct.Solution(Mechanism)
        self._cached_T = None
        self._cached_solution = None

    def get_solution(self, thermo):
        # If using Cantera thermodynamics, then the solution is
        # already available:
        if thermo.THERMO_TYPE == ThermoType.Cantera:
            return thermo.solution

        # Otherwise we must create it, but first check if we have
        # already done so by comparing a cached copy of the energy:
        if self._cached_T is thermo.T:
            return self._cached_solution

        # Otherwise create a fresh solution and cache it:
        solution = ct.Solution(self.gas, thermo.T.shape)
        solution.TDY = thermo.T, thermo.rho, thermo.Y

        self._cached_T = thermo.T
        self._cached_solution = solution

        return solution

    def get_viscosity(self, thermo):
        solution = self.get_solution(thermo)

        return np.atleast_3d(solution.viscosity)

    def get_thermal_conductivity(self, thermo):
        solution = self.get_solution(thermo)

        return np.atleast_3d(solution.thermal_conductivity)

    def get_diffusion_coefficients(self, thermo):
        solution = self.get_solution(thermo)

        return solution.mix_diff_coeffs
